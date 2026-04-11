"""
AutoStream AI Agent — LangGraph State Machine.

This module wires up the full conversational agent using LangGraph:

  ┌──────────┐
  │  START   │
  └────┬─────┘
       ▼
  ┌──────────┐     info / greeting
  │ classify │ ──────────────────────► respond_info ──► END
  └────┬─────┘
       │ signup
       ▼
  ┌──────────────┐   missing fields
  │ collect_lead │ ◄─────────────────┐
  └──────┬───────┘                   │
         │ all fields present        │
         ▼                           │
  ┌──────────────┐                   │
  │  call_tool   │ (mock_lead_capture)
  └──────┬───────┘                   │
         ▼                           │
       END                           │
         ▲                           │
         └───── user replies ────────┘

Nodes:
  1. classify_intent  — LLM decides: greeting / info / signup
  2. respond_info     — LLM answers using RAG context
  3. collect_lead     — Checks state for name/email/platform, asks for missing
  4. call_tool        — Fires mock_lead_capture once all fields are present
"""

import os
import re
from typing import Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.tools import mock_lead_capture
from src.rag_engine import get_full_context, query_knowledge_base

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
_REQUESTED_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"
_MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5flash": "gemini-2.5-flash",
    "models/gemini-1.5-flash": "gemini-2.5-flash",
}
_GEMINI_MODEL = _MODEL_ALIASES.get(_REQUESTED_GEMINI_MODEL.lower(), _REQUESTED_GEMINI_MODEL)
_GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-flash-latest,gemini-2.0-flash").split(",")
    if model.strip()
]

_MODEL_CANDIDATES: list[str] = []
for model in [_GEMINI_MODEL, *_GEMINI_FALLBACK_MODELS]:
    if model not in _MODEL_CANDIDATES:
        _MODEL_CANDIDATES.append(model)

_ACTIVE_GEMINI_MODEL = _MODEL_CANDIDATES[0]

_SIGNUP_MARKERS = (
    "i want to sign up",
    "i'd like to sign up",
    "help me sign up",
    "let me sign up",
    "ready to sign up",
    "i want to try",
    "want to try",
    "try the pro plan",
    "try pro",
    "i want the pro plan",
    "i'll take the pro plan",
    "get started",
    "start a trial",
    "start trial",
    "book a demo",
    "schedule a demo",
    "subscribe",
    "buy now",
)

_INFO_MARKERS = (
    "price",
    "pricing",
    "cost",
    "refund",
    "cancel",
    "policy",
    "support",
    "faq",
    "plan",
    "feature",
    "trial",
)

_GREETING_MARKERS = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening")

_PLATFORMS = ("youtube", "tiktok", "instagram", "facebook", "linkedin", "x", "twitter", "twitch")

def _get_llm(temperature: float = 0.3):
    """Return a ChatGoogleGenerativeAI instance bound with the lead-capture tool."""
    return ChatGoogleGenerativeAI(
        model=_ACTIVE_GEMINI_MODEL,
        google_api_key=_GEMINI_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


def _is_retryable_model_error(exc: ChatGoogleGenerativeAIError) -> bool:
    message = str(exc).upper()
    return (
        ("RESOURCE_EXHAUSTED" in message)
        or ("429" in message)
        or ("NOT_FOUND" in message and "MODEL" in message)
        or ("UNAVAILABLE" in message)
        or ("503" in message)
    )


def _invoke_with_model_failover(messages: list[Any], temperature: float) -> Any:
    """Invoke Gemini and fail over to alternative models on quota/model errors."""
    global _ACTIVE_GEMINI_MODEL

    ordered_models = [_ACTIVE_GEMINI_MODEL, *[m for m in _MODEL_CANDIDATES if m != _ACTIVE_GEMINI_MODEL]]
    last_chat_error: ChatGoogleGenerativeAIError | None = None

    for model in ordered_models:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_GEMINI_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
        try:
            model_response_payload = llm.invoke(messages)
            _ACTIVE_GEMINI_MODEL = model
            return model_response_payload
        except ChatGoogleGenerativeAIError as exc:
            last_chat_error = exc
            if not _is_retryable_model_error(exc):
                raise
            continue

    if last_chat_error is not None:
        raise last_chat_error
    raise RuntimeError("No Gemini model candidates configured")


# ---------------------------------------------------------------------------
# System prompt (injected with RAG context)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are the **AutoStream AI Assistant**. Your goal is to help creators automate their video editing.

## Guidelines

1. **Identify Intent** — Determine if the user is:
   - Just saying hello or making small talk → respond warmly.
   - Asking about pricing, policies, features, or FAQs → use the **Knowledge Base** below to answer accurately.
   - Expressing interest in signing up or getting started (High Intent) → switch to **lead capture** mode.

2. **Knowledge Retrieval** — For pricing and policy questions use ONLY the context provided below. If the information is not in the knowledge base, say you don't have that information and suggest contacting support.

3. **Lead Capture Logic** — If the user shows high intent (wants to sign up, start a trial, get a demo, etc.), you MUST collect **all three** of the following before calling the `mock_lead_capture` tool:
   - **Full Name**
   - **Email Address**
   - **Creator Platform** (e.g., YouTube, TikTok, Instagram)

   Ask for them politely, ONE at a time if the user hasn't provided them.
   DO NOT call mock_lead_capture until you have ALL THREE.
   Once you have all three, call the tool immediately.

4. **Tone** — Professional, helpful, concise, and enthusiastic about helping creators.

5. **Boundaries** — You are ONLY an AutoStream assistant. Politely decline unrelated questions.

---

## Knowledge Base

{context}
"""


def _build_system_message() -> SystemMessage:
    context = get_full_context()
    return SystemMessage(content=_SYSTEM_PROMPT.format(context=context))


def _extract_string_from_llm_payload(content: Any) -> str:
    """Normalize model content that may be either plain text or structured blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        raw_message_string = content.get("text")
        if isinstance(raw_message_string, str):
            return raw_message_string
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                raw_message_string = item.get("text")
                if isinstance(raw_message_string, str):
                    parts.append(raw_message_string)
                    continue
            raw_message_string_attr = getattr(item, "text", None)
            if isinstance(raw_message_string_attr, str):
                parts.append(raw_message_string_attr)
        return "\n".join(parts).strip()
    return str(content)


def _last_user_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            return _extract_string_from_llm_payload(getattr(msg, "content", "")).strip()
    return ""


def _fast_intent_from_text(user_text: str) -> str | None:
    text = user_text.lower()
    if not text:
        return None
    if any(marker in text for marker in _SIGNUP_MARKERS):
        return "signup"
    if any(marker in text for marker in _INFO_MARKERS):
        return "info"
    if any(marker in text for marker in _GREETING_MARKERS) and len(text.split()) <= 12:
        return "greeting"
    return None


def format_detected_intent(intent: str | None, source: str | None = None) -> str:
    """Return a human-friendly label for the detected intent."""
    if not intent:
        return "Unknown"

    label_map = {
        "greeting": "Casual Greeting",
        "info": "Product/Pricing Inquiry",
        "signup": "High-Intent Lead",
    }
    source_map = {
        "rule_based": "rule-based",
        "llm": "LLM",
        "lead_progress": "lead-progress",
    }

    label = label_map.get(intent, intent.title())
    if source:
        return f"{label} ({source_map.get(source, source)})"
    return label


def _lead_in_progress(state: AgentState) -> bool:
    """Return True when signup has started but lead capture is not complete yet."""
    if state.get("lead_captured"):
        return False

    has_any = bool(state.get("user_name") or state.get("user_email") or state.get("user_platform"))
    has_all = bool(state.get("user_name") and state.get("user_email") and state.get("user_platform"))
    if has_any and not has_all:
        return True

    return state.get("intent") == "signup" and not has_all


def _extract_email(text: str) -> str | None:
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group(0) if match else None


def _extract_platform(text: str) -> str | None:
    lowered = text.lower()
    for platform in _PLATFORMS:
        if platform in lowered:
            return platform.capitalize() if platform != "x" else "X"
    return None


def _extract_name(text: str) -> str | None:
    lowered = text.lower()
    patterns = (
        r"(?:my\s+full\s+name\s+is)\s+([A-Za-z][A-Za-z\s.'-]{1,60})",
        r"(?:my name is|i am|i'm|this is)\s+([A-Za-z][A-Za-z\s.'-]{1,60})",
        r"name\s*[:\-]\s*([A-Za-z][A-Za-z\s.'-]{1,60})",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip(" .,!?")
            return " ".join(part.capitalize() for part in raw.split())
    # If user replies with two-ish words and no obvious metadata, treat as name.
    if "@" not in text and len(text.split()) in (2, 3) and text.replace(" ", "").isalpha():
        return " ".join(part.capitalize() for part in text.split())
    return None


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> dict:
    """
    Use a lightweight LLM call to classify the latest user message
    into one of: greeting, info, signup.
    """
    # Keep lead capture sticky once signup has begun.
    if _lead_in_progress(state):
        return {"intent": "signup", "intent_source": "lead_progress"}

    fast_intent = _fast_intent_from_text(_last_user_text(state))
    if fast_intent:
        return {"intent": fast_intent, "intent_source": "rule_based"}

    classification_prompt = [
        SystemMessage(
            content=(
                "You are an intent classifier. Given the user message AND the "
                "conversation history, reply with EXACTLY one word: "
                "'greeting', 'info', or 'signup'. "
                "'signup' means the user wants to sign up, start a trial, get started, "
                "get a demo, or buy a plan. "
                "'info' means they are asking about pricing, features, policies, or FAQs. "
                "'greeting' means casual hello or small-talk."
            )
        ),
        *state["messages"],
    ]

    model_response_payload = _invoke_with_model_failover(classification_prompt, temperature=0.0)
    intent = _extract_string_from_llm_payload(model_response_payload.content).strip().lower()

    # Normalise edge cases
    if "signup" in intent or "sign" in intent:
        intent = "signup"
    elif "info" in intent:
        intent = "info"
    else:
        intent = "greeting"

    return {"intent": intent, "intent_source": "llm"}


def respond_info(state: AgentState) -> dict:
    """Handle greetings and informational questions using the full RAG context."""
    recent_user_message = _last_user_text(state)
    latest_lower = recent_user_message.lower()

    # Fast-path local answers for common FAQ/pricing/policy requests.
    if any(marker in latest_lower for marker in _INFO_MARKERS):
        return {"messages": [AIMessage(content=query_knowledge_base(recent_user_message))]}

    if any(marker in latest_lower for marker in _GREETING_MARKERS) and len(latest_lower.split()) <= 12:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Hello! I can help with AutoStream pricing, policies, and getting started. "
                        "What would you like to know?"
                    )
                )
            ]
        }

    messages = [_build_system_message(), *state["messages"]]
    model_response_payload = _invoke_with_model_failover(messages, temperature=0.4)
    return {"messages": [AIMessage(content=_extract_string_from_llm_payload(model_response_payload.content))]}


def collect_lead(state: AgentState) -> dict:
    """
    Check which lead fields are still missing and ask for the next one.
    If all fields are present, let the router send us to call_tool.
    """
    name = state.get("user_name")
    email = state.get("user_email")
    platform = state.get("user_platform")

    state_mutations: dict = {}
    recent_user_message = _last_user_text(state)

    if not name:
        name_val = _extract_name(recent_user_message)
        if name_val:
            state_mutations["user_name"] = name_val
            name = name_val

    if not email:
        email_val = _extract_email(recent_user_message)
        if email_val:
            state_mutations["user_email"] = email_val
            email = email_val

    if not platform:
        platform_val = _extract_platform(recent_user_message)
        if platform_val:
            state_mutations["user_platform"] = platform_val
            platform = platform_val

    if name and email and platform:
        state_mutations["user_name"] = name
        state_mutations["user_email"] = email
        state_mutations["user_platform"] = platform
        return state_mutations

    if not name:
        ask = "That's great to hear you're interested! To get you set up, could you share your **full name**?"
    elif not email:
        ask = f"Thanks, {name}! What's the best **email address** to reach you at?"
    else:
        ask = f"Almost there! Which **creator platform** do you primarily use? (e.g., YouTube, TikTok, Instagram)"

    state_mutations["messages"] = [AIMessage(content=ask)]
    return state_mutations


def call_tool(state: AgentState) -> dict:
    """Invoke mock_lead_capture with the collected fields."""
    name = state["user_name"]
    email = state["user_email"]
    platform = state["user_platform"]

    result = mock_lead_capture.invoke({"name": name, "email": email, "platform": platform})

    confirmation = (
        f"Awesome — you're all set!\n\n{result}\n\n"
        "Is there anything else I can help you with?"
    )
    return {
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
    }


# ---------------------------------------------------------------------------
# Router functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> str:
    intent = state.get("intent", "greeting")
    if intent == "signup":
        return "collect_lead"
    return "respond_info"


def route_after_collect(state: AgentState) -> str:
    """If all three fields are collected, go to tool call; otherwise end
    (we've already asked the user for the next field)."""
    if state.get("user_name") and state.get("user_email") and state.get("user_platform"):
        return "call_tool"
    return END


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------

def build_agent_graph():
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond_info", respond_info)
    graph.add_node("collect_lead", collect_lead)
    graph.add_node("call_tool", call_tool)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional edge after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "respond_info": "respond_info",
            "collect_lead": "collect_lead",
        },
    )

    # respond_info always ends the turn
    graph.add_edge("respond_info", END)

    # After collecting lead info, either call tool or end (wait for user)
    graph.add_conditional_edges(
        "collect_lead",
        route_after_collect,
        {
            "call_tool": "call_tool",
            END: END,
        },
    )

    # After calling the tool, end the turn
    graph.add_edge("call_tool", END)

    return graph.compile()
