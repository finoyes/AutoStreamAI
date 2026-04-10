"""
AutoStream AI Agent — CLI Demo Entry Point.

Run:
    python main.py

Type 'quit' or 'exit' to end the conversation.
"""

import sys
import re
from typing import cast
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import AIMessage, HumanMessage
from src.agent import build_agent_graph
from src.rag_engine import query_knowledge_base
from src.state import AgentState

# ANSI colour helpers (works on Windows 10+ and all *nix terminals)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════╗
║            AutoStream AI Assistant               ║
╠══════════════════════════════════════════════════╣
║  I can help you with:                            ║
║    • Pricing & plan comparisons                  ║
║    • Policies (refunds, cancellation, trials)    ║
║    • Getting started / signing up                ║
║                                                  ║
║  Type 'quit' or 'exit' to leave.                 ║
╚══════════════════════════════════════════════════╝{RESET}
"""


def _safe_print(text: str) -> None:
    """Print safely on terminals that do not support full Unicode output."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="ignore").decode("ascii"))


def _print_llm_error(exc: Exception) -> None:
    """Render a friendly, actionable message for model invocation failures."""
    message = str(exc)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        retry_hint = ""
        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
        if match:
            retry_hint = f" Retry in about {int(float(match.group(1)))} seconds."
        print(
            f"\n{YELLOW}Gemini quota/rate limit reached.{RESET}"
            f"\n{DIM}Please wait and try again.{retry_hint}{RESET}"
            f"\n{DIM}Usage: https://ai.dev/rate-limit{RESET}\n"
        )
        return

    if "NOT_FOUND" in message and "model" in message.lower():
        print(
            f"\n{YELLOW}Configured Gemini model is not available for this API key/project.{RESET}"
            f"\n{DIM}Use one of: gemini-flash-latest, gemini-2.0-flash, gemini-2.5-flash.{RESET}\n"
        )
        return

    print(
        f"\n{YELLOW}The model request failed.{RESET}"
        f"\n{DIM}{message}{RESET}\n"
    )


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message or "QUOTA" in message


def _is_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "NOT_FOUND" in message and "MODEL" in message


def _is_service_unavailable_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "UNAVAILABLE" in message or "503" in message


def _offline_fallback_reply(user_input: str, state: AgentState) -> str:
    """Return a local response when online model calls are unavailable."""
    q = user_input.lower()

    # Explicit high-intent signup phrases only.
    signup_markers = (
        "sign up",
        "signup",
        "get started",
        "start a trial",
        "start trial",
        "book a demo",
        "schedule a demo",
        "subscribe",
        "buy now",
    )

    # Use local JSON knowledge for pricing/policy/FAQ questions.
    info_markers = ("price", "pricing", "cost", "refund", "cancel", "policy", "support", "faq", "plan", "trial")

    if any(w in q for w in info_markers):
        kb_answer = query_knowledge_base(user_input)
        return (
            "I am temporarily in offline mode because the online model is unavailable. "
            "Here is the answer from the local knowledge base:\n\n"
            f"{kb_answer}"
        )

    # Keep lead capture usable even while the model is unavailable.
    if any(w in q for w in signup_markers):
        if not state.get("user_name"):
            return "I am temporarily in offline mode because the online model is unavailable. To continue signup, please share your full name."
        if not state.get("user_email"):
            return "I am temporarily in offline mode because the online model is unavailable. Please share your email address next."
        if not state.get("user_platform"):
            return "I am temporarily in offline mode because the online model is unavailable. Which creator platform do you use (YouTube, TikTok, Instagram)?"
        return "Thanks. I have your signup details and can continue once API quota is available again."

    return (
        "I cannot call Gemini right now because the online model is unavailable. "
        "You can still ask pricing or policy questions and I will answer from the local knowledge base."
    )


def main():
    _safe_print(BANNER)

    # Compile the graph once
    agent = build_agent_graph()

    # Persistent state across turns (enables multi-turn lead collection)
    state: AgentState = {
        "messages": [],
        "user_name": None,
        "user_email": None,
        "user_platform": None,
        "lead_captured": False,
        "intent": None,
    }

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You ▶  {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n{DIM}Thanks for chatting — see you next time!{RESET}")
            break

        # Append the user message
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        # Run the graph
        print(f"\n{DIM}AutoStream is thinking...{RESET}", flush=True)
        try:
            result = cast(AgentState, agent.invoke(state))
        except KeyboardInterrupt:
            state["messages"] = list(state["messages"][:-1])
            print(f"\n{YELLOW}Request interrupted. Please try again.{RESET}\n")
            continue
        except ChatGoogleGenerativeAIError as exc:
            if _is_quota_error(exc) or _is_model_not_found_error(exc) or _is_service_unavailable_error(exc):
                _print_llm_error(exc)
                fallback = _offline_fallback_reply(user_input, state)
                state["messages"] = list(state["messages"]) + [AIMessage(content=fallback)]
                print(f"\n{CYAN}{BOLD}AutoStream ▶  {RESET}{fallback}\n")
                continue

            # Roll back the user message so retries do not duplicate turns.
            state["messages"] = list(state["messages"][:-1])
            _print_llm_error(exc)
            continue
        except Exception as exc:
            state["messages"] = list(state["messages"][:-1])
            _print_llm_error(exc)
            continue

        # Replace with returned state snapshot
        state = result

        # Print the last AI message
        ai_messages = [m for m in result.get("messages", []) if hasattr(m, "content") and m.type == "ai"]
        if ai_messages:
            last_reply = ai_messages[-1].content
            print(f"\n{CYAN}{BOLD}AutoStream ▶  {RESET}{last_reply}\n")
        else:
            # Fallback — shouldn't normally happen
            print(f"\n{YELLOW}(No response generated — please try again.){RESET}\n")


if __name__ == "__main__":
    main()
