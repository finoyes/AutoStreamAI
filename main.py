
import sys
import re
from typing import cast
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import AIMessage, HumanMessage
from src.agent import build_agent_graph, format_detected_intent
from src.rag_engine import query_knowledge_base
from src.state import AgentState
from src.tools import mock_lead_capture

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





def _offline_fallback_reply(raw_user_string: str, state: AgentState) -> str:
    """Return a local response when online model calls are unavailable."""
    q = raw_user_string.lower()

    def is_signup_intent_text(text: str) -> bool:
        lowered = text.lower()
        explicit_markers = (
            "sign up",
            "signup",
            "get started",
            "start a trial",
            "start trial",
            "book a demo",
            "schedule a demo",
            "subscribe",
            "buy now",
            "i want pro plan",
            "i want the pro plan",
            "try pro",
        )
        if any(marker in lowered for marker in explicit_markers):
            return True
        return bool(
            re.search(
                r"\b(i\s*(want|would\s+like|choose|need)|i[' ]?ll\s+take|go\s+with)\s+(the\s+)?(pro|basic)\s+plan\b",
                lowered,
            )
        )

    def extract_email(text: str) -> str | None:
        match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
        return match.group(0) if match else None

    def extract_platform(text: str) -> str | None:
        lowered = text.lower()
        if re.search(r"\byoutube\b", lowered):
            return "Youtube"
        if re.search(r"\btiktok\b", lowered):
            return "Tiktok"
        if re.search(r"\binstagram\b", lowered):
            return "Instagram"
        if re.search(r"\bfacebook\b", lowered):
            return "Facebook"
        if re.search(r"\blinkedin\b", lowered):
            return "Linkedin"
        if re.search(r"\btwitch\b", lowered):
            return "Twitch"
        if re.search(r"\b(?:x|twitter)\b", lowered):
            return "X"
        return None

    def extract_name(text: str) -> str | None:
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
        if "@" not in text and len(text.split()) in (2, 3) and text.replace(" ", "").isalpha():
            return " ".join(part.capitalize() for part in text.split())
        return None

    # Use local JSON knowledge for pricing/policy/FAQ questions.
    info_markers = ("price", "pricing", "cost", "refund", "cancel", "policy", "support", "faq", "plan", "trial")

    if not state.get("user_name"):
        extracted_name = extract_name(raw_user_string)
        if extracted_name:
            state["user_name"] = extracted_name

    if not state.get("user_email"):
        extracted_email = extract_email(raw_user_string)
        if extracted_email:
            state["user_email"] = extracted_email

    if not state.get("user_platform"):
        extracted_platform = extract_platform(raw_user_string)
        if extracted_platform:
            state["user_platform"] = extracted_platform

    has_any = bool(state.get("user_name") or state.get("user_email") or state.get("user_platform"))
    has_all = bool(state.get("user_name") and state.get("user_email") and state.get("user_platform"))
    lead_in_progress = has_any and not has_all

    if is_signup_intent_text(q) or lead_in_progress or (has_all and not state.get("lead_captured")):
        state["intent"] = "signup"
        state["intent_source"] = "offline_rule"

        if not state.get("user_name"):
            return "I am temporarily in offline mode because the online model is unavailable. To continue signup, please share your full name."
        if not state.get("user_email"):
            return f"Thanks, {state['user_name']}! I am temporarily in offline mode because the online model is unavailable. Please share your email address next."
        if not state.get("user_platform"):
            return "I am temporarily in offline mode because the online model is unavailable. Which creator platform do you use (YouTube, TikTok, Instagram)?"

        if not state.get("lead_captured"):
            result = mock_lead_capture.invoke(
                {
                    "name": state["user_name"],
                    "email": state["user_email"],
                    "platform": state["user_platform"],
                }
            )
            state["lead_captured"] = True
            return (
                "Gemini is unavailable right now, but I completed your signup intake in offline mode.\n\n"
                f"{result}"
            )

        return "Your signup details are already captured. I can still answer pricing/policy questions while Gemini is unavailable."

    if any(w in q for w in info_markers):
        kb_answer = query_knowledge_base(raw_user_string)
        return (
            "I am temporarily in offline mode because the online model is unavailable. "
            "Here is the answer from the local knowledge base:\n\n"
            f"{kb_answer}"
        )

    return (
        "I cannot call Gemini right now because the online model is unavailable. "
        "You can still ask pricing or policy questions and I will answer from the local knowledge base."
    )


def main():
    try:
        print(BANNER)
    except UnicodeEncodeError:
        print(BANNER.encode("ascii", errors="ignore").decode("ascii"))

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
        "intent_source": None,
    }

    while True:
        try:
            raw_user_string = input(f"{GREEN}{BOLD}You ▶  {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not raw_user_string:
            continue
        if raw_user_string.lower() in ("quit", "exit", "q"):
            print(f"\n{DIM}Thanks for chatting — see you next time!{RESET}")
            break

        # Append the user message
        state["messages"] = list(state["messages"]) + [HumanMessage(content=raw_user_string)]

        # Run the graph
        print(f"\n{DIM}AutoStream is thinking...{RESET}", flush=True)
        try:
            updated_state = cast(AgentState, agent.invoke(state))
        except KeyboardInterrupt:
            state["messages"] = list(state["messages"][:-1])
            print(f"\n{YELLOW}Request interrupted. Please try again.{RESET}\n")
            continue
        except ChatGoogleGenerativeAIError as exc:
            err_str = str(exc).upper()
            is_offline_condition = (
                "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "QUOTA" in err_str or
                "NOT_FOUND" in err_str or
                "UNAVAILABLE" in err_str or "503" in err_str
            )
            
            if is_offline_condition:
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    retry_hint = ""
                    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
                    if match:
                        retry_hint = f" Retry in about {int(float(match.group(1)))} seconds."
                    print(f"\n{YELLOW}Gemini quota/rate limit reached.{RESET}\n{DIM}Please wait and try again.{retry_hint}{RESET}")
                elif "NOT_FOUND" in err_str:
                    print(f"\n{YELLOW}Configured Gemini model is not available.{RESET}\n{DIM}Check API key and model name.{RESET}")
                else:
                    print(f"\n{YELLOW}Model API temporarily unavailable.{RESET}\n{DIM}{str(exc)}{RESET}")

                fallback_text = _offline_fallback_reply(raw_user_string, state)
                state["messages"] = list(state["messages"]) + [AIMessage(content=fallback_text)]
                print(f"\n{CYAN}{BOLD}AutoStream ▶  {RESET}{fallback_text}\n")
                continue

            # Roll back the user message so retries do not duplicate turns.
            state["messages"] = list(state["messages"][:-1])
            print(f"\n{YELLOW}The model request failed.{RESET}\n{DIM}{str(exc)}{RESET}\n")
            continue

        # Replace with returned state snapshot
        state = updated_state

        intent_badge = format_detected_intent(state.get("intent"), state.get("intent_source"))

        # Print the last AI message
        assistant_messages = [m for m in updated_state.get("messages", []) if hasattr(m, "content") and m.type == "ai"]
        if assistant_messages:
            assistant_reply_text = assistant_messages[-1].content
            print(f"\n{DIM}Detected intent: {intent_badge}{RESET}")
            print(f"\n{CYAN}{BOLD}AutoStream ▶  {RESET}{assistant_reply_text}\n")
        else:
            print(f"\n{YELLOW}(No response generated — please try again.){RESET}\n")


if __name__ == "__main__":
    main()
