
import sys
import re
from typing import cast
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import AIMessage, HumanMessage
from src.agent import build_agent_graph, format_detected_intent
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





def _offline_fallback_reply(raw_user_string: str, state: AgentState) -> str:
    """Return a local response when online model calls are unavailable."""
    q = raw_user_string.lower()

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
        kb_answer = query_knowledge_base(raw_user_string)
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
