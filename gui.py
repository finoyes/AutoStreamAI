"""
AutoStream AI Agent - Desktop GUI.

Run:
    python gui.py

This launches a simple chat interface backed by the same LangGraph agent
used by the CLI entry point.
"""

import queue
import threading
import time
import os
from datetime import datetime
from typing import cast
import re
import tkinter as tk
from tkinter import scrolledtext, ttk

from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import build_agent_graph
from src.rag_engine import query_knowledge_base
from src.state import AgentState


COLORS = {
    "app_bg": "#F3F6FB",
    "card_bg": "#FFFFFF",
    "header_bg": "#0F172A",
    "header_title": "#F8FAFC",
    "header_subtitle": "#CBD5E1",
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "status_ready_bg": "#DCFCE7",
    "status_ready_fg": "#166534",
    "status_busy_bg": "#DBEAFE",
    "status_busy_fg": "#1E40AF",
    "text": "#111827",
    "muted": "#6B7280",
    "assistant_text": "#111827",
    "user_text": "#0F172A",
    "chat_bg": "#F9FBFF",
    "border": "#DCE3EE",
}


def _sanitize_for_gui(text: str) -> str:
    """Convert basic markdown-style formatting to plain readable text."""
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*]\s+", "• ", cleaned, flags=re.MULTILINE)
    return cleaned




def _offline_fallback_reply(user_input: str, state: AgentState) -> str:
    """Return a local response when online model calls are unavailable."""
    q = user_input.lower()

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

    info_markers = ("price", "pricing", "cost", "refund", "cancel", "policy", "support", "faq", "plan", "trial")

    if any(w in q for w in info_markers):
        kb_answer = query_knowledge_base(user_input)
        return (
            "I am in offline mode right now. Here is the answer from the local knowledge base:\n\n"
            f"{kb_answer}"
        )

    if any(w in q for w in signup_markers):
        if not state.get("user_name"):
            return "I am in offline mode right now. To continue signup, please share your full name."
        if not state.get("user_email"):
            return "I am in offline mode right now. Please share your email address next."
        if not state.get("user_platform"):
            return "I am in offline mode right now. Which creator platform do you use (YouTube, TikTok, Instagram)?"
        return "Thanks. I have your signup details and can continue once API access is available again."

    return (
        "I cannot call Gemini right now. You can still ask pricing or policy questions "
        "and I will answer from the local knowledge base."
    )




class AutoStreamGUI:
    def __init__(self) -> None:
        # GUI should actively use Gemini unless user explicitly disables this flag.
        os.environ.setdefault("AUTOSTREAM_FORCE_GEMINI", "true")

        self.root = tk.Tk()
        self.root.title("AutoStream AI Assistant")
        self.root.geometry("980x700")
        self.root.minsize(560, 420)
        self.root.configure(bg=COLORS["app_bg"])

        self.agent = build_agent_graph()
        self.state: AgentState = {
            "messages": [],
            "user_name": None,
            "user_email": None,
            "user_platform": None,
            "lead_captured": False,
            "intent": None,
            "intent_source": None,
        }

        self._event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._pending = False
        self._thinking_job: str | None = None
        self._thinking_tick = 0
        self._thinking_start_index: str | None = None
        self._thinking_end_index: str | None = None
        self._offline_hard_quota = False
        self._offline_until_ts = 0.0
        self._compact_layout = False
        self._force_gemini_mode = os.getenv("AUTOSTREAM_FORCE_GEMINI", "true").strip().lower() in ("1", "true", "yes", "on")

        self.container: ttk.Frame
        self.header_card: ttk.Frame
        self.header_left: ttk.Frame
        self.header_sub: ttk.Label
        self.input_row: ttk.Frame

        self._configure_styles()
        self._build_ui()
        self.root.bind("<Configure>", self._on_window_resize)
        self.root.after(50, self._apply_responsive_layout)
        self._append_message("assistant", "Welcome to AutoStream AI. Ask me about pricing, policies, or signup.")
        self.root.after(100, self._drain_event_queue)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=COLORS["app_bg"])
        style.configure("Card.TFrame", background=COLORS["card_bg"], relief="solid", borderwidth=1)
        style.configure("Header.TFrame", background=COLORS["header_bg"])

        style.configure(
            "HeaderTitle.TLabel",
            background=COLORS["header_bg"],
            foreground=COLORS["header_title"],
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=COLORS["header_bg"],
            foreground=COLORS["header_subtitle"],
            font=("Segoe UI", 10),
        )

        style.configure(
            "StatusReady.TLabel",
            background=COLORS["status_ready_bg"],
            foreground=COLORS["status_ready_fg"],
            font=("Segoe UI", 9, "bold"),
            padding=(10, 4),
        )
        style.configure(
            "StatusBusy.TLabel",
            background=COLORS["status_busy_bg"],
            foreground=COLORS["status_busy_fg"],
            font=("Segoe UI", 9, "bold"),
            padding=(10, 4),
        )

        style.configure(
            "Input.TEntry",
            fieldbackground="#FFFFFF",
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=8,
        )

        style.configure(
            "Send.TButton",
            background=COLORS["primary"],
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
            borderwidth=0,
        )
        style.map(
            "Send.TButton",
            background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_hover"])],
            foreground=[("disabled", "#E5E7EB")],
        )

    def _build_ui(self) -> None:
        self.container = ttk.Frame(self.root, padding=14, style="App.TFrame")
        self.container.pack(fill=tk.BOTH, expand=True)

        self.header_card = ttk.Frame(self.container, style="Header.TFrame", padding=(16, 12))
        self.header_card.pack(fill=tk.X, pady=(0, 12))

        self.header_left = ttk.Frame(self.header_card, style="Header.TFrame")
        self.header_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        header_title = ttk.Label(self.header_left, text="AutoStream AI Assistant", style="HeaderTitle.TLabel")
        header_title.pack(anchor=tk.W)
        self.header_sub = ttk.Label(
            self.header_left,
            text="Professional support for pricing, policies, and creator onboarding.",
            style="HeaderSub.TLabel",
            wraplength=560,
            justify=tk.LEFT,
        )
        self.header_sub.pack(anchor=tk.W, pady=(2, 0))

        self.status_var = tk.StringVar(value="Ready")
        self.status_badge = ttk.Label(self.header_card, textvariable=self.status_var, style="StatusReady.TLabel")
        self.status_badge.pack(side=tk.RIGHT, padx=(10, 0))

        chat_card = ttk.Frame(self.container, style="Card.TFrame", padding=1)
        chat_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(
            chat_card,
            wrap=tk.WORD,
            height=26,
            font=("Segoe UI", 11),
            padx=14,
            pady=14,
            relief=tk.FLAT,
            background=COLORS["chat_bg"],
            foreground=COLORS["text"],
            insertbackground=COLORS["text"],
            borderwidth=0,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.configure(state=tk.DISABLED)

        self.chat.tag_configure("assistant_header", foreground=COLORS["primary"], font=("Segoe UI", 9, "bold"))
        self.chat.tag_configure("assistant_body", foreground=COLORS["assistant_text"], font=("Segoe UI", 11))
        self.chat.tag_configure("user_header", foreground="#0F766E", font=("Segoe UI", 9, "bold"))
        self.chat.tag_configure("user_body", foreground=COLORS["user_text"], font=("Segoe UI", 11))
        self.chat.tag_configure("thinking_header", foreground=COLORS["muted"], font=("Segoe UI", 9, "bold"))
        self.chat.tag_configure("thinking_body", foreground=COLORS["muted"], font=("Segoe UI", 11, "italic"))
        self.chat.tag_configure("thinking_block")

        self.chat.tag_configure("msg", lmargin1=4, lmargin2=4, spacing1=3, spacing3=8)

        self.input_row = ttk.Frame(self.container, style="App.TFrame")
        self.input_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))

        self.user_input = ttk.Entry(self.input_row, font=("Segoe UI", 11), style="Input.TEntry")
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self._on_send)

        self.send_button = ttk.Button(
            self.input_row,
            text="Send",
            style="Send.TButton",
            command=lambda: self._on_send(None),
        )
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))

    def _set_header_layout(self, compact: bool) -> None:
        self.status_badge.pack_forget()
        if compact:
            self.status_badge.pack(anchor=tk.W, pady=(8, 0))
        else:
            self.status_badge.pack(side=tk.RIGHT, padx=(10, 0))

    def _set_input_layout(self, compact: bool) -> None:
        self.user_input.pack_forget()
        self.send_button.pack_forget()
        if compact:
            self.user_input.pack(fill=tk.X)
            self.send_button.pack(fill=tk.X, pady=(8, 0))
        else:
            self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.send_button.pack(side=tk.LEFT, padx=(8, 0))

    def _apply_responsive_layout(self) -> None:
        width = self.root.winfo_width()
        compact = width < 760

        if compact != self._compact_layout:
            self._compact_layout = compact
            self._set_header_layout(compact)
            self._set_input_layout(compact)

        if compact:
            wraplength = max(260, width - 80)
        else:
            wraplength = max(340, width - 360)
        self.header_sub.configure(wraplength=wraplength)

    def _on_window_resize(self, event) -> None:
        if event.widget is self.root:
            self._apply_responsive_layout()

    def _append_message(self, role: str, text: str) -> None:
        display_text = _sanitize_for_gui(text) if role == "assistant" else text
        timestamp = datetime.now().strftime("%H:%M")
        if role == "user":
            header = f"You  {timestamp}\n"
            header_tag = "user_header"
            body_tag = "user_body"
        else:
            header = f"AutoStream  {timestamp}\n"
            header_tag = "assistant_header"
            body_tag = "assistant_body"

        self.chat.configure(state=tk.NORMAL)
        # Guard against line-join artifacts so each message starts on a new line.
        if self.chat.index(tk.END) != "1.0":
            last_char = self.chat.get("end-2c", "end-1c")
            if last_char and last_char != "\n":
                self.chat.insert(tk.END, "\n")
        self.chat.insert(tk.END, header, (header_tag, "msg"))
        self.chat.insert(tk.END, f"{display_text}\n\n", (body_tag, "msg"))
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _show_thinking_message(self) -> None:
        self.chat.configure(state=tk.NORMAL)
        if self.chat.index(tk.END) != "1.0":
            last_char = self.chat.get("end-2c", "end-1c")
            if last_char and last_char != "\n":
                self.chat.insert(tk.END, "\n")
        self._thinking_start_index = self.chat.index(tk.END)
        self.chat.insert(tk.END, "Thinking...\n\n", ("thinking_body", "msg"))
        self._thinking_end_index = self.chat.index(tk.END)
        self.chat.tag_add("thinking_block", self._thinking_start_index, self._thinking_end_index)
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _clear_thinking_message(self) -> None:
        self.chat.configure(state=tk.NORMAL)
        ranges = self.chat.tag_ranges("thinking_block")
        if len(ranges) >= 2:
            # Remove all pending thinking placeholders (from newest to oldest).
            for idx in range(len(ranges) - 2, -1, -2):
                self.chat.delete(ranges[idx], ranges[idx + 1])
        self.chat.configure(state=tk.DISABLED)
        self._thinking_start_index = None
        self._thinking_end_index = None

    def _is_offline_mode_active(self) -> bool:
        if self._offline_hard_quota:
            return True
        return time.time() < self._offline_until_ts

    def _offline_notice(self) -> str:
        if self._offline_hard_quota:
            return "Gemini daily quota appears exhausted. Continuing in offline mode for now."
        remaining = max(0, int(self._offline_until_ts - time.time()))
        if remaining > 0:
            return f"Gemini rate limit active. Continuing in offline mode (about {remaining}s remaining)."
        return "Gemini is temporarily unavailable. Continuing in offline mode."

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.user_input.configure(state=state)
        self.send_button.configure(state=state)
        if enabled:
            self.user_input.focus_set()

    def _animate_thinking(self) -> None:
        if not self._pending:
            return
        dots = "." * ((self._thinking_tick % 3) + 1)
        self.status_var.set(f"Thinking{dots}")
        self.status_badge.configure(style="StatusBusy.TLabel")
        self._thinking_tick += 1
        self._thinking_job = self.root.after(400, self._animate_thinking)

    def _start_thinking_animation(self) -> None:
        if self._thinking_job is not None:
            self.root.after_cancel(self._thinking_job)
            self._thinking_job = None
        self._thinking_tick = 0
        self._animate_thinking()

    def _stop_thinking_animation(self) -> None:
        if self._thinking_job is not None:
            self.root.after_cancel(self._thinking_job)
            self._thinking_job = None
        self.status_var.set("Ready")
        self.status_badge.configure(style="StatusReady.TLabel")

    def _on_send(self, _event) -> str:
        if self._pending:
            return "break"

        chat_input_text = self.user_input.get().strip()
        if not chat_input_text:
            return "break"

        self.user_input.delete(0, tk.END)
        self._append_message("user", chat_input_text)

        self.state["messages"] = list(self.state["messages"]) + [HumanMessage(content=chat_input_text)]

        self._pending = True
        self._set_controls_enabled(False)
        self._show_thinking_message()
        self._start_thinking_animation()

        worker = threading.Thread(target=self._process_message, args=(chat_input_text,), daemon=True)
        worker.start()
        return "break"

    def _process_message(self, chat_input_text: str) -> None:
        assistant_response: str

        if self._is_offline_mode_active() and not self._force_gemini_mode:
            fallback = _offline_fallback_reply(chat_input_text, self.state)
            self.state["messages"] = list(self.state["messages"]) + [AIMessage(content=fallback)]
            assistant_response = f"{self._offline_notice()}\n\n{fallback}"
            self._event_queue.put(("assistant", assistant_response))
            self._event_queue.put(("status", "Ready"))
            self._event_queue.put(("controls", "enabled"))
            return

        try:
            updated_state = cast(AgentState, self.agent.invoke(self.state))
            self.state = updated_state

            assistant_messages = [
                m for m in updated_state.get("messages", [])
                if hasattr(m, "content") and getattr(m, "type", "") == "ai"
            ]
            if assistant_messages:
                assistant_response = str(assistant_messages[-1].content)
            else:
                assistant_response = "No response generated. Please try again."

        except ChatGoogleGenerativeAIError as exc:
            err_str = str(exc).upper()
            is_exhausted = "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "QUOTA" in err_str
            is_offline_condition = is_exhausted or "NOT_FOUND" in err_str or "UNAVAILABLE" in err_str or "503" in err_str
            
            if is_offline_condition:
                if is_exhausted:
                    if "PERDAY" in err_str or "PER DAY" in err_str or "REQUESTSPERDAY" in err_str:
                        self._offline_hard_quota = True
                    else:
                        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
                        retry_seconds = max(1, int(float(match.group(1)))) if match else 15
                        self._offline_until_ts = time.time() + retry_seconds
                    friendly_error = "Gemini quota/rate limit reached."
                elif "NOT_FOUND" in err_str:
                    friendly_error = "Configured Gemini model is not available for this API key/project."
                else:
                    friendly_error = "Gemini is temporarily unavailable due to high demand."

                fallback = _offline_fallback_reply(chat_input_text, self.state)
                self.state["messages"] = list(self.state["messages"]) + [AIMessage(content=fallback)]
                assistant_response = f"{friendly_error}\n\n{fallback}"
            else:
                self.state["messages"] = list(self.state["messages"][:-1])
                assistant_response = f"Model request failed: {exc}"

        self._event_queue.put(("assistant", assistant_response))
        self._event_queue.put(("status", "Ready"))
        self._event_queue.put(("controls", "enabled"))

    def _drain_event_queue(self) -> None:
        while True:
            try:
                kind, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "assistant":
                self._clear_thinking_message()
                self._append_message("assistant", payload)
            elif kind == "status":
                if not self._pending:
                    self.status_var.set(payload)
            elif kind == "controls":
                self._pending = False
                self._stop_thinking_animation()
                self._set_controls_enabled(True)

        self.root.after(100, self._drain_event_queue)

    def run(self) -> None:
        self.user_input.focus_set()
        self.root.mainloop()


def main() -> None:
    app = AutoStreamGUI()
    app.run()


if __name__ == "__main__":
    main()
