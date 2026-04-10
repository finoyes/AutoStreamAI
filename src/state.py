"""
State schema for the AutoStream AI Agent.

Defines the TypedDict used by LangGraph to track conversation state,
including chat history and lead capture fields.
"""

from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The central state object that flows through every node in the graph.

    Attributes:
        messages: The full conversation history (managed by LangGraph's
                  add_messages reducer so appends are merged automatically).
        user_name: The lead's full name, once collected.
        user_email: The lead's email address, once collected.
        user_platform: The creator platform (YouTube, TikTok, etc.).
        lead_captured: Flag indicating whether mock_lead_capture has fired.
        intent: The last classified intent (greeting / info / signup).
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_name: Optional[str]
    user_email: Optional[str]
    user_platform: Optional[str]
    lead_captured: bool
    intent: Optional[str]
