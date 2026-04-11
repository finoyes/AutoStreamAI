"""
RAG Engine for the AutoStream AI Agent.

Loads the knowledge base JSON and provides a simple retrieval function
that matches user queries against known topics (pricing, policies, FAQ).
No vector DB needed — the dataset is small enough for keyword matching.
"""

import json
from pathlib import Path
from typing import Optional

_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

_knowledge_base: Optional[dict] = None


def _load_kb() -> dict:
    """Lazy-load the knowledge base from disk."""
    global _knowledge_base
    if _knowledge_base is None:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            _knowledge_base = json.load(f)
    if _knowledge_base is None:
        raise RuntimeError("Failed to load knowledge base")
    return _knowledge_base


def get_full_context() -> str:
    """
    Return a formatted string of the ENTIRE knowledge base so the LLM
    can ground its answers in real data.

    For a small KB like this, injecting the whole thing into the system
    prompt is simpler and more reliable than keyword retrieval.
    """
    knowledge_dictionary = _load_kb()
    sections: list[str] = []

    # --- Pricing ---
    sections.append("## Pricing Plans\n")
    for plan_name, plan_specifications in knowledge_dictionary.get("pricing", {}).items():
        features = ", ".join(plan_specifications.get("features", []))
        sections.append(
            f"**{plan_name}** — {plan_specifications['price']}\n"
            f"  - Video limit: {plan_specifications.get('limit', 'N/A')}\n"
            f"  - Max resolution: {plan_specifications.get('resolution', 'N/A')}\n"
            f"  - Features: {features}\n"
        )

    # --- Policies ---
    sections.append("\n## Policies\n")
    for policy_key, policy_text in knowledge_dictionary.get("policies", {}).items():
        nice_key = policy_key.replace("_", " ").title()
        sections.append(f"**{nice_key}:** {policy_text}\n")

    # --- Supported Platforms ---
    platforms = knowledge_dictionary.get("platforms_supported", [])
    if platforms:
        sections.append("\n## Supported Creator Platforms\n")
        sections.append(", ".join(platforms) + "\n")

    # --- FAQ ---
    sections.append("\n## Frequently Asked Questions\n")
    for q_key, answer in knowledge_dictionary.get("faq", {}).items():
        nice_q = q_key.replace("_", " ").title()
        sections.append(f"**{nice_q}:** {answer}\n")

    return "\n".join(sections)


def query_knowledge_base(query: str) -> str:
    """
    Simple keyword-based retrieval for targeted queries.

    Falls back to the full context if no specific section matches.
    Useful if you later want to swap in a vector-store retriever.
    """
    lower_user_query = query.lower()
    knowledge_dictionary = _load_kb()

    # Check for pricing intent
    if any(word in lower_user_query for word in ("price", "pricing", "cost", "plan", "tier", "basic", "pro", "enterprise")):
        section = knowledge_dictionary.get("pricing", {})
        lines = ["Here are the AutoStream pricing plans:\n"]
        for plan, plan_specifications in section.items():
            features = ", ".join(plan_specifications.get("features", []))
            lines.append(
                f"**{plan}** — {plan_specifications['price']}  |  "
                f"{plan_specifications.get('limit', '')}  |  "
                f"{plan_specifications.get('resolution', '')}  |  "
                f"Features: {features}"
            )
        return "\n".join(lines)

    # Check for policy / refund / cancel intent
    if any(word in lower_user_query for word in ("refund", "cancel", "policy", "support", "trial", "data")):
        section = knowledge_dictionary.get("policies", {})
        lines = ["Relevant AutoStream policies:\n"]
        for key, policy_description in section.items():
            if any(w in lower_user_query for w in key.split("_")) or "policy" in lower_user_query:
                formatted_policy_title = key.replace("_", " ").title()
                lines.append(f"**{formatted_policy_title}:** {policy_description}")
        if len(lines) == 1:  # nothing matched — return all policies
            for key, policy_description in section.items():
                formatted_policy_title = key.replace("_", " ").title()
                lines.append(f"**{formatted_policy_title}:** {policy_description}")
        return "\n".join(lines)

    # Default: return everything
    return get_full_context()
