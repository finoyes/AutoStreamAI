"""
Tool definitions for the AutoStream AI Agent.

"""

from langchain_core.tools import tool


@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Capture a lead by recording the creator's name, email, and preferred platform.

    This tool should ONLY be called when all three arguments are available.
    It simulates saving the lead to a CRM system.

    Args:
        name: The full name of the prospective customer.
        email: The email address of the prospective customer.
        platform: The creator platform they primarily use (e.g. YouTube, TikTok).

    Returns:
        A confirmation string indicating the lead was captured successfully.
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")

    return (
        f"Lead successfully captured!\n"
        f"   Name: {name}\n"
        f"   Email: {email}\n"
        f"   Platform: {platform}\n"
        f"Our sales team will reach out within 24 hours."
    )
