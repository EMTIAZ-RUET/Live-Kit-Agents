"""Communication tools for BS23 Frontdesk Agent."""

from langchain_core.tools import tool
from datetime import datetime

@tool
def send_email(subject: str, message: str, to_email: str) -> str:
    """Send email to specified recipient."""
    print(f"\n=== EMAIL SENT ===")
    print(f"To: {to_email}")
    print(f"Subject: {subject}")
    print(f"Message: {message}")
    print(f"Timestamp: {datetime.now()}")
    print("==================\n")
    return "Email sent successfully!"

@tool
def collect_caller_info(name: str, email: str, phone: str = "", purpose: str = "") -> str:
    """Collect and store caller information."""
    info = {
        "name": name,
        "email": email,
        "phone": phone,
        "purpose": purpose,
        "timestamp": datetime.now().isoformat()
    }
    return f"Caller information collected: {info}"

# Tool collection
communication_tools = [send_email, collect_caller_info]
