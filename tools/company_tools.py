"""Company information tools for BS23 Frontdesk Agent."""

from langchain_core.tools import tool

# Company data
COMPANY_DATA = {
    "services": "Brain Station 23 offers software development, mobile app development, web development, AI/ML solutions, and digital transformation services.",
    "location": "Brain Station 23 is located in Dhaka, Bangladesh. Main office: Plot 15, Block B, Bashundhara R/A, Dhaka 1229.",
    "contact": "Phone: +880-2-8401010, Email: info@brainstation-23.com",
    "hours": "Working hours: Sunday to Thursday, 9:00 AM to 6:00 PM"
}

@tool
def get_company_services(query: str) -> str:
    """Get information about Brain Station 23 services."""
    return COMPANY_DATA["services"]

@tool
def get_company_location(query: str) -> str:
    """Get Brain Station 23 location and address information."""
    return COMPANY_DATA["location"]

@tool
def get_company_contact(query: str) -> str:
    """Get Brain Station 23 contact information."""
    return COMPANY_DATA["contact"]

@tool
def get_company_hours(query: str) -> str:
    """Get Brain Station 23 working hours."""
    return COMPANY_DATA["hours"]

# Tool collection
company_info_tools = [get_company_services, get_company_location, get_company_contact, get_company_hours]
