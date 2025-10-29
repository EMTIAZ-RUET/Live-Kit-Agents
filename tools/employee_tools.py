"""Employee information tools for BS23 Frontdesk Agent."""

from langchain_core.tools import tool

# Employee data
EMPLOYEE_DATA = {
    "john doe": {"name": "John Doe", "title": "Senior Developer", "department": "Engineering", "email": "john.doe@brainstation-23.com"},
    "jane smith": {"name": "Jane Smith", "title": "Project Manager", "department": "Operations", "email": "jane.smith@brainstation-23.com"},
    "ahmed hassan": {"name": "Ahmed Hassan", "title": "HR Manager", "department": "Human Resources", "email": "ahmed.hassan@brainstation-23.com"}
}

@tool
def search_employee(employee_name: str) -> str:
    """Search for employee information by name."""
    name_lower = employee_name.lower()
    for name, info in EMPLOYEE_DATA.items():
        if name in name_lower or any(part in name_lower for part in name.split()):
            return f"Found: {info['name']}, {info['title']} in {info['department']}"
    return "Employee not found in our directory."

# Tool collection
employee_info_tools = [search_employee]
