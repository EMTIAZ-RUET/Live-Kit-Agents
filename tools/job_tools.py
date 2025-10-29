"""Job opportunity tools for BS23 Frontdesk Agent."""

from langchain_core.tools import tool

# Job data
JOB_DATA = {
    "developer": ["Senior Software Engineer", "Frontend Developer", "Backend Developer", "Full Stack Developer"],
    "manager": ["Project Manager", "Product Manager", "Team Lead"],
    "designer": ["UI/UX Designer", "Graphic Designer"]
}

@tool
def get_available_positions(job_type: str = "") -> str:
    """Get available job positions at Brain Station 23."""
    if job_type:
        job_type_lower = job_type.lower()
        for category, jobs in JOB_DATA.items():
            if category in job_type_lower:
                return f"Available {category} positions: " + ", ".join(jobs)
    
    all_jobs = []
    for jobs in JOB_DATA.values():
        all_jobs.extend(jobs)
    return "Available positions: " + ", ".join(all_jobs[:5])

# Tool collection
job_opportunity_tools = [get_available_positions]
