import logging
from typing import Annotated, TypedDict
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AnyMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, silero
from livekit.plugins.groq import LLM as GroqLLM
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("bs23-frontdesk-agent")
load_dotenv(".env")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Initialize memory components
checkpointer = MemorySaver()
in_memory_store = InMemoryStore()

# Dummy data for vector search responses
COMPANY_DATA = {
    "services": "Brain Station 23 offers software development, mobile app development, web development, AI/ML solutions, and digital transformation services.",
    "location": "Brain Station 23 is located in Dhaka, Bangladesh. Main office: Plot 15, Block B, Bashundhara R/A, Dhaka 1229.",
    "contact": "Phone: +880-2-8401010, Email: info@brainstation-23.com",
    "hours": "Working hours: Sunday to Thursday, 9:00 AM to 6:00 PM"
}

EMPLOYEE_DATA = {
    "john doe": {"name": "John Doe", "title": "Senior Developer", "department": "Engineering", "email": "john.doe@brainstation-23.com"},
    "jane smith": {"name": "Jane Smith", "title": "Project Manager", "department": "Operations", "email": "jane.smith@brainstation-23.com"},
    "ahmed hassan": {"name": "Ahmed Hassan", "title": "HR Manager", "department": "Human Resources", "email": "ahmed.hassan@brainstation-23.com"}
}

JOB_DATA = {
    "developer": ["Senior Software Engineer", "Frontend Developer", "Backend Developer", "Full Stack Developer"],
    "manager": ["Project Manager", "Product Manager", "Team Lead"],
    "designer": ["UI/UX Designer", "Graphic Designer"]
}

class State(TypedDict):
    """State schema for the BS23 frontdesk workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    current_intent: str

# Company Information Tools
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

# Employee Information Tools
@tool
def search_employee(employee_name: str) -> str:
    """Search for employee information by name."""
    name_lower = employee_name.lower()
    for name, info in EMPLOYEE_DATA.items():
        if name in name_lower or any(part in name_lower for part in name.split()):
            return f"Found: {info['name']}, {info['title']} in {info['department']}"
    return "Employee not found in our directory."

# Job Opportunity Tools
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

# Communication Tools
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

# Tool collections for different sub-agents
company_info_tools = [get_company_services, get_company_location, get_company_contact, get_company_hours]
project_discussion_tools = [send_email, collect_caller_info, get_company_services]
employee_info_tools = [search_employee, collect_caller_info, send_email]
job_opportunity_tools = [get_available_positions, collect_caller_info, send_email]
admin_finance_tools = [collect_caller_info, send_email]

# Simple Node Functions
def company_info_node(state: State, config: RunnableConfig):
    """Handle company information queries."""
    user_query = state["messages"][-1].content.lower()
    
    if "service" in user_query:
        info = COMPANY_DATA["services"]
    elif "location" in user_query or "address" in user_query:
        info = COMPANY_DATA["location"]
    elif "contact" in user_query or "phone" in user_query:
        info = COMPANY_DATA["contact"]
    elif "hours" in user_query or "time" in user_query:
        info = COMPANY_DATA["hours"]
    else:
        info = f"{COMPANY_DATA['services']} {COMPANY_DATA['location']} {COMPANY_DATA['contact']}"
    
    response = f"Here's the information you requested: {info}"
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

def employee_info_node(state: State, config: RunnableConfig):
    """Handle employee contact requests."""
    user_query = state["messages"][-1].content.lower()
    
    found_employee = None
    for name, info in EMPLOYEE_DATA.items():
        if name in user_query:
            found_employee = info
            break
    
    if found_employee:
        response = f"I found {found_employee['name']}, who is a {found_employee['title']} in {found_employee['department']}. I can help you connect with them. Please provide your name and purpose for contact."
    else:
        response = "I couldn't find that employee. Could you please provide the full name or try a different spelling?"
    
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

def job_opportunity_node(state: State, config: RunnableConfig):
    """Handle job opportunity inquiries."""
    user_query = state["messages"][-1].content.lower()
    
    available_jobs = []
    for category, jobs in JOB_DATA.items():
        if category in user_query:
            available_jobs.extend(jobs)
    
    if not available_jobs:
        available_jobs = [job for jobs in JOB_DATA.values() for job in jobs[:2]]
    
    response = f"Here are some available positions: {', '.join(available_jobs[:5])}. Please send your resume to careers@brainstation-23.com for application."
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

def project_discussion_node(state: State, config: RunnableConfig):
    """Handle project inquiries."""
    response = "I'd be happy to help with your project inquiry. Please provide your name, company, email, and project details. Our team will contact you within 24 hours."
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

def admin_finance_node(state: State, config: RunnableConfig):
    """Handle admin/finance inquiries."""
    response = "For administrative, finance, or compliance matters, please provide your contact details and inquiry details. I'll forward this to the appropriate department."
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

# Project Discussion Sub-agent
def project_subagent(state: State, config: RunnableConfig):
    """Handle project discussion inquiries with lead collection."""
    messages = state["messages"]
    user_query = messages[-1].content.lower()
    
    # Check if this is initial project inquiry
    if any(keyword in user_query for keyword in ["project", "development", "hire", "build", "software"]):
        response = "I'd be happy to help with your project inquiry! To better assist you, I'll need some information. Could you please provide:\n\n1. Your name and company\n2. Your email address\n3. Project type and requirements\n4. Timeline and budget range\n\nOur team will review your requirements and contact you within 24 hours."
        return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}
    
    # If user is providing information, collect it
    response = "Thank you for the information. I'm forwarding your project inquiry to our team at khairahmad6@gmail.com. You can expect to hear from us within 24 hours with next steps."
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

# Employee Information Sub-agent
def employee_subagent(state: State, config: RunnableConfig):
    """Handle employee contact requests with privacy protection."""
    user_query = state["messages"][-1].content.lower()
    
    # Search for employee in the query
    found_employee = None
    for name, info in EMPLOYEE_DATA.items():
        if name in user_query or any(part in user_query for part in name.split()):
            found_employee = info
            break
    
    if found_employee:
        response = f"I found {found_employee['name']}, who is a {found_employee['title']} in {found_employee['department']}. For security reasons, I cannot share direct contact information. However, I can help you connect with them.\n\nPlease provide:\n1. Your name\n2. Your email address\n3. Purpose of contact\n\nI'll forward your message to {found_employee['name']} and they will contact you directly."
    else:
        response = "I couldn't find that employee in our directory. Could you please provide the full name or check the spelling? Alternatively, you can describe the department or role you're looking for."
    
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

# Job Opportunity Sub-agent
def job_subagent(state: State, config: RunnableConfig):
    """Handle job opportunity inquiries and candidate guidance."""
    user_query = state["messages"][-1].content.lower()
    
    # Find relevant job categories
    available_jobs = []
    for category, jobs in JOB_DATA.items():
        if category in user_query:
            available_jobs.extend(jobs)
    
    if not available_jobs:
        # Show general available positions
        available_jobs = [job for jobs in JOB_DATA.values() for job in jobs[:2]]
    
    response = f"Great to hear about your interest in joining Brain Station 23! Here are some current openings:\n\n{', '.join(available_jobs[:5])}\n\nTo apply, please send your resume to careers@brainstation-23.com with the position title in the subject line.\n\nIf you'd like to discuss your background and interests, please share:\n1. Your name and current role\n2. Years of experience\n3. Preferred technology stack\n4. Career goals\n\nOur HR team will review your application and get back to you soon!"
    
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

# Admin/Finance/Compliance Sub-agent
def admin_subagent(state: State, config: RunnableConfig):
    """Handle administrative, finance, and compliance inquiries."""
    user_query = state["messages"][-1].content.lower()
    
    # Determine the type of inquiry
    if any(keyword in user_query for keyword in ["finance", "billing", "payment", "invoice"]):
        department = "Finance"
        email = "khairahammed01@gmail.com"
    elif any(keyword in user_query for keyword in ["compliance", "legal", "regulation"]):
        department = "Compliance"
        email = "khair@brainstation-23.com"
    else:
        department = "Administration"
        email = "khairahmad6@gmail.com"
    
    response = f"I'll help you with your {department.lower()} inquiry. To ensure your request is handled properly, please provide:\n\n1. Your name and company (if applicable)\n2. Your email address\n3. Detailed description of your inquiry\n4. Any relevant reference numbers or documents\n\nI'll forward your request to our {department} team at {email}, and they will contact you within 1-2 business days."
    
    return {"messages": [SystemMessage(content=response)], "current_intent": "complete"}

# Supervisor Agent for Multi-Agent Orchestration
supervisor_prompt = """You are Sabnam, the expert virtual receptionist and supervisor for Brain Station 23.
You coordinate a team of specialized assistants to provide exceptional customer service.

Your team consists of:
1. company_subagent: Handles company information queries (services, location, contact, hours)
2. project_subagent: Manages project discussions and lead generation
3. employee_subagent: Facilitates employee contact and connection requests
4. job_subagent: Provides career opportunities and job information
5. admin_subagent: Handles administrative, compliance, and finance matters

ROUTING GUIDELINES:
- Analyze the caller's request to determine the most appropriate specialist
- Route to the sub-agent best equipped to handle the specific inquiry
- Ensure smooth transitions between agents when needed
- Maintain professional and friendly communication throughout

Based on the conversation context, determine which sub-agent should handle the next step.
This could involve multiple sub-agent calls for complex inquiries.
"""

# Create supervisor using a simple routing approach (following sample format)
def create_supervisor_agent():
    """Create supervisor agent for routing to sub-agents."""
    
    def supervisor_node(state: State, config: RunnableConfig):
        """Supervisor node that routes to appropriate sub-agents."""
        
        # Welcome message for first interaction
        if len(state["messages"]) == 1:
            welcome_msg = "Thank you for calling Brain Station 23. This is Sabnam, how may I help you today?"
            return {"messages": [SystemMessage(content=welcome_msg)], "current_intent": "greeting"}
        
        # Analyze intent and route to appropriate sub-agent
        last_message = state["messages"][-1].content.lower()
        
        # Determine routing based on keywords and context
        if any(keyword in last_message for keyword in ["service", "company", "location", "contact", "hours", "about"]):
            intent = "company_info"
        elif any(keyword in last_message for keyword in ["project", "development", "hire", "build", "software"]):
            intent = "project_discussion"
        elif any(keyword in last_message for keyword in ["employee", "staff", "person", "contact", "speak to"]):
            intent = "employee_information"
        elif any(keyword in last_message for keyword in ["job", "career", "position", "hiring", "work", "apply"]):
            intent = "job_opportunity"
        elif any(keyword in last_message for keyword in ["admin", "finance", "billing", "compliance", "payment"]):
            intent = "admin_finance"
        else:
            # Ask for clarification
            clarification_msg = "I'd be happy to help you! Could you please let me know if you're looking for information about our company, discussing a project, contacting an employee, exploring job opportunities, or handling administrative matters?"
            return {"messages": [SystemMessage(content=clarification_msg)], "current_intent": "clarify"}
        
        return {"current_intent": intent}
    
    return supervisor_node

supervisor_agent = create_supervisor_agent()

# Routing function for conditional edges
def route_to_subagent(state: State, config: RunnableConfig):
    """Route to appropriate sub-agent based on intent."""
    intent = state.get("current_intent", "")
    
    if intent == "company_info":
        return "company_subagent"
    elif intent == "project_discussion":
        return "project_subagent"
    elif intent == "employee_information":
        return "employee_subagent"
    elif intent == "job_opportunity":
        return "job_subagent"
    elif intent == "admin_finance":
        return "admin_subagent"
    elif intent == "complete":
        return "end"
    else:
        return "supervisor"

# Create Multi-Agent Workflow with Supervisor Architecture
def create_bs23_frontdesk_graph():
    """Create the BS23 frontdesk multi-agent graph following sample format."""
    
    # Create main workflow StateGraph
    workflow = StateGraph(State)
    
    # Add supervisor node
    workflow.add_node("supervisor", supervisor_agent)
    
    # Add all sub-agent nodes
    workflow.add_node("company_subagent", company_info_node)
    workflow.add_node("project_subagent", project_subagent)
    workflow.add_node("employee_subagent", employee_subagent)
    workflow.add_node("job_subagent", job_subagent)
    workflow.add_node("admin_subagent", admin_subagent)
    
    # Set entry point to supervisor
    workflow.add_edge(START, "supervisor")
    
    # Add conditional routing from supervisor to sub-agents
    workflow.add_conditional_edges(
        "supervisor",
        route_to_subagent,
        {
            "company_subagent": "company_subagent",
            "project_subagent": "project_subagent",
            "employee_subagent": "employee_subagent",
            "job_subagent": "job_subagent",
            "admin_subagent": "admin_subagent",
            "supervisor": "supervisor",
            "end": END
        }
    )
    
    # All sub-agents return to END after completion
    workflow.add_edge("company_subagent", END)
    workflow.add_edge("project_subagent", END)
    workflow.add_edge("employee_subagent", END)
    workflow.add_edge("job_subagent", END)
    workflow.add_edge("admin_subagent", END)
    
    # Compile without checkpointer for LiveKit compatibility
    return workflow.compile()

def prewarm(proc: JobProcess):
    """Preload components for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the BS23 frontdesk agent."""
    
    # Create agent with direct LLM (without bound tools for LiveKit compatibility)
    agent = Agent(
        instructions="""You are Sabnam, the virtual receptionist for Brain Station 23. 

GREETING: Always start conversations with: "Thank you for calling Brain Station 23. This is Sabnam, how may I help you today?"

CAPABILITIES & RESPONSES:
- Company Information: 
  * Services: "Brain Station 23 offers software development, mobile app development, web development, AI/ML solutions, and digital transformation services."
  * Location: "Brain Station 23 is located in Dhaka, Bangladesh. Main office: Plot 15, Block B, Bashundhara R/A, Dhaka 1229."
  * Contact: "Phone: +880-2-8401010, Email: info@brainstation-23.com"
  * Hours: "Working hours: Sunday to Thursday, 9:00 AM to 6:00 PM"

- Project Discussions: Collect project requirements (name, company, email, project type, timeline, budget) and inform them our team will contact within 24 hours at khairahmad6@gmail.com

- Employee Contacts: Help connect with employees like John Doe (Senior Developer), Jane Smith (Project Manager), Ahmed Hassan (HR Manager). Always collect caller details for security.

- Job Opportunities: Available positions include Senior Software Engineer, Frontend Developer, Backend Developer, Project Manager, UI/UX Designer. Direct them to careers@brainstation-23.com

- Admin/Finance: Route to appropriate departments - Admin: khairahmad6@gmail.com, Compliance: khair@brainstation-23.com, Finance: khairahammed01@gmail.com

RESPONSE STYLE:
- Be professional, friendly, and helpful
- Ask clarifying questions when needed
- Collect necessary information before forwarding requests
- Provide clear next steps to callers""",
        llm=GroqLLM(model="llama-3.3-70b-versatile"),
    )
    
    # Create session with voice components and adjusted turn detection
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-general"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        turn_detection=MultilingualModel(),
    )
    
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )
    
    logger.info("BS23 Frontdesk Agent started successfully with direct LLM integration")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
