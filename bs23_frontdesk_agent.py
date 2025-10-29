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
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

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
    remaining_steps: RemainingSteps

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

# Bind tools to language model for each sub-agent
llm_with_company_tools = llm.bind_tools(company_info_tools)
llm_with_project_tools = llm.bind_tools(project_discussion_tools)
llm_with_employee_tools = llm.bind_tools(employee_info_tools)
llm_with_job_tools = llm.bind_tools(job_opportunity_tools)
llm_with_admin_tools = llm.bind_tools(admin_finance_tools)

# Create tool nodes for each sub-agent
company_tool_node = ToolNode(company_info_tools)
project_tool_node = ToolNode(project_discussion_tools)
employee_tool_node = ToolNode(employee_info_tools)
job_tool_node = ToolNode(job_opportunity_tools)
admin_tool_node = ToolNode(admin_finance_tools)

# Prompt generation functions following sample format
def generate_company_assistant_prompt() -> str:
    """
    Generate a system prompt for the company information assistant.
    
    Returns:
        str: Formatted system prompt for the company assistant
    """
    return """
    You are a member of the Brain Station 23 assistant team, your role specifically is focused on providing accurate company information.
    You have access to tools that can retrieve information about our services, location, contact details, and working hours.
    
    CORE RESPONSIBILITIES:
    - Provide accurate information about Brain Station 23 services and capabilities
    - Share location and contact information when requested
    - Inform about working hours and availability
    - Maintain professional and helpful communication
    - You are routed only when there are questions related to company information; focus on these queries.
    
    RESPONSE GUIDELINES:
    1. Always use the available tools to get the most current information
    2. Provide complete and accurate details
    3. Be concise but informative
    4. If you cannot find specific information, acknowledge this clearly
    
    Message history is attached for context.
    """

def generate_project_assistant_prompt() -> str:
    """
    Generate a system prompt for the project discussion assistant.
    
    Returns:
        str: Formatted system prompt for the project assistant
    """
    return """
    You are a member of the Brain Station 23 assistant team, your role specifically is focused on handling project inquiries and lead generation.
    You have access to tools for collecting client information, sending emails, and gathering project requirements.
    
    CORE RESPONSIBILITIES:
    - Collect comprehensive project requirements from potential clients
    - Gather client contact information securely
    - Forward project inquiries to appropriate team members
    - Provide initial project guidance and next steps
    - You are routed only when there are questions related to project discussions; focus on these queries.
    
    INFORMATION TO COLLECT:
    1. Client name and company
    2. Email address for follow-up
    3. Project type and requirements
    4. Timeline and budget considerations
    5. Technical specifications if available
    
    Message history is attached for context.
    """

def generate_employee_assistant_prompt() -> str:
    """
    Generate a system prompt for the employee contact assistant.
    
    Returns:
        str: Formatted system prompt for the employee assistant
    """
    return """
    You are a member of the Brain Station 23 assistant team, your role specifically is focused on facilitating employee connections while maintaining privacy and security.
    You have access to tools for searching employee information and collecting caller details.
    
    CORE RESPONSIBILITIES:
    - Help callers connect with appropriate Brain Station 23 employees
    - Protect employee privacy by not sharing direct contact information
    - Collect caller information for security purposes
    - Facilitate proper introductions between callers and employees
    - You are routed only when there are questions related to employee contact; focus on these queries.
    
    SECURITY PROTOCOL:
    1. Never share direct employee contact information
    2. Always collect caller details before facilitating connections
    3. Verify the purpose of contact
    4. Forward connection requests through proper channels
    
    Message history is attached for context.
    """

def generate_job_assistant_prompt() -> str:
    """
    Generate a system prompt for the job opportunities assistant.
    
    Returns:
        str: Formatted system prompt for the job assistant
    """
    return """
    You are a member of the Brain Station 23 assistant team, your role specifically is focused on providing career opportunities and guiding job seekers.
    You have access to tools for retrieving available positions and collecting candidate information.
    
    CORE RESPONSIBILITIES:
    - Provide information about current job openings
    - Guide candidates through the application process
    - Collect candidate information and preferences
    - Direct candidates to appropriate application channels
    - You are routed only when there are questions related to job opportunities; focus on these queries.
    
    APPLICATION GUIDANCE:
    1. Share relevant open positions based on candidate interests
    2. Explain application process and requirements
    3. Collect candidate background information
    4. Direct to careers@brainstation-23.com for formal applications
    5. Provide timeline expectations for hiring process
    
    Message history is attached for context.
    """

def generate_admin_assistant_prompt() -> str:
    """
    Generate a system prompt for the admin/finance assistant.
    
    Returns:
        str: Formatted system prompt for the admin assistant
    """
    return """
    You are a member of the Brain Station 23 assistant team, your role specifically is focused on handling administrative, finance, and compliance matters.
    You have access to tools for collecting inquiry details and routing to appropriate departments.
    
    CORE RESPONSIBILITIES:
    - Handle administrative inquiries and route to proper departments
    - Collect detailed information for finance and billing matters
    - Manage compliance and regulatory questions
    - Ensure proper documentation and follow-up
    - You are routed only when there are questions related to admin/finance/compliance; focus on these queries.
    
    ROUTING GUIDELINES:
    - Finance/Billing: Route to khairahammed01@gmail.com
    - Compliance/Legal: Route to khair@brainstation-23.com
    - General Admin: Route to khairahmad6@gmail.com
    
    INFORMATION TO COLLECT:
    1. Caller name and company
    2. Contact information
    3. Detailed inquiry description
    4. Relevant reference numbers or documents
    5. Urgency level
    
    Message history is attached for context.
    """

# Sub-agent node functions following sample format
def company_assistant(state: State, config: RunnableConfig):
    """
    Company information assistant node that handles company-related queries.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the company assistant agent
    company_assistant_prompt = generate_company_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_company_tools.invoke([SystemMessage(company_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def project_assistant(state: State, config: RunnableConfig):
    """
    Project discussion assistant node that handles project inquiries.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the project assistant agent
    project_assistant_prompt = generate_project_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_project_tools.invoke([SystemMessage(project_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def employee_assistant(state: State, config: RunnableConfig):
    """
    Employee contact assistant node that handles employee connection requests.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the employee assistant agent
    employee_assistant_prompt = generate_employee_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_employee_tools.invoke([SystemMessage(employee_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def job_assistant(state: State, config: RunnableConfig):
    """
    Job opportunities assistant node that handles career inquiries.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the job assistant agent
    job_assistant_prompt = generate_job_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_job_tools.invoke([SystemMessage(job_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def admin_assistant(state: State, config: RunnableConfig):
    """
    Admin/finance assistant node that handles administrative matters.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the admin assistant agent
    admin_assistant_prompt = generate_admin_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_admin_tools.invoke([SystemMessage(admin_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

# Conditional edge function for ReAct pattern
def should_continue(state: State, config: RunnableConfig):
    """
    Conditional edge function that determines the next step in the ReAct agent workflow.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        str: Either "continue" to execute tools or "end" to finish the workflow
    """
    # Get all messages from the current state
    messages = state["messages"]
    
    # Examine the most recent message to check for tool calls
    last_message = messages[-1]
    
    # If the last message doesn't contain any tool calls, the agent is done
    if not last_message.tool_calls:
        return "end"
    # If there are tool calls present, continue to execute them
    else:
        return "continue"


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


# Create sub-agent workflows following sample format with ReAct pattern
def create_company_subagent():
    """Create company information sub-agent with ReAct pattern."""
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("company_assistant", company_assistant)
    workflow.add_node("company_tool_node", company_tool_node)
    
    # Set entry point
    workflow.add_edge(START, "company_assistant")
    
    # Add conditional edge from assistant based on whether tools need to be called
    workflow.add_conditional_edges(
        "company_assistant",
        should_continue,
        {
            "continue": "company_tool_node",
            "end": END,
        },
    )
    
    # After tool execution, return to assistant
    workflow.add_edge("company_tool_node", "company_assistant")
    
    return workflow.compile(name="company_subagent", checkpointer=checkpointer, store=in_memory_store)

def create_project_subagent():
    """Create project discussion sub-agent with ReAct pattern."""
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("project_assistant", project_assistant)
    workflow.add_node("project_tool_node", project_tool_node)
    
    # Set entry point
    workflow.add_edge(START, "project_assistant")
    
    # Add conditional edge from assistant based on whether tools need to be called
    workflow.add_conditional_edges(
        "project_assistant",
        should_continue,
        {
            "continue": "project_tool_node",
            "end": END,
        },
    )
    
    # After tool execution, return to assistant
    workflow.add_edge("project_tool_node", "project_assistant")
    
    return workflow.compile(name="project_subagent", checkpointer=checkpointer, store=in_memory_store)

def create_employee_subagent():
    """Create employee information sub-agent with ReAct pattern."""
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("employee_assistant", employee_assistant)
    workflow.add_node("employee_tool_node", employee_tool_node)
    
    # Set entry point
    workflow.add_edge(START, "employee_assistant")
    
    # Add conditional edge from assistant based on whether tools need to be called
    workflow.add_conditional_edges(
        "employee_assistant",
        should_continue,
        {
            "continue": "employee_tool_node",
            "end": END,
        },
    )
    
    # After tool execution, return to assistant
    workflow.add_edge("employee_tool_node", "employee_assistant")
    
    return workflow.compile(name="employee_subagent", checkpointer=checkpointer, store=in_memory_store)

def create_job_subagent():
    """Create job opportunity sub-agent with ReAct pattern."""
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("job_assistant", job_assistant)
    workflow.add_node("job_tool_node", job_tool_node)
    
    # Set entry point
    workflow.add_edge(START, "job_assistant")
    
    # Add conditional edge from assistant based on whether tools need to be called
    workflow.add_conditional_edges(
        "job_assistant",
        should_continue,
        {
            "continue": "job_tool_node",
            "end": END,
        },
    )
    
    # After tool execution, return to assistant
    workflow.add_edge("job_tool_node", "job_assistant")
    
    return workflow.compile(name="job_subagent", checkpointer=checkpointer, store=in_memory_store)

def create_admin_subagent():
    """Create admin/finance sub-agent with ReAct pattern."""
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("admin_assistant", admin_assistant)
    workflow.add_node("admin_tool_node", admin_tool_node)
    
    # Set entry point
    workflow.add_edge(START, "admin_assistant")
    
    # Add conditional edge from assistant based on whether tools need to be called
    workflow.add_conditional_edges(
        "admin_assistant",
        should_continue,
        {
            "continue": "admin_tool_node",
            "end": END,
        },
    )
    
    # After tool execution, return to assistant
    workflow.add_edge("admin_tool_node", "admin_assistant")
    
    return workflow.compile(name="admin_subagent", checkpointer=checkpointer, store=in_memory_store)

# Create individual sub-agents
company_subagent_workflow = create_company_subagent()
project_subagent_workflow = create_project_subagent()
employee_subagent_workflow = create_employee_subagent()
job_subagent_workflow = create_job_subagent()
admin_subagent_workflow = create_admin_subagent()

# Create supervisor workflow using LangGraph's pre-built supervisor
# The supervisor coordinates between multiple subagents based on the incoming queries
supervisor_workflow = create_supervisor(
    agents=[company_subagent_workflow, project_subagent_workflow, employee_subagent_workflow, job_subagent_workflow, admin_subagent_workflow],  # List of subagents to supervise
    output_mode="last_message",  # Return only the final response (alternative: "full_history")
    model=llm,  # Language model for supervisor reasoning and routing decisions
    prompt=(supervisor_prompt),  # System instructions for the supervisor agent
    state_schema=State  # State schema defining data flow structure
)

# Compile the supervisor workflow with memory components
# - checkpointer: Enables short-term memory within conversation threads
# - store: Provides long-term memory storage across conversations
def create_bs23_frontdesk_graph():
    """Create the BS23 frontdesk multi-agent graph following sample format."""
    return supervisor_workflow.compile(
        name="bs23_frontdesk_supervisor", 
        checkpointer=checkpointer, 
        store=in_memory_store
    )

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
