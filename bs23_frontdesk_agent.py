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

# Import modular components
from subagents.company_agent import create_company_subagent
from subagents.project_agent import create_project_subagent
from subagents.employee_agent import create_employee_subagent
from subagents.job_agent import create_job_subagent
from subagents.admin_agent import create_admin_subagent

class State(TypedDict):
    """State schema for the BS23 frontdesk workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    current_intent: str
    remaining_steps: RemainingSteps



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


# Create individual sub-agents using modular functions
company_subagent_workflow = create_company_subagent(llm, checkpointer, in_memory_store, State)
project_subagent_workflow = create_project_subagent(llm, checkpointer, in_memory_store, State)
employee_subagent_workflow = create_employee_subagent(llm, checkpointer, in_memory_store, State)
job_subagent_workflow = create_job_subagent(llm, checkpointer, in_memory_store, State)
admin_subagent_workflow = create_admin_subagent(llm, checkpointer, in_memory_store, State)

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
