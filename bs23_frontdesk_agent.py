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
from livekit.plugins import deepgram, silero, langchain
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



# Note: supervisor_prompt removed - using inline system_message in supervisor_node instead


# Create individual sub-agents using modular functions (without memory for LiveKit compatibility)
company_subagent_workflow = create_company_subagent(llm, None, None, State)
project_subagent_workflow = create_project_subagent(llm, None, None, State)
employee_subagent_workflow = create_employee_subagent(llm, None, None, State)
job_subagent_workflow = create_job_subagent(llm, None, None, State)
admin_subagent_workflow = create_admin_subagent(llm, None, None, State)

# Create simple supervisor using exact same pattern as working langraph_implementation.py
def create_bs23_frontdesk_graph():
    """Create LiveKit-compatible supervisor using exact same pattern as working example."""
    
    def supervisor_node(state: State):
        # Simple supervisor with embedded employee intelligence (no sub-agent routing)
        messages = state["messages"]
        last_message = messages[-1].content.lower() if messages else ""
        
        # Embedded employee intelligence in supervisor (no sub-agent calls)
        if any(keyword in last_message for keyword in ["employee", "contact", "person", "speak to", "connect", "john", "jane", "ahmed", "david", "johnson"]):
            print("🎯 Using supervisor with employee intelligence (no sub-agent)")
            system_message = """You are Sabnam, the Employee Contact Specialist for Brain Station 23.

GREETING: "Thank you for calling Brain Station 23. This is Sabnam, how may I help you today?"

EMPLOYEE DIRECTORY:
- John Doe: Senior Developer, Engineering Department, john.doe@brainstation-23.com
- Jane Smith: Project Manager, Operations Department, jane.smith@brainstation-23.com  
- Ahmed Hassan: HR Manager, Human Resources Department, ahmed.hassan@brainstation-23.com
- David Johnson: Senior Developer, Engineering Department, john.doe@brainstation-23.com

SECURITY PROTOCOL:
- Always collect caller information before connecting to employees
- Ask for: caller name, company, purpose of contact
- For security, verify the request is legitimate

Handle employee-related requests professionally and securely."""
        else:
            print("🎯 Using general supervisor response")
            system_message = """You are Sabnam, the expert virtual receptionist for Brain Station 23.

GREETING: "Thank you for calling Brain Station 23. This is Sabnam, how may I help you today?"

You provide information about:
- Company services and information
- Project discussions and lead generation
- Employee contacts and connections
- Career opportunities and jobs
- Administrative and compliance matters

Be professional, friendly, and helpful."""
        
        # Create enhanced messages with context-specific prompt
        enhanced_messages = [{"role": "system", "content": system_message}] + messages
        response = llm.invoke(enhanced_messages)
        return {"messages": [response]}
    
    # Use exact same StateGraph pattern as working langraph_implementation.py
    from langgraph.graph import StateGraph, START
    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor_node)
    builder.add_edge(START, "supervisor")
    return builder.compile()  # Simple compile like working example - NO NAME, NO CHECKPOINTER



def prewarm(proc: JobProcess):
    """Preload components for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()




async def entrypoint(ctx: JobContext):
    """Main entrypoint for the BS23 frontdesk agent."""
    
    # Use your original complex supervisor workflow (without memory for LiveKit compatibility)
    bs23_graph = create_bs23_frontdesk_graph()
    
    # Create agent with your original LangGraph supervisor
    agent = Agent(
        instructions="""You are Sabnam, the virtual receptionist for Brain Station 23. 

GREETING BEHAVIOR: 
- ONLY greet with "Thank you for calling Brain Station 23. This is Sabnam, how may I help you today?" at the very beginning of the conversation
- After the initial greeting, respond naturally to the caller's requests without repeating the greeting
- Continue the conversation flow normally based on what the caller is asking

You have access to a team of specialized assistants through your supervisor system:
- Company information and services
- Project discussions and lead generation  
- Employee contact and connections
- Career opportunities and jobs
- Administrative and compliance matters
""",
        llm=langchain.LLMAdapter(bs23_graph),
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
    
    # Generate initial greeting to activate TTS
    await session.generate_reply(
        instructions="Greet the user with your standard Brain Station 23 greeting."
    )
    
    logger.info("BS23 Frontdesk Agent started successfully with original multi-agent supervisor")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
