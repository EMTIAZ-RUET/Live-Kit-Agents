import logging
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

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

# Memory components removed - not needed for LiveKit compatibility

# Import specialist functions from sub-agents
from subagents.employee_agent import intent_analyzer, employee_specialist
from subagents.company_agent import company_specialist, general_receptionist
from subagents.project_agent import project_specialist
from subagents.job_agent import job_specialist
from subagents.admin_agent import admin_specialist

class State(TypedDict):
    """State schema for the BS23 frontdesk workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    remaining_steps: RemainingSteps



# LangGraph-based frontdesk agent with modular prompts
def create_bs23_frontdesk_graph():
    """Create LiveKit-compatible supervisor using modular LangGraph approach."""
    
    # Wrapper functions to pass llm parameter to imported functions
    def intent_analyzer_wrapper(state: State):
        return intent_analyzer(state, llm)
    
    def employee_specialist_wrapper(state: State):
        return employee_specialist(state, llm)
    
    def company_specialist_wrapper(state: State):
        return company_specialist(state, llm)
    
    def project_specialist_wrapper(state: State):
        return project_specialist(state, llm)
    
    def job_specialist_wrapper(state: State):
        return job_specialist(state, llm)
    
    def admin_specialist_wrapper(state: State):
        return admin_specialist(state, llm)
    
    def general_receptionist_wrapper(state: State):
        return general_receptionist(state, llm)
    
    def route_intent(state: State):
        """Route based on detected intent."""
        intent = state.get("intent", "GENERAL")
        
        if intent == "EMPLOYEE":
            return "employee_specialist"
        elif intent == "COMPANY":
            return "company_specialist"
        elif intent == "PROJECT":
            return "project_specialist"
        elif intent == "JOB":
            return "job_specialist"
        elif intent == "ADMIN":
            return "admin_specialist"
        else:
            return "general_receptionist"
    
    # Create LangGraph with intelligent routing
    from langgraph.graph import StateGraph, START, END
    builder = StateGraph(State)
    
    # Add all specialist nodes
    builder.add_node("intent_analyzer", intent_analyzer_wrapper)
    builder.add_node("employee_specialist", employee_specialist_wrapper)
    builder.add_node("company_specialist", company_specialist_wrapper)
    builder.add_node("project_specialist", project_specialist_wrapper)
    builder.add_node("job_specialist", job_specialist_wrapper)
    builder.add_node("admin_specialist", admin_specialist_wrapper)
    builder.add_node("general_receptionist", general_receptionist_wrapper)
    
    # Set entry point
    builder.add_edge(START, "intent_analyzer")
    
    # Add conditional routing based on intent
    builder.add_conditional_edges(
        "intent_analyzer",
        route_intent,
        {
            "employee_specialist": "employee_specialist",
            "company_specialist": "company_specialist", 
            "project_specialist": "project_specialist",
            "job_specialist": "job_specialist",
            "admin_specialist": "admin_specialist",
            "general_receptionist": "general_receptionist"
        }
    )
    
    # All specialists end the conversation
    builder.add_edge("employee_specialist", END)
    builder.add_edge("company_specialist", END)
    builder.add_edge("project_specialist", END)
    builder.add_edge("job_specialist", END)
    builder.add_edge("admin_specialist", END)
    builder.add_edge("general_receptionist", END)
    
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
