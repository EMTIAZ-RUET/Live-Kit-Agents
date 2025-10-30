"""Employee contact subagent."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from tools.employee_tools import employee_info_tools
from tools.communication_tools import communication_tools
from subagent_prompts.employee_prompts import generate_employee_assistant_prompt

# Combine tools for employee agent
employee_tools_combined = employee_info_tools + communication_tools

def employee_assistant(state, config: RunnableConfig, llm_with_tools):
    """
    Employee contact assistant node that handles employee connection requests.
    
    Args:
        state: Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        llm_with_tools: LLM with bound employee tools
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the employee assistant agent
    employee_assistant_prompt = generate_employee_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_tools.invoke([SystemMessage(employee_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def should_continue(state, config: RunnableConfig):
    """
    Conditional edge function that determines the next step in the ReAct agent workflow.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def create_employee_subagent(llm, checkpointer: MemorySaver, store: InMemoryStore, state_schema):
    """Create LiveKit-compatible employee sub-agent (no tools to prevent hanging)."""
    
    def employee_node(state):
        """Simple employee assistant without tools (LiveKit compatible)."""
        messages = state["messages"]
        
        # Employee knowledge embedded in prompt (no tools needed)
        employee_prompt = """You are the Employee Contact Specialist for Brain Station 23.
You help callers connect with employees and provide employee information.

EMPLOYEE DIRECTORY:
- John Doe: Senior Developer, Engineering Department, john.doe@brainstation-23.com
- Jane Smith: Project Manager, Operations Department, jane.smith@brainstation-23.com  
- Ahmed Hassan: HR Manager, Human Resources Department, ahmed.hassan@brainstation-23.com
- David Johnson: Senior Developer, Engineering Department, john.doe@brainstation-23.com

SECURITY PROTOCOL:
- Always collect caller information before connecting to employees
- Ask for: caller name, company, purpose of contact
- For security, verify the request is legitimate

RESPONSES:
- Provide employee information when requested
- Offer to connect callers to appropriate employees
- Collect caller details for security purposes
- Be professional and helpful

Handle employee-related requests professionally and securely."""
        
        # Create enhanced messages with employee context
        enhanced_messages = [{"role": "system", "content": employee_prompt}] + messages
        response = llm.invoke(enhanced_messages)
        return {"messages": [response]}
    
    # Simple StateGraph like working langraph_implementation.py (no tools, no conditional edges)
    workflow = StateGraph(state_schema)
    workflow.add_node("employee_assistant", employee_node)
    workflow.add_edge(START, "employee_assistant")
    
    # Simple compile without checkpointer for LiveKit compatibility
    return workflow.compile()
