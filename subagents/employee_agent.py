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
    """Create employee information sub-agent with ReAct pattern."""
    
    # Bind tools to LLM
    llm_with_employee_tools = llm.bind_tools(employee_tools_combined)
    
    # Create tool node
    employee_tool_node = ToolNode(employee_tools_combined)
    
    # Create wrapper function with bound LLM
    def employee_assistant_wrapper(state, config: RunnableConfig):
        return employee_assistant(state, config, llm_with_employee_tools)
    
    workflow = StateGraph(state_schema)
    
    # Add nodes to the graph
    workflow.add_node("employee_assistant", employee_assistant_wrapper)
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
    
    return workflow.compile(name="employee_subagent", checkpointer=checkpointer, store=store)
