"""Admin/finance subagent."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from tools.communication_tools import communication_tools
from subagent_prompts.admin_prompts import generate_admin_assistant_prompt

# Admin tools are just communication tools
admin_finance_tools = communication_tools

def admin_assistant(state, config: RunnableConfig, llm_with_tools):
    """
    Admin/finance assistant node that handles administrative matters.
    
    Args:
        state: Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        llm_with_tools: LLM with bound admin tools
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the admin assistant agent
    admin_assistant_prompt = generate_admin_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_tools.invoke([SystemMessage(admin_assistant_prompt)] + state["messages"])
    
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

def create_admin_subagent(llm, checkpointer: MemorySaver, store: InMemoryStore, state_schema):
    """Create admin/finance sub-agent with ReAct pattern."""
    
    # Bind tools to LLM
    llm_with_admin_tools = llm.bind_tools(admin_finance_tools)
    
    # Create tool node
    admin_tool_node = ToolNode(admin_finance_tools)
    
    # Create wrapper function with bound LLM
    def admin_assistant_wrapper(state, config: RunnableConfig):
        return admin_assistant(state, config, llm_with_admin_tools)
    
    workflow = StateGraph(state_schema)
    
    # Add nodes to the graph
    workflow.add_node("admin_assistant", admin_assistant_wrapper)
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
    
    return workflow.compile(name="admin_subagent", checkpointer=checkpointer, store=store)
