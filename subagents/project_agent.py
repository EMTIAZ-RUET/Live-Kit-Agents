"""Project discussion subagent."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from tools.communication_tools import communication_tools
from tools.company_tools import get_company_services
from subagent_prompts.project_prompts import generate_project_assistant_prompt

# Combine tools for project agent
project_discussion_tools = communication_tools + [get_company_services]

def project_assistant(state, config: RunnableConfig, llm_with_tools):
    """
    Project discussion assistant node that handles project inquiries.
    
    Args:
        state: Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        llm_with_tools: LLM with bound project tools
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the project assistant agent
    project_assistant_prompt = generate_project_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_tools.invoke([SystemMessage(project_assistant_prompt)] + state["messages"])
    
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

def create_project_subagent(llm, checkpointer: MemorySaver, store: InMemoryStore, state_schema):
    """Create project discussion sub-agent with ReAct pattern."""
    
    # Bind tools to LLM
    llm_with_project_tools = llm.bind_tools(project_discussion_tools)
    
    # Create tool node
    project_tool_node = ToolNode(project_discussion_tools)
    
    # Create wrapper function with bound LLM
    def project_assistant_wrapper(state, config: RunnableConfig):
        return project_assistant(state, config, llm_with_project_tools)
    
    workflow = StateGraph(state_schema)
    
    # Add nodes to the graph
    workflow.add_node("project_assistant", project_assistant_wrapper)
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
    
    return workflow.compile(name="project_subagent", checkpointer=checkpointer, store=store)
