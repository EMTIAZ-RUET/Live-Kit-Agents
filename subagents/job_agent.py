"""Job opportunities subagent."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from tools.job_tools import job_opportunity_tools
from tools.communication_tools import communication_tools
from subagent_prompts.job_prompts import generate_job_assistant_prompt

# Combine tools for job agent
job_tools_combined = job_opportunity_tools + communication_tools

def job_assistant(state, config: RunnableConfig, llm_with_tools):
    """
    Job opportunities assistant node that handles career inquiries.
    
    Args:
        state: Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        llm_with_tools: LLM with bound job tools
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the job assistant agent
    job_assistant_prompt = generate_job_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_tools.invoke([SystemMessage(job_assistant_prompt)] + state["messages"])
    
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

def create_job_subagent(llm, checkpointer: MemorySaver, store: InMemoryStore, state_schema):
    """Create job opportunity sub-agent with ReAct pattern."""
    
    # Bind tools to LLM
    llm_with_job_tools = llm.bind_tools(job_tools_combined)
    
    # Create tool node
    job_tool_node = ToolNode(job_tools_combined)
    
    # Create wrapper function with bound LLM
    def job_assistant_wrapper(state, config: RunnableConfig):
        return job_assistant(state, config, llm_with_job_tools)
    
    workflow = StateGraph(state_schema)
    
    # Add nodes to the graph
    workflow.add_node("job_assistant", job_assistant_wrapper)
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
    
    return workflow.compile(name="job_subagent", checkpointer=checkpointer, store=store)
