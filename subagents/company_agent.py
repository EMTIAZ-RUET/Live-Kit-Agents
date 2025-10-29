"""Company information subagent."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from tools.company_tools import company_info_tools
from subagent_prompts.company_prompts import generate_company_assistant_prompt

def company_assistant(state, config: RunnableConfig, llm_with_tools):
    """
    Company information assistant node that handles company-related queries.
    
    Args:
        state: Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        llm_with_tools: LLM with bound company tools
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Generate instructions for the company assistant agent
    company_assistant_prompt = generate_company_assistant_prompt()

    # Invoke the language model with tools and system prompt
    response = llm_with_tools.invoke([SystemMessage(company_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def should_continue(state, config: RunnableConfig):
    """
    Conditional edge function that determines the next step in the ReAct agent workflow.
    
    Args:
        state: Current state containing messages and other workflow data
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

def create_company_subagent(llm, checkpointer: MemorySaver, store: InMemoryStore, state_schema):
    """Create company information sub-agent with ReAct pattern."""
    
    # Bind tools to LLM
    llm_with_company_tools = llm.bind_tools(company_info_tools)
    
    # Create tool node
    company_tool_node = ToolNode(company_info_tools)
    
    # Create wrapper function with bound LLM
    def company_assistant_wrapper(state, config: RunnableConfig):
        return company_assistant(state, config, llm_with_company_tools)
    
    # Create workflow with proper state schema
    workflow = StateGraph(state_schema)
    
    # Add nodes to the graph
    workflow.add_node("company_assistant", company_assistant_wrapper)
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
    
    return workflow.compile(name="company_subagent", checkpointer=checkpointer, store=store)
