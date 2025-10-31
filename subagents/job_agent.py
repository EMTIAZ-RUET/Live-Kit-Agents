"""Job opportunities subagent."""

from subagent_prompts.job_prompts import generate_job_specialist_prompt

# Only specialist functions needed for modular LangGraph approach

def job_specialist(state, llm):
    """Handle career and job queries with modular prompt."""
    messages = state["messages"]
    system_message = generate_job_specialist_prompt()
    
    enhanced_messages = [{"role": "system", "content": system_message}] + [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, 'type')]
    response = llm.invoke(enhanced_messages)
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}
