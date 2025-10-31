"""Project discussion subagent."""

from subagent_prompts.project_prompts import generate_project_specialist_prompt

# Only specialist functions needed for modular LangGraph approach

def project_specialist(state, llm):
    """Handle project discussion queries with modular prompt."""
    messages = state["messages"]
    system_message = generate_project_specialist_prompt()
    
    enhanced_messages = [{"role": "system", "content": system_message}] + [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, 'type')]
    response = llm.invoke(enhanced_messages)
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}
