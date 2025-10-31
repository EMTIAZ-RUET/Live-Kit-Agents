"""Administrative and compliance subagent."""

from subagent_prompts.admin_prompts import generate_admin_specialist_prompt

# Only specialist functions needed for modular LangGraph approach

def admin_specialist(state, llm):
    """Handle administrative queries with modular prompt."""
    messages = state["messages"]
    system_message = generate_admin_specialist_prompt()
    
    enhanced_messages = [{"role": "system", "content": system_message}] + [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, 'type')]
    response = llm.invoke(enhanced_messages)
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}
