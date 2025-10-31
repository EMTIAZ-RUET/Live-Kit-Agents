"""Company information subagent."""

from subagent_prompts.company_prompts import generate_company_specialist_prompt, generate_general_receptionist_prompt

# Only specialist functions needed for modular LangGraph approach

def company_specialist(state, llm):
    """Handle company information queries with modular prompt."""
    messages = state["messages"]
    system_message = generate_company_specialist_prompt()
    
    enhanced_messages = [{"role": "system", "content": system_message}] + [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, 'type')]
    response = llm.invoke(enhanced_messages)
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}

def general_receptionist(state, llm):
    """Handle general inquiries and greetings with modular prompt."""
    messages = state["messages"]
    system_message = generate_general_receptionist_prompt()
    
    enhanced_messages = [{"role": "system", "content": system_message}] + [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, 'type')]
    response = llm.invoke(enhanced_messages)
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}
