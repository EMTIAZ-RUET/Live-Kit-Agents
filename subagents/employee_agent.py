"""Employee contact subagent."""

from subagent_prompts.employee_prompts import generate_employee_specialist_prompt, generate_intent_classifier_prompt

# Only specialist functions needed for modular LangGraph approach

def employee_specialist(state, llm):
    """Handle employee-related queries with modular prompt."""
    messages = state["messages"]
    system_message = generate_employee_specialist_prompt()
    
    enhanced_messages = [{"role": "system", "content": system_message}] + [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, 'type')]
    response = llm.invoke(enhanced_messages)
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}

def intent_analyzer(state, llm):
    """Analyze user intent using LLM and route accordingly."""
    messages = state["messages"]
    last_user_message = messages[-1].content if messages else ""
    
    # Get intent analysis prompt from modular file
    intent_prompt = generate_intent_classifier_prompt()
    
    # Single LLM call for intent analysis
    intent_response = llm.invoke([{"role": "user", "content": intent_prompt.format(user_message=last_user_message)}])
    intent = intent_response.content.strip().upper()
    
    print(f"🎯 Intent detected: {intent}")
    
    # Store intent for routing
    return {"messages": messages, "intent": intent}
