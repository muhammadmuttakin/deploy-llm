from rag.llm import ask_llm
from rag.prompt_loader import load_prompt
from typing import Literal

QueryType = Literal["tracking", "learning", "recommendation"]

def classify_query(query: str) -> QueryType:
    """
    Classify user query into 'tracking', 'learning', or 'recommendation' category.
    
    Args:
        query: User's question
    
    Returns:
        'tracking' if asking about progress/courses completed
        'learning' if asking for study material/explanations
        'recommendation' if asking for course suggestions/next steps
    """
    system_prompt = load_prompt("classifier")
    
    prompt = f"""Klasifikasikan pertanyaan berikut:

"{query}"

Jawab hanya dengan satu kata: tracking, learning, atau recommendation"""
    
    response = ask_llm(prompt, system_prompt=system_prompt).strip().lower()
    
    # Extract the classification from response
    if "tracking" in response:
        return "tracking"
    elif "recommendation" in response:
        return "recommendation"
    else:
        return "learning"  # Default to learning if unclear