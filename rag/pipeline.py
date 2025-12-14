from rag.llm import ask_llm
from rag.classifier import classify_query
from rag.tracking import get_tracker
from rag.recommendation import get_recommendation_engine
from rag.history import get_history_manager
from rag.prompt_loader import load_prompt


def learning_answer(query: str, session_id: str = None) -> str:
    """
    ✅ Answer learning queries WITHOUT RAG.
    Langsung pass ke LLM dengan system prompt saja.
    
    Args:
        query: User's learning question
        session_id: Optional session ID for conversation history
    
    Returns:
        Answer from LLM
    """
    # Get conversation history if available
    history_context = ""
    if session_id:
        history_manager = get_history_manager()
        history_context = history_manager.get_conversation_context(session_id, last_n=3)
    
    # Load system prompt untuk learning
    system_prompt = load_prompt("learning")
    
    # Simple prompt tanpa RAG context - langsung ke LLM
    prompt = f"""
{history_context}

PERTANYAAN:
{query}

JAWABAN:
"""
    
    return ask_llm(prompt, system_prompt=system_prompt)


def smart_answer(query: str, session_id: str = None) -> dict:
    """
    Main entry point - classify query dan route ke handler yang tepat.
    
    Flow:
    - tracking → retrieve dari API user progress
    - recommendation → retrieve dari JSON data
    - learning → langsung ke LLM (no RAG)
    """
    
    # Step 1: Classify query type
    query_type = classify_query(query)
    
    # Step 2: Route ke handler yang sesuai
    if query_type == "tracking":
        # ✅ Tracking: Retrieve data dari API, pass ke LLM
        tracker = get_tracker()
        answer = tracker.answer_tracking_query(query, session_id=session_id)
    
    elif query_type == "recommendation":
        # ✅ Recommendation: Retrieve dari JSON, pass ke LLM
        recommendation_engine = get_recommendation_engine()
        answer = recommendation_engine.answer_recommendation_query(query, session_id=session_id)
    
    else:  # learning
        # ✅ Learning: LANGSUNG KE LLM (no RAG, no embedding, no vector search)
        answer = learning_answer(query, session_id=session_id)
    
    # Step 3: Save ke history jika ada session_id
    if session_id:
        history_manager = get_history_manager()
        history_manager.save_message(
            session_id=session_id,
            query=query,
            answer=answer,
            query_type=query_type
        )
    
    return {
        "answer": answer,
        "type": query_type,
        "session_id": session_id
    }