from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
from rag.pipeline import smart_answer
from rag.history import get_history_manager
from rag.recommendation import get_recommendation_engine
from rag.tracking import get_tracker
from rag.data_loader import get_data_loader

app = FastAPI(
    title="Dicoding Personal Learning Assistant API",
    description="Smart Assistant with Progress Tracking, Chat History & Course Recommendations (No RAG)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


class QueryIn(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    type: str 
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]

class SessionInfo(BaseModel):
    session_id: str
    created_at: Optional[str]
    last_updated: Optional[str]
    message_count: int

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    learning_paths_count: int
    courses_count: int

class ProgressResponse(BaseModel):
    user_name: str
    learning_path: str
    total_courses: int
    completed_courses: int
    in_progress_courses: int
    completed_names: List[str]
    in_progress_details: List[Dict]

@app.post("/chat", response_model=ChatResponse)
async def chat(payload: QueryIn):
    """
    Main chat endpoint - automatically routes to tracking/learning/recommendation
    based on query classification
    """
    try:
        session_id = payload.session_id or str(uuid.uuid4())
        result = smart_answer(payload.query, session_id=session_id)
        
        return ChatResponse(
            answer=result["answer"],
            type=result["type"],
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str, limit: Optional[int] = None):
    """Get chat history for a specific session"""
    try:
        history_manager = get_history_manager()
        messages = history_manager.get_history(session_id, limit=limit)
        
        return HistoryResponse(
            session_id=session_id,
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear history for a specific session"""
    try:
        history_manager = get_history_manager()
        deleted = history_manager.clear_session(session_id)
        
        if deleted:
            return {"message": "History cleared successfully", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all chat sessions"""
    try:
        history_manager = get_history_manager()
        sessions = history_manager.list_sessions()
        
        return [SessionInfo(**session) for session in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history")
async def clear_all_history():
    """Clear all chat history"""
    try:
        history_manager = get_history_manager()
        sessions = history_manager.list_sessions()
        
        deleted_count = 0
        for session in sessions:
            if history_manager.clear_session(session["session_id"]):
                deleted_count += 1
        
        return {
            "message": f"Cleared {deleted_count} session(s)",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(limit: int = 5):
    """
    Get course recommendations based on available learning paths
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 5)
    """
    try:
        engine = get_recommendation_engine()
        recommendations = engine.get_recommended_courses(limit=limit)
        
        all_paths = engine.get_all_learning_paths()
        all_courses = engine.data_loader.courses
        
        return RecommendationResponse(
            recommendations=recommendations,
            learning_paths_count=len(all_paths),
            courses_count=len(all_courses)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/next")
async def get_next_course():
    """Get the single best next course recommendation"""
    try:
        engine = get_recommendation_engine()
        recommendations = engine.get_recommended_courses(limit=1)
        
        if not recommendations:
            return {
                "message": "No recommendations available",
                "recommendation": None
            }
        
        return {
            "message": "Next recommended course",
            "recommendation": recommendations[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/progress", response_model=ProgressResponse)
async def get_progress():
    """Get user's current learning progress"""
    try:
        tracker = get_tracker()
        user = tracker.user_data["user"]
        
        completed = [c for c in user["courses"] if c["progress"] == 100]
        in_progress = [c for c in user["courses"] if 0 < c["progress"] < 100]
        
        return ProgressResponse(
            user_name=user["name"],
            learning_path=user["learning_path"],
            total_courses=len(user["courses"]),
            completed_courses=len(completed),
            in_progress_courses=len(in_progress),
            completed_names=[c["course_name"] for c in completed],
            in_progress_details=in_progress
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/summary")
async def get_progress_summary():
    """Get a quick summary of user progress"""
    try:
        tracker = get_tracker()
        user = tracker.user_data["user"]
        
        total = len(user["courses"])
        completed = sum(1 for c in user["courses"] if c["progress"] == 100)
        avg_progress = sum(c["progress"] for c in user["courses"]) / total if total > 0 else 0
        
        return {
            "user_name": user["name"],
            "learning_path": user["learning_path"],
            "completion_rate": f"{(completed/total*100):.1f}%" if total > 0 else "0%",
            "average_progress": f"{avg_progress:.1f}%",
            "courses_completed": completed,
            "courses_total": total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/courses")
async def list_courses(learning_path_id: Optional[int] = None):
    """
    List all courses or filter by learning path
    
    Query Parameters:
    - learning_path_id: Filter by learning path (optional)
    """
    try:
        loader = get_data_loader()
        
        if learning_path_id is not None:
            courses = loader.get_courses_by_learning_path(learning_path_id)
        else:
            courses = loader.courses
        
        return {
            "total": len(courses),
            "courses": courses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning-paths")
async def list_learning_paths():
    """List all available learning paths"""
    try:
        loader = get_data_loader()
        
        return {
            "total": len(loader.learning_paths),
            "learning_paths": loader.learning_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/course-levels")
async def list_course_levels():
    """List all course difficulty levels"""
    try:
        loader = get_data_loader()
        
        return {
            "total": len(loader.course_levels),
            "levels": loader.course_levels
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Quick validation - no need to load FAISS anymore
        loader = get_data_loader()
        engine = get_recommendation_engine()
        
        return {
            "status": "ok",
            "service": "Dicoding Personal Learning Assistant",
            "version": "3.0.0",
            "features": [
                "direct_llm",  # Changed from "rag" to "direct_llm"
                "tracking",
                "recommendation",
                "history"
            ],
            "data_stats": {
                "courses": len(loader.courses),
                "learning_paths": len(loader.learning_paths),
                "course_levels": len(loader.course_levels)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "Dicoding Personal Learning Assistant",
            "error": str(e)
        }

@app.get("/")
async def root():
    """API root - documentation redirect"""
    return {
        "message": "Dicoding Personal Learning Assistant API",
        "version": "3.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "chat": "POST /chat",
            "recommendations": "GET /recommendations",
            "progress": "GET /progress",
            "history": "GET /history/{session_id}",
            "health": "GET /health"
        }
    }


# ✅ FIXED: Exception handlers sekarang return JSONResponse
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "detail": str(exc)
        }
    )