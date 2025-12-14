from typing import List, Dict, Optional
from rag.data_loader import get_data_loader
from rag.llm import ask_llm
from rag.prompt_loader import load_prompt
from rag.history import get_history_manager
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    ✅ Refactored: Smart course recommendation.
    Retrieve data dari JSON → Pass ke LLM (no RAG, no embedding)
    """

    def __init__(self):
        self.data_loader = get_data_loader()

    def get_all_learning_paths(self) -> List[Dict]:
        """Get all available learning paths"""
        return self.data_loader.learning_paths

    def get_courses_by_learning_path(self, learning_path_id: int) -> List[Dict]:
        """Get all courses in a specific learning path"""
        return self.data_loader.get_courses_by_learning_path(learning_path_id)

    def get_learning_path_by_name(self, path_name: str) -> Optional[Dict]:
        """Get learning path by name"""
        for path in self.data_loader.learning_paths:
            if path.get("learning_path_name", "").lower() == path_name.lower():
                return path
        return None

    def get_recommended_courses(
        self,
        learning_path_name: str = None,
        course_level: int = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get recommended courses based on learning path and/or level.
        
        Args:
            learning_path_name: Filter by learning path name (optional)
            course_level: Filter by course level ID (optional)
            limit: Maximum number of courses to return
        
        Returns:
            List of recommended courses
        """
        try:
            courses = self.data_loader.courses

            # Filter by learning path if specified
            if learning_path_name:
                path = self.get_learning_path_by_name(learning_path_name)
                if path:
                    path_id = path.get("learning_path_id")
                    courses = [c for c in courses if c.get("learning_path_id") == path_id]
                else:
                    logger.warning(f"Learning path not found: {learning_path_name}")
                    return []

            # Filter by course level if specified
            if course_level:
                courses = [c for c in courses if c.get("course_level_str") == course_level]

            # Sort by level (ascending)
            courses.sort(key=lambda c: c.get("course_level_str", 1))

            # Add level name to each course
            recommendations = []
            for course in courses[:limit]:
                level_id = course.get("course_level_str", 1)
                level_name = self.data_loader.get_level_name(level_id)
                path_name = self.data_loader.get_learning_path_name(
                    course.get("learning_path_id")
                )

                recommendations.append({
                    "course_id": course.get("course_id"),
                    "course_name": course.get("course_name"),
                    "level_id": level_id,
                    "level_name": level_name,
                    "learning_path_id": course.get("learning_path_id"),
                    "learning_path_name": path_name,
                })

            return recommendations
        
        except Exception as e:
            logger.error(f"Error in get_recommended_courses: {e}")
            raise

    def get_learning_path_overview(self, learning_path_name: str) -> str:
        """Get overview of courses in a learning path"""
        path = self.get_learning_path_by_name(learning_path_name)
        if not path:
            return f"Learning path '{learning_path_name}' tidak ditemukan"

        path_id = path.get("learning_path_id")
        courses = self.get_courses_by_learning_path(path_id)

        if not courses:
            return f"Tidak ada kursus tersedia untuk {learning_path_name}"

        overview = f"📚 {learning_path_name}\n"
        overview += f"Total Kursus: {len(courses)}\n\n"

        # Group by level
        by_level = {}
        for course in courses:
            level_id = course.get("course_level_str", 1)
            level_name = self.data_loader.get_level_name(level_id)
            if level_name not in by_level:
                by_level[level_name] = []
            by_level[level_name].append(course)

        for level_name in sorted(by_level.keys()):
            overview += f"\n{level_name}:\n"
            for course in by_level[level_name]:
                overview += f"  - {course.get('course_name')}\n"

        return overview

    def answer_recommendation_query(
        self,
        query: str,
        session_id: str = None
    ) -> str:
        """
        ✅ Answer recommendation queries.
        Retrieve data dari JSON → Pass ke LLM (no RAG)
        
        Args:
            query: User's question
            session_id: Optional session ID for conversation history
        
        Returns:
            Recommendation answer from LLM
        """
        try:
            history_context = ""
            if session_id:
                history_manager = get_history_manager()
                history_context = history_manager.get_conversation_context(
                    session_id, last_n=3
                )

            # Get all learning paths
            all_paths = self.get_all_learning_paths()
            paths_list = "\n".join(
                [f"- {p.get('learning_path_name')}" for p in all_paths]
            )

            # Get all levels
            all_levels = self.data_loader.course_levels
            levels_list = "\n".join(
                [f"- {l.get('course_level')} (Level {l.get('id')})" for l in all_levels]
            )

            # Get all courses (limit untuk context, no embedding)
            all_courses = self.data_loader.courses[:20]  # Limit ke 20 courses
            
            context = "LEARNING PATHS TERSEDIA:\n"
            context += paths_list + "\n\n"

            context += "COURSE LEVELS:\n"
            context += levels_list + "\n\n"

            context += "SAMPLE KURSUS (dari total {}):\n".format(len(self.data_loader.courses))
            for i, course in enumerate(all_courses, 1):
                level_name = self.data_loader.get_level_name(
                    course.get("course_level_str", 1)
                )
                path_name = self.data_loader.get_learning_path_name(
                    course.get("learning_path_id")
                )
                context += (
                    f"{i}. {course.get('course_name')}\n"
                    f"   Path: {path_name} | Level: {level_name}\n"
                )

            # Load system prompt
            system_prompt = load_prompt("recommendation")

            # Build final prompt
            prompt = f"""
{history_context}

{context}

PERTANYAAN USER:
{query}

JAWABAN:
"""
            return ask_llm(prompt, system_prompt=system_prompt)
        
        except Exception as e:
            logger.error(f"Error in answer_recommendation_query: {e}")
            raise


# Singleton
_recommendation_engine = None


def get_recommendation_engine() -> RecommendationEngine:
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine