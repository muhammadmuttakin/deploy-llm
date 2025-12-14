import json
import os
import requests
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from rag.llm import ask_llm
from rag.prompt_loader import load_prompt
from rag.history import get_history_manager


class ProgressTracker:
    """
    ✅ Refactored: Retrieve user progress dari API, pass ke LLM.
    No RAG, no embedding. Langsung retrieve + prompt.
    """
    def __init__(self):
        """Initialize progress tracker with user data from API (.env)"""
        load_dotenv()

        self.api_url = os.getenv("USER_API_URL")
        self.user_id = os.getenv("USER_ID")

        if not self.api_url:
            raise ValueError("USER_API_URL not found in .env")
        if not self.user_id:
            raise ValueError("USER_ID not found in .env")

        self.user_data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load user progress data from API instead of JSON file"""
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Cari user berdasarkan ID
            target_user = None
            for u in data.get("users", []):
                if u["_id"] == self.user_id:
                    target_user = u
                    break

            if not target_user:
                raise ValueError(f"User with id {self.user_id} not found")

            # Konversi struktur API → struktur lama
            converted = {
                "user": {
                    "name": target_user.get("name", ""),
                    "learning_path": "Dicoding Learning Path",
                    "courses": []
                }
            }

            for c in target_user.get("classes", []):
                converted["user"]["courses"].append({
                    "course_id": c.get("course_id"),
                    "course_name": c.get("course_name"),
                    "progress": int(c.get("progress", 0)),
                    "deadline": c.get("deadline")
                })

            return converted

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to load data from API: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error when loading API data: {e}")

    def get_progress_context(self) -> str:
        """Build context string from user progress data"""
        user = self.user_data["user"]
        today = datetime.today().date()

        # Basic statistics
        total_courses = len(user["courses"])
        completed_courses = sum(1 for c in user["courses"] if c["progress"] == 100)
        in_progress_courses = [c for c in user["courses"] if 0 < c["progress"] < 100]
        not_started = total_courses - completed_courses - len(in_progress_courses)

        avg_progress = (
            sum(c["progress"] for c in user["courses"]) / total_courses
            if total_courses > 0 else 0
        )

        # Deadline insights
        deadlines = []
        overdue_courses = []

        for c in user["courses"]:
            if "deadline" in c and c["deadline"]:
                try:
                    deadline_date = datetime.strptime(c["deadline"], "%Y-%m-%d").date()
                    days_left = (deadline_date - today).days
                    deadlines.append((c["course_name"], deadline_date, days_left))

                    if days_left < 0 and c["progress"] < 100:
                        overdue_courses.append(
                            (c["course_name"], abs(days_left))
                        )
                except ValueError:
                    pass

        # Nearest deadline
        nearest_deadline = min(deadlines, key=lambda x: x[2]) if deadlines else None

        # Estimate remaining progress
        remaining_progress = sum(100 - c["progress"] for c in user["courses"])
        max_total = total_courses * 100
        remaining_percent = (remaining_progress / max_total * 100) if max_total > 0 else 0

        context = f"""
DATA PROGRESS PENGGUNA:
- Nama: {user['name']}
- Learning Path: {user['learning_path']}

STATISTIK UTAMA:
- Total Kursus: {total_courses}
- Selesai: {completed_courses} kursus ({(completed_courses/total_courses*100) if total_courses else 0:.0f}%)
- Sedang Berjalan: {len(in_progress_courses)} kursus
- Belum Dimulai: {not_started} kursus
- Progress Rata-rata: {avg_progress:.0f}%
- Sisa Progress Total: {remaining_percent:.0f}%

INSIGHT DEADLINE:
"""

        if nearest_deadline:
            context += f"- Deadline terdekat: {nearest_deadline[0]} ({nearest_deadline[2]} hari lagi)\n"

        if overdue_courses:
            context += "- Kursus terlambat:\n"
            for name, days in overdue_courses:
                context += f"  ⚠️ {name} (terlambat {days} hari)\n"
        else:
            context += "- Tidak ada kursus yang terlambat ✅\n"

        context += "\nDETAIL KURSUS:\n"

        # Completed
        if completed_courses > 0:
            context += "\n✅ KURSUS SELESAI:\n"
            for course in user["courses"]:
                if course["progress"] == 100:
                    context += f"  - {course['course_name']}\n"

        # In progress
        if in_progress_courses:
            context += "\n⏳ SEDANG BERJALAN:\n"
            for course in in_progress_courses:
                days_info = ""
                if "deadline" in course and course["deadline"]:
                    try:
                        d = datetime.strptime(course["deadline"], "%Y-%m-%d").date()
                        days_left = (d - today).days
                        days_info = f" | {days_left} hari menuju deadline"
                    except ValueError:
                        pass

                context += (
                    f"  - {course['course_name']}: {course['progress']}%{days_info}\n"
                )

        # Not started
        if not_started > 0:
            context += f"\n📋 BELUM DIMULAI: {not_started} kursus\n"

        return context

    def answer_tracking_query(self, query: str, session_id: str = None) -> str:
        """
        ✅ Answer progress tracking questions.
        Retrieve data dari API → Pass ke LLM (no RAG)
        """
        history_context = ""
        if session_id:
            history_manager = get_history_manager()
            history_context = history_manager.get_conversation_context(session_id, last_n=3)

        context = self.get_progress_context()
        system_prompt = load_prompt("tracking")

        prompt = f"""
{history_context}

{context}

PERTANYAAN USER:
{query}

JAWABAN:
"""
        return ask_llm(prompt, system_prompt=system_prompt)


# Singleton instance
_tracker = None

def get_tracker() -> ProgressTracker:
    """Get or create ProgressTracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = ProgressTracker()
    return _tracker