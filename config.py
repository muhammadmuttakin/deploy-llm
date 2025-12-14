import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Data Paths (untuk tracking & recommendation saja, bukan RAG)
COURSES_PATH = os.getenv("COURSES_PATH", "data/courses.json")
LEARNING_PATHS_PATH = os.getenv("LEARNING_PATHS_PATH", "data/learning_paths.json")
COURSE_LEVELS_PATH = os.getenv("COURSE_LEVELS_PATH", "data/course_levels.json")
TUTORIALS_PATH = os.getenv("TUTORIALS_PATH", "data/tutorials.json")

