import json
from typing import Dict, List, Optional
from pathlib import Path

class DataLoader:
    """Centralized data loader for all JSON files"""
    
    def __init__(
        self,
        courses_path: str = "data/courses.json",
        learning_paths_path: str = "data/learning_paths.json",
        course_levels_path: str = "data/course_levels.json",
        tutorials_path: str = "data/tutorials.json"
    ):
        self.courses_path = courses_path
        self.learning_paths_path = learning_paths_path
        self.course_levels_path = course_levels_path
        self.tutorials_path = tutorials_path
        
        # Cache data
        self._courses: Optional[List[Dict]] = None
        self._learning_paths: Optional[List[Dict]] = None
        self._course_levels: Optional[List[Dict]] = None
        self._tutorials: Optional[List[Dict]] = None
        
        # Lookup dictionaries for fast access
        self._course_by_id: Optional[Dict[int, Dict]] = None
        self._path_by_id: Optional[Dict[int, Dict]] = None
        self._level_by_id: Optional[Dict[int, Dict]] = None
    
    def _load_json(self, path: str) -> List[Dict]:
        """Load JSON file with error handling"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: {path} not found, returning empty list")
            return []
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Invalid JSON in {path}: {e}")
            return []
    
    @property
    def courses(self) -> List[Dict]:
        """Get all courses"""
        if self._courses is None:
            self._courses = self._load_json(self.courses_path)
        return self._courses
    
    @property
    def learning_paths(self) -> List[Dict]:
        """Get all learning paths"""
        if self._learning_paths is None:
            self._learning_paths = self._load_json(self.learning_paths_path)
        return self._learning_paths
    
    @property
    def course_levels(self) -> List[Dict]:
        """Get all course levels"""
        if self._course_levels is None:
            self._course_levels = self._load_json(self.course_levels_path)
        return self._course_levels
    
    @property
    def tutorials(self) -> List[Dict]:
        """Get all tutorials"""
        if self._tutorials is None:
            self._tutorials = self._load_json(self.tutorials_path)
        return self._tutorials
    
    def get_course_by_id(self, course_id: int) -> Optional[Dict]:
        """Get course by ID"""
        if self._course_by_id is None:
            self._course_by_id = {c["course_id"]: c for c in self.courses}
        return self._course_by_id.get(course_id)
    
    def get_learning_path_by_id(self, path_id: int) -> Optional[Dict]:
        """Get learning path by ID"""
        if self._path_by_id is None:
            self._path_by_id = {p["learning_path_id"]: p for p in self.learning_paths}
        return self._path_by_id.get(path_id)
    
    def get_level_by_id(self, level_id: int) -> Optional[Dict]:
        """Get course level by ID"""
        if self._level_by_id is None:
            self._level_by_id = {l["id"]: l for l in self.course_levels}
        return self._level_by_id.get(level_id)
    
    def get_courses_by_learning_path(self, learning_path_id: int) -> List[Dict]:
        """Get all courses for a specific learning path"""
        return [
            c for c in self.courses 
            if c.get("learning_path_id") == learning_path_id
        ]
    
    def get_tutorials_by_course(self, course_id: int) -> List[Dict]:
        """Get all tutorials for a specific course"""
        return [
            t for t in self.tutorials
            if t.get("course_id") == course_id
        ]
    
    def get_level_name(self, level_id: int) -> str:
        """Get level name by ID"""
        level = self.get_level_by_id(level_id)
        return level.get("course_level", "Unknown") if level else "Unknown"
    
    def get_learning_path_name(self, path_id: int) -> str:
        """Get learning path name by ID"""
        path = self.get_learning_path_by_id(path_id)
        return path.get("learning_path_name", "Unknown") if path else "Unknown"


# Singleton instance
_data_loader = None

def get_data_loader() -> DataLoader:
    """Get or create DataLoader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader