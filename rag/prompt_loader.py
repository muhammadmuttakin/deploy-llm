"""
Utility untuk load system prompts dari file .txt
"""

from pathlib import Path
from typing import Dict, Optional

class PromptLoader:
    """Load and cache system prompts from text files"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
        
        # Create prompts directory if not exists
        self.prompts_dir.mkdir(exist_ok=True)
    
    def load(self, prompt_name: str) -> str:
        """
        Load prompt from file. Uses cache if already loaded.
        
        Args:
            prompt_name: Name of prompt file (without .txt extension)
        
        Returns:
            Content of prompt file
        
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Check cache first
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        # Load from file
        prompt_path = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Please create the file with appropriate system prompt."
            )
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        # Cache it
        self._cache[prompt_name] = content
        
        return content
    
    def reload(self, prompt_name: str) -> str:
        """
        Force reload prompt from file (bypass cache).
        Useful for development when prompts are being updated.
        """
        if prompt_name in self._cache:
            del self._cache[prompt_name]
        return self.load(prompt_name)
    
    def clear_cache(self):
        """Clear all cached prompts"""
        self._cache.clear()


# Singleton instance
_prompt_loader: Optional[PromptLoader] = None

def get_prompt_loader() -> PromptLoader:
    """Get or create PromptLoader instance"""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader

def load_prompt(prompt_name: str) -> str:
    """Convenience function to load a prompt"""
    return get_prompt_loader().load(prompt_name)