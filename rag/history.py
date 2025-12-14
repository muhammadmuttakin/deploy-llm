import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class ChatHistory:
    def __init__(self, storage_dir: str = "chat_history"):
        """
        Initialize chat history manager.
        
        Args:
            storage_dir: Directory to store chat history files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get file path for a session"""
        return self.storage_dir / f"{session_id}.json"
    
    def save_message(
        self, 
        session_id: str, 
        query: str, 
        answer: str, 
        query_type: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save a chat message to history.
        
        Args:
            session_id: Unique session identifier
            query: User's question
            answer: Bot's response
            query_type: Type of query ('tracking' or 'learning')
            metadata: Additional metadata (e.g., timestamp, user_id)
        """
        session_file = self._get_session_file(session_id)
        
        # Load existing history
        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        # Add new message
        message = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "type": query_type
        }
        
        if metadata:
            message["metadata"] = metadata
        
        history["messages"].append(message)
        history["last_updated"] = datetime.now().isoformat()
        
        # Save to file
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of recent messages to return
            
        Returns:
            List of messages (most recent first if limit is set)
        """
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            return []
        
        with open(session_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        messages = history.get("messages", [])
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_context(
        self, 
        session_id: str, 
        last_n: int = 5
    ) -> str:
        """
        Get formatted conversation context for LLM.
        
        Args:
            session_id: Session identifier
            last_n: Number of recent messages to include
            
        Returns:
            Formatted conversation history string
        """
        messages = self.get_history(session_id, limit=last_n)
        
        if not messages:
            return ""
        
        context_parts = ["RIWAYAT PERCAKAPAN SEBELUMNYA:"]
        
        for msg in messages:
            context_parts.append(f"\nUser: {msg['query']}")
            
            # ✅ FIX: Jangan truncate answer, atau perbesar limitnya
            answer = msg['answer']
            if len(answer) > 500:  # Naik dari 200 ke 500
                answer = answer[:500] + "..."
            
            context_parts.append(f"Dico: {answer}")
        
        context_parts.append("\n---\nPERTANYAAN SAAT INI MUNGKIN TERKAIT DENGAN PERCAKAPAN DI ATAS.\n")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear history for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if session doesn't exist
        """
        session_file = self._get_session_file(session_id)
        
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def list_sessions(self) -> List[Dict]:
        """
        List all available sessions.
        
        Returns:
            List of session info (id, created_at, message_count)
        """
        sessions = []
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                
                sessions.append({
                    "session_id": history["session_id"],
                    "created_at": history.get("created_at"),
                    "last_updated": history.get("last_updated"),
                    "message_count": len(history.get("messages", []))
                })
            except Exception as e:
                print(f"Error reading session {session_file}: {e}")
                continue
        
        # Sort by last updated (most recent first)
        sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        
        return sessions


# Singleton instance
_history_manager = None

def get_history_manager() -> ChatHistory:
    """Get or create ChatHistory instance"""
    global _history_manager
    if _history_manager is None:
        _history_manager = ChatHistory()
    return _history_manager