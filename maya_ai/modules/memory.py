#!/usr/bin/env python3
"""
MAYA AI Memory Module
Long-term memory and personalization
Like: ChatGPT Memory, Claude Memory
"""

import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class UserProfile:
    """User profile with preferences and history"""
    user_id: str
    name: str = ""
    preferences: Dict = field(default_factory=dict)
    conversation_topics: List = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

class MemoryStore:
    """
    Long-term memory storage
    Like: ChatGPT Memory, Claude persistent memory
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".maya_ai" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, UserProfile] = {}
        self.memories: Dict[str, List[Dict]] = {}
    
    def create_profile(self, user_id: str, name: str = "", 
                     preferences: Dict = None) -> UserProfile:
        """Create new user profile"""
        profile = UserProfile(
            user_id=user_id,
            name=name,
            preferences=preferences or {}
        )
        self.profiles[user_id] = profile
        self._save_profile(user_id)
        return profile
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        if user_id not in self.profiles:
            self._load_profile(user_id)
        return self.profiles.get(user_id)
    
    def update_preference(self, user_id: str, key: str, value: Any):
        """Update user preference"""
        if user_id not in self.profiles:
            self.create_profile(user_id)
        
        self.profiles[user_id].preferences[key] = value
        self.profiles[user_id].last_active = time.time()
        self._save_profile(user_id)
    
    def add_memory(self, user_id: str, memory_type: str, 
                   content: str, importance: int = 1):
        """Add a memory for user"""
        if user_id not in self.memories:
            self.memories[user_id] = []
        
        self.memories[user_id].append({
            'type': memory_type,
            'content': content,
            'importance': importance,
            'timestamp': time.time()
        })
        
        self._save_memory(user_id)
    
    def get_memories(self, user_id: str, memory_type: str = None,
                    limit: int = 10) -> List[Dict]:
        """Get memories for user"""
        memories = self.memories.get(user_id, [])
        
        if memory_type:
            memories = [m for m in memories if m['type'] == memory_type]
        
        # Sort by importance and recency
        memories.sort(key=lambda x: (x['importance'], x['timestamp']), 
                     reverse=True)
        
        return memories[:limit]
    
    def _save_profile(self, user_id: str):
        """Save profile to disk"""
        profile = self.profiles.get(user_id)
        if profile:
            path = self.storage_path / f"{user_id}_profile.json"
            with open(path, 'w') as f:
                json.dump(profile.__dict__, f)
    
    def _load_profile(self, user_id: str):
        """Load profile from disk"""
        path = self.storage_path / f"{user_id}_profile.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.profiles[user_id] = UserProfile(**data)
    
    def _save_memory(self, user_id: str):
        """Save memories to disk"""
        memories = self.memories.get(user_id, [])
        path = self.storage_path / f"{user_id}_memories.json"
        with open(path, 'w') as f:
            json.dump(memories, f)
    
    def _load_memory(self, user_id: str):
        """Load memories from disk"""
        path = self.storage_path / f"{user_id}_memories.json"
        if path.exists():
            with open(path, 'r') as f:
                self.memories[user_id] = json.load(f)