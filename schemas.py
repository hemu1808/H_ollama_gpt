from pydantic import BaseModel, validator
from typing import List, Optional, Literal
import re

# --- ADD THIS MISSING CLASS ---
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class QueryInput(BaseModel):
    question: str
    top_k: int = 4
    mode: Literal["fast", "deep"] = "deep"
    chat_history: List[ChatMessage] = []
    
    
    @validator('question')
    def validate_question(cls, v):
        if not v or len(v.strip()) < 2: 
            raise ValueError("Question too short")
        return re.sub(r'[<>]', '', v).strip()

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    metadata: dict
    processing_time: float
    thoughts: Optional[str] = None