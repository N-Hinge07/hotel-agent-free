from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    guest_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    reply: str
    intent: Optional[str] = None
    suggested_actions: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


