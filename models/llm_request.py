from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class LLMRequest(BaseModel):
    user_prompt: str
    system_prompt: str
    context: Optional[str] = None # Context received from user
    additional_context: Optional[str] = None # Context retrieved from RAG
    max_output_tokens: int = 512
    temperature: float = 0.7
    extra_metadata: Optional[Dict[str, Any]] = None
