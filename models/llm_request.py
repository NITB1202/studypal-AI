from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from models.rag_chunk import RAGChunk

class LLMRequest(BaseModel):
    user_prompt: str
    system_prompt: str
    context: Optional[str] = None # Context received from user
    attachment_chunks: Optional[List[RAGChunk]] = None  # Chunks from attachments
    kb_chunks: Optional[List[RAGChunk]] = None # Knowledge base chunks retrieved from RAG
    max_output_tokens: int = 512
    temperature: float = 0.7
    extra_metadata: Optional[Dict[str, Any]] = None
