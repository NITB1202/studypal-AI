from typing import List, Optional
from pydantic import BaseModel, Field
from models.document import Document

class QuestionRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="User prompt"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context"
    )
    attachments: Optional[List[Document]] = Field(
        default=None,
        description="List of attachments"
    )
    max_output_tokens: int = Field(
        ...,
        gt=0,
        description="Maximum number of tokens for LLM output"
    )
