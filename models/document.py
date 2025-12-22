from pydantic import BaseModel, Field

class Document(BaseModel):
    fileName: str = Field(..., description="Document name")
    content: str = Field(..., description="Document text content")
