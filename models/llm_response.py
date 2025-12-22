from pydantic import BaseModel

class LLMResponse(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int