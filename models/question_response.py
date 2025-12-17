from pydantic import BaseModel, Field

class QuestionResponse(BaseModel):
    reply: str = Field(
        ...,
        description="LLM generated reply"
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input tokens"
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Number of output tokens"
    )
