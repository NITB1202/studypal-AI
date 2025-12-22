import openai
from openai import OpenAI

from config.settings import settings
from models.llm_request import LLMRequest
from models.llm_response import LLMResponse

class LLMService:
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def _build_instructions(self, request: LLMRequest):
        instructions = request.system_prompt

        if request.context:
            instructions += f"\nUser context:\n{request.context}"

        if request.additional_context:
            instructions += f"\nRelevant data:\n{request.additional_context}"

        return instructions

    def send_request(self, request: LLMRequest) -> LLMResponse:
        try:
            # Build instructions
            instructions = self._build_instructions(request)

            # Call Responses API
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=request.user_prompt,
            )

            # Extract answer and token usage
            if hasattr(response, "output_text"):
                answer_text = response.output_text.strip()
            else:
                answer_text = response.output[0].content[0].text.strip()

            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0)
                output_tokens = getattr(response.usage, "output_tokens", 0)
            else:
                input_tokens = 0
                output_tokens = 0

            return LLMResponse(
                answer=answer_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            return LLMResponse(
                answer=f"Error calling OpenAI API: {e}",
                input_tokens=0,
                output_tokens=0
            )
