import openai

from config.settings import settings
from models.llm_request import LLMRequest
from models.llm_response import LLMResponse

class LLMService:
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        self.model = model

    def _build_messages(self, request: LLMRequest):
        system_prompt = request.system_prompt

        if request.context:
            system_prompt += f"\nUser context:\n{request.context}"

        if request.additional_context:
            system_prompt += f"\nRelevant data:\n{request.additional_context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_prompt}
        ]

        return messages

    def send_request(self, request: LLMRequest) -> LLMResponse:
        try:
            # Build messages from LLMRequest
            messages = self._build_messages(request)

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                api_key=self.api_key
            )

            # Extract answer and token usage
            answer_text = response.choices[0].message["content"].strip()
            usage = response.usage
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            return LLMResponse(
                answer=answer_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            # Return LLMResponse with error message in answer
            return LLMResponse(
                answer=f"Error calling OpenAI API: {e}",
                input_tokens=0,
                output_tokens=0
            )
