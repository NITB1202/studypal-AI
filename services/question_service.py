from models.question_request import QuestionRequest
from models.question_response import QuestionResponse

class QuestionService:

    def handle_question(self, request: QuestionRequest) -> QuestionResponse:
        input_tokens = len(request.prompt.split())
        output_tokens = min(request.max_output_tokens, 50)

        mock_reply = (
            f"[MOCK RESPONSE]\n"
            f"Prompt: {request.prompt}\n"
            f"Context: {request.context}\n"
            f"Attachments: {request.attachments}"
        )

        return QuestionResponse(
            reply=mock_reply,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
