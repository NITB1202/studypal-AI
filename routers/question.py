import json

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse
from models.question_request import QuestionRequest
from services.providers import get_question_service
from services.question_service import QuestionService

router = APIRouter()

@router.post(
    "/ask",
    summary="Ask a question"
)
def ask_question(
        request: QuestionRequest,
        question_service: QuestionService = Depends(get_question_service)
):
    generator = question_service.handle_question(request)

    async def event_stream():
        for dto in generator:
            json_str = json.dumps(dto.dict(), ensure_ascii=False)
            yield f"data: {json_str}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
