from fastapi import APIRouter, Depends
from models.question_request import QuestionRequest
from models.question_response import QuestionResponse
from services.providers import get_question_service
from services.question_service import QuestionService

router = APIRouter()

@router.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask a question"
)
def ask_question(
    req: QuestionRequest,
    question_service: QuestionService = Depends(get_question_service)
):
    return question_service.handle_question(req)