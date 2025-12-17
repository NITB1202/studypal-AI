from fastapi import APIRouter
from models.question_request import QuestionRequest
from models.question_response import QuestionResponse
from services.question_service import QuestionService

router = APIRouter()
question_service = QuestionService()

@router.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask a question"
)
def ask_question(req: QuestionRequest):
    return question_service.handle_question(req)
