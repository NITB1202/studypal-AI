from typing import Optional, List

from models.llm_request import LLMRequest
from models.llm_response import LLMResponse
from models.question_request import QuestionRequest
from models.question_response import QuestionResponse
from models.rag_chunk import RAGChunk


class LLMMapper:
    @staticmethod
    def to_llm_request(
            user_prompt: Optional[str] = None,
            system_prompt: Optional[str] = None,
            request: Optional[QuestionRequest] = None,
            attachment_chunks: Optional[List[RAGChunk]] = None,
            kb_chunks: Optional[List[RAGChunk]] = None
    ) -> LLMRequest:

        if request is not None:
            return LLMRequest(
                user_prompt=request.prompt,
                system_prompt=system_prompt,
                context=request.context,
                attachment_chunks=attachment_chunks,
                kb_chunks=kb_chunks,
                max_output_tokens=request.max_output_tokens,
            )
        elif user_prompt is not None:
            return LLMRequest(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
        else:
            raise ValueError("Either 'user_prompt' or 'request' must be provided.")

    @staticmethod
    def to_question_response(response: LLMResponse) -> QuestionResponse:
        return QuestionResponse(
            reply=response.answer,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens
        )