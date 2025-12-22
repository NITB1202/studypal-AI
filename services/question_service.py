from typing import List

from enums.preprocess_strategy import PreprocessStrategy
from mappers.llm_mapper import LLMMapper
from models.question_request import QuestionRequest
from models.question_response import QuestionResponse
from models.rag_chunk import RAGChunk
from services.llm_service import LLMService
from services.rag_service import RAGService
from services.system_prompt_factory import SystemPromptFactory
from services.text_preprocessing_service import TextPreprocessingService

class QuestionService:

    def __init__(self,
                 text_preprocessing_service: TextPreprocessingService,
                 rag_service: RAGService,
                 llm_service: LLMService):
        self.text_preprocessing_service = text_preprocessing_service
        self.rag_service = rag_service
        self.llm_service = llm_service

    def handle_question(self, request: QuestionRequest) -> QuestionResponse:
        merged_attachments_content = self.text_preprocessing_service.merge_attachment_content(request.attachments)
        strategy = self.text_preprocessing_service.get_preprocess_strategy(
            merged_attachments_content,
            request.max_output_tokens
        )

        all_rag_chunks: List[RAGChunk] = []

        for attachment in request.attachments:
            if strategy == PreprocessStrategy.DIRECT:
                text = attachment.content
                chunks = [text]
            elif strategy == PreprocessStrategy.CHUNK:
                chunks = self.text_preprocessing_service.chunk(attachment.content, request.max_output_tokens)
            else:
                system_prompt = SystemPromptFactory.get_summarize_prompt()
                llm_request = LLMMapper.to_llm_request(user_prompt=attachment.content, system_prompt=system_prompt)

                llm_response = self.llm_service.send_request(llm_request)
                summary = llm_response.answer

                cleaned_summary = self.text_preprocessing_service.clean_summary(summary)
                next_strategy = self.text_preprocessing_service.get_preprocess_strategy(
                    cleaned_summary, request.max_output_tokens
                )

                if next_strategy == PreprocessStrategy.CHUNK:
                    chunks = self.text_preprocessing_service.chunk(cleaned_summary, request.max_output_tokens)
                else:
                    chunks = [cleaned_summary]

            rag_chunks = self.rag_service.to_rag_chunks(chunks, attachment.fileName)

            all_rag_chunks.extend(rag_chunks)
            self.rag_service.index_chunks(rag_chunks)

        retrieved_chunks = self.rag_service.retrieve(request.prompt)

        additional_context = self.rag_service.build_context(retrieved_chunks)
        system_prompt = SystemPromptFactory.get_planner_prompt()

        llm_request = LLMMapper.to_llm_request(request=request, system_prompt=system_prompt,
                                               additional_context=additional_context)

        llm_response = self.llm_service.send_request(llm_request)
        return LLMMapper.to_question_response(llm_response)
