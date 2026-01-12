from typing import List

from enums.preprocess_strategy import PreprocessStrategy
from mappers.llm_mapper import LLMMapper
from models.question_request import QuestionRequest
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

    def handle_question(self, request: QuestionRequest):
        merged_attachments_content = self.text_preprocessing_service.merge_attachment_content(request.attachments)
        strategy = self.text_preprocessing_service.get_preprocess_strategy(merged_attachments_content)

        all_rag_chunks: List[RAGChunk] = []

        if request.attachments:
            for attachment in request.attachments:
                if strategy == PreprocessStrategy.DIRECT:
                    chunks = [attachment.content]
                elif strategy == PreprocessStrategy.CHUNK:
                    chunks = self.text_preprocessing_service.chunk(attachment.content, request.max_output_tokens)
                else:
                    system_prompt = SystemPromptFactory.get_summarize_prompt()
                    llm_request = LLMMapper.to_llm_request(user_prompt=attachment.content, system_prompt=system_prompt)

                    llm_response = self.llm_service.send_request(llm_request)
                    summary = llm_response.answer

                    cleaned_summary = self.text_preprocessing_service.clean_summary(summary)
                    next_strategy = self.text_preprocessing_service.get_preprocess_strategy(cleaned_summary)

                    if next_strategy == PreprocessStrategy.CHUNK:
                        chunks = self.text_preprocessing_service.chunk(cleaned_summary, request.max_output_tokens)
                    else:
                        chunks = [cleaned_summary]

                rag_chunks = self.rag_service.to_rag_chunks(chunks, attachment.fileName)
                all_rag_chunks.extend(rag_chunks)

        top_k = self.rag_service.top_k
        kb_chunks: List[RAGChunk] = []

        if len(all_rag_chunks) > top_k:
            attachment_chunks = self.rag_service.rank_chunks(all_rag_chunks, request.prompt, top_k)
        elif len(all_rag_chunks) == top_k:
            attachment_chunks = all_rag_chunks
        else:
            needed = top_k - len(all_rag_chunks)
            attachment_chunks = all_rag_chunks
            kb_chunks = self.rag_service.retrieve(request.prompt, needed)

        self.rag_service.index_chunks(all_rag_chunks)

        system_prompt = SystemPromptFactory.get_planner_prompt()
        llm_request = LLMMapper.to_llm_request(request=request, system_prompt=system_prompt,
                                               attachment_chunks=attachment_chunks,
                                               kb_chunks=kb_chunks)

        for llm_response in self.llm_service.stream_request(llm_request):
            yield LLMMapper.to_question_response(llm_response)