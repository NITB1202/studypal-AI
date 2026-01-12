import faiss
from fastapi import Depends
from langchain_community.docstore import InMemoryDocstore

from services.llm_service import LLMService
from services.question_service import QuestionService
from services.rag_service import RAGService
from services.text_preprocessing_service import TextPreprocessingService
from config.settings import settings

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# LLMService provider
def get_llm_service() -> LLMService:
    return LLMService(model=settings.model, api_key=settings.openai_api_key)

# TextPreprocessingService provider
def get_text_preprocessing_service() -> TextPreprocessingService:
    return TextPreprocessingService(model=settings.model, model_max_tokens=settings.model_max_tokens)

# RAGService provider
def get_rag_service() -> RAGService:
    embedding_client = OpenAIEmbeddings(model=settings.embedding_model)

    if os.path.exists(settings.vector_store_path):
        vector_store = FAISS.load_local(
            settings.vector_store_path,
            embedding_client
        )
    else:
        dim = len(embedding_client.embed_query("dimension_check"))
        index = faiss.IndexFlatL2(dim)

        vector_store = FAISS(
            embedding_function=embedding_client,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )

    return RAGService(embedding_client=embedding_client, vector_store=vector_store)

# QuestionService provider
def get_question_service(
    text_preprocessing_service: TextPreprocessingService = Depends(get_text_preprocessing_service),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> QuestionService:
    return QuestionService(
        text_preprocessing_service=text_preprocessing_service,
        rag_service=rag_service,
        llm_service=llm_service
    )
