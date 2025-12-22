import uuid

from models.rag_chunk import RAGChunk
from typing import List, Dict, Any, Optional

class RAGService:
    def __init__(
        self,
        embedding_client,
        vector_store,
        top_k: int = 5,
    ):
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.top_k = top_k

    def to_rag_chunks(
            self,
            chunks: List[str],
            source_id: str,
            extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RAGChunk]:
        if not chunks:
            return []

        extra_metadata = extra_metadata or {}
        rag_chunks: List["RAGChunk"] = []

        for idx, chunk in enumerate(chunks):
            clean_chunk = chunk.strip()
            if not clean_chunk:
                continue  # Skip empty chunks

            rag_chunks.append(
                RAGChunk(
                    id=f"{source_id}_{idx}_{uuid.uuid4().hex[:8]}",  # unique ID
                    content=clean_chunk,
                    metadata={
                        "source_id": source_id,
                        "chunk_index": idx,
                        **extra_metadata,
                    },
                )
            )

        return rag_chunks

    def index_chunks(self, chunks: List[RAGChunk]) -> None:
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = self.embedding_client.embed(texts)

        self.vector_store.add(
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[c.id for c in chunks],
        )

    def retrieve(self, query: str) -> List[RAGChunk]:
        results = self.vector_store.similarity_search(
            query=query,
            k=self.top_k
        )

        return [
            RAGChunk(
                id=r.metadata.get("id", ""),
                content=r.page_content,
                metadata=r.metadata
            )
            for r in results
        ]

    def build_context(self, chunks: List[RAGChunk], max_chunks: int = 5) -> str:
        selected_chunks = [
            c for c in chunks[:max_chunks]
            if c.content and c.content.strip()
        ]

        if not selected_chunks:
            return ""

        return "\n\n".join(
            f"[Source {i + 1}]\n{chunk.content}"
            for i, chunk in enumerate(selected_chunks)
        )
