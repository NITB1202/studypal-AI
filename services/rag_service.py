import uuid
import numpy as np

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
        metadatas = [c.metadata for c in chunks]
        ids = [c.id for c in chunks]

        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> List[RAGChunk]:
        results = self.vector_store.similarity_search(
            query=query,
            k=top_k or self.top_k
        )

        return [
            RAGChunk(
                id=r.metadata.get("id", ""),
                content=r.page_content,
                metadata=r.metadata
            )
            for r in results
        ]

    def rank_chunks(
            self,
            chunks: List[RAGChunk],
            query: str,
            top_k: int | None = None,
    ) -> List[RAGChunk]:
        if not chunks:
            return []

        top_k = top_k or self.top_k
        query_embedding = self.embedding_client.embed_query(query)
        chunk_embeddings = self.embedding_client.embed_documents(
            [c.content for c in chunks]
        )

        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scored_chunks = []
        for chunk, emb in zip(chunks, chunk_embeddings):
            score = cosine_similarity(query_embedding, emb)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
