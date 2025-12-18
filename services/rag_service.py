from typing import List
from models.rag_chunk import RAGChunk

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
        query_embedding = self.embedding_client.embed([query])[0]

        results = self.vector_store.search(
            embedding=query_embedding,
            top_k=self.top_k,
        )

        return [
            RAGChunk(
                id=r["id"],
                content=r["document"],
                metadata=r.get("metadata", {}),
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
