import re
import tiktoken

from typing import Optional, List
from models.document import Document
from enums.preprocess_strategy import PreprocessStrategy

class TextPreprocessingService:
    META_LINE_MAX_LEN = 40

    def __init__(
                self, model: str = "gpt-4",
                model_max_tokens: int = 8000,
                safety_buffer_tokens: int = 200):
        self.model = model
        self.model_max_tokens = model_max_tokens
        self.encoder = tiktoken.encoding_for_model(model)
        self.safety_buffer_tokens = safety_buffer_tokens

    def _estimate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def merge_attachment_content(self, documents: Optional[List[Document]] = None) -> str:
        parts = []

        if documents:
            for doc in documents:
                parts.append(doc.content)

        return "\n".join(parts)

    def get_preprocess_strategy(self, text: str, max_output_tokens) -> PreprocessStrategy:
        total_tokens = self._estimate_tokens(text)

        if total_tokens + max_output_tokens <= self.model_max_tokens:
            return PreprocessStrategy.DIRECT

        if total_tokens > self.model_max_tokens * 2:
            return PreprocessStrategy.SUMMARIZE

        return PreprocessStrategy.CHUNK

    def chunk(
            self,
            text: str,
            max_output_tokens: int,
            overlap_tokens: int = 100
    ) -> list[str]:
        max_input_tokens = self.model_max_tokens - max_output_tokens - self.safety_buffer_tokens

        chunks: list[list[int]] = []
        current_chunk_tokens: list[int] = []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            para_tokens = self.encoder.encode(para)

            if len(para_tokens) > max_input_tokens:
                # fallback sentence
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self.encoder.encode(sent)

                    if len(sent_tokens) > max_input_tokens:
                        # fallback token-level
                        for i in range(0, len(sent_tokens), max_input_tokens):
                            chunks.append(sent_tokens[i:i + max_input_tokens])
                        continue

                    if len(current_chunk_tokens) + len(sent_tokens) > max_input_tokens:
                        chunks.append(current_chunk_tokens)
                        current_chunk_tokens = sent_tokens
                    else:
                        current_chunk_tokens.extend(sent_tokens)
            else:
                if len(current_chunk_tokens) + len(para_tokens) > max_input_tokens:
                    if current_chunk_tokens:
                        chunks.append(current_chunk_tokens)
                    current_chunk_tokens = para_tokens
                else:
                    current_chunk_tokens.extend(para_tokens)

        if current_chunk_tokens:
            chunks.append(current_chunk_tokens)

        # Apply overlap
        if overlap_tokens > 0 and len(chunks) > 1:
            overlapped_chunks = [chunks[0]]
            for prev, curr in zip(chunks, chunks[1:]):
                overlap = prev[-overlap_tokens:] if overlap_tokens < len(prev) else prev
                overlapped_chunks.append(overlap + curr)
            chunks = overlapped_chunks

        # Decode all chunks
        return [self.encoder.decode(chunk_tokens) for chunk_tokens in chunks]

    def clean_summary(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # 1. Skip empty lines
            if not line:
                continue

            # 2. Remove mark down headers
            line = re.sub(r'^#{1,6}\s*', '', line)

            # 3. Normalize bullet points
            line = re.sub(r'^[-•*]\s*', '', line)

            # 4. Drop meta lines (language-agnostic heuristics)
            # Short title-like lines ending with ":" (e.g. "Summary:")
            if len(line) <= self.META_LINE_MAX_LEN and line.endswith(":"):
                continue

            # Conversational / assistant-like offers (questions)
            if "?" in line:
                continue

            # 5. Remove excessive markdown emphasis
            line = re.sub(r'(\*\*|__)(.*?)\1', r'\2', line)
            line = re.sub(r'([*_])(.*?)\1', r'\2', line)

            cleaned_lines.append(line)

        # 6. Normalize newlines
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text.strip()