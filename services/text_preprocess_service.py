import re
import tiktoken

from enums.preprocess_strategy import PreprocessStrategy
from models.question_request import QuestionRequest

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

    def _merge_request_text(self, request: QuestionRequest) -> str:
        parts = [request.prompt]

        if request.context:
            parts.append(request.context)

        if request.attachments:
            parts.extend(request.attachments)

        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def get_preprocess_strategy(self, request) -> PreprocessStrategy:
        total_text = self._merge_request_text(request)
        total_tokens = self._estimate_tokens(total_text)

        if total_tokens + request.max_output_tokens <= self.model_max_tokens:
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
        max_input_tokens = (
                self.model_max_tokens
                - max_output_tokens
                - self.safety_buffer_tokens
        )

        chunks: list[str] = []
        current_chunk = ""
        current_tokens = 0

        # Split text into small paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)

            # If the paragraph is too long → fallback sentence
            if para_tokens > max_input_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)

                for sent in sentences:
                    sent_tokens = self.estimate_tokens(sent)

                    # If the sentence is too long → fallback token
                    if sent_tokens > max_input_tokens:
                        token_ids = self.encoder.encode(sent)

                        for i in range(0, len(token_ids), max_input_tokens):
                            sub_tokens = token_ids[i:i + max_input_tokens]
                            chunks.append(self.encoder.decode(sub_tokens))
                        continue

                    if current_tokens + sent_tokens > max_input_tokens:
                        chunks.append(current_chunk)
                        current_chunk = sent
                        current_tokens = sent_tokens
                    else:
                        current_chunk = (
                            sent if not current_chunk else current_chunk + " " + sent
                        )
                        current_tokens += sent_tokens

                continue

            # Normal paragraph
            if current_tokens + para_tokens > max_input_tokens:
                chunks.append(current_chunk)
                current_chunk = para
                current_tokens = para_tokens
            else:
                current_chunk = (
                    para if not current_chunk else current_chunk + "\n\n" + para
                )
                current_tokens += para_tokens

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        if overlap_tokens > 0 and len(chunks) > 1:
            overlapped_chunks = []

            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                    continue

                prev_tokens = self.encoder.encode(chunks[i - 1])
                overlap = prev_tokens[-overlap_tokens:]
                overlap_text = self.encoder.decode(overlap)

                overlapped_chunks.append(overlap_text + "\n" + chunk)

            chunks = overlapped_chunks

        return chunks

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