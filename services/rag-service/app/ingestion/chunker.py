from fastembed import TextEmbedding
import numpy as np
import re


class SemanticChunker:
    def __init__(self, model_name: str, similarity_threshold: float = 0.55, max_chunk_sentences: int = 8):
        self.embedder = TextEmbedding(model_name=model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_sentences = max_chunk_sentences

    def _split_sentences(self, text: str) -> list[str]:
        # split sederhana berbasis tanda baca akhir kalimat
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences

        embeddings = list(self.embedder.embed(sentences))

        chunks: list[str] = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_sim(embeddings[i -1], embeddings[i])
            same_topic = sim >= self.similarity_threshold
            room_left = len(current) < self.max_chunk_sentences

            if same_topic and room_left:
                current.append(sentences[i])
            else:
                chunks.append(" ".join(current))
                current = [sentences[i]]
        
        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))