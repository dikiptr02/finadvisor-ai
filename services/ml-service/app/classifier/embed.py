from fastembed import TextEmbedding
import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)))