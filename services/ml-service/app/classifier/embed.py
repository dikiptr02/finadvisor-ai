from fastembed import TextEmbedding
import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Model ini multilingual (termasuk Bahasa Indonesia) dan TIDAK butuh prefix
        # khusus seperti keluarga E5 ("query: "/"passage: ") -- beda dari eksperimen
        # sebelumnya yang gagal karena multilingual-e5-small tidak didukung fastembed Python.
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)))