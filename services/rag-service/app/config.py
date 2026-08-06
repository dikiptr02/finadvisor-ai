from pydantic_settings import BaseSettings


class Setting(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "finadvisor_docs"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k_retrieve: int = 20
    top_k_final: int = 5

settings = Setting()