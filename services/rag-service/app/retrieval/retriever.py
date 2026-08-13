from app.config import settings
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models


class Retriever:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.dense_model = TextEmbedding(model_name=settings.dense_model)
        self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
        self.reranker = TextCrossEncoder(model_name=settings.reranker_model)

    def search(self, query: str) -> list[dict]:
        dense_vec = next(iter(self.dense_model.embed([query])))
        sparse_vec = next(iter(self.sparse_model.embed([query])))

        # Hybrid: prefetch dense & sparse, gabung dengan Reciprocal Rank Fusion
        results = self.client.query_points(
            collection_name=settings.collection_name,
            prefetch=[
                models.Prefetch(query=dense_vec.tolist(), using="dense", limit=settings.top_k_retrieve),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                    using="sparse",
                    limit=settings.top_k_retrieve,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=settings.top_k_retrieve,
        ).points

        candidates = [p.payload["text"] for p in results]
        if not candidates:
            return []

        # Rerank dengan cross-encoder, ambil top_k_final
        scores = list(self.reranker.rerank(query, candidates))
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        return [
            {"text": point.payload["text"], "source": point.payload["source"], "score": float(score)}
            for point, score in ranked[: settings.top_k_final]
        ]