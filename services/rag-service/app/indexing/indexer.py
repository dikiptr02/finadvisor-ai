import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from app.config import settings


class Indexer:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.dense_model = TextEmbedding(model_name=settings.dense_model)
        self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.collection_exists(settings.collection_name):
            return
        self.client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    
    def index_chunks(self, chunks: list[str], source: str):
        dense_vecs = list(self.dense_model.embed(chunks))
        sparse_vecs = list(self.sparse_model.embed(chunks))

        points = []
        for text, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense.tolist(),
                        "sparse": models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
                    },
                    payload={"text": text, "source": source},
                )
            )

        self.client.upsert(collection_name=settings.collection_name, points=points)