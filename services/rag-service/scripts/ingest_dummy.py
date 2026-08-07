import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.indexing.indexer import Indexer
from app.ingestion.chunker import SemanticChunker
from data.dummy_articles import DUMMY_ARTICLES


def main():
    chunker = SemanticChunker(model_name=settings.dense_model)
    indexer = Indexer()
    
    total_chunks = 0
    for article in DUMMY_ARTICLES:
        chunks = chunker.chunk(article["text"])
        indexer.index_chunks(chunks, source=article["source"])
        total_chunks += len(chunks)
        print(f"Indexed {len(chunks)} chunks from {article['source']}")

    print(f"Done. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()