import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from app.retrieval.retriever import Retriever
from eval.golden_dataset import GOLDEN_DATASET

OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def build_eval_dataset():
    retriever = Retriever()

    rows = []
    for item in GOLDEN_DATASET:
        results = retriever.search(item["question"])
        contexts = [r["text"] for r in results]

        rows.append(
            {
                "question": item["question"],
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                # answer sementara pakai gabungan context teratas;
                # nanti diganti hasil generate dari Agent Service di Fase 4
                "answer": contexts[0] if contexts else "",
            }
        )
    return Dataset.from_list(rows)


def main():
    dataset = build_eval_dataset()

    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    run_config = RunConfig(timeout=300, max_workers=1)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print(result)
    result.to_pandas().to_csv("eval/ragas_results.csv", index=False)
    print("Saved to eval/ragas_results.csv")


if __name__ == "__main__":
    main()