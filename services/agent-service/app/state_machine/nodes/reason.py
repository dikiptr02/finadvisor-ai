import httpx

from app.state_machine.state import AgentState

LLM_ROUTER_URL = "http://llm-router-service:8000/internal/v1/generate"

REASON_PROMPT = """Kamu adalah asisten finansial. Jawab pertanyaan user HANYA berdasarkan
konteks berikut. Jangan mengarang informasi yang tidak ada di konteks.

Konteks:
{context}

Pertanyaan: {query}

Jawaban:"""


def reason_node(state: AgentState) -> dict:
    context_text = "\n\n".join(c["text"] for c in state["retrieved_context"])
    prompt = REASON_PROMPT.format(context=context_text, query=state["query"])

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(LLM_ROUTER_URL, json={"prompt": prompt})
        resp.raise_for_status()
        draft = resp.json()["data"]["text"]

    return {"draft_answer": draft}