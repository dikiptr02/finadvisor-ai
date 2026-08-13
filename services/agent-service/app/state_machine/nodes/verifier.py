import httpx
from app.state_machine.state import AgentState

LLM_ROUTER_URL = "http://llm-router-service:8000/internal/v1/generate"

VERIFIER_PROMPT = """Kamu adalah fact-checker yang ketat. Tugasmu HANYA memeriksa apakah
JAWABAN di bawah ini didukung oleh KONTEKS yang diberikan — bukan menilai apakah jawaban
itu benar secara umum, bukan pula menjawab pertanyaannya sendiri.

Aturan:
- Jika SEMUA klaim di jawaban bisa ditelusuri balik ke konteks, jawab: GROUNDED
- Jika ADA klaim di jawaban yang tidak didukung konteks (mengarang/menambah informasi
  di luar konteks), jawab: NOT_GROUNDED
- Jawab HANYA dengan satu kata: GROUNDED atau NOT_GROUNDED. Tanpa penjelasan tambahan.

Konteks:
{context}

Jawaban yang diperiksa:
{answer}

Verdict:"""


def verifier_node(state: AgentState) -> dict:
    context_text = "\n\n".join(c["text"] for c in state["retrieved_context"])
    prompt = VERIFIER_PROMPT.format(context=context_text, answer=state["draft_answer"])

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(LLM_ROUTER_URL, json={"prompt": prompt})
        resp.raise_for_status()
        result = resp.json()["data"]["text"].strip().upper()

    verdict = "grounded" if "NOT_GROUNDED" not in result and "GROUNDED" in result else "not_grounded"
    return {"verifier_verdict": verdict}