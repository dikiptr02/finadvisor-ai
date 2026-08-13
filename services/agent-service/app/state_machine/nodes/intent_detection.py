import httpx
from app.state_machine.state import AgentState

LLM_ROUTER_URL = "http://llm-router-service:8000/internal/v1/generate"

INTENT_PROMPT = """Klasifikasikan pertanyaan berikut ke dalam salah satu kategori:
- "informational": pertanyaan edukasi/informasi umum finansial, tidak menyangkut kondisi personal user
- "actionable": pertanyaan yang meminta rekomendasi/keputusan personal terkait uang user (misal alokasi investasi, apakah harus beli/jual, evaluasi kondisi keuangan pribadi)

Jawab HANYA dengan satu kata: "informational" atau "actionable".

Pertanyaan: {query}"""


def intent_detection_node(state: AgentState) -> dict:
    prompt = INTENT_PROMPT.format(query=state["query"])

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(LLM_ROUTER_URL, json={"prompt": prompt})
        resp.raise_for_status()
        result = resp.json()["data"]["text"].strip().lower()
    
    intent = "actionable" if "actionable" in result else "informational"
    return {"intent": intent}