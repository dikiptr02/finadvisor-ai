import httpx

from app.state_machine.state import AgentState

RAG_SERVICE_URL = "http://rag-service:8000/internal/v1/retrieve"


def retrieve_context_node(state: AgentState) -> dict:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(RAG_SERVICE_URL, json={"query": state["query"]})
        resp.raise_for_status()
        chunks = resp.json()["data"]["chunks"]
    
    return {"retrieved_context": chunks}