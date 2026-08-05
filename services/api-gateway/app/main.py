import httpx
from fastapi import FastAPI

app = FastAPI(title="API Gateway", version="0.1.0")

SERVICE_URLS = {
    "rag": "http://rag-service:8000",
    "agent": "http://agent-service:8000",
    "llm-router": "http://llm-router-service:8000",
    "ml": "http://ml-service:8000",
    "forecast": "http://forecast-service:8000",
}

@app.get("/api/v1/health")
async def health_check():
    results = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, base_url in SERVICE_URLS.items():
            try:
                resp = await client.get(f"{base_url}/internal/v1/health")
                results[name] = resp.json()["data"]["status"]
            except Exception:
                results[name] = "unreachable"
    return {"success": True, "data": results, "trace_id": None, "error": None}