from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    resp = client.get("/internal/v1/health")
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "ok"