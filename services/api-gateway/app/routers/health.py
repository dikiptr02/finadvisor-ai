from fastapi import APIRouter

router = APIRouter()

@router.get("/internal/v1/health")
def health_check():
    return {"success": True, "data": {"status": "oke"}, "trace_id": None, "error":None}