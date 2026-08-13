from app.router.router import AllProvidersFailedError, generate_with_fallback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str



@router.post("/internal/v1/generate")
def generate(req: GenerateRequest):
    try:
        result = generate_with_fallback(req.prompt)
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "success": True,
        "data": result,
        "trace_id": None,
        "error": None,
    }