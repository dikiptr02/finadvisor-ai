from fastapi import APIRouter
from pydantic import BaseModel

from app.retrieval.retriever import Retriever

router = APIRouter()
Retriever = Retriever()


class RetrieveRequest(BaseModel):
    query: str


@router.post("/internal/v1/retrieve")
def retrieve(req: RetrieveRequest):
    chunks = Retriever.search(req.query)
    return {"success": True, "data": {"chunks": chunks}, "trace_id": None, "error": None}