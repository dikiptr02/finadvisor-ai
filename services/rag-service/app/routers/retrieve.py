from app.retrieval.retriever import Retriever
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
retriever = Retriever()


class RetrieveRequest(BaseModel):
    query: str


@router.post("/internal/v1/retrieve")
def retrieve(req: RetrieveRequest):
    chunks = retriever.search(req.query)
    return {"success": True, "data": {"chunks": chunks}, "trace_id": None, "error": None}