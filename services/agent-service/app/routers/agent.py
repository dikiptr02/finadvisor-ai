from fastapi import APIRouter
from langgraph.types import Command
from pydantic import BaseModel

from app.state_machine.graph import build_graph

router = APIRouter()
agent_graph = build_graph()


class StartRequest(BaseModel):
    user_id: str
    query: str


class ResumeRequest(BaseModel):
    thread_id: str
    approval_status: str    # "approved" | "rejected"


@router.post("/internal/v1/agent/start")
def start_agent(req: StartRequest):
    config = {"configurable": {"thread_id": req.user_id}}
    result = agent_graph.invoke({"user_id": req.user_id, "query": req.query}, config=config)

    if "final_answer" not in result:
        return {
            "success": True,
            "data": {
                "status": "pending_approval", 
                "thread_id": req.user_id, 
                "detail": {
                    "message": "Rekomendasi berikut butuh persetujuan kamu sebelum dilanjutkan.",
                    "draft_answer": result.get("draft_answer")
                }
            },
            "trace_id": None,
            "error": None,
        }

    return {"success": True, "data": {"status": "done", "final_answer": result["final_answer"]}, "trace_id": None, "error": None}


@router.post("/internal/v1/agent/resume")
def resume_agent(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = agent_graph.invoke(Command(resume={"approval_status": req.approval_status}), config=config)

    return {"success": True, "data": {"status": "done", "final_answer": result["final_answer"]}, "trace_id": None, "error": None}