from app.state_machine.state import AgentState
from langgraph.types import interrupt


def pending_approval_node(state: AgentState) -> dict:
    decision = interrupt(
        {
            "message": "Rekomendasi berikut butuh persetujuan kamu sebelum dilanjutkan.",
            "draft_answer": state["draft_answer"],
        }
    )
    # decision datang dari luar (endpoint resume), bentuknya: {"approval_status": "approved"|"rejected"}
    return {"approval_status": decision["approval_status"]}