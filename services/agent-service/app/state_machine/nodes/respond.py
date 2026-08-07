from app.state_machine.state import AgentState


def respond_node(state: AgentState) -> dict:
    if state.get("approval_status") == "rejected":
        return {"final_answer": "Rekomendasi dibatalkan sesuai keputusan kamu."}
    
    if state.get("verifier_verdict") == "not_grounded":
        return {
            "final_answer": (
                "Maaf, saya tidak bisa memberikan jawaban yang cukup didukung data "
                "yang tersedia untuk pertanyaan ini. Silakan konsultasikan dengan "
                "penasihat finansial untuk keputusan ini."
            )
        }
    
    return {"final_answer": state["draft_answer"]}