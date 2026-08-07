from typing import Literal, Optional, TypedDict


class AgentState(TypedDict):
    user_id: str
    query: str
    intent: Optional[Literal["informational", "actionable"]]
    retrieved_context: Optional[list[dict]]
    draft_answer: Optional[str]
    approval_status: Optional[Literal["approved", "rejected"]]
    verifier_verdict: Optional[Literal["grounded", "not_grounded"]]
    final_answer: Optional[str]