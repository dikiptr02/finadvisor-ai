from typing import Literal, TypedDict


class AgentState(TypedDict):
    user_id: str
    query: str
    intent: Literal["informational", "actionable"] | None
    retrieved_context: list[dict] | None
    draft_answer: str | None
    approval_status: Literal["approved", "rejected"] | None
    verifier_verdict: Literal["grounded", "not_grounded"] | None
    final_answer: str | None