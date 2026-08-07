from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.state_machine.nodes.approval import pending_approval_node
from app.state_machine.nodes.intent_detection import intent_detection_node
from app.state_machine.nodes.reason import reason_node
from app.state_machine.nodes.respond import respond_node
from app.state_machine.nodes.retrieve import retrieve_context_node
from app.state_machine.nodes.verifier import verifier_node
from app.state_machine.state import AgentState


def route_after_reason(state: AgentState) -> str:
    return "pending_approval" if state["intent"] == "actionable" else "verifier"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_detection", intent_detection_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("reason", reason_node)
    graph.add_node("pending_approval", pending_approval_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("intent_detection")
    graph.add_edge("intent_detection", "retrieve_context")
    graph.add_edge("retrieve_context", "reason")
    graph.add_conditional_edges("reason", route_after_reason, {"pending_approval": "pending_approval", "verifier": "verifier"})
    graph.add_edge("pending_approval", "verifier")
    graph.add_edge("verifier", "respond")
    graph.add_edge("respond", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)