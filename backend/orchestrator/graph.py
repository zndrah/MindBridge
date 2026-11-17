"""
MindBridge 2.0 - Central LangGraph State Machine (`graph.py`)
=============================================================

Defines the central, deterministic LangGraph workflow coordinating all conversational,
safety, classification, planning, and retrieval transformations.

Architecture Flow:
1. User Input -> `input_node` (PII sanitization & ephemeral boundary)
2. `input_node` -> `deterministic_safety_node` (pre-ML deterministic emergency scan)
3. Router `route_after_deterministic_safety`:
   - "emergency_node" -> `emergency_node` -> END
   - "followup_node" -> `followup_node` -> `route_after_followup`
   - "classifier_node" -> `classifier_node` -> `classifier_decision_node`
4. Router `route_after_classifier_decision`:
   - "emergency_node" -> `emergency_node` -> END
   - "followup_node" -> `followup_node` -> END
   - "deescalation_node" -> `deescalation_node` -> END
   - "planning_node" -> `planning_node` -> END (RAG retrieval)
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from backend.orchestrator.input_node import input_node
from backend.orchestrator.planning_node import planning_node
from backend.orchestrator.response_node import response_node
from backend.orchestrator.verification_node import verification_node
from backend.orchestrator.safety_node import (
    classifier_decision_node,
    classifier_node,
    deterministic_safety_node,
    format_emergency_guidance,
    route_after_classifier_decision,
    route_after_deterministic_safety,
    route_after_followup,
)
from backend.orchestrator.state import (
    JurisdictionCode,
    MindBridgeState,
)
from backend.retrieval import RAGRetriever

# Sentinel nodes for graph traversal
START = "__start__"
END = "__end__"


# ============================================================================
# Terminal & Supportive Transition Nodes
# ============================================================================

def emergency_node(state: MindBridgeState) -> Dict[str, Any]:
    """
    Emits deterministic emergency guidance using authoritative contacts from hotlines.json.
    Halts open generation and prevents vector search execution.
    """
    if state.get("final_response"):
        guidance = state["final_response"]
    else:
        from backend.config import settings
        jur: JurisdictionCode = state.get("jurisdiction_filter") or getattr(settings, "default_jurisdiction", "IN") or "IN"
        is_already_asked = state.get("emergency_trigger_id") in ("SAFETY_FOLLOWUP_CONFIRMED", "EMERGENCY_STATE_PRESERVED")
        guidance = format_emergency_guidance(jur, already_asked_danger=is_already_asked)
    return {
        "final_response": guidance,
        "overall_safety_route": "emergency_bypass",
        "deterministic_danger_flag": True,
        "dialog_goal": "general",
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieval_status": "not_attempted",
        "llm_telemetry": {
            "provider": "skipped",
            "model": "bypassed_by_safety",
            "fallback_triggered": False,
            "local_fallback_used": False,
            "duration_ms": 0.0,
            "success": True,
        },
    }


def deescalation_node(state: MindBridgeState) -> Dict[str, Any]:
    """
    Supportive de-escalation fallback for non-crisis distress or unassessed classifier risk.
    Emits neutral supportive language and avoids unconditional somatic prescriptions.
    """
    if state.get("final_response"):
        return {
            "final_response": state["final_response"],
            "overall_safety_route": state.get("overall_safety_route", "ml_risk_unassessed"),
            "dialog_goal": "general",
            "retrieval_status": "not_attempted",
        }
    supportive_message = (
        "I am here with you. I hear how difficult things feel right now. "
        "I am ready to support you with helpful information or resources whenever you are ready."
    )
    return {
        "final_response": supportive_message,
        "overall_safety_route": "supportive_deescalation",
        "dialog_goal": "general",
        "retrieval_status": "not_attempted",
    }


def followup_node(state: MindBridgeState) -> Dict[str, Any]:
    """
    Structured multi-turn PHQ-9 Item 9 follow-up node.
    Maintains continuity of clinical risk dimension evaluation.
    """
    status = state.get("phq9_followup_status", "not_started")
    if status in ("not_started", "interrupted"):
        status = "in_progress"

    followup_response = state.get("final_response") or (
        "I hear how much pain you are experiencing right now, and I want to support you. "
        "Your safety is very important. Can you tell me if you are in immediate danger, "
        "or if you have any thoughts of ending your life or hurting yourself?"
    )

    return {
        "phq9_followup_status": status,
        "overall_safety_route": "pending_safety_follow_up",
        "final_response": followup_response,
    }


# ============================================================================
# Deterministic StateGraph Engine (LangGraph-Compatible)
# ============================================================================

class CompiledGraph:
    """
    Executable compiled graph executing nodes immutably over MindBridgeState.
    """
    def __init__(
        self,
        nodes: Dict[str, Callable[[MindBridgeState], Dict[str, Any]]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Tuple[Callable[[MindBridgeState], str], Optional[Dict[str, str]]]],
        entry_point: str,
    ):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point

    def invoke(
        self,
        state: MindBridgeState,
        raw_user_text: Optional[str] = None,
    ) -> MindBridgeState:
        """
        Executes the graph deterministically from entry_point until reaching END.
        Ephemeral input text is scrubbed and never persisted in output state.
        """
        current_state = dict(state)  # type: ignore
        ephemeral_text = (
            raw_user_text
            or current_state.pop("raw_user_text", None)
            or current_state.get("sanitized_text")
        )
        if not ephemeral_text and current_state.get("messages"):
            for m in reversed(current_state["messages"]):
                if m.get("role") == "user":
                    ephemeral_text = m.get("content")
                    break

        if ephemeral_text:
            current_state["_ephemeral_raw_text"] = ephemeral_text

        current_node = self.entry_point
        max_hops = 20
        hops = 0

        while current_node != END and hops < max_hops:
            hops += 1
            if current_node not in self.nodes:
                raise ValueError(f"Graph execution encountered unknown node: '{current_node}'")

            # Execute node transformation
            node_fn = self.nodes[current_node]
            updates = node_fn(current_state)  # type: ignore
            if updates and isinstance(updates, dict):
                current_state.update(updates)

            # Scrub ephemeral helper keys after input_node executes
            if current_node == "input_node":
                current_state.pop("_ephemeral_raw_text", None)

            # Determine next node
            if current_node in self.conditional_edges:
                router_fn, path_map = self.conditional_edges[current_node]
                next_target = router_fn(current_state)  # type: ignore
                if path_map and next_target in path_map:
                    next_node = path_map[next_target]
                else:
                    next_node = next_target
            elif current_node in self.edges:
                next_node = self.edges[current_node]
            else:
                next_node = END

            current_node = next_node

        current_state.pop("_ephemeral_raw_text", None)
        return current_state  # type: ignore


class MindBridgeStateGraph:
    """
    StateGraph builder for MindBridge orchestration workflow.
    Provides complete API compatibility with LangGraph's StateGraph.
    """
    def __init__(self, state_schema: Any = MindBridgeState):
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[MindBridgeState], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[
            str, Tuple[Callable[[MindBridgeState], str], Optional[Dict[str, str]]]
        ] = {}
        self.entry_point: str = "input_node"

    def add_node(self, name: str, func: Callable[[MindBridgeState], Dict[str, Any]]) -> "MindBridgeStateGraph":
        self.nodes[name] = func
        return self

    def add_edge(self, start_node: str, end_node: str) -> "MindBridgeStateGraph":
        self.edges[start_node] = end_node
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[[MindBridgeState], str],
        path_map: Optional[Dict[str, str]] = None,
    ) -> "MindBridgeStateGraph":
        self.conditional_edges[source] = (path, path_map)
        return self

    def set_entry_point(self, key: str) -> "MindBridgeStateGraph":
        self.entry_point = key
        return self

    def compile(self) -> CompiledGraph:
        return CompiledGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
        )


def build_mindbridge_graph(
    retriever: Optional[RAGRetriever] = None,
    generator: Optional[Any] = None,
) -> CompiledGraph:
    """
    Factory constructing and compiling the authoritative MindBridge 2.0 orchestration workflow.

    Nodes:
    - input_node: PII masking and message history sanitization
    - deterministic_safety_node: Regex-based emergency triage and follow-up checks
    - classifier_node: RoBERTa ML inference
    - classifier_decision_node: RoBERTa threshold decision logic
    - planning_node: Query understanding, goal arbitration, and RAG retrieval
    - response_node: Grounded non-clinical supportive response generation
    - verification_node: Deterministic post-generation safety, citation & grounding gate
    - emergency_node: Static hotline guidance emission
    - deescalation_node: Non-crisis supportive de-escalation
    - followup_node: Multi-turn PHQ-9 risk assessment continuity
    """
    graph = MindBridgeStateGraph(MindBridgeState)

    # 1. Register Nodes
    def _input_wrapper(state: MindBridgeState) -> Dict[str, Any]:
        raw_text = state.get("_ephemeral_raw_text") or state.get("sanitized_text")
        if not raw_text and state.get("messages"):
            for m in reversed(state["messages"]):
                if m.get("role") == "user":
                    raw_text = m.get("content")
                    break
        return input_node(state, raw_user_text=raw_text)

    def _planning_wrapper(state: MindBridgeState) -> Dict[str, Any]:
        return planning_node(state, retriever=retriever)

    def _response_wrapper(state: MindBridgeState) -> Dict[str, Any]:
        return response_node(state, generator=generator)

    def _verification_wrapper(state: MindBridgeState) -> Dict[str, Any]:
        return verification_node(state)

    graph.add_node("input_node", _input_wrapper)
    graph.add_node("deterministic_safety_node", deterministic_safety_node)
    graph.add_node("classifier_node", classifier_node)
    graph.add_node("classifier_decision_node", classifier_decision_node)
    graph.add_node("planning_node", _planning_wrapper)
    graph.add_node("response_node", _response_wrapper)
    graph.add_node("verification_node", _verification_wrapper)
    graph.add_node("emergency_node", emergency_node)
    graph.add_node("deescalation_node", deescalation_node)
    graph.add_node("followup_node", followup_node)

    # 2. Register Entry Point & Linear Edges
    graph.set_entry_point("input_node")
    graph.add_edge("input_node", "deterministic_safety_node")
    graph.add_edge("classifier_node", "classifier_decision_node")

    # 3. Register Conditional Routers
    graph.add_conditional_edges(
        "deterministic_safety_node",
        route_after_deterministic_safety,
        {
            "emergency_node": "emergency_node",
            "followup_node": "followup_node",
            "classifier_node": "classifier_node",
            "deescalation_node": "deescalation_node",
        },
    )

    graph.add_conditional_edges(
        "classifier_decision_node",
        route_after_classifier_decision,
        {
            "emergency_node": "emergency_node",
            "followup_node": "followup_node",
            "deescalation_node": "deescalation_node",
            "planning_node": "planning_node",
        },
    )

    graph.add_conditional_edges(
        "followup_node",
        route_after_followup,
        {
            "emergency_node": "emergency_node",
            "deescalation_node": "deescalation_node",
            "planning_node": "planning_node",
            "__end__": END,
        },
    )

    # 4. Response Generation, Verification & Terminal Edges
    graph.add_edge("planning_node", "response_node")
    graph.add_edge("response_node", "verification_node")
    graph.add_edge("verification_node", END)
    graph.add_edge("emergency_node", END)
    graph.add_edge("deescalation_node", END)

    return graph.compile()
