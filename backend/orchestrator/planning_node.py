"""
MindBridge 2.0 - Planning & Knowledge Retrieval Node (`planning_node`)
======================================================================

Implements the deterministic query planning, goal arbitration, and knowledge
retrieval execution layer for standard (non-crisis) conversations.

Authoritative Governance & Safety Invariants:
1. Universal Emergency Bypass: If safety detection flagged an emergency (imminent risk
   or active danger flag), this node immediately returns without executing retrieval queries.
2. Original Message Preservation: The user's original message is never mutated or overwritten.
3. Grounding Integrity: Translates informal user disclosures into canonical psychoeducational
   search queries to prevent dense retrieval score collapse.
4. Privacy & Telemetry: Zero raw user messages or stack traces enter execution traces.
5. Graceful Failure: If retrieval times out or fails, sets retrieval_status="error" with
   a controlled ErrorCode, never crashing the orchestrator.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Union

from backend.orchestrator.query_understanding import (
    QueryUnderstandingResult,
    understand_query,
)
from backend.orchestrator.state import (
    DialogGoal,
    ErrorCode,
    JurisdictionCode,
    KnowledgeChunk,
    MindBridgeState,
    RetrievalStatus,
    TraceEntry,
)
from backend.retrieval import (
    RAGRetriever,
    RetrievalAuthorization,
    build_retriever_from_knowledge_base,
)

# Global cached retriever instance for production graph runs
_CACHED_RETRIEVER: Optional[Any] = None


def get_shared_retriever() -> Any:
    """
    Returns the singleton application retriever instance.
    Combines canonical clinical KB with CounselChat retrieval if index exists.
    Lazily initialized upon first invocation.
    """
    global _CACHED_RETRIEVER
    if _CACHED_RETRIEVER is None:
        canonical_retriever, _ = build_retriever_from_knowledge_base()
        try:
            from backend.retrieval.counselchat_retriever import (
                CounselChatRetriever,
                MultiCollectionRetriever,
                DEFAULT_CONSOLIDATED_FAISS_DIR,
                DEFAULT_EXPERIMENTAL_FAISS_DIR,
            )
            from backend.retrieval.vector_store import FAISSVectorStore

            cc_dir = DEFAULT_CONSOLIDATED_FAISS_DIR if (DEFAULT_CONSOLIDATED_FAISS_DIR / "index.faiss").exists() else DEFAULT_EXPERIMENTAL_FAISS_DIR
            if (cc_dir / "index.faiss").exists():
                cc_store = FAISSVectorStore.load(cc_dir)
                cc_retriever = CounselChatRetriever(
                    vector_store=cc_store,
                    embedding_engine=canonical_retriever.embedding_engine,
                    default_threshold=0.60,
                    is_experimental=True,
                )
                _CACHED_RETRIEVER = MultiCollectionRetriever(
                    canonical_retriever=canonical_retriever,
                    counselchat_retriever=cc_retriever,
                    max_total_chunks=5,
                )
            else:
                _CACHED_RETRIEVER = canonical_retriever
        except Exception:
            _CACHED_RETRIEVER = canonical_retriever
    return _CACHED_RETRIEVER


def set_shared_retriever(retriever: Optional[Any]) -> None:
    """Explicitly inject a retriever (e.g. mock retriever in hermetic unit tests)."""
    global _CACHED_RETRIEVER
    _CACHED_RETRIEVER = retriever


def planning_node(
    state: MindBridgeState,
    retriever: Optional[Any] = None,
    authorization: Optional[RetrievalAuthorization] = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Query Understanding, Goal Arbitration & RAG Knowledge Retrieval.

    Args:
        state: Immutable MindBridgeState passed from upstream nodes.
        retriever: Optional injected RAGRetriever instance (for tests/offline harnesses).
        authorization: Optional trusted RetrievalAuthorization instance.

    Returns:
        State update dictionary containing dialog_goal, retrieval_queries,
        retrieved_chunks, retrieval_status, and execution_trace updates.
    """
    start_time = time.perf_counter()

    # Invariant 1: Universal Emergency Bypass
    # If explicit immediate danger or emergency bypass route is active, halt retrieval.
    if (
        state.get("overall_safety_route") == "emergency_bypass"
        or state.get("deterministic_danger_flag") is True
    ):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace: TraceEntry = {
            "node": "planning_node",
            "status": "skipped",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "EMERGENCY_BYPASS_SKIPPED",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "dialog_goal": "general",
            "retrieval_queries": [],
            "retrieved_chunks": [],
            "retrieval_status": "not_attempted",
            "execution_trace": current_traces,
        }

    # Extract sanitized text from state
    sanitized_text = state.get("sanitized_text", "")
    if not sanitized_text and state.get("messages"):
        # Retrieve the latest user message content if sanitized_text is unset
        for m in reversed(state["messages"]):
            if m.get("role") == "user":
                sanitized_text = m.get("content", "")
                break

    # Determine user jurisdiction
    from backend.config import settings
    jurisdiction_filter: JurisdictionCode = state.get("jurisdiction_filter") or getattr(settings, "default_jurisdiction", "IN") or "IN"

    # Invariant 2: Deterministic Query Understanding
    understanding: QueryUnderstandingResult = understand_query(
        sanitized_text=sanitized_text,
        overall_safety_route=state.get("overall_safety_route"),
        deterministic_danger_flag=state.get("deterministic_danger_flag", False),
    )

    # Invariant 3: Crisis disclosure defense-in-depth
    # If the text was flagged as crisis-related, refuse to synthesize mental-health search queries
    if understanding.is_crisis_related:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace = {
            "node": "planning_node",
            "status": "skipped",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "CRISIS_DISCLOSURE_REFUSED",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "dialog_goal": "general",
            "retrieval_queries": [],
            "retrieved_chunks": [],
            "retrieval_status": "not_attempted",
            "execution_trace": current_traces,
        }

    # Invariant 4: Out-of-Domain handling
    # If query is completely out-of-domain, return safe empty retrieval state
    if understanding.is_out_of_domain or not understanding.retrieval_queries:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace = {
            "node": "planning_node",
            "status": "success",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "OOD_NO_CONTEXT",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "dialog_goal": understanding.dialog_goal,
            "retrieval_queries": [],
            "retrieved_chunks": [],
            "retrieval_status": "empty",
            "jurisdiction_filter": jurisdiction_filter,
            "execution_trace": current_traces,
            "retrieval_concepts": understanding.identified_concepts,
        }

    # Invariant 5: Knowledge Retrieval Execution
    active_retriever = retriever or _CACHED_RETRIEVER or get_shared_retriever()
    retrieved_chunks: List[KnowledgeChunk] = []
    retrieval_status: RetrievalStatus = "empty"
    error_code: Optional[ErrorCode] = None
    error_component: Optional[str] = None

    # Handle restricted authorization context for clinical rubrics
    # Security Rule (Restricted Screening Access):
    # A user query matching "screening" or referencing clinical rubrics must NOT automatically
    # grant access to restricted clinical rubrics.
    # Restricted retrieval remains disabled unless an explicit, trusted authorization mechanism
    # (provided via state["retrieval_authorization"] or an explicit `authorization` argument)
    # is passed in.
    trusted_auth: Optional[RetrievalAuthorization] = authorization or state.get("retrieval_authorization")
    include_restricted = False
    auth_context: Optional[RetrievalAuthorization] = None

    if (
        isinstance(trusted_auth, RetrievalAuthorization)
        and bool(trusted_auth.authorized_by)
        and state.get("overall_safety_route") == "standard_conversation"
    ):
        include_restricted = True
        auth_context = trusted_auth

    try:
        primary_query = understanding.retrieval_queries[0]
        retrieved_chunks = active_retriever.retrieve(
            query=primary_query,
            jurisdiction=jurisdiction_filter,
            include_restricted=include_restricted,
            authorization=auth_context,
            top_k=5,
        )
        retrieval_status = "success" if len(retrieved_chunks) > 0 else "empty"
    except Exception:
        # Failure containment: Log controlled error code, zero raw exception strings in state
        retrieval_status = "error"
        retrieved_chunks = []
        error_code = "RETRIEVAL_FAILURE"
        error_component = "planning_node"

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    trace = {
        "node": "planning_node",
        "status": "success" if retrieval_status in ("success", "empty") else "error",
        "duration_ms": round(elapsed_ms, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_code": f"GOAL_{understanding.dialog_goal.upper()}_{retrieval_status.upper()}",
    }
    current_traces = list(state.get("execution_trace", []))
    current_traces.append(trace)

    updates: Dict[str, Any] = {
        "dialog_goal": understanding.dialog_goal,
        "retrieval_queries": understanding.retrieval_queries,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_status": retrieval_status,
        "jurisdiction_filter": jurisdiction_filter,
        "execution_trace": current_traces,
        "retrieval_concepts": understanding.identified_concepts,
    }

    if error_code is not None:
        updates["error_code"] = error_code
        updates["error_component"] = error_component

    return updates
