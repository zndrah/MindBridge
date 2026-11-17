"""
MindBridge 2.0 - FastAPI Route Handlers (`routes.py`)
=====================================================

Implements the public API endpoints:
- GET  /        : Service information & navigation
- GET  /health  : Liveness probe
- GET  /ready   : Readiness probe for local models and vector index
- POST /chat    : Conversational pipeline processing
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import (
    app_state,
    get_app_graph,
    get_concurrency_semaphore,
)
from backend.api.schemas import (
    AimlInferenceRequest,
    AimlInferenceResponse,
    ChatRequest,
    ChatResponse,
    CitationItem,
    EmergencyResourceItem,
    HealthResponse,
    LLMTelemetry,
    ModelReadiness,
    RAGTelemetry,
    ReadyResponse,
    RetrievedChunkTelemetry,
    RoBERTaTelemetry,
    RootResponse,
    SafetyRoutingTelemetry,
)
from backend.orchestrator.graph import CompiledGraph
from backend.orchestrator.state import MindBridgeState

router = APIRouter()


@router.get("/", response_model=RootResponse, tags=["General"])
async def root() -> RootResponse:
    """Returns basic service metadata and documentation links."""
    return RootResponse()


@router.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health() -> HealthResponse:
    """
    Fast liveness probe verifying that the application process is running.
    Does not evaluate deep model loading to prevent unnecessary CPU load.
    """
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, tags=["Diagnostics"])
async def ready() -> ReadyResponse:
    """
    Readiness probe evaluating operational readiness of local RoBERTa,
    BGE embeddings, FAISS vector index, and orchestration graph.
    Returns HTTP 503 if required models or indexes are unavailable.
    """
    info = app_state.get_readiness_info()
    readiness = ReadyResponse(
        status="ready" if info["is_ready"] else "not_ready",
        models=ModelReadiness(**info["models"]),
        details=info["details"],
    )

    if not info["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness.model_dump(),
        )

    return readiness


@router.get(
    "/safety/resources",
    response_model=List[EmergencyResourceItem],
    tags=["Safety"],
    summary="Get verified emergency directory resources by jurisdiction",
)
@router.get(
    "/emergency-resources",
    response_model=List[EmergencyResourceItem],
    tags=["Safety"],
    summary="Get verified emergency directory resources by jurisdiction (alias)",
)
async def get_emergency_resources(
    jurisdiction: Optional[str] = None,
) -> List[EmergencyResourceItem]:
    """
    Returns authoritative emergency and crisis support contacts sourced strictly
    from backend/safety/hotlines.json for the requested jurisdiction.
    Defaults to settings.default_jurisdiction ("IN").
    """
    from backend.config import settings
    from backend.orchestrator.safety_node import get_hotlines_for_jurisdiction
    from backend.api.schemas import VALID_JURISDICTIONS

    jur_code: JurisdictionCode = "IN"
    if jurisdiction and isinstance(jurisdiction, str):
        cleaned = jurisdiction.strip().upper()
        if cleaned in VALID_JURISDICTIONS:
            jur_code = cleaned  # type: ignore
        else:
            jur_code = getattr(settings, "default_jurisdiction", "IN") or "IN"
    else:
        jur_code = getattr(settings, "default_jurisdiction", "IN") or "IN"

    raw_hotlines = get_hotlines_for_jurisdiction(jur_code)
    if not raw_hotlines and jur_code != "GL":
        raw_hotlines = get_hotlines_for_jurisdiction("GL")

    return [EmergencyResourceItem(**h) for h in raw_hotlines]


@router.post("/chat", response_model=ChatResponse, tags=["Conversational"])
async def chat(
    request: ChatRequest,
    graph: CompiledGraph = Depends(get_app_graph),
    semaphore: asyncio.Semaphore = Depends(get_concurrency_semaphore),
) -> ChatResponse:
    """
    Main conversational first-response endpoint.
    Processes user utterances through the complete, verified MindBridge 2.0 state machine:
    PII Masking -> Deterministic Safety -> RoBERTa Classifier -> Goal Planning ->
    RAG Retrieval -> Grounded Generation -> Response Verification Gate.

    Security & Privacy Guarantees:
    - Never trusts client-supplied safety states or authorization.
    - Raw user messages are ephemeral and never logged.
    - Client cannot inject restricted retrieval permissions.
    """
    # 1. Establish session identifiers and privacy-preserving user hash
    session_id = request.session_id or str(uuid.uuid4())
    user_id_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]

    # 2. Format sanitized message history (if provided by client)
    messages: List[Dict[str, str]] = []
    if request.conversation_history:
        for turn in request.conversation_history:
            messages.append({"role": turn.role, "content": turn.content})

    # 3. Construct clean initial MindBridgeState with ZERO client-supplied trust
    # Incorporates server-side session safety state to maintain multi-turn safety continuity
    from backend.safety.session_store import session_safety_store
    from backend.config import settings
    stored_safety_state = session_safety_store.get_session_state(session_id)
    initial_safety_route = (
        stored_safety_state.get("safety_route")
        if stored_safety_state
        else "standard_conversation"
    )
    initial_followup_status = (
        "in_progress"
        if initial_safety_route == "pending_safety_follow_up"
        else "not_started"
    )

    # Determine effective jurisdiction:
    # 1. Explicit in current request (e.g. user selected another region in UI)
    # 2. Existing stored jurisdiction in session (preserves user selection across turns)
    # 3. Application default jurisdiction from settings (default: "IN")
    stored_jurisdiction = stored_safety_state.get("jurisdiction") if stored_safety_state else None
    effective_jurisdiction: JurisdictionCode = (
        request.jurisdiction
        or stored_jurisdiction
        or getattr(settings, "default_jurisdiction", "IN")
        or "IN"
    )

    initial_state: MindBridgeState = {
        "session_id": session_id,
        "user_id_hash": user_id_hash,
        "messages": messages,  # type: ignore
        "sanitized_text": "",  # Populated by input_node inside graph
        "deterministic_danger_flag": (initial_safety_route == "emergency_bypass"),
        "emergency_trigger_id": stored_safety_state.get("emergency_trigger_id") if stored_safety_state else None,
        "roberta_prob": None,
        "roberta_crisis_flag": False,
        "classifier_status": "not_run",
        "overall_safety_route": initial_safety_route or "standard_conversation",
        "phq9_item9_endorsed": False,
        "phq9_followup_status": initial_followup_status,
        "phq9_risk_level": None,
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": "general",
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieval_status": "not_attempted",
        "jurisdiction_filter": effective_jurisdiction,
        "retrieval_authorization": None,  # NEVER allow client to grant restricted access
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }

    # 4. Execute compiled orchestration graph with concurrency bounds
    try:
        async with semaphore:
            final_state = await asyncio.to_thread(
                graph.invoke,
                initial_state,
                raw_user_text=request.message,
            )
        # Update server-side session safety store with resulting safety route and jurisdiction
        final_route = final_state.get("overall_safety_route", "standard_conversation")
        final_jurisdiction: JurisdictionCode = final_state.get("jurisdiction_filter") or effective_jurisdiction
        session_safety_store.update_session_safety_state(
            session_id=session_id,
            safety_route=final_route,
            pending_question=final_state.get("final_response") if final_route == "pending_safety_follow_up" else None,
            emergency_trigger_id=final_state.get("emergency_trigger_id"),
            jurisdiction=final_jurisdiction,
        )
    except Exception:
        # Failure containment: Zero raw exception strings or tracebacks in API response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "An internal error occurred during request processing.",
                "error_code": "GRAPH_EXECUTION_FAILURE",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    # 5. Extract provenance citations from verified retrieved chunks
    citations: List[CitationItem] = []
    # Only expose citations if the final response is grounded in retrieved chunks
    # and strictly filter to chunks actually used to support the final response
    if (
        final_state.get("retrieval_status") == "success"
        and final_state.get("verification_status") in ("verified", "passed")
        and final_state.get("grounding_status") == "grounded"
    ):
        from backend.retrieval import filter_used_citations
        used_chunks = filter_used_citations(
            response_text=final_state.get("final_response", ""),
            retrieved_chunks=final_state.get("retrieved_chunks", []),
        )
        for chunk in used_chunks:
            citations.append(
                CitationItem(
                    title=chunk.get("title", ""),
                    document_id=chunk.get("document_id", ""),
                    category=chunk.get("category", ""),
                    jurisdiction=chunk.get("jurisdiction", "GL"),
                )
            )

    # 6. Extract listed directory emergency resources if emergency_bypass was triggered
    emergency_resources: List[EmergencyResourceItem] = []
    if final_state.get("overall_safety_route") == "emergency_bypass":
        from backend.orchestrator.safety_node import get_hotlines_for_jurisdiction
        jur_code = final_state.get("jurisdiction_filter") or effective_jurisdiction
        hotline_entries = get_hotlines_for_jurisdiction(jur_code)
        for entry in hotline_entries:
            emergency_resources.append(
                EmergencyResourceItem(
                    resource_id=entry.get("resource_id", ""),
                    resource_name=entry.get("resource_name", ""),
                    authority=entry.get("authority"),
                    telephone=entry.get("telephone"),
                    alternate_telephone=entry.get("alternate_telephone"),
                    dialing_instructions=entry.get("dialing_instructions"),
                    sms_instructions=entry.get("sms_instructions"),
                    resource_type=entry.get("resource_type", "crisis_counselling"),
                    modality=entry.get("modality", []),
                    availability=entry.get("availability"),
                    toll_free=entry.get("toll_free"),
                    official_url=entry.get("official_url"),
                    operational_notes=entry.get("operational_notes"),
                )
            )

    # 7. Build and return structured public response
    return ChatResponse(
        session_id=session_id,
        response=final_state.get("final_response", ""),
        safety_route=final_state.get("overall_safety_route", "standard_conversation"),
        verification_status=final_state.get("verification_status", "not_run"),
        grounding_status=final_state.get("grounding_status"),
        citations=citations,
        emergency_resources=emergency_resources,
        jurisdiction=final_jurisdiction,
        error_code=final_state.get("error_code"),
    )


@router.post(
    "/aiml/inference",
    response_model=AimlInferenceResponse,
    tags=["Diagnostics"],
    summary="Execute live AI/ML inference demonstration with empirical telemetry",
)
async def aiml_inference(
    request: AimlInferenceRequest,
    graph: CompiledGraph = Depends(get_app_graph),
    semaphore: asyncio.Semaphore = Depends(get_concurrency_semaphore),
) -> AimlInferenceResponse:
    """
    Dedicated AI/ML live inference and evaluation endpoint for academic presentation.
    Processes user input through the live graph pipeline and extracts genuine
    RoBERTa classification metrics, safety routing decisions, FAISS similarity scores,
    and cascading LLM orchestration telemetry without modifying production state.
    """
    start_time = time.perf_counter()
    session_id = f"aiml-eval-{uuid.uuid4().hex[:8]}"
    user_id_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]

    from backend.config import settings
    effective_jurisdiction: JurisdictionCode = (
        request.jurisdiction
        or getattr(settings, "default_jurisdiction", "IN")
        or "IN"
    )

    initial_state: MindBridgeState = {
        "session_id": session_id,
        "user_id_hash": user_id_hash,
        "messages": [],
        "sanitized_text": "",
        "deterministic_danger_flag": False,
        "emergency_trigger_id": None,
        "roberta_prob": None,
        "roberta_crisis_flag": False,
        "classifier_status": "not_run",
        "overall_safety_route": "standard_conversation",
        "phq9_item9_endorsed": False,
        "phq9_followup_status": "not_started",
        "phq9_risk_level": None,
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": "general",
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieval_status": "not_attempted",
        "jurisdiction_filter": effective_jurisdiction,
        "retrieval_authorization": None,
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }

    try:
        async with semaphore:
            final_state = await asyncio.to_thread(
                graph.invoke,
                initial_state,
                raw_user_text=request.message,
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Pipeline execution failed during AIML inference.",
                "error_code": "GRAPH_EXECUTION_FAILURE",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    traces = final_state.get("execution_trace", [])

    # 1. RoBERTa Classifier Telemetry
    clf_trace = next((t for t in traces if t.get("node") == "classifier_node"), None)
    clf_latency = clf_trace.get("duration_ms", 0.0) if clf_trace else 0.0
    r_prob = final_state.get("roberta_prob")
    r_flag = final_state.get("roberta_crisis_flag", False)
    r_status = final_state.get("classifier_status", "not_run")

    predicted_label = "Crisis-Risk" if r_flag else "Non-Crisis"
    if r_status != "completed":
        predicted_label = "Unassessed"

    clf_decision = "safe_to_route"
    if r_flag:
        clf_decision = "elevated_crisis_risk"
    elif final_state.get("deterministic_danger_flag"):
        clf_decision = "deterministic_emergency_override"

    roberta_telemetry = RoBERTaTelemetry(
        predicted_label=predicted_label,
        probability=round(r_prob, 4) if r_prob is not None else None,
        threshold=0.5098,
        crisis_flag=r_flag,
        latency_ms=clf_latency,
        status=r_status,
        decision=clf_decision,
    )

    # 2. Safety Routing Telemetry
    overall_route = final_state.get("overall_safety_route", "standard_conversation")
    is_emergency = overall_route == "emergency_bypass"
    safety_telemetry = SafetyRoutingTelemetry(
        safety_route=overall_route,
        emergency_bypass_triggered=is_emergency,
        rag_retrieval_bypassed=is_emergency or (final_state.get("retrieval_status") == "not_attempted"),
        llm_generation_skipped=is_emergency,
        emergency_trigger_id=final_state.get("emergency_trigger_id"),
    )

    # 3. RAG Retrieval Telemetry
    plan_trace = next((t for t in traces if t.get("node") == "planning_node"), None)
    rag_latency = plan_trace.get("duration_ms", 0.0) if plan_trace else 0.0
    raw_chunks = final_state.get("retrieved_chunks", [])
    chunk_items: List[RetrievedChunkTelemetry] = []
    for c in raw_chunks:
        chunk_items.append(
            RetrievedChunkTelemetry(
                title=c.get("title", "Resource"),
                document_id=c.get("document_id", ""),
                category=c.get("category", "General"),
                score=round(float(c.get("score", 0.0)), 4),
                source=c.get("source"),
                collection=c.get("collection"),
            )
        )

    rag_telemetry = RAGTelemetry(
        retrieval_status=final_state.get("retrieval_status", "not_attempted"),
        chunks_count=len(chunk_items),
        chunks=chunk_items,
        latency_ms=rag_latency,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimensions=384,
        similarity_metric="FAISS IndexFlatIP (Cosine Similarity)",
    )

    # 4. LLM Telemetry
    llm_raw = final_state.get("llm_telemetry", {})
    resp_trace = next((t for t in traces if t.get("node") == "response_node"), None)
    gen_latency = resp_trace.get("duration_ms", 0.0) if resp_trace else llm_raw.get("duration_ms", 0.0)

    llm_telemetry = LLMTelemetry(
        provider=llm_raw.get("provider", "local" if not is_emergency else "skipped"),
        model_name=llm_raw.get("model", "deterministic_grounded_fallback" if not is_emergency else "bypassed_by_safety"),
        fallback_triggered=bool(llm_raw.get("fallback_triggered", False)),
        local_fallback_used=bool(llm_raw.get("local_fallback_used", False)),
        generation_latency_ms=gen_latency,
        success=bool(llm_raw.get("success", True)),
        error_code=llm_raw.get("error_code"),
    )

    # 5. Provenance Citations
    citations: List[CitationItem] = []
    if not is_emergency and final_state.get("retrieval_status") == "success":
        from backend.retrieval import filter_used_citations
        used = filter_used_citations(
            response_text=final_state.get("final_response", ""),
            retrieved_chunks=raw_chunks,
        )
        for u in used:
            citations.append(
                CitationItem(
                    title=u.get("title", ""),
                    document_id=u.get("document_id", ""),
                    category=u.get("category", ""),
                    jurisdiction=u.get("jurisdiction", "GL"),
                )
            )

    # 6. Emergency Resources if triggered
    emergency_resources: List[EmergencyResourceItem] = []
    if is_emergency:
        from backend.orchestrator.safety_node import get_hotlines_for_jurisdiction
        hotlines = get_hotlines_for_jurisdiction(effective_jurisdiction)
        for h in hotlines:
            emergency_resources.append(EmergencyResourceItem(**h))

    return AimlInferenceResponse(
        session_id=session_id,
        user_message_length=len(request.message),
        roberta=roberta_telemetry,
        safety=safety_telemetry,
        rag=rag_telemetry,
        llm=llm_telemetry,
        verification_status=final_state.get("verification_status", "not_run"),
        grounding_status=final_state.get("grounding_status"),
        final_response=final_state.get("final_response", ""),
        citations=citations,
        emergency_resources=emergency_resources,
        jurisdiction=effective_jurisdiction,
        total_pipeline_latency_ms=total_latency_ms,
    )
