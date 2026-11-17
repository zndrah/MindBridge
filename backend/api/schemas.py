"""
MindBridge 2.0 - API Request & Response Schemas (`schemas.py`)
==============================================================

Pydantic v2 schemas enforcing input boundary validation, forbidden extra fields,
and safe non-leaking response structures.
"""

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.orchestrator.state import JurisdictionCode, SafetyState, VerificationStatus

VALID_JURISDICTIONS = {"IN", "US", "UK", "CA", "AU", "GL", "UNKNOWN"}


class HealthResponse(BaseModel):
    """Liveness probe response model."""
    status: str = "healthy"
    version: str = "0.2.0"
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelReadiness(BaseModel):
    """Readiness breakdown of internal local models and indexes."""
    crisis_detector: bool
    embedding_engine: bool
    vector_store: bool
    graph: bool
    llm_generator: bool = False
    llm_provider: str = "local"
    gemini_generator: bool = False
    groq_generator: bool = False


class ReadyResponse(BaseModel):
    """Readiness probe response model."""
    status: Literal["ready", "not_ready"]
    models: ModelReadiness
    details: Dict[str, Any]
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RootResponse(BaseModel):
    """Service information root response model."""
    name: str = "MindBridge 2.0 API"
    version: str = "0.2.0"
    description: str = "Safety-first, non-clinical mental health first-response and knowledge-grounded support system."
    documentation: str = "/docs"
    health: str = "/health"
    ready: str = "/ready"


class ConversationTurn(BaseModel):
    """Sanitized conversation history turn."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=10000)

    model_config = ConfigDict(extra="forbid")


class ChatRequest(BaseModel):
    """
    Chat endpoint request payload.
    Rejects any client-supplied authorization or safety flags.
    """
    message: str = Field(..., min_length=1, max_length=10000, description="User input message.")
    session_id: Optional[str] = Field(default=None, max_length=64, description="Client session identifier.")
    jurisdiction: Optional[JurisdictionCode] = Field(default=None, description="User jurisdiction code.")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None, max_length=20, description="Optional previous conversation turns."
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        trimmed = v.strip()
        if not trimmed:
            return None
        if len(trimmed) > 64:
            raise ValueError("session_id must not exceed 64 characters.")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", trimmed):
            raise ValueError(
                "session_id must contain only alphanumeric characters, underscores, or hyphens."
            )
        return trimmed

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message must not be empty or whitespace only.")
        return trimmed

    @field_validator("jurisdiction")
    @classmethod
    def validate_jurisdiction(cls, v: Optional[str]) -> Optional[JurisdictionCode]:
        if v is None:
            return None
        trimmed = v.strip().upper()
        if not trimmed:
            return None
        if trimmed not in VALID_JURISDICTIONS:
            return None
        return trimmed  # type: ignore


class CitationItem(BaseModel):
    """Authoritative knowledge chunk provenance citation."""
    title: str
    document_id: str
    category: str
    jurisdiction: str


class EmergencyResourceItem(BaseModel):
    """Listed emergency/crisis support contact from published directory records."""
    resource_id: str
    resource_name: str
    authority: Optional[str] = None
    telephone: Optional[str] = None
    alternate_telephone: Optional[str] = None
    dialing_instructions: Optional[str] = None
    sms_instructions: Optional[str] = None
    resource_type: str = "crisis_counselling"
    modality: List[str] = Field(default_factory=list)
    availability: Optional[str] = None
    toll_free: Optional[bool] = None
    official_url: Optional[str] = None
    operational_notes: Optional[str] = None
    source_screen: Optional[str] = None
    verification_status: str = "source_extracted"
    uptime_verification: str = "unverified_operational_availability"
    governing_authority: Optional[str] = None
    url_source: Optional[str] = None


class ChatResponse(BaseModel):
    """
    Public conversational response. Excludes raw model logits, internal traces,
    or private error strings.
    """
    session_id: str
    response: str
    safety_route: SafetyState
    verification_status: VerificationStatus
    grounding_status: Optional[str] = None
    citations: List[CitationItem] = Field(default_factory=list)
    emergency_resources: List[EmergencyResourceItem] = Field(default_factory=list)
    jurisdiction: Optional[JurisdictionCode] = Field(default=None, description="Active user jurisdiction code.")
    error_code: Optional[str] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorResponse(BaseModel):
    """Controlled public error response model."""
    error: str
    error_code: Optional[str] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# Live AIML Inference & Telemetry Schemas
# ============================================================================

class RetrievedChunkTelemetry(BaseModel):
    """Detailed telemetry for individual retrieved vector chunks."""
    title: str
    document_id: str
    category: str
    score: float
    source: Optional[str] = None
    collection: Optional[str] = None


class RoBERTaTelemetry(BaseModel):
    """Actual runtime outputs from the trained RoBERTa crisis detector."""
    predicted_label: str = "Non-Crisis"
    probability: Optional[float] = None
    threshold: float = 0.5098
    crisis_flag: bool = False
    latency_ms: float = 0.0
    status: str = "completed"
    decision: str = "safe_to_route"
    note: str = "Experimental dataset-derived signal; not clinical diagnostic risk assessment."


class SafetyRoutingTelemetry(BaseModel):
    """Deterministic & ML-driven safety router decision telemetry."""
    safety_route: SafetyState
    emergency_bypass_triggered: bool = False
    rag_retrieval_bypassed: bool = False
    llm_generation_skipped: bool = False
    emergency_trigger_id: Optional[str] = None


class RAGTelemetry(BaseModel):
    """Dense vector retrieval telemetry from FAISS indexes."""
    retrieval_status: str
    chunks_count: int = 0
    chunks: List[RetrievedChunkTelemetry] = Field(default_factory=list)
    latency_ms: float = 0.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    similarity_metric: str = "FAISS IndexFlatIP (Cosine Similarity)"


class LLMTelemetry(BaseModel):
    """Cascading LLM provider execution telemetry."""
    provider: str = "local"
    model_name: Optional[str] = None
    fallback_triggered: bool = False
    local_fallback_used: bool = False
    generation_latency_ms: float = 0.0
    success: bool = True
    error_code: Optional[str] = None


class AimlInferenceRequest(BaseModel):
    """Request payload for live AIML inference evaluation."""
    message: str = Field(..., min_length=1, max_length=10000, description="Test input utterance.")
    jurisdiction: Optional[JurisdictionCode] = Field(default="IN", description="Target emergency jurisdiction code.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message must not be empty or whitespace only.")
        return trimmed


class AimlInferenceResponse(BaseModel):
    """Complete empirical telemetry for live demonstration."""
    session_id: str
    user_message_length: int
    roberta: RoBERTaTelemetry
    safety: SafetyRoutingTelemetry
    rag: RAGTelemetry
    llm: LLMTelemetry
    verification_status: str
    grounding_status: Optional[str] = None
    final_response: str
    citations: List[CitationItem] = Field(default_factory=list)
    emergency_resources: List[EmergencyResourceItem] = Field(default_factory=list)
    jurisdiction: JurisdictionCode = "IN"
    total_pipeline_latency_ms: float = 0.0
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
