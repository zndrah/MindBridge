"""
MindBridge 2.0 - LangGraph State Container Schema
=================================================

This module defines the foundational strongly-typed state schema (`MindBridgeState`)
governing all state transitions across the MindBridge 2.0 LangGraph state machine.

CRITICAL PRIVACY RULE (Privacy-by-Design & Hardened Boundary):
--------------------------------------------------------------
Raw, unmasked user input must NEVER enter or be stored in persistent graph state,
checkpoints, execution traces, or disk logs.

1. BOUNDARY SEPARATION:
   Raw user text is strictly temporary input-boundary data held in ephemeral
   in-memory buffers within `input_node`. It is discarded immediately upon PII scrubbing.
   Only sanitized text (`sanitized_text`) and sanitized conversation history
   (`messages`) are permitted in this schema and persisted downstream.

2. RUNTIME SANITIZATION RESPONSIBILITY:
   `TypedDict` provides static structural typing and compile-time verification,
   but does NOT enforce runtime sanitization. Runtime boundary sanitization is the
   mandatory responsibility of `input_node` and upstream boundary middleware before
   any payload or field is merged into graph state.

3. CONTROLLED ENUMS & PROHIBITION OF ARBITRARY FREE-TEXT:
   To eliminate accidental leakage channels:
   - Errors are recorded via controlled error codes (`error_code`) and component IDs;
     raw exception messages (`str(e)`), tracebacks, and unfiltered error strings are prohibited.
   - Emergency triggers are recorded via predefined pattern identifiers (`emergency_trigger_id`);
     matched substrings of user text are strictly prohibited.
   - Timeframes are recorded via normalized categories (`timeframe`); free-form text is prohibited.
   - Protective factors are recorded via normalized category tags (`protective_factors`);
     verbatim user quotes are strictly prohibited.
   - Execution traces are strictly typed (`TraceEntry`) to exclude arbitrary payloads.
   - Retrieval queries (`retrieval_queries`) must be sanitized queries synthesized by planning nodes.

Authoritative References:
- docs/PRD.md §5.2 (NFR-2: Privacy-by-Design & Input Safety)
- docs/ARCHITECTURE.md §3.1 (LangGraph State Container Schema)
- docs/AGENTS.md §2 (Input Processing Node & PII Boundary)
- docs/SAFETY_POLICY.md §4 (PHQ-9 Seven-Dimensional Clinical Framework)
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, NotRequired


# ============================================================================
# 1. System Enums & Literal Type Definitions
# ============================================================================

JurisdictionCode = Literal[
    "IN",       # India
    "US",       # United States
    "UK",       # United Kingdom
    "CA",       # Canada
    "AU",       # Australia
    "GL",       # Global / International non-profit resources
    "UNKNOWN"   # Unresolved / Unknown jurisdiction
]
"""Supported jurisdiction codes for emergency hotline and knowledge retrieval routing."""

SafetyState = Literal[
    "emergency_bypass",           # Explicit immediate danger disclosed; halts open generation
    "pending_safety_follow_up",   # Multi-turn 7-dimension risk assessment in progress
    "supportive_deescalation",     # Passive ideation or chronic distress without imminent peril
    "standard_conversation",      # Non-crisis conversational or psychoeducational dialogue
    "elevated_classifier_signal", # RoBERTa p >= tau without explicit regex triggers
    "ml_risk_unassessed"          # RoBERTa timeout or runtime exception; safe fallback active
]
"""The six mutually exclusive operational safety states governing system-wide triage."""

PHQ9FollowupStatus = Literal[
    "not_started",  # Follow-up has not been triggered
    "in_progress",  # Multi-turn structured clinical assessment underway
    "completed",    # Assessment finished; risk level established
    "interrupted"   # Evaluation interrupted, timed out, or encountered an unhandled error
]
"""Consistent multi-turn PHQ-9 Item 9 follow-up assessment progress states."""

PHQ9DimensionName = Literal[
    "passive_death_wishes",      # Dimension 1
    "active_suicidal_ideation",  # Dimension 2
    "intent",                    # Dimension 3
    "plan",                      # Dimension 4
    "means",                     # Dimension 5
    "access",                    # Dimension 6
    "timeframe"                  # Dimension 7
]
"""Exact names of the seven documented clinical risk dimensions."""

TimeframeCategory = Literal[
    "immediate_imminent",  # Acute/immediate crisis (e.g., right now, within hours, tonight)
    "near_future",         # Defined upcoming timeframe (e.g., within days/weeks)
    "unspecified_future",  # Chronic, vague, or non-specific timeframe
    "none_reported"        # No suicidal timeframe disclosed
]
"""Normalized timeframe categories to prevent raw user text storage in risk dimensions."""

ProtectiveFactorCategory = Literal[
    "family_support",        # Family, children, partner
    "social_network",        # Friends, community, peers
    "professional_care",     # Therapist, psychiatrist, medical team
    "spiritual_faith",       # Religious, philosophical, moral convictions
    "future_goals",          # Aspirations, responsibilities, projects
    "pet_companionship",     # Pets, caregiving duties
    "other_sanitized",       # Other vetted non-PII protective factor tag
    "none_reported"          # No protective factors identified
]
"""Normalized protective factor categories to prevent raw user quote storage in state."""

EmergencyTriggerID = Literal[
    "IMMINENT_SUICIDE_INTENT",      # Explicit statements of active suicidal intent
    "EXPLICIT_SUICIDAL_INTENT",      # Explicit expressions of suicidal thoughts, urges, or intent
    "SAFETY_FOLLOWUP_CONFIRMED",    # Positive confirmation of suicidal thoughts or danger during safety follow-up
    "EMERGENCY_STATE_PRESERVED",    # Active crisis emergency bypass state maintained
    "LETHAL_MEANS_DISCLOSURE",      # Disclosures of possession/access to lethal means
    "IMMINENT_TIMEFRAME_CRISIS",    # Disclosures of immediate/tonight crisis timeline
    "SUICIDE_NOTE_POSTING",         # Final farewell communications or suicide notes
    "SELF_HARM_DIRECT_THREAT",      # Active acute physical self-harm threats
    "UNKNOWN_EMERGENCY_TRIGGER"     # Fallback emergency trigger identifier
]
"""Predefined emergency pattern identifiers; storing matched user substrings is prohibited."""

ErrorCode = Literal[
    "CLASSIFIER_TIMEOUT",           # RoBERTa inference exceeded latency budget (>200ms)
    "CLASSIFIER_FAILURE",           # RoBERTa tokenization or forward pass exception
    "RETRIEVAL_TIMEOUT",            # Vector search exceeded latency budget (>100ms)
    "RETRIEVAL_FAILURE",            # FAISS index or embedding model exception
    "LLM_GENERATION_TIMEOUT",       # Generative process exceeded cutoff timeout
    "LLM_GENERATION_FAILURE",       # Generative LLM exception or malformed token stream
    "VERIFICATION_VIOLATION",       # Safety validator flagged unauthorized content or numbers
    "VERIFICATION_FAILURE",         # Verification gate runtime exception
    "INPUT_PARSING_FAILURE",        # Ephemeral input sanitization or encoding error
    "FOLLOWUP_PROCESSING_FAILURE",  # Multi-turn follow-up dialogue state failure
    "STATE_TRANSITION_ERROR",       # Router transition failure or invalid safety state
    "UNKNOWN_ERROR"                 # Uncategorized runtime exception
]
"""Controlled error codes preventing raw exception strings or stack traces from entering state."""

RiskLevel = Literal[
    "imminent",        # Immediate suicide risk disclosed (active ideation + means/intent/access)
    "passive_unclear", # Passive death wishes or ambiguous distress without imminent peril
    "none",            # No suicidal ideation detected
    "unassessed"       # Risk level unassessed due to failure or interruption
]
"""Clinical risk stratification levels derived from the seven-dimensional evaluation."""

DialogGoal = Literal[
    "somatic_grounding",    # Grounding exercises (box breathing, physiological sigh)
    "cognitive_reframing",  # Educational self-reflection worksheets
    "screening",            # Educational discussion of screening tools (PHQ-9, GAD-7, PSS-10)
    "general"               # Supportive informational or general conversational dialogue
]
"""Conversational goal determined by planning node for knowledge retrieval and prompting."""

ClassifierStatus = Literal[
    "not_run",    # Classifier inference has not been attempted
    "completed",  # Inference succeeded and produced a valid probability
    "timeout",    # Inference exceeded latency threshold (>200ms)
    "error"       # Runtime exception encountered during tokenization or model forward pass
]
"""Operational execution status of the RoBERTa crisis classifier."""

RetrievalStatus = Literal[
    "not_attempted",  # Knowledge retrieval not invoked (e.g. emergency bypass active)
    "success",        # Top-k chunks retrieved meeting cosine similarity threshold (>= 0.65)
    "empty",          # No knowledge chunks met the similarity threshold
    "timeout",        # Vector search exceeded latency threshold (>100ms)
    "error"           # FAISS index or embedding model runtime exception
]
"""Operational execution status of dense vector knowledge retrieval."""

VerificationStatus = Literal[
    "not_run",              # Verification gate not invoked (e.g. emergency bypass)
    "passed",               # Synonym for verified (backward compatible)
    "verified",             # Draft response satisfies all safety, grounding, and contact checks
    "failed",               # General rejection (backward compatible)
    "rejected_unsafe",      # Safety policy violation (diagnostic claim, prescription, fabricated contact)
    "rejected_ungrounded",  # Ungrounded factual claim or invalid citation
    "fallback_used",        # Safe fallback substituted due to empty retrieval or suppressed draft
    "error",                # General error (backward compatible)
    "verification_error",   # Exception encountered during verification execution
]
"""Operational execution status of the post-generation verification gate."""


# ============================================================================
# 2. Sub-State Schemas (Clinical Dimensions, Metadata, Trace & Chunks)
# ============================================================================

class SanitizedMessage(TypedDict):
    """
    Sanitized conversational turn record.
    Must pass through the input sanitization boundary before inclusion.
    Raw unmasked user utterances are strictly prohibited in `content`.
    """
    role: Literal["user", "assistant", "system"]
    content: str  # PII-masked, sanitized text only


class TraceEntry(TypedDict):
    """
    Restricted execution trace entry containing strictly controlled telemetry metadata.
    Unfiltered user disclosures, prompt bodies, or raw payloads are strictly forbidden.
    """
    node: str                                                    # Name of executed node (e.g. "safety_node")
    status: Literal["success", "fallback", "error", "skipped"]   # Node execution status
    duration_ms: float                                           # Monotonic duration in milliseconds
    timestamp_utc: str                                           # ISO-8601 UTC timestamp
    metadata_code: Optional[str]                                 # Controlled code (e.g. "REGEX_MATCH", "CACHE_HIT")


class ClinicalRiskDimensions(TypedDict, total=False):
    """
    The exact seven PHQ-9 Item 9 clinical risk dimensions documented in:
    docs/PHASE3_STEP4_1_CORRECTIONS.md and docs/SAFETY_POLICY.md §4.1.

    All fields default to None until explicitly evaluated during multi-turn follow-up.
    `timeframe` uses normalized categories to eliminate free-form user text storage.
    """
    passive_death_wishes: Optional[bool]              # Dimension 1: Thoughts of wanting relief or wishing not to wake up
    active_suicidal_ideation: Optional[bool]          # Dimension 2: Explicit thoughts of taking one's life
    intent: Optional[bool]                            # Dimension 3: Stated commitment or determination to act
    plan: Optional[bool]                              # Dimension 4: Formulated method, location, or sequence of actions
    means: Optional[bool]                             # Dimension 5: Knowledge of or possession of lethal mechanisms
    access: Optional[bool]                            # Dimension 6: Physical availability and proximity to lethal means
    timeframe: Optional[TimeframeCategory]            # Dimension 7: Normalized timeframe category; never free-form text


class SupplementaryContext(TypedDict, total=False):
    """
    Supplementary contextual considerations informing conversational de-escalation tone.
    Must NOT be presented as replacements or additional dimensions in the 7-dimensional framework.
    `protective_factors` uses normalized categories to prevent storing raw user disclosures.
    """
    protective_factors: Optional[List[ProtectiveFactorCategory]]  # Normalized category tags; never free-form text
    emotional_agitation: Optional[bool]                           # Acute behavioral turbulence, panic, or impulsivity


class KnowledgeChunk(TypedDict):
    """
    Structured representation of a dense vector knowledge chunk retrieved from FAISS.
    Grounded in docs/ARCHITECTURE.md §5 and data/knowledge_base/.
    """
    chunk_id: str
    document_id: str
    title: str
    category: str
    jurisdiction: JurisdictionCode
    content: str
    score: float  # Cosine similarity score in [-1.0, 1.0] (faiss.IndexFlatIP on unit L2 vectors)
    eligibility: Literal["approved", "restricted"]
    section_header: NotRequired[str]
    verification_status: NotRequired[str]
    guardrails: NotRequired[List[str]]
    disclaimer_text: NotRequired[str]
    content_hash: NotRequired[str]
    clinical_safety_level: NotRequired[str]
    source: NotRequired[str]
    retrieval_tier: NotRequired[str]
    safety_level: NotRequired[str]
    disclaimer: NotRequired[str]
    chunk_index: NotRequired[int]


# ============================================================================
# 3. Master MindBridge State Schema (LangGraph Compatible)
# ============================================================================

class MindBridgeState(TypedDict):
    """
    Foundational shared state container for the MindBridge 2.0 LangGraph state machine.

    Passed immutably between graph nodes. Every field represents verified, typed
    conversational, safety, retrieval, or verification state.

    PRIVACY GUARANTEE:
    Raw unmasked user utterances, raw exception strings, and unfiltered user disclosures
    are strictly prohibited across all fields. All user-derived text must pass through
    the sanitization boundary before entering persistent graph state.
    """
    # ------------------------------------------------------------------------
    # Session Context & Sanitized Conversation History
    # ------------------------------------------------------------------------
    session_id: str
    user_id_hash: str
    messages: List[SanitizedMessage]                 # Sanitized conversation history only; never raw input
    sanitized_text: str                              # PII-masked user utterance entering downstream graph state

    # ------------------------------------------------------------------------
    # Safety Intelligence & Classifier Outputs
    # ------------------------------------------------------------------------
    deterministic_danger_flag: bool                  # True if deterministic regex emergency pattern triggered
    emergency_trigger_id: Optional[EmergencyTriggerID] # Predefined trigger identifier; NEVER matched user text
    roberta_prob: Optional[float]                    # RoBERTa Crisis-Risk probability (0.0 to 1.0)
    roberta_crisis_flag: bool                        # True if roberta_prob >= 0.5098
    classifier_status: ClassifierStatus              # "not_run" | "completed" | "timeout" | "error"
    overall_safety_route: SafetyState                # One of 6 operational safety states

    # ------------------------------------------------------------------------
    # PHQ-9 Item 9 Structured Multi-Turn Follow-Up
    # ------------------------------------------------------------------------
    phq9_item9_endorsed: bool                        # True if PHQ-9 Item 9 endorsed
    phq9_followup_status: PHQ9FollowupStatus         # "not_started" | "in_progress" | "completed" | "interrupted"
    phq9_risk_level: Optional[RiskLevel]             # "imminent" | "passive_unclear" | "none" | "unassessed"
    phq9_dimensions: ClinicalRiskDimensions          # The 7 clinical risk dimensions (normalized categories)
    phq9_supplementary_context: SupplementaryContext  # Supplementary factors (normalized categories)
    phq9_current_dimension: Optional[PHQ9DimensionName] # Current dimension evaluated in multi-turn dialogue

    # ------------------------------------------------------------------------
    # Dialog Planning & Knowledge Retrieval
    # ------------------------------------------------------------------------
    dialog_goal: Optional[DialogGoal]                # "somatic_grounding" | "cognitive_reframing" | "screening" | "general"
    retrieval_queries: List[str]                     # Sanitized search queries synthesized by planning; never raw input
    retrieved_chunks: List[KnowledgeChunk]           # Top-k retrieved chunks
    retrieval_status: RetrievalStatus                # "not_attempted" | "success" | "empty" | "timeout" | "error"
    jurisdiction_filter: JurisdictionCode            # Jurisdiction code for retrieval filtering
    retrieval_authorization: NotRequired[Optional[Any]]  # Explicit trusted backend authorization for restricted retrieval

    # ------------------------------------------------------------------------
    # Response Generation, Verification & Error Handling
    # ------------------------------------------------------------------------
    draft_response: Optional[str]                    # Candidate response text from local LLM
    verification_passed: bool                        # True if all verification checks passed
    verification_flags: List[str]                    # Controlled violation category flags if verification fails
    verification_status: VerificationStatus          # Operational verification status
    grounding_status: NotRequired[Optional[str]]     # "grounded" | "unsupported" | "fallback"
    llm_telemetry: NotRequired[Dict[str, Any]]       # Telemetry dictionary from generation node (provider, model, duration_ms, fallback_triggered)
    final_response: str                              # Validated response string emitted to user
    error_code: Optional[ErrorCode]                  # Controlled error code; raw exception strings prohibited
    error_component: Optional[str]                   # Subsystem/node name (e.g. "safety_node"); no raw exception text
    execution_trace: List[TraceEntry]                # Strictly typed telemetry metadata; arbitrary payloads prohibited
