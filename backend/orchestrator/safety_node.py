"""
MindBridge 2.0 - Deterministic Safety Node & Classifier Decision Node
=====================================================================

This module implements the deterministic safety arbitration layer and post-classifier
decision logic for MindBridge 2.0.

Authoritative Governance:
- docs/SAFETY_POLICY.md
- docs/ARCHITECTURE.md
- backend/orchestrator/state.py
- backend/safety/hotlines.json
- Safety Architecture Contract v2.1.0

Core Invariants:
1. Universal Emergency Override: Explicit immediate danger always overrides ML classifiers,
   follow-up status, and standard conversation routes.
2. Follow-Up Continuity: Active (in_progress) or interrupted follow-ups take priority over
   classifier execution and ordinary RAG.
3. Decoupling of Signals: Preliminary regex and RoBERTa signals are screening alerts only;
   they NEVER set phq9_item9_endorsed = True.
4. Strict Ternary Clinical Logic: True = affirmed, False = explicitly denied, None = unresolved.
   None is NEVER interpreted as False.
5. Contextual Safety Analysis: Negation, quotations, hypotheticals, third-party reports,
   and ordinary medication use do not trigger emergency bypass.
6. Privacy-Safe Telemetry: Zero raw user messages, matched regex substrings, or raw exception
   strings enter execution traces or error fields.
7. Jurisdiction-Aware Hotlines: Sourced strictly from hotlines.json; zero invented numbers;
   never claim live dial-testing; zero auto-dispatch claims.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from backend.safety.crisis_detector import (
    CrisisDetector,
    CrisisPredictionResult,
    DEFAULT_THRESHOLD as ROBERTA_CRISIS_THRESHOLD,
)
from backend.orchestrator.input_node import (
    CLAUSE_DELIMITERS,
    EMERGENCY_TRIGGERS,
    HYPOTHETICAL_PATTERNS,
    NEGATION_PATTERNS,
    ORDINARY_MEDICATION_PATTERNS,
    PASSIVE_DEATH_WISHES_REGEX,
    QUOTE_OR_ACADEMIC_PATTERNS,
    SELF_HARM_DISCLOSURE_PATTERNS,
    SUICIDAL_IDEATION_REGEX,
    THIRD_PARTY_PATTERNS,
    evaluate_clauses_for_safety,
)
from backend.orchestrator.state import (
    ClassifierStatus,
    ClinicalRiskDimensions,
    EmergencyTriggerID,
    ErrorCode,
    JurisdictionCode,
    MindBridgeState,
    PHQ9DimensionName,
    PHQ9FollowupStatus,
    RiskLevel,
    SafetyState,
    TimeframeCategory,
    TraceEntry,
)

# Cache for hotlines.json to eliminate redundant disk I/O
_HOTLINES_DATA: Optional[Dict[str, Any]] = None


# ============================================================================
# Hotline Directory Loading & Emergency Guidance Synthesis
# ============================================================================

def load_hotlines_directory(reload: bool = False) -> Dict[str, Any]:
    """
    Loads and caches the authoritative hotlines directory from backend/safety/hotlines.json.
    Pass reload=True to bust cache and re-read from disk.
    """
    global _HOTLINES_DATA
    if _HOTLINES_DATA is not None and not reload:
        return _HOTLINES_DATA

    hotlines_path = Path(__file__).resolve().parent.parent / "safety" / "hotlines.json"
    if not hotlines_path.exists():
        # Fallback to local project path if running from working directory
        hotlines_path = Path("backend/safety/hotlines.json")

    try:
        with open(hotlines_path, "r", encoding="utf-8") as f:
            _HOTLINES_DATA = json.load(f)
    except Exception:
        _HOTLINES_DATA = {"jurisdictions": {}}

    return _HOTLINES_DATA


def set_hotlines_directory_for_testing(data: Optional[Dict[str, Any]]) -> None:
    """Helper to set or reset cached hotlines data in test harnesses."""
    global _HOTLINES_DATA
    _HOTLINES_DATA = data


def get_hotlines_for_jurisdiction(jurisdiction: JurisdictionCode) -> List[Dict[str, Any]]:
    """
    Extracts source-extracted directory hotline entries applicable to the user's jurisdiction.
    Provenance is sourced from hotlines.json without live telephony operational testing.
    """
    data = load_hotlines_directory()
    jurisdictions = data.get("jurisdictions", {})

    if jurisdiction in jurisdictions:
        jur_data = jurisdictions[jurisdiction]
        entries: List[Dict[str, Any]] = []
        for category in (
            "emergency_dispatch",
            "crisis_counselling",
            "crisis_text_services",
            "text_support",
            "secondary_hospital_facilities",
            "hospital_emergency",
        ):
            if category in jur_data and isinstance(jur_data[category], list):
                entries.extend(jur_data[category])
        return entries

    return []


def is_usable_sms_instruction(sms_inst: Optional[str]) -> bool:
    """
    Checks if an SMS instruction string provides a clear, usable destination and directive.
    Requires the string to start with 'Text ' and contain at least one numeric digit.
    """
    if not sms_inst or not isinstance(sms_inst, str):
        return False
    trimmed = sms_inst.strip()
    if not trimmed:
        return False
    if not trimmed.lower().startswith("text "):
        return False
    if not re.search(r"\d", trimmed):
        return False
    return True


def _format_hotline_markdown_entry(entry: Dict[str, Any]) -> str:
    """
    Formats a single hotline entry into standard markdown using strictly the fields
    present in the directory entry. Does not invent contact methods or numbers.
    Never infers SMS capability or destination from a voice telephone number.
    """
    name = entry.get("resource_name", "Support Resource")
    telephone = entry.get("telephone")
    alt_tel = entry.get("alternate_telephone")
    dialing_inst = entry.get("dialing_instructions")
    sms_inst = entry.get("sms_instructions")
    modality = entry.get("modality", [])
    availability = entry.get("availability")
    toll_free = entry.get("toll_free")
    url = entry.get("official_url")
    res_type = entry.get("resource_type", "")

    action_parts: List[str] = []

    # 1. Voice dialing or special dialing instructions
    if dialing_inst:
        action_parts.append(dialing_inst)
    elif telephone and ("voice" in modality or not modality):
        if alt_tel and res_type != "emergency_dispatch":
            action_parts.append(f"Dial **{telephone}** or **{alt_tel}**")
        else:
            action_parts.append(f"Dial **{telephone}**")

    # 2. SMS instructions: strictly require explicit, usable instructions
    if "sms" in modality:
        if is_usable_sms_instruction(sms_inst):
            action_parts.append(sms_inst.strip())
        else:
            action_parts.append("SMS instructions unavailable")
    elif is_usable_sms_instruction(sms_inst):
        action_parts.append(sms_inst.strip())

    # 3. Web directory fallback if no phone/sms exists
    if not action_parts and url:
        action_parts.append(f"Directory: {url}")

    # Metadata badges: modality, availability, toll-free
    meta_badges: List[str] = []
    if "voice" in modality and "sms" in modality:
        meta_badges.append("Voice & Text")
    elif "voice" in modality:
        meta_badges.append("Voice")
    elif "sms" in modality:
        meta_badges.append("SMS")

    if availability:
        meta_badges.append(availability)
    if toll_free:
        meta_badges.append("Toll-Free")

    actions_str = " | ".join(action_parts) if action_parts else "Listed Resource"
    meta_str = f" ({', '.join(meta_badges)})" if meta_badges else ""

    return f"- **{name}:** {actions_str}{meta_str}"


def format_emergency_guidance(jurisdiction: JurisdictionCode, already_asked_danger: bool = False) -> str:
    """
    Synthesizes pre-approved static emergency guidance using directory entries strictly
    applicable to the user's jurisdiction.
    Strictly prohibits claims of automated calling, emergency dispatch, or live dial-testing.
    """
    contacts = get_hotlines_for_jurisdiction(jurisdiction)

    lines = [
        "I hear how much pain and distress you are experiencing right now. Please know that you are not alone, and your safety is the most important thing.",
        "",
    ]

    if not already_asked_danger:
        lines.extend([
            "Are you in immediate danger right now, or have you already harmed yourself? If you have already hurt yourself or need urgent medical care, please reach out for emergency medical assistance immediately.",
            "",
        ])
    else:
        lines.extend([
            "Thank you for sharing that with me. Because you are in distress and having thoughts of ending your life, your safety is the absolute priority right now.",
            "",
        ])

    lines.extend([
        "If you are in immediate danger, please contact local emergency services immediately or go to the nearest hospital emergency department. "
        "Please also put safe distance between yourself and anything you might use to hurt yourself, and reach out to a trusted family member, friend, or person you feel safe with right now so you do not have to be alone.",
        "",
        "### Immediate Emergency & Crisis Support Resources",
    ])

    if contacts:
        for entry in contacts:
            line = _format_hotline_markdown_entry(entry)
            if line:
                lines.append(line)
    else:
        # Safe generic guidance without inventing numbers or mixing foreign jurisdictions
        lines.extend([
            "- **Local Emergency Services:** If you are in immediate danger or need urgent medical attention, please contact your local emergency dispatch service (police, fire, or ambulance) immediately.",
            "- **Nearest Medical Facility:** Go to the nearest hospital emergency department or medical casualty ward.",
            "- **International Crisis Directories:** International directories may help you find local support services. Check the service's own privacy and availability information (such as **Befrienders Worldwide** at https://www.befrienders.org or **Find A Helpline** at https://findahelpline.com). *(Note: International directory websites provide helpline listings and cannot dispatch local emergency responders).*",
            "",
            "If you would like more specific local resources, please let me know your country or location.",
        ])

    lines.extend([
        "",
        "> [!IMPORTANT]",
        "> **Direct Calling Required:** MindBridge is an automated conversational tool and cannot contact "
        "emergency services, dial hotlines, or dispatch emergency responders on your behalf. "
        "Please use the contact numbers above directly or ask someone with you to help.",
        "> ",
        "> **Operational Notice:** Emergency contacts are compiled from official institutional directories. "
        "Operational availability and live telecommunications routing have not been independently tested.",
    ])

    return "\n".join(lines)


# ============================================================================
# Contextual Analysis Helpers
# ============================================================================

def is_contextual_negation_or_third_party(text: str) -> bool:
    """
    Checks whether user text represents syntactic negation, a third-party report,
    an academic/literary quote, or a hypothetical discussion.
    """
    if not text:
        return False
    return bool(
        NEGATION_PATTERNS.search(text)
        or THIRD_PARTY_PATTERNS.search(text)
        or QUOTE_OR_ACADEMIC_PATTERNS.search(text)
        or HYPOTHETICAL_PATTERNS.search(text)
    )


def is_ordinary_medication_mention(text: str) -> bool:
    """
    Checks whether user text represents routine, non-crisis prescription medication use.
    """
    if not text:
        return False
    return bool(ORDINARY_MEDICATION_PATTERNS.search(text))


# ============================================================================
# Deterministic Safety Node (`deterministic_safety_node`)
# ============================================================================

def deterministic_safety_node(state: MindBridgeState) -> Dict[str, Any]:
    """
    Authoritative deterministic safety arbitration node.

    Precedence:
    1. Explicit Immediate Danger -> emergency_bypass (Universal Override).
    2. Active or Interrupted Follow-Up -> pending_safety_follow_up (Continuity).
    3. Preliminary Screening Trigger -> pending_safety_follow_up (Initiates Follow-Up).
    4. Passthrough -> standard_conversation (Allows classifier execution).

    Strict Guarantees:
    - Never sets phq9_item9_endorsed = True.
    - Never overwrites confirmed follow-up dimensions.
    - Never leaks raw text or exceptions into traces.
    """
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    existing_trace = state.get("execution_trace", [])
    sanitized_text = state.get("sanitized_text", "")
    from backend.config import settings
    jurisdiction = state.get("jurisdiction_filter") or getattr(settings, "default_jurisdiction", "IN") or "IN"

    # ------------------------------------------------------------------------
    # 1. Universal Emergency Override Check
    # ------------------------------------------------------------------------
    is_emergency_flagged = (
        state.get("overall_safety_route") == "emergency_bypass"
        or state.get("deterministic_danger_flag") is True
    )

    emergency_trigger_id: Optional[EmergencyTriggerID] = state.get("emergency_trigger_id")

    # Locally scoped, clause-aware safety evaluation on sanitized text
    has_clause_emergency, clause_trigger_id, has_clause_followup = evaluate_clauses_for_safety(sanitized_text)

    # Disjoint acute peril check during ongoing follow-up:
    # Active ideation combined with immediate access to lethal means
    dimensions = state.get("phq9_dimensions", {})
    has_active_ideation = (
        dimensions.get("active_suicidal_ideation") is True
        or bool(SUICIDAL_IDEATION_REGEX.search(sanitized_text))
    )
    has_means_access = dimensions.get("means") is True and dimensions.get("access") is True
    has_acute_timeframe = dimensions.get("timeframe") == "immediate_imminent"

    has_acute_disjoint_peril = has_active_ideation and (has_means_access or has_acute_timeframe)

    if has_clause_emergency:
        is_emergency_flagged = True
        emergency_trigger_id = emergency_trigger_id or clause_trigger_id
    elif has_acute_disjoint_peril:
        is_emergency_flagged = True
        emergency_trigger_id = emergency_trigger_id or "IMMINENT_SUICIDE_INTENT"

    if is_emergency_flagged:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry: TraceEntry = {
            "node": "deterministic_safety_node",
            "status": "success",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "EMERGENCY_INTERCEPT",
        }
        is_already_asked = emergency_trigger_id in ("SAFETY_FOLLOWUP_CONFIRMED", "EMERGENCY_STATE_PRESERVED")
        emergency_guidance = format_emergency_guidance(jurisdiction, already_asked_danger=is_already_asked)
        return {
            "deterministic_danger_flag": True,
            "emergency_trigger_id": emergency_trigger_id or "UNKNOWN_EMERGENCY_TRIGGER",
            "overall_safety_route": "emergency_bypass",
            "draft_response": None,  # Halt open generation
            "final_response": emergency_guidance,
            "verification_passed": True,  # Pre-approved static template
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # 2. Active or Interrupted Follow-Up Continuity Check
    # ------------------------------------------------------------------------
    followup_status = state.get("phq9_followup_status")
    is_followup_active = followup_status in ("in_progress", "interrupted")

    # If the user utterance is an explicit negation, clarification, or third-party/fictional discussion,
    # the follow-up inquiry has been answered or negated; do NOT loop follow-up question.
    from backend.safety.conversational_safety import is_safety_negation
    if is_followup_active and (is_safety_negation(sanitized_text) or is_contextual_negation_or_third_party(sanitized_text)):
        is_followup_active = False

    if is_followup_active:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry = {
            "node": "deterministic_safety_node",
            "status": "success",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "FOLLOWUP_CONTINUITY",
        }
        # Preserve active follow-up route and all existing dimensions
        return {
            "overall_safety_route": "pending_safety_follow_up",
            "phq9_followup_status": followup_status,
            "deterministic_danger_flag": False,
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # 3. Preliminary Screening Trigger Check
    # ------------------------------------------------------------------------
    is_preliminary_followup = (
        state.get("overall_safety_route") == "pending_safety_follow_up"
        or has_clause_followup
    )

    if is_preliminary_followup:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry = {
            "node": "deterministic_safety_node",
            "status": "success",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "FOLLOWUP_PRELIMINARY_FLAG",
        }
        return {
            "overall_safety_route": "pending_safety_follow_up",
            "phq9_followup_status": "in_progress",
            "deterministic_danger_flag": False,
            # phq9_item9_endorsed remains False until confirmed by dialogue
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # 4. Input Error / Unassessed Risk Preservation
    # ------------------------------------------------------------------------
    if (
        state.get("overall_safety_route") == "ml_risk_unassessed"
        or (state.get("error_code") == "INPUT_PARSING_FAILURE" and state.get("final_response"))
    ):
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry = {
            "node": "deterministic_safety_node",
            "status": "fallback",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "INPUT_REJECTION_PRESERVED",
        }
        return {
            "overall_safety_route": "ml_risk_unassessed",
            "deterministic_danger_flag": False,
            "final_response": state.get("final_response"),
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # 5. Safety Passthrough to Classifier
    # ------------------------------------------------------------------------
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    trace_entry = {
        "node": "deterministic_safety_node",
        "status": "success",
        "duration_ms": duration_ms,
        "timestamp_utc": timestamp_utc,
        "metadata_code": "SAFETY_PASSTHROUGH",
    }
    return {
        "overall_safety_route": "standard_conversation",
        "deterministic_danger_flag": False,
        "execution_trace": existing_trace + [trace_entry],
    }


# ============================================================================
# Classifier Inference & Decision Nodes
# ============================================================================

def classifier_node(
    state: MindBridgeState,
    detector: Optional[CrisisDetector] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Authoritative LangGraph classifier node for MindBridge 2.0.

    Executes fine-tuned RoBERTa crisis classification within bounded execution time,
    strictly enforcing safety precedence:
    1. Universal Emergency Override: Skips inference if emergency is flagged.
    2. Follow-Up Continuity: Skips inference if structured follow-up is active.
    3. Elevated Signal: Initiates structured follow-up; NEVER declares emergency;
       NEVER sets phq9_item9_endorsed = True.
    4. Classifier Failure / Timeout: Enforces fail-closed safety, setting roberta_prob = None,
       roberta_crisis_flag = False, overall_safety_route = "ml_risk_unassessed",
       and emitting safe non-clinical fallback guidance.
    5. Low Score: Permits standard_conversation only when deterministic checks are clear.
    """
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    existing_trace = state.get("execution_trace", [])
    overall_route = state.get("overall_safety_route")

    # Failsafe 1: Universal Emergency Override (Skip inference to save latency)
    if overall_route == "emergency_bypass" or state.get("deterministic_danger_flag") is True:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry: TraceEntry = {
            "node": "classifier_node",
            "status": "skipped",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "EMERGENCY_PRESERVED",
        }
        return {
            "overall_safety_route": "emergency_bypass",
            "classifier_status": "not_run",
            "execution_trace": existing_trace + [trace_entry],
        }

    # Failsafe 2: Follow-Up Preservation (Skip inference during active follow-up)
    if (
        overall_route == "pending_safety_follow_up"
        or state.get("phq9_followup_status") in ("in_progress", "interrupted")
        or state.get("phq9_item9_endorsed") is True
    ):
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry: TraceEntry = {
            "node": "classifier_node",
            "status": "skipped",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "FOLLOWUP_PRESERVED",
        }
        return {
            "overall_safety_route": "pending_safety_follow_up",
            "classifier_status": "not_run",
            "execution_trace": existing_trace + [trace_entry],
        }

    # Execute bounded ML inference
    active_detector = detector or CrisisDetector.get_instance()
    sanitized_text = state.get("sanitized_text", "")

    result: CrisisPredictionResult = active_detector.predict(
        sanitized_text,
        timeout_seconds=timeout_seconds,
    )

    # Map status to ClassifierStatus: "not_run" | "completed" | "timeout" | "error"
    if result.status == "completed":
        status_for_state: ClassifierStatus = "completed"
    elif result.status in ("timeout", "busy"):
        status_for_state = "timeout"
    else:
        status_for_state = "error"

    inference_trace: TraceEntry = {
        "node": "classifier_node",
        "status": "success" if result.status == "completed" else "timeout" if result.status in ("timeout", "busy") else "error",
        "duration_ms": result.latency_ms,
        "timestamp_utc": timestamp_utc,
        "metadata_code": f"INFERENCE_{result.status.upper()}",
    }

    inference_updates: Dict[str, Any] = {
        "roberta_prob": result.probability,
        "roberta_crisis_flag": result.is_crisis,
        "classifier_status": status_for_state,
    }

    # State presented to classifier_decision_node
    decision_state: MindBridgeState = {
        **state,
        **inference_updates,
        "execution_trace": existing_trace + [inference_trace],
    }

    decision_updates = classifier_decision_node(decision_state)

    return {**inference_updates, **decision_updates}


def classifier_decision_node(state: MindBridgeState) -> Dict[str, Any]:
    """
    Evaluates ML classifier results and manages fail-closed transitions.

    Rules:
    - Emergency state preservation: Low classifier score CANNOT override emergency.
    - Follow-up preservation: Classifier cannot dismiss active follow-up.
    - Elevated classifier signal (p >= threshold): Routes to followup_node; NEVER emergency.
      Does NOT set phq9_item9_endorsed = True.
    - Classifier failure/timeout: Routes to ml_risk_unassessed; emits safe fallback.
    - Standard conversation: Allowed only when other conditions permit.
    """
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    existing_trace = state.get("execution_trace", [])
    overall_route = state.get("overall_safety_route")
    classifier_status = state.get("classifier_status", "not_run")
    roberta_prob = state.get("roberta_prob")
    roberta_crisis_flag = state.get("roberta_crisis_flag", False)

    # ------------------------------------------------------------------------
    # Failsafe 1: Emergency State Preservation
    # ------------------------------------------------------------------------
    if overall_route == "emergency_bypass" or state.get("deterministic_danger_flag") is True:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry: TraceEntry = {
            "node": "classifier_decision_node",
            "status": "skipped",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "EMERGENCY_PRESERVED",
        }
        return {
            "overall_safety_route": "emergency_bypass",
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # Failsafe 2: Follow-Up Preservation
    # ------------------------------------------------------------------------
    if overall_route == "pending_safety_follow_up" or state.get("phq9_followup_status") in ("in_progress", "interrupted"):
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry = {
            "node": "classifier_decision_node",
            "status": "skipped",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "FOLLOWUP_PRESERVED",
        }
        return {
            "overall_safety_route": "pending_safety_follow_up",
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # Case A: Completed Inference
    # ------------------------------------------------------------------------
    if classifier_status == "completed":
        is_elevated = roberta_crisis_flag or (
            roberta_prob is not None and roberta_prob >= ROBERTA_CRISIS_THRESHOLD
        )
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if is_elevated:
            from backend.safety.conversational_safety import is_safety_negation
            sanitized_text = state.get("sanitized_text", "")
            # If the user explicitly negates risk, clarifies a fictional/hypothetical situation,
            # or reports a third-party statement without personal harm disclosure,
            # do NOT allow RoBERTa token-level false positives to force a crisis follow-up.
            if is_contextual_negation_or_third_party(sanitized_text) or is_safety_negation(sanitized_text):
                trace_entry = {
                    "node": "classifier_decision_node",
                    "status": "success",
                    "duration_ms": duration_ms,
                    "timestamp_utc": timestamp_utc,
                    "metadata_code": "ROBERTA_NEGATION_SUPPRESSED",
                }
                return {
                    "overall_safety_route": "standard_conversation",
                    "phq9_followup_status": "not_started" if state.get("phq9_followup_status") != "completed" else "completed",
                    "execution_trace": existing_trace + [trace_entry],
                }

            # Elevated signal initiates structured follow-up; NEVER emergency
            trace_entry = {
                "node": "classifier_decision_node",
                "status": "success",
                "duration_ms": duration_ms,
                "timestamp_utc": timestamp_utc,
                "metadata_code": "ROBERTA_ELEVATED",
            }
            return {
                "overall_safety_route": "elevated_classifier_signal",
                "phq9_followup_status": "in_progress",
                # Crucial invariant: phq9_item9_endorsed is NOT set to True by ML prediction
                "execution_trace": existing_trace + [trace_entry],
            }
        else:
            trace_entry = {
                "node": "classifier_decision_node",
                "status": "success",
                "duration_ms": duration_ms,
                "timestamp_utc": timestamp_utc,
                "metadata_code": "ROBERTA_BELOW_THRESHOLD",
            }
            return {
                "overall_safety_route": "standard_conversation",
                "execution_trace": existing_trace + [trace_entry],
            }

    # ------------------------------------------------------------------------
    # Case B: Classifier Timeout or Failure
    # ------------------------------------------------------------------------
    if classifier_status in ("timeout", "error"):
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        error_code: ErrorCode = "CLASSIFIER_TIMEOUT" if classifier_status == "timeout" else "CLASSIFIER_FAILURE"
        meta_code = "CLASSIFIER_TIMEOUT_HANDLED" if classifier_status == "timeout" else "CLASSIFIER_ERROR_HANDLED"

        trace_entry = {
            "node": "classifier_decision_node",
            "status": "fallback",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": meta_code,
        }

        # Safe non-clinical fallback template
        safe_fallback_response = (
            "I am here with you. I am ready to support you with information or resources "
            "whenever you are ready."
        )

        return {
            "roberta_prob": None,
            "roberta_crisis_flag": False,
            "overall_safety_route": "ml_risk_unassessed",
            "error_code": error_code,
            "error_component": "classifier_node",
            "final_response": safe_fallback_response,
            "execution_trace": existing_trace + [trace_entry],
        }

    # Default fallback
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    trace_entry = {
        "node": "classifier_decision_node",
        "status": "fallback",
        "duration_ms": duration_ms,
        "timestamp_utc": timestamp_utc,
        "metadata_code": "CLASSIFIER_UNASSESSED",
    }
    return {
        "overall_safety_route": "ml_risk_unassessed",
        "execution_trace": existing_trace + [trace_entry],
    }


# ============================================================================
# Clinical Risk Dimension & Route Eligibility Validators
# ============================================================================

def evaluate_deescalation_eligibility(dimensions: ClinicalRiskDimensions) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether the 7 clinical dimensions satisfy all conditions for
    supportive_deescalation under the documented safety policy.

    Conditions:
    1. passive_death_wishes is affirmed (True).
    2. active_suicidal_ideation is explicitly False (or not True).
    3. intent is explicitly False.
    4. plan is explicitly False.
    5. means is explicitly False.
    6. access is explicitly False.
    7. timeframe is explicitly "none_reported" or "unspecified_future".
       (Must NOT be "immediate_imminent", "near_future", or None).

    Ternary Invariant:
    None represents unknown, ambiguous, or unanswered data. None is NEVER
    interpreted as False. Any acute dimension remaining None blocks de-escalation.
    """
    if not dimensions.get("passive_death_wishes"):
        return False, "PASSIVE_DEATH_WISHES_NOT_AFFIRMED"

    if dimensions.get("active_suicidal_ideation") is True:
        return False, "ACTIVE_SUICIDAL_IDEATION_PRESENT"

    # Acute dimensions must be explicitly False; None blocks de-escalation
    acute_dims: List[PHQ9DimensionName] = ["intent", "plan", "means", "access"]
    for dim in acute_dims:
        val = dimensions.get(dim)
        if val is None:
            return False, f"UNRESOLVED_ACUTE_DIMENSION_{dim.upper()}"
        if val is True:
            return False, f"ACUTE_DIMENSION_AFFIRMED_{dim.upper()}"

    timeframe = dimensions.get("timeframe")
    if timeframe is None:
        return False, "UNRESOLVED_TIMEFRAME"
    if timeframe == "immediate_imminent":
        return False, "IMMEDIATE_IMMINENT_TIMEFRAME"
    if timeframe == "near_future":
        # A specific near-future timeframe blocks premature de-escalation
        return False, "NEAR_FUTURE_TIMEFRAME"
    if timeframe not in ("none_reported", "unspecified_future"):
        return False, "INVALID_TIMEFRAME"

    return True, None


def evaluate_standard_conversation_eligibility(
    status: PHQ9FollowupStatus,
    dimensions: ClinicalRiskDimensions
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether structured follow-up has successfully concluded and allows
    transitioning back to standard_conversation.

    Conditions:
    1. Follow-up status must be "completed".
    2. Zero dimensions may be affirmed (True).
    3. Zero acute dimensions may be unresolved (None).
    """
    if status != "completed":
        return False, "FOLLOWUP_NOT_COMPLETED"

    for k, v in dimensions.items():
        if k != "timeframe" and v is True:
            return False, f"DIMENSION_AFFIRMED_{k.upper()}"

    for k in ("intent", "plan", "means", "access"):
        if dimensions.get(k) is None:  # type: ignore
            return False, f"UNRESOLVED_DIMENSION_{k.upper()}"

    return True, None


# ============================================================================
# LangGraph Conditional Edge Routers
# ============================================================================

def route_after_deterministic_safety(
    state: MindBridgeState,
) -> Literal["emergency_node", "followup_node", "classifier_node", "deescalation_node"]:
    """
    Conditional edge router following deterministic_safety_node execution.

    Precedence:
    1. Explicit Emergency -> emergency_node.
    2. Active/Interrupted Follow-Up or Preliminary Signal -> followup_node.
    3. Input Rejection / Unassessed Risk -> deescalation_node.
    4. Passthrough -> classifier_node.
    """
    if state.get("overall_safety_route") == "emergency_bypass" or state.get("deterministic_danger_flag") is True:
        return "emergency_node"

    followup_status = state.get("phq9_followup_status")
    if (
        state.get("overall_safety_route") == "pending_safety_follow_up"
        or followup_status in ("in_progress", "interrupted")
        or state.get("phq9_item9_endorsed") is True
    ):
        return "followup_node"

    if state.get("overall_safety_route") == "ml_risk_unassessed" or (
        state.get("error_code") == "INPUT_PARSING_FAILURE" and state.get("final_response")
    ):
        return "deescalation_node"

    return "classifier_node"


def route_after_classifier_decision(
    state: MindBridgeState,
) -> Literal["emergency_node", "followup_node", "deescalation_node", "planning_node"]:
    """
    Conditional edge router following classifier_decision_node execution.

    Precedence:
    1. Emergency Preservation -> emergency_node.
    2. Elevated Classifier Signal / Active Follow-Up -> followup_node (NEVER emergency_node).
    3. Classifier Failure (ml_risk_unassessed) -> deescalation_node.
    4. Standard Conversation -> planning_node.
    """
    if state.get("overall_safety_route") == "emergency_bypass":
        return "emergency_node"

    if state.get("overall_safety_route") in ("elevated_classifier_signal", "pending_safety_follow_up"):
        return "followup_node"

    if state.get("overall_safety_route") == "ml_risk_unassessed":
        return "deescalation_node"

    return "planning_node"


def route_after_followup(
    state: MindBridgeState,
) -> Literal["emergency_node", "deescalation_node", "planning_node", "__end__"]:
    """
    Conditional edge router following followup_node execution.
    """
    route = state.get("overall_safety_route")
    status = state.get("phq9_followup_status")

    if route == "emergency_bypass":
        return "emergency_node"
    if route == "supportive_deescalation":
        return "deescalation_node"
    if route == "standard_conversation" and status == "completed":
        return "planning_node"

    return "__end__"
