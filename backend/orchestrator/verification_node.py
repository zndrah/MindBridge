"""
MindBridge 2.0 - Deterministic Response Verification Node (`verification_node.py`)
==================================================================================

Audits every generated response before it is exposed to the user as the final safety gate.
Does NOT independently diagnose users or make clinical risk assessments.

Mandatory Verification Dimensions (docs/AGENTS.md §10 & docs/SAFETY_POLICY.md):
1. Safety Policy Verification:
   - Prohibits diagnostic claims and clinical labeling.
   - Prohibits medication prescriptions and dosage instructions.
   - Prohibits fabricated/unverified telephone numbers or hotlines (validated against hotlines.json).
   - Prohibits false claims of emergency responder dispatch.
   - Prohibits unsupported guarantees of safety.
   - Prohibits impersonation of licensed doctors, therapists, or healthcare professionals.
   - Prohibits unsolicited or unconditional somatic commands.
2. Grounding & Citation Integrity:
   - Verifies that factual psychoeducational claims are supported by authorized retrieved chunks.
   - Validates all citation tags against actual retrieved chunks.
   - Rejects citations to nonexistent or fabricated chunk identifiers.
   - Prohibits restricted or excluded content without explicit trusted authorization.
   - Rejects excluded content under all circumstances.
   - Enforces that presence of a citation alone does not prove factual support.
3. Screening Safeguards:
   - Ensures clinical screening rubrics (PHQ-9, GAD-7, PSS-10) retain the mandatory non-diagnostic disclaimer.
   - Prohibits interpreting screening scores as confirmed clinical diagnoses.
   - Prohibits presenting screening tools as replacements for professional healthcare evaluations.
4. Response Integrity & Safety Invariants:
   - Strictly preserves original sanitized user utterances and conversation history.
   - Preserves classifier outputs, safety routes, and PHQ-9 follow-up assessment progress.
   - Halts open verification on emergency bypass and preserves authoritative hotline guidance.
   - Replaces failing drafts with vetted neutral supportive fallbacks without exposing invalid text.
   - Never silently marks fallbacks as grounded in knowledge base chunks.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.orchestrator.fallbacks import (
    MANDATORY_SCREENING_DISCLAIMER,
    NEUTRAL_SUPPORTIVE_FALLBACK,
)
from backend.orchestrator.safety_node import load_hotlines_directory
from backend.orchestrator.state import (
    ErrorCode,
    JurisdictionCode,
    KnowledgeChunk,
    MindBridgeState,
    TraceEntry,
    VerificationStatus,
)
from backend.retrieval import RetrievalAuthorization

# Cached set of verified hotline numbers extracted from hotlines.json
_VERIFIED_HOTLINE_DIGITS: Optional[Set[str]] = None


# ============================================================================
# 1. Authoritative Hotline Whitelist Compilation
# ============================================================================

def get_verified_hotline_digits() -> Set[str]:
    """
    Builds and caches a set of normalized phone number digit strings
    from backend/safety/hotlines.json to prevent fabricated contact leakage.
    """
    global _VERIFIED_HOTLINE_DIGITS
    if _VERIFIED_HOTLINE_DIGITS is not None:
        return _VERIFIED_HOTLINE_DIGITS

    data = load_hotlines_directory()
    digits_set: Set[str] = set()

    # Common international emergency dispatch codes
    for standard_code in ("112", "911", "999", "000", "102", "111", "988", "14416", "741741", "85258", "686868"):
        digits_set.add(standard_code)

    jurisdictions = data.get("jurisdictions", {})
    for jur_code, jur_data in jurisdictions.items():
        if not isinstance(jur_data, dict):
            continue
        for cat, items in jur_data.items():
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for key in ("telephone", "alternate_telephone"):
                        val = item.get(key)
                        if val and isinstance(val, str):
                            digits = re.sub(r"\D", "", val)
                            if digits:
                                digits_set.add(digits)
                                # Normalize country codes
                                if digits.startswith("91") and len(digits) > 10:
                                    digits_set.add(digits[2:])
                                if digits.startswith("1") and len(digits) == 11:
                                    digits_set.add(digits[1:])
                                if digits.startswith("44") and len(digits) > 10:
                                    digits_set.add(digits[2:])

    _VERIFIED_HOTLINE_DIGITS = digits_set
    return _VERIFIED_HOTLINE_DIGITS


# ============================================================================
# 2. Prohibited Clinical & Policy Regex Lexicons
# ============================================================================

# Diagnostic claims
DIAGNOSTIC_CLAIM_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:you\s+have|diagnosed\s+with|i\s+diagnose\s+you\s+with|you\s+suffer\s+from|my\s+diagnosis\s+is)\s+"
        r"(?:major\s+depressive\s+disorder|depression|bipolar|panic\s+disorder|schizophrenia|ptsd|"
        r"generalized\s+anxiety\s+disorder|clinical\s+insomnia|borderline|adhd|ocd)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:a\s+diagnosis\s+of\s+(?:depression|panic|anxiety|bipolar|schizophrenia))\b", re.IGNORECASE),
)

# Medication prescriptions or dosage instructions
MEDICATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(?:take|prescribe|administer|dose|dosage\s+of)\s+\d+\s*(?:mg|milligrams|g|ml)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:you\s+should\s+take|start\s+taking|prescribe\s+you)\s+"
        r"(?:ssri|lexapro|prozac|zoloft|xanax|ativan|klonopin|valium|ambien|sertraline|fluoxetine|escitalopram)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+\s*(?:mg|milligrams)\s+of\s+[a-zA-Z]+\b", re.IGNORECASE),
)

# False emergency responder dispatch claims
FALSE_DISPATCH_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:we\s+have\s+dispatched|dispatching\s+emergency|calling\s+(?:911|112|emergency|police|ambulance)\s+for\s+you|"
        r"sending\s+an\s+ambulance|help\s+is\s+on\s+the\s+way|contacting\s+authorities\s+for\s+you)\b",
        re.IGNORECASE,
    ),
)

# Unsupported safety guarantees
SAFETY_GUARANTEE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:you\s+are\s+(?:completely|totally|100%|guaranteed\s+to\s+be)\s+safe|"
        r"i\s+guarantee\s+(?:your\s+safety|you\s+will\s+be\s+fine)|nothing\s+bad\s+can\s+happen|"
        r"everything\s+will\s+definitely\s+be\s+fine|guaranteed\s+cure)\b",
        re.IGNORECASE,
    ),
)

# Impersonation of medical/clinical professionals
IMPERSONATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:as\s+your\s+(?:therapist|psychologist|psychiatrist|doctor|physician|clinician|counselor)|"
        r"i\s+am\s+your\s+(?:therapist|doctor|clinician)|in\s+my\s+clinical\s+practice|my\s+medical\s+opinion)\b",
        re.IGNORECASE,
    ),
)

# Unsolicited or unconditional somatic commands
UNSOLICITED_SOMATIC_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:take\s+a\s+deep\s+breath|inhale\s+for\s+\d+|exhale\s+for\s+\d+|hold\s+your\s+breath)\b",
        re.IGNORECASE,
    ),
)

# Screening score interpreted as confirmed diagnosis
SCREENING_DIAGNOSIS_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:score\s+(?:of\s+\d+\s+)?(?:means|confirms|proves)\s+you\s+have|"
        r"score\s+(?:diagnoses|identifies)\s+you\s+with)\b",
        re.IGNORECASE,
    ),
)

# Screening tool as substitute for professional evaluation
SCREENING_SUBSTITUTE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:replaces?\s+(?:a\s+)?(?:doctor|therapist|clinical\s+evaluation|professional\s+assessment)|"
        r"instead\s+of\s+seeing\s+a\s+(?:doctor|therapist)|no\s+need\s+to\s+see\s+a\s+doctor)\b",
        re.IGNORECASE,
    ),
)


# ============================================================================
# 3. Deterministic Verification Helper Functions
# ============================================================================

def scan_for_fabricated_contacts(text: str) -> List[str]:
    """
    Scans draft text for telephone numbers and crisis codes, filtering out
    common numeric false positives (years, scores, durations, counts).
    Returns list of unrecognized/fabricated contact numbers.
    """
    if not text:
        return []

    verified_digits = get_verified_hotline_digits()

    # Mask known numeric false positives
    t = text
    # 1. Breathing / grounding counts: 4-4-4-4, 5-4-3-2-1, 4-7-8
    t = re.sub(r"\b4-4-4-4\b", "MASKED_COUNT", t)
    t = re.sub(r"\b5-4-3-2-1\b", "MASKED_COUNT", t)
    t = re.sub(r"\b4-7-8\b", "MASKED_COUNT", t)
    # 2. 24/7 availability tag
    t = re.sub(r"\b24/7\b", "MASKED_TIME", t)
    # 3. Score ranges: 0-27, 0–4, 5–9, 10–14, 15–19, 20–27
    t = re.sub(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b", "MASKED_RANGE", t)
    # 4. Calendar years: 19xx, 20xx
    t = re.sub(r"\b(?:19|20)\d{2}\b", "MASKED_YEAR", t)
    # 5. Times / Durations: 20 minutes, 4 seconds, etc.
    t = re.sub(r"\b\d+\s*(?:minutes?|seconds?|hours?|days?|cycles?)\b", "MASKED_DUR", t, flags=re.IGNORECASE)

    candidates: List[str] = []

    # Match explicit call/dial/text commands with numbers
    for m in re.finditer(r"\b(?:call|dial|text|hotline|helpline)\s+(?:at\s+|to\s+)?(\+?[\d\s\-()]{3,20})\b", t, re.IGNORECASE):
        candidates.append(m.group(1))

    # Match standalone phone numbers and short codes
    for m in re.finditer(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", t):
        candidates.append(m.group(0))

    unrecognized: List[str] = []
    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        # Check if digits meet phone number length threshold
        if len(digits) >= 3 and digits not in verified_digits:
            unrecognized.append(cand.strip())

    return unrecognized


def extract_response_citations(text: str) -> List[str]:
    """
    Extracts citation references from text (e.g. [somatic_regulation:01...], [citation: ...]).
    Filters out Markdown alerts and hyperlinks.
    """
    if not text:
        return []

    # Strip markdown alerts like [!CRITICAL] or [!IMPORTANT]
    t = re.sub(r"\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION|CRITICAL)\]", "", text)
    # Strip markdown links [label](url)
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", "", t)

    citations: List[str] = []
    for m in re.finditer(r"\[(?:citation|source|doc|chunk)?\s*:?\s*([a-zA-Z0-9_\-:\s]+)\]", t, re.IGNORECASE):
        tag = m.group(1).strip()
        # Require citation characteristics (contains colon, underscore, or explicit citation keyword)
        if any(c in tag for c in (":", "_")) or "citation" in m.group(0).lower() or "source" in m.group(0).lower():
            citations.append(tag)

    return citations


def check_factual_grounding(
    draft_response: str,
    retrieved_chunks: List[KnowledgeChunk],
    dialog_goal: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Deterministically evaluates whether the draft response is factually supported
    by the retrieved knowledge chunks.
    
    Returns:
        (is_grounded, failure_reason)
    """
    if not draft_response:
        return False, "EMPTY_DRAFT"

    # Neutral supportive fallback is explicitly non-grounded safe fallback
    if draft_response.strip() == NEUTRAL_SUPPORTIVE_FALLBACK.strip():
        return True, "SAFE_FALLBACK"

    # If no chunks were retrieved, substantive factual claims are ungrounded
    if not retrieved_chunks:
        return False, "UNGROUNDED_EMPTY_CHUNKS"

    # Strip citation tags before checking text grounding so mere citation presence doesn't fake grounding
    clean_draft = re.sub(r"\[(?:citation|source|doc|chunk)?\s*:?\s*[a-zA-Z0-9_\-:\s]+\]", "", draft_response)
    draft_lower = clean_draft.lower()

    # Build corpus of chunk words and titles
    all_chunk_words: Set[str] = set()
    title_matched = False

    for chunk in retrieved_chunks:
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        title_lower = title.lower()
        title_core_words = [w for w in re.findall(r"\b[a-z]{3,}\b", title_lower) if w not in {"the", "and", "for", "with"}]
        if title_lower and title_lower in draft_lower:
            title_matched = True
        elif title_core_words and sum(1 for w in title_core_words if w in draft_lower) >= max(1, len(title_core_words) // 2):
            title_matched = True

        words = re.findall(r"\b[a-z]{3,}\b", f"{title} {content}".lower())
        all_chunk_words.update(words)

    # Extract substantive words from draft (excluding general conversational greeting words)
    stopwords = {
        "the", "and", "that", "this", "with", "for", "you", "your", "are", "here",
        "hear", "what", "ready", "support", "information", "resources", "whenever",
        "take", "gentle", "step", "pace", "control", "pause", "stop", "time", "feel",
        "thoughts", "write", "down", "journal", "notebook", "calmly", "speed", "patient",
        "kind", "remember", "help", "from", "can", "also", "some", "overview", "reflect",
        "instrument", "designed", "individuals", "over", "patterns", "symptoms"
    }
    substantive_draft_words = [w for w in re.findall(r"\b[a-z]{3,}\b", draft_lower) if w not in stopwords]

    if not substantive_draft_words:
        return True, "MINIMAL_SUBSTANTIVE_CLAIMS"

    matched_substantive = {w for w in substantive_draft_words if w in all_chunk_words}

    # Grounding requires either title alignment with at least 1 protocol term, OR at least 3 distinct matching substantive tokens
    if title_matched and len(matched_substantive) >= 1:
        return True, "GROUNDED_TITLE_AND_TERMS"

    if len(matched_substantive) >= 3:
        return True, f"GROUNDED_MATCHED_TERMS_{len(matched_substantive)}"

    return False, f"INSUFFICIENT_CHUNK_OVERLAP_{len(matched_substantive)}"


# ============================================================================
# 4. Master Verification Node
# ============================================================================

def verification_node(
    state: MindBridgeState,
    auto_correct_disclaimer: bool = False,
) -> Dict[str, Any]:
    """
    LangGraph Node: Deterministic Response Verification & Post-Generation Safety Gate.

    Args:
        state: Immutable MindBridgeState container passed from response_node.
        auto_correct_disclaimer: If True, appends missing screening disclaimers
                                 instead of rejecting the draft.

    Returns:
        State update dictionary with verification_passed, verification_status,
        verification_flags, final_response, and execution_trace updates.
    """
    start_time = time.perf_counter()

    # Invariant 1: Universal Emergency Bypass & Precedence
    # Emergency bypass halts open verification; preserves authoritative hotline guidance
    overall_safety_route = state.get("overall_safety_route", "standard_conversation")
    if (
        overall_safety_route == "emergency_bypass"
        or state.get("deterministic_danger_flag") is True
    ):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace: TraceEntry = {
            "node": "verification_node",
            "status": "skipped",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "EMERGENCY_BYPASS_SKIPPED",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "verification_passed": True,
            "verification_status": "not_run",
            "verification_flags": [],
            "execution_trace": current_traces,
        }

    # Invariant 2: Active Multi-Turn Follow-Up & De-escalation Preservation
    if overall_safety_route in ("pending_safety_follow_up", "supportive_deescalation"):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace = {
            "node": "verification_node",
            "status": "skipped",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "SAFETY_STATE_PRESERVED",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "verification_passed": True,
            "verification_status": "not_run",
            "verification_flags": [],
            "execution_trace": current_traces,
        }

    # Extract state fields
    draft_response = state.get("draft_response") or ""
    sanitized_text = state.get("sanitized_text", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    dialog_goal = state.get("dialog_goal")
    retrieval_status = state.get("retrieval_status", "empty")
    trusted_auth = state.get("retrieval_authorization")

    verification_flags: List[str] = []
    verification_passed = True
    verification_status: VerificationStatus = "verified"
    grounding_status: str = "grounded"
    error_code: Optional[ErrorCode] = None
    error_component: Optional[str] = None
    final_response = draft_response

    try:
        # Check 1: Empty draft handling
        if not draft_response.strip():
            verification_flags.append("EMPTY_DRAFT_RESPONSE")
            verification_passed = False
            verification_status = "fallback_used"
            final_response = NEUTRAL_SUPPORTIVE_FALLBACK
            grounding_status = "fallback"

        # Check 2: Pre-approved neutral fallback check
        elif draft_response.strip() == NEUTRAL_SUPPORTIVE_FALLBACK.strip():
            # If draft is already the vetted neutral supportive fallback, verify it safely
            verification_passed = True
            verification_status = "fallback_used"
            final_response = NEUTRAL_SUPPORTIVE_FALLBACK
            grounding_status = "fallback"

        else:
            # ----------------------------------------------------------------
            # Check 3: Safety Policy - Prohibited Diagnostic Phrasing
            # ----------------------------------------------------------------
            for pattern in DIAGNOSTIC_CLAIM_PATTERNS:
                if pattern.search(draft_response):
                    verification_flags.append("DIAGNOSTIC_CLAIM_DETECTED")
                    verification_passed = False
                    verification_status = "rejected_unsafe"
                    break

            # ----------------------------------------------------------------
            # Check 4: Safety Policy - Medication & Dosage Instructions
            # ----------------------------------------------------------------
            for pattern in MEDICATION_PATTERNS:
                if pattern.search(draft_response):
                    verification_flags.append("MEDICATION_PRESCRIPTION_DETECTED")
                    verification_passed = False
                    verification_status = "rejected_unsafe"
                    break

            # ----------------------------------------------------------------
            # Check 5: Safety Policy - Fabricated Emergency Hotlines
            # ----------------------------------------------------------------
            unrecognized_contacts = scan_for_fabricated_contacts(draft_response)
            if unrecognized_contacts:
                verification_flags.append("FABRICATED_HOTLINE_DETECTED")
                verification_passed = False
                verification_status = "rejected_unsafe"

            # ----------------------------------------------------------------
            # Check 6: Safety Policy - False Responder Dispatch Claims
            # ----------------------------------------------------------------
            for pattern in FALSE_DISPATCH_PATTERNS:
                if pattern.search(draft_response):
                    verification_flags.append("FALSE_DISPATCH_CLAIM_DETECTED")
                    verification_passed = False
                    verification_status = "rejected_unsafe"
                    break

            # ----------------------------------------------------------------
            # Check 7: Safety Policy - Unsupported Safety Guarantees
            # ----------------------------------------------------------------
            for pattern in SAFETY_GUARANTEE_PATTERNS:
                if pattern.search(draft_response):
                    verification_flags.append("UNSUPPORTED_SAFETY_GUARANTEE_DETECTED")
                    verification_passed = False
                    verification_status = "rejected_unsafe"
                    break

            # ----------------------------------------------------------------
            # Check 8: Safety Policy - Clinician Impersonation
            # ----------------------------------------------------------------
            for pattern in IMPERSONATION_PATTERNS:
                if pattern.search(draft_response):
                    verification_flags.append("IMPERSONATION_OF_MEDICAL_PROFESSIONAL")
                    verification_passed = False
                    verification_status = "rejected_unsafe"
                    break

            # ----------------------------------------------------------------
            # Check 9: Safety Policy - Unsolicited Somatic Commands
            # ----------------------------------------------------------------
            has_somatic_chunk = any(c.get("category") == "somatic_regulation" for c in retrieved_chunks)
            if not has_somatic_chunk and dialog_goal != "somatic_grounding":
                for pattern in UNSOLICITED_SOMATIC_PATTERNS:
                    if pattern.search(draft_response):
                        verification_flags.append("UNSOLICITED_SOMATIC_INSTRUCTION")
                        verification_passed = False
                        verification_status = "rejected_unsafe"
                        break

            # ----------------------------------------------------------------
            # Check 10: Citation Integrity & Fabricated Citation Validation
            # ----------------------------------------------------------------
            citations = extract_response_citations(draft_response)
            if citations:
                chunk_ids = {c.get("chunk_id", "").lower() for c in retrieved_chunks}
                doc_ids = {c.get("document_id", "").lower() for c in retrieved_chunks}
                titles = {c.get("title", "").lower() for c in retrieved_chunks}

                for cit in citations:
                    cit_norm = cit.lower()
                    matched = (
                        cit_norm in chunk_ids
                        or cit_norm in doc_ids
                        or cit_norm in titles
                        or any(cit_norm in cid for cid in chunk_ids)
                        or any(cit_norm in did for did in doc_ids)
                    )
                    if not matched:
                        verification_flags.append(f"INVALID_CITATION_DETECTED:{cit}")
                        verification_passed = False
                        if verification_status != "rejected_unsafe":
                            verification_status = "rejected_ungrounded"

            # ----------------------------------------------------------------
            # Check 11: Restricted & Excluded Knowledge Governance
            # ----------------------------------------------------------------
            for chunk in retrieved_chunks:
                eligibility = chunk.get("eligibility", "approved")
                if eligibility == "excluded" or "excluded" in chunk.get("document_id", "").lower():
                    verification_flags.append("EXCLUDED_CONTENT_USED")
                    verification_passed = False
                    verification_status = "rejected_unsafe"
                    break

                if eligibility == "restricted":
                    is_auth = (
                        isinstance(trusted_auth, RetrievalAuthorization)
                        and bool(trusted_auth.authorized_by)
                    )
                    if not is_auth:
                        verification_flags.append("UNAUTHORIZED_RESTRICTED_CONTENT_USED")
                        verification_passed = False
                        verification_status = "rejected_unsafe"
                        break

            # ----------------------------------------------------------------
            # Check 12: Factual Grounding Evaluation
            # ----------------------------------------------------------------
            is_grounded, grounding_detail = check_factual_grounding(
                draft_response=draft_response,
                retrieved_chunks=retrieved_chunks,
                dialog_goal=dialog_goal,
            )
            if not is_grounded:
                verification_flags.append(f"UNGROUNDED_FACTUAL_CLAIM:{grounding_detail}")
                verification_passed = False
                if verification_status != "rejected_unsafe":
                    verification_status = "rejected_ungrounded"
                grounding_status = "unsupported"

            # ----------------------------------------------------------------
            # Check 13: Screening Safeguards (Disclaimer & Diagnostic Score Claims)
            # ----------------------------------------------------------------
            is_screening = (
                dialog_goal == "screening"
                or any(c.get("category") == "clinical_rubrics" for c in retrieved_chunks)
                or any(k in draft_response.lower() for k in ("phq-9", "gad-7", "pss-10", "screening rubric"))
            )

            if is_screening:
                # Prohibit screening scores as confirmed diagnosis
                for pattern in SCREENING_DIAGNOSIS_PATTERNS:
                    if pattern.search(draft_response):
                        verification_flags.append("SCREENING_SCORE_DIAGNOSIS_CLAIM")
                        verification_passed = False
                        verification_status = "rejected_unsafe"
                        break

                # Prohibit screening tools as substitute for professional evaluation
                for pattern in SCREENING_SUBSTITUTE_PATTERNS:
                    if pattern.search(draft_response):
                        verification_flags.append("SUBSTITUTE_FOR_PROFESSIONAL_ASSESSMENT")
                        verification_passed = False
                        verification_status = "rejected_unsafe"
                        break

                # Mandatory non-diagnostic disclaimer verification
                if MANDATORY_SCREENING_DISCLAIMER not in draft_response:
                    if auto_correct_disclaimer and verification_passed:
                        final_response = f"{draft_response}\n\n{MANDATORY_SCREENING_DISCLAIMER}"
                        verification_flags.append("DISCLAIMER_SAFE_CORRECTION_APPLIED")
                    else:
                        verification_flags.append("MISSING_SCREENING_DISCLAIMER")
                        verification_passed = False
                        verification_status = "rejected_unsafe"

        # Final Outcome Resolution
        if not verification_passed:
            # Never expose the rejected draft to the user!
            final_response = NEUTRAL_SUPPORTIVE_FALLBACK
            grounding_status = "unsupported"
            error_code = "VERIFICATION_VIOLATION"
            error_component = "verification_node"

    except Exception:
        # Failure containment: Log controlled error code, zero raw exception strings in state
        verification_passed = False
        verification_status = "verification_error"
        verification_flags = ["INTERNAL_VERIFICATION_EXCEPTION"]
        final_response = NEUTRAL_SUPPORTIVE_FALLBACK
        grounding_status = "unsupported"
        error_code = "VERIFICATION_FAILURE"
        error_component = "verification_node"

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status_code = "success" if verification_passed else ("error" if verification_status == "verification_error" else "fallback")
    metadata_code = f"VERIFICATION_{verification_status.upper()}"

    trace = {
        "node": "verification_node",
        "status": status_code,
        "duration_ms": round(elapsed_ms, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_code": metadata_code,
    }
    current_traces = list(state.get("execution_trace", []))
    current_traces.append(trace)

    result_updates: Dict[str, Any] = {
        "verification_passed": verification_passed,
        "verification_status": verification_status,
        "verification_flags": verification_flags,
        "final_response": final_response,
        "grounding_status": grounding_status,
        "execution_trace": current_traces,
    }

    if error_code is not None:
        result_updates["error_code"] = error_code
        result_updates["error_component"] = error_component

    return result_updates
