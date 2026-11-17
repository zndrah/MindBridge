"""
MindBridge 2.0 - Grounded Response Generation Node (`response_node.py`)
======================================================================

Synthesizes empathetic, non-clinical supportive responses strictly grounded
in retrieved, authorized knowledge chunks.

Authoritative Governance & Safety Requirements:
1. Grounded Synthesis: Ground responses strictly in authorized retrieved knowledge chunks.
2. Non-Diagnostic Framing: Strictly prohibit diagnostic labeling, clinical assertions,
   medical advice, and prescription dosing.
3. Cognitive Reframing Boundaries: Present cognitive reframing strictly as educational
   self-reflection worksheets, never as clinical CBT psychotherapy.
4. User-Initiated Emergency Guidance: Strictly prohibit phrasing implying automated
   system dialing, emergency responder dispatch, or active dispatch services.
5. Mandatory Clinical Disclaimer: Automatically append the mandatory non-diagnostic
   disclaimer whenever clinical screening rubrics (e.g. PHQ-9, GAD-7, PSS-10) are retrieved.
6. Safe Neutral Fallback: When retrieval returns empty, out-of-domain, or unassessed context,
   emit the vetted neutral supportive fallback, strictly prohibiting unconditional
   breathing exercises or unsolicited somatic prescriptions.
7. Universal Safety Precedence: Emergency bypass and active safety routes immediately
   halt open generation.

Authoritative References:
- docs/AGENTS.md §9 (Response Generation Node & Verification Gate)
- docs/ARCHITECTURE.md §3.1 & §5 (State Schema & RAG Knowledge Retrieval)
- docs/PRD.md §5.2 (NFR-2: Privacy-by-Design & Input Safety)
- docs/SAFETY_POLICY.md §4 (Non-Clinical Safety Boundaries)
"""

from datetime import datetime, timezone
import re
import time
from typing import Any, Callable, Dict, List, Optional

from backend.orchestrator.state import (
    ErrorCode,
    JurisdictionCode,
    KnowledgeChunk,
    MindBridgeState,
    TraceEntry,
)
from backend.safety.fallbacks import (
    MANDATORY_SCREENING_DISCLAIMER,
    NEUTRAL_SUPPORTIVE_FALLBACK,
    contains_prohibited_clinical_claims,
)
from backend.retrieval.counselchat_rag_design import COUNSELCHAT_DISCLAIMER


def _extract_protocol_steps_or_summary(content: str) -> str:
    """
    Extracts step-by-step instructions or key bullet points from a markdown knowledge chunk.
    Prioritizes Section 3 ('Step-by-Step Instructions'), Section 2 ('Summary'), or numbered lists.
    """
    if not content:
        return ""

    # Check for Section 3: Step-by-Step Instructions & Protocol Details
    step_match = re.search(
        r"##\s*3\.\s*Step-by-Step Instructions.*?\n(.*?)(?=\n##|\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if step_match:
        section_text = step_match.group(1).strip()
        lines = [line.strip() for line in section_text.splitlines() if line.strip()]
        # Filter out markdown blockquotes or notes inside step section
        step_lines = [l for l in lines if not l.startswith(">")]
        if step_lines:
            return "\n".join(step_lines)

    # Check for Section 2: Summary & Operational Objective
    summary_match = re.search(
        r"##\s*2\.\s*Summary.*?\n(.*?)(?=\n##|\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if summary_match:
        summary_text = summary_match.group(1).strip()
        lines = [line.strip() for line in summary_text.splitlines() if line.strip() and not line.startswith(">")]
        if lines:
            return "\n".join(lines)

    # Fallback: Find any numbered or bulleted lines in the document body
    list_lines = []
    for line in content.splitlines():
        trimmed = line.strip()
        if re.match(r"^(?:\d+[\.\)]|[-*•])\s+", trimmed):
            list_lines.append(trimmed)

    if list_lines:
        return "\n".join(list_lines[:8])

    # Ultimate fallback: Clean snippet
    cleaned = re.sub(r"---.*?---", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"#+.*?\n", "", cleaned)
    cleaned = re.sub(r">.*?\n", "", cleaned)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    return paragraphs[0] if paragraphs else ""


def _sanitize_against_clinical_claims(text: str) -> str:
    """
    Sanitizes any inadvertent diagnostic labels, medical assertions,
    or automated dispatch phrases to maintain strict non-clinical boundaries.
    """
    # Replace diagnostic assertions
    text = re.sub(
        r"\b(?:you\s+have|diagnosed\s+with|i\s+diagnose\s+you\s+with)\s+(?:major\s+depressive\s+disorder|depression|bipolar|panic\s+disorder|schizophrenia|ptsd|generalized\s+anxiety\s+disorder)\b",
        "you are experiencing these symptoms",
        text,
        flags=re.IGNORECASE,
    )
    # Remove prescription dosing and medication mentions
    text = re.sub(
        r"\b(?:take|prescribe|administer|dose|dosage\s+of)\s+\d+\s*(?:mg|milligrams|g|ml)(?:\s+of\s+[a-zA-Z\-]+)?\b",
        "discuss medication guidance with a licensed physician",
        text,
        flags=re.IGNORECASE,
    )
    # Scrub specific psychiatric medication names if mentioned
    text = re.sub(
        r"\b(?:ssris?|lexapro|prozac|zoloft|xanax|ativan|klonopin|valium|ambien|sertraline|fluoxetine|escitalopram)\b",
        "medication",
        text,
        flags=re.IGNORECASE,
    )
    # Prohibit automated dispatch claims
    text = re.sub(
        r"\b(?:we\s+have\s+dispatched|dispatching\s+emergency|calling\s+(?:911|112|emergency)\s+for\s+you|sending\s+an\s+ambulance)\b",
        "please contact local emergency services immediately",
        text,
        flags=re.IGNORECASE,
    )
    return text


def synthesize_grounded_response(
    sanitized_text: str,
    dialog_goal: Optional[str],
    retrieved_chunks: List[KnowledgeChunk],
    jurisdiction_filter: JurisdictionCode = "GL",
) -> str:
    """
    Deterministically synthesizes an empathetic, grounded supportive response
    using retrieved knowledge chunks under non-clinical boundaries.
    """
    if not retrieved_chunks:
        return NEUTRAL_SUPPORTIVE_FALLBACK

    primary_chunk = retrieved_chunks[0]
    category = primary_chunk.get("category", "")
    title = primary_chunk.get("title", "Supportive Exercise")
    extracted_steps = _extract_protocol_steps_or_summary(primary_chunk.get("content", ""))

    response_parts: List[str] = []

    # 1. Somatic Regulation (Grounding & Breathing)
    if category == "somatic_regulation" or dialog_goal == "somatic_grounding":
        response_parts.append(
            "I hear what you are experiencing right now. Let's take a pause together. "
            f"Here is a gentle, supportive grounding exercise—**{title}**—that you can explore at your own pace:"
        )
        if extracted_steps:
            response_parts.append(extracted_steps)
        response_parts.append(
            "Take this at whatever pace feels comfortable. You are in control of this exercise, "
            "and you can pause or stop at any time."
        )

    # 2. Cognitive Reframing (Self-Reflection Worksheets)
    elif category == "cognitive_reframing" or dialog_goal == "cognitive_reframing":
        response_parts.append(
            "I hear how heavy and overwhelming these thoughts feel. Here is an educational self-reflection "
            f"worksheet—**{title}**—designed to help examine catastrophic or racing thought patterns "
            "from a grounded perspective (note: this is a self-guided educational reflection tool, not clinical psychotherapy):"
        )
        if extracted_steps:
            response_parts.append(extracted_steps)
        response_parts.append(
            "You can write these reflection prompts down in a journal or notebook to work through them "
            "at your own pace."
        )

    # 3. Clinical Screening Rubrics (Informational Overview)
    elif category == "clinical_rubrics" or dialog_goal == "screening":
        response_parts.append(
            f"Here is an informational educational overview of the **{title}**:"
        )
        if extracted_steps:
            response_parts.append(extracted_steps)
        response_parts.append(
            "This instrument is designed to help individuals reflect on symptom patterns over time."
        )

    # 4. Peer Counseling Examples (CounselChat Community Perspectives)
    elif category == "peer_counseling_example" or "counselchat" in str(primary_chunk.get("chunk_id", "")).lower():
        peer_chunks = [
            c for c in retrieved_chunks
            if c.get("category") == "peer_counseling_example"
            or "counselchat" in str(c.get("chunk_id", "")).lower()
            or str(c.get("collection", "")).lower() == "counselchat_examples"
        ]

        if len(peer_chunks) <= 1:
            therapist_name = primary_chunk.get("therapist_name")
            therapist_info = f" shared by {therapist_name}" if therapist_name else ""
            response_parts.append(
                f"I hear what you are experiencing. Here is an illustrative peer counseling perspective{therapist_info} regarding **{title}** (note: this is community peer discussion for illustrative coping context, not clinical psychotherapy):"
            )
            if extracted_steps:
                response_parts.append(extracted_steps)
            response_parts.append(
                "Everyone's situation is unique, so consider whether these community perspectives feel helpful or relevant to you."
            )
        else:
            therapist_names = [c.get("therapist_name") for c in peer_chunks if c.get("therapist_name")]
            distinct_therapists: List[str] = []
            for name in therapist_names:
                if name and name not in distinct_therapists:
                    distinct_therapists.append(name)

            if len(distinct_therapists) > 1:
                therapist_summary = f" (including perspectives from {', '.join(distinct_therapists)})"
            elif distinct_therapists:
                therapist_summary = f" shared by {distinct_therapists[0]}"
            else:
                therapist_summary = ""

            response_parts.append(
                f"I hear what you are experiencing. Here are illustrative peer counseling perspectives{therapist_summary} regarding **{title}** (note: this is community peer discussion for illustrative coping context, not clinical psychotherapy):"
            )
            for idx, p_chunk in enumerate(peer_chunks[:3]):
                t_name = p_chunk.get("therapist_name") or f"Counselor {idx+1}"
                p_title = p_chunk.get("title") or title
                p_content = _extract_protocol_steps_or_summary(p_chunk.get("content", ""))
                response_parts.append(f"**Perspective {idx + 1} ({t_name} — {p_title}):**\n{p_content}")

            response_parts.append(
                "Everyone's situation is unique, so consider whether these community perspectives feel helpful or relevant to you."
            )

    # 5. Psychoeducation (Sleep, Exam Stress, Burnout)
    else:
        response_parts.append(
            f"I hear what you are experiencing. Here are some evidence-informed educational insights and practical strategies regarding **{title}**:"
        )
        if extracted_steps:
            response_parts.append(extracted_steps)
        response_parts.append(
            "Remember to be patient and kind with yourself as you navigate these adjustments."
        )

        # Check if complementary CounselChat peer chunks were also retrieved
        peer_chunks = [
            c for c in retrieved_chunks
            if c.get("category") == "peer_counseling_example"
            or "counselchat" in str(c.get("chunk_id", "")).lower()
            or str(c.get("collection", "")).lower() == "counselchat_examples"
        ]
        if peer_chunks:
            response_parts.append(
                "### Complementary Community Counseling Perspectives\n"
                "Here are practical peer perspectives shared by community counselors (note: unverified illustrative context, not clinical psychotherapy):"
            )
            for idx, p_chunk in enumerate(peer_chunks[:2]):
                t_name = p_chunk.get("therapist_name") or f"Counselor {idx+1}"
                p_title = p_chunk.get("title") or "Peer Example"
                p_content = _extract_protocol_steps_or_summary(p_chunk.get("content", ""))
                response_parts.append(f"**Perspective from {t_name} ({p_title}):**\n{p_content}")

    body = "\n\n".join(response_parts)

    # Check if mandatory screening disclaimer is required
    requires_disclaimer = any(
        c.get("category") == "clinical_rubrics"
        or c.get("eligibility") == "restricted"
        or "screening" in str(dialog_goal).lower()
        for c in retrieved_chunks
    )

    if requires_disclaimer:
        body = f"{body}\n\n{MANDATORY_SCREENING_DISCLAIMER}"

    # Check if CounselChat peer disclaimer is required
    has_peer_example = any(
        c.get("category") == "peer_counseling_example"
        or "counselchat" in str(c.get("chunk_id", "")).lower()
        or "counselchat" in str(c.get("document_id", "")).lower()
        or "counselchat" in str(c.get("source", "")).lower()
        or "counselchat" in str(c.get("source_type", "")).lower()
        or str(c.get("collection", "")).lower() == "counselchat_examples"
        for c in retrieved_chunks
    )
    if has_peer_example and COUNSELCHAT_DISCLAIMER not in body:
        body = f"{body}\n\n*Note: {COUNSELCHAT_DISCLAIMER}*"

    # Enforce non-diagnostic framing and dispatch phrasing sanitization
    sanitized_response = _sanitize_against_clinical_claims(body)

    # Final sanity check: If prohibited clinical assertions remain, fall back to safe template
    if contains_prohibited_clinical_claims(sanitized_response):
        return NEUTRAL_SUPPORTIVE_FALLBACK

    return sanitized_response


def response_node(
    state: MindBridgeState,
    generator: Optional[Callable[[MindBridgeState], str]] = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Grounded Natural Language Response Generation.

    Args:
        state: Immutable MindBridgeState passed from upstream nodes.
        generator: Optional custom LLM generator callable (for pluggable generation).

    Returns:
        State update dictionary containing draft_response, final_response,
        and strictly typed execution_trace updates.
    """
    start_time = time.perf_counter()

    # Invariant 1: Universal Emergency Bypass & Precedence
    # If acute crisis or deterministic danger is active, halt open generation immediately.
    overall_safety_route = state.get("overall_safety_route", "standard_conversation")
    if (
        overall_safety_route == "emergency_bypass"
        or state.get("deterministic_danger_flag") is True
    ):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace: TraceEntry = {
            "node": "response_node",
            "status": "skipped",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "EMERGENCY_BYPASS_SKIPPED",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "draft_response": None,
            "execution_trace": current_traces,
            "llm_telemetry": {
                "provider": "skipped",
                "model": "bypassed_by_safety",
                "fallback_triggered": False,
                "local_fallback_used": False,
                "duration_ms": 0.0,
                "success": True,
            },
        }

    # Invariant 2: Active Safety Follow-Up Preservation
    # If structured clinical follow-up or de-escalation is active, preserve safety state
    if overall_safety_route in ("pending_safety_follow_up", "supportive_deescalation"):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        trace = {
            "node": "response_node",
            "status": "skipped",
            "duration_ms": round(elapsed_ms, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_code": "SAFETY_STATE_PRESERVED",
        }
        current_traces = list(state.get("execution_trace", []))
        current_traces.append(trace)
        return {
            "draft_response": None,
            "execution_trace": current_traces,
            "llm_telemetry": {
                "provider": "skipped",
                "model": "safety_followup_active",
                "fallback_triggered": False,
                "local_fallback_used": False,
                "duration_ms": 0.0,
                "success": True,
            },
        }

    # Extract state fields
    sanitized_text = state.get("sanitized_text", "")
    dialog_goal = state.get("dialog_goal")
    retrieved_chunks = state.get("retrieved_chunks", [])
    retrieval_status = state.get("retrieval_status", "empty")
    from backend.config import settings
    jurisdiction_filter: JurisdictionCode = state.get("jurisdiction_filter") or getattr(settings, "default_jurisdiction", "IN") or "IN"

    draft_response: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    error_component: Optional[str] = None
    metadata_code = "GROUNDED_RESPONSE_SYNTHESIZED"
    status_tag = "success"
    llm_telemetry: Dict[str, Any] = {
        "provider": "local",
        "model": "deterministic_grounded_fallback",
        "fallback_triggered": False,
        "local_fallback_used": True,
        "duration_ms": 0.0,
        "success": True,
    }

    try:
        # Invariant 3: Safe Neutral Fallback on Empty Context / OOD
        # When no chunks are retrieved or retrieval status is empty/not_attempted,
        # emit vetted neutral supportive template (strictly NO unconditional breathing exercises).
        if retrieval_status in ("empty", "not_attempted") or not retrieved_chunks:
            draft_response = NEUTRAL_SUPPORTIVE_FALLBACK
            metadata_code = "NO_CONTEXT_SUPPORTIVE_FALLBACK"
            status_tag = "fallback"
            llm_telemetry = {
                "provider": "local",
                "model": "neutral_supportive_template",
                "fallback_triggered": False,
                "local_fallback_used": True,
                "duration_ms": 0.0,
                "success": True,
            }
        else:
            # Invariant 4: Grounded Synthesis
            if generator is not None:
                # Pluggable generator callable
                raw_candidate = generator(state)
                gen_result = getattr(generator, "last_result", None)

                if raw_candidate is None or (isinstance(raw_candidate, str) and not raw_candidate.strip()):
                    # Generator returned None or empty (controlled fallback to deterministic grounded synthesis)
                    draft_response = synthesize_grounded_response(
                        sanitized_text=sanitized_text,
                        dialog_goal=dialog_goal,
                        retrieved_chunks=retrieved_chunks,
                        jurisdiction_filter=jurisdiction_filter,
                    )
                    metadata_code = "LOCAL_GROUNDED_FALLBACK_ON_LLM_FAILURE"
                    status_tag = "fallback"
                    llm_telemetry = {
                        "provider": "local",
                        "model": "deterministic_grounded_fallback",
                        "fallback_triggered": True,
                        "local_fallback_used": True,
                        "duration_ms": getattr(gen_result, "duration_ms", 0.0) if gen_result else 0.0,
                        "success": True,
                        "error_code": getattr(gen_result, "error_code", None) if gen_result else "NO_GENERATOR",
                    }
                elif contains_prohibited_clinical_claims(raw_candidate):
                    draft_response = NEUTRAL_SUPPORTIVE_FALLBACK
                    metadata_code = "PROHIBITED_CLAIM_SUPPRESSED"
                    status_tag = "fallback"
                    llm_telemetry = {
                        "provider": getattr(gen_result, "provider", "gemini") if gen_result else "gemini",
                        "model": getattr(gen_result, "model", "llm"),
                        "fallback_triggered": getattr(gen_result, "fallback_triggered", False) if gen_result else False,
                        "local_fallback_used": True,
                        "duration_ms": getattr(gen_result, "duration_ms", 0.0) if gen_result else 0.0,
                        "success": False,
                        "error_code": "PROHIBITED_CLINICAL_CLAIM",
                    }
                else:
                    candidate = _sanitize_against_clinical_claims(raw_candidate)
                    draft_response = candidate
                    llm_telemetry = {
                        "provider": getattr(gen_result, "provider", "gemini") if gen_result else "gemini",
                        "model": getattr(gen_result, "model", "gemini-2.5-flash") if gen_result else "gemini-2.5-flash",
                        "fallback_triggered": getattr(gen_result, "fallback_triggered", False) if gen_result else False,
                        "local_fallback_used": False,
                        "duration_ms": getattr(gen_result, "duration_ms", 0.0) if gen_result else round((time.perf_counter() - start_time) * 1000, 2),
                        "success": True,
                    }
                    # Ensure screening disclaimer is appended if needed
                    if any(
                        c.get("category") == "clinical_rubrics" or c.get("eligibility") == "restricted"
                        for c in retrieved_chunks
                    ) and MANDATORY_SCREENING_DISCLAIMER not in draft_response:
                        draft_response = f"{draft_response}\n\n{MANDATORY_SCREENING_DISCLAIMER}"
                    # Ensure CounselChat peer disclaimer is appended if peer counseling content was used
                    has_counselchat_used = any(
                        c.get("category") == "peer_counseling_example"
                        or "counselchat" in str(c.get("chunk_id", "")).lower()
                        or "counselchat" in str(c.get("document_id", "")).lower()
                        or str(c.get("chunk_id", "")).startswith("cc_")
                        or str(c.get("collection", "")).lower() == "counselchat_examples"
                        for c in retrieved_chunks
                    )
                    if has_counselchat_used and COUNSELCHAT_DISCLAIMER not in draft_response:
                        draft_response = f"{draft_response}\n\n*Note: {COUNSELCHAT_DISCLAIMER}*"
            else:
                # Built-in modular deterministic grounded generator
                draft_response = synthesize_grounded_response(
                    sanitized_text=sanitized_text,
                    dialog_goal=dialog_goal,
                    retrieved_chunks=retrieved_chunks,
                    jurisdiction_filter=jurisdiction_filter,
                )
                llm_telemetry = {
                    "provider": "local",
                    "model": "deterministic_grounded_fallback",
                    "fallback_triggered": False,
                    "local_fallback_used": True,
                    "duration_ms": 0.0,
                    "success": True,
                }

    except Exception:
        # Failure containment: Log controlled error code, zero raw exception strings in state
        draft_response = NEUTRAL_SUPPORTIVE_FALLBACK
        error_code = "LLM_GENERATION_FAILURE"
        error_component = "response_node"
        metadata_code = "RESPONSE_GENERATION_FAILURE_FALLBACK"
        status_tag = "error"
        llm_telemetry = {
            "provider": "local",
            "model": "error_fallback",
            "fallback_triggered": True,
            "local_fallback_used": True,
            "duration_ms": 0.0,
            "success": False,
            "error_code": "LLM_GENERATION_FAILURE",
        }

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    trace = {
        "node": "response_node",
        "status": status_tag,
        "duration_ms": round(elapsed_ms, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_code": metadata_code,
    }
    current_traces = list(state.get("execution_trace", []))
    current_traces.append(trace)

    result_updates: Dict[str, Any] = {
        "draft_response": draft_response,
        "final_response": draft_response or state.get("final_response", ""),
        "execution_trace": current_traces,
        "llm_telemetry": llm_telemetry,
    }

    if error_code is not None:
        result_updates["error_code"] = error_code
        result_updates["error_component"] = error_component

    return result_updates
