"""
Unit & Integration Tests for MindBridge 2.0 Response Verification Node
=====================================================================

Comprehensive test suite verifying all 18 mandatory post-generation safety gate requirements:
1. Valid grounded response accepted.
2. Unsupported factual claim rejected.
3. Fabricated citation rejected.
4. Citation to a nonexistent chunk rejected.
5. Restricted-content citation rejected without authorization.
6. Excluded content never accepted under any conditions.
7. Diagnostic claim rejected.
8. Medication dosage advice rejected.
9. Fabricated hotline number rejected.
10. False emergency dispatch claim rejected.
11. Missing screening disclaimer causes rejection or safe correction.
12. Empty retrieval produces an explicitly ungrounded safe fallback.
13. Verification exception fails safely.
14. Original sanitized message remains unchanged.
15. Emergency bypass skips response generation and verification.
16. Existing safety decisions and classifier outputs remain unchanged.
17. Jurisdiction metadata is preserved.
18. Complete graph integration works with verification gate active.
"""

from datetime import datetime, timezone
import unittest
from typing import Any, Dict, List, Optional

from backend.orchestrator.fallbacks import (
    MANDATORY_SCREENING_DISCLAIMER,
    NEUTRAL_SUPPORTIVE_FALLBACK,
)
from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.state import (
    KnowledgeChunk,
    MindBridgeState,
    TraceEntry,
)
from backend.orchestrator.verification_node import (
    extract_response_citations,
    scan_for_fabricated_contacts,
    verification_node,
)
from backend.retrieval import RetrievalAuthorization


def create_verification_test_state(
    sanitized_text: str = "",
    draft_response: str = "",
    dialog_goal: str = "general",
    retrieved_chunks: Optional[List[KnowledgeChunk]] = None,
    retrieval_status: str = "success",
    overall_safety_route: str = "standard_conversation",
    deterministic_danger_flag: bool = False,
    jurisdiction_filter: str = "GL",
    retrieval_authorization: Optional[RetrievalAuthorization] = None,
    roberta_prob: Optional[float] = 0.05,
    roberta_crisis_flag: bool = False,
    phq9_item9_endorsed: bool = False,
    phq9_followup_status: str = "not_started",
    phq9_risk_level: Optional[str] = "none",
) -> MindBridgeState:
    """Helper constructing conforming MindBridgeState for testing verification_node."""
    chunks = retrieved_chunks if retrieved_chunks is not None else []
    return {
        "session_id": "test-verif-session-1234",
        "user_id_hash": "test-verif-user-hash",
        "messages": [{"role": "user", "content": sanitized_text}] if sanitized_text else [],
        "sanitized_text": sanitized_text,
        "deterministic_danger_flag": deterministic_danger_flag,
        "emergency_trigger_id": None,
        "roberta_prob": roberta_prob,
        "roberta_crisis_flag": roberta_crisis_flag,
        "classifier_status": "completed",
        "overall_safety_route": overall_safety_route,  # type: ignore
        "phq9_item9_endorsed": phq9_item9_endorsed,
        "phq9_followup_status": phq9_followup_status,  # type: ignore
        "phq9_risk_level": phq9_risk_level,  # type: ignore
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": dialog_goal,  # type: ignore
        "retrieval_queries": ["test search query"],
        "retrieved_chunks": chunks,
        "retrieval_status": retrieval_status,  # type: ignore
        "jurisdiction_filter": jurisdiction_filter,  # type: ignore
        "retrieval_authorization": retrieval_authorization,
        "draft_response": draft_response,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class TestVerificationNode(unittest.TestCase):
    """Test suite for post-generation deterministic response verification."""

    # ------------------------------------------------------------------------
    # Test 1: Valid grounded response accepted
    # ------------------------------------------------------------------------
    def test_01_valid_grounded_response_accepted(self):
        """Valid grounded response is accepted and passed into final_response."""
        chunk: KnowledgeChunk = {
            "chunk_id": "somatic:box_breathing",
            "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
            "title": "Box Breathing",
            "category": "somatic_regulation",
            "jurisdiction": "GL",
            "content": "Inhale 4, hold 4, exhale 4, hold 4.",
            "score": 0.88,
            "eligibility": "approved",
        }
        draft = (
            "I hear what you are experiencing right now. Let's take a pause together. "
            "Here is a gentle, supportive grounding exercise—**Box Breathing**—that you can explore at your own pace:\n\n"
            "Inhale 4, hold 4, exhale 4, hold 4.\n\n"
            "Take this at whatever pace feels comfortable. You are in control of this exercise, and you can pause or stop at any time."
        )
        state = create_verification_test_state(
            sanitized_text="I feel acute panic and need help breathing",
            draft_response=draft,
            dialog_goal="somatic_grounding",
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertTrue(res["verification_passed"])
        self.assertEqual(res["verification_status"], "verified")
        self.assertEqual(res["verification_flags"], [])
        self.assertEqual(res["final_response"], draft)
        self.assertEqual(res["grounding_status"], "grounded")
        self.assertEqual(res["execution_trace"][-1]["status"], "success")

    # ------------------------------------------------------------------------
    # Test 2: Unsupported factual claim rejected
    # ------------------------------------------------------------------------
    def test_02_unsupported_factual_claim_rejected(self):
        """Unsupported factual claim unrelated to retrieved chunk is rejected and replaced with safe fallback."""
        chunk: KnowledgeChunk = {
            "chunk_id": "somatic:box_breathing",
            "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
            "title": "Box Breathing",
            "category": "somatic_regulation",
            "jurisdiction": "GL",
            "content": "Inhale 4, hold 4, exhale 4, hold 4.",
            "score": 0.88,
            "eligibility": "approved",
        }
        # Draft makes unsupported claims about ketogenic diet and cold showers
        unsupported_draft = (
            "You should follow a strict ketogenic diet and drink apple cider vinegar "
            "every morning to reset vagus nerve tone and cure panic attacks."
        )
        state = create_verification_test_state(
            sanitized_text="What should I do for anxiety",
            draft_response=unsupported_draft,
            dialog_goal="somatic_grounding",
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_ungrounded")
        self.assertTrue(any("UNGROUNDED_FACTUAL_CLAIM" in f for f in res["verification_flags"]))
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(res["grounding_status"], "unsupported")

    # ------------------------------------------------------------------------
    # Test 3: Fabricated citation rejected
    # ------------------------------------------------------------------------
    def test_03_fabricated_citation_rejected(self):
        """Fabricated citation tag referencing an invented source is rejected."""
        chunk: KnowledgeChunk = {
            "chunk_id": "somatic:box_breathing",
            "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
            "title": "Box Breathing",
            "category": "somatic_regulation",
            "jurisdiction": "GL",
            "content": "Inhale 4, hold 4, exhale 4, hold 4.",
            "score": 0.88,
            "eligibility": "approved",
        }
        draft_with_fake_citation = (
            "Here is Box Breathing [somatic:fake_fabricated_source_999]. Inhale 4, hold 4, exhale 4, hold 4."
        )
        state = create_verification_test_state(
            sanitized_text="How to do box breathing",
            draft_response=draft_with_fake_citation,
            dialog_goal="somatic_grounding",
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_ungrounded")
        self.assertTrue(any("INVALID_CITATION_DETECTED" in f for f in res["verification_flags"]))
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 4: Citation to a nonexistent chunk rejected
    # ------------------------------------------------------------------------
    def test_04_citation_to_nonexistent_chunk_rejected(self):
        """Citation referencing a chunk not present in retrieved_chunks is rejected."""
        chunk: KnowledgeChunk = {
            "chunk_id": "somatic:box_breathing",
            "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
            "title": "Box Breathing",
            "category": "somatic_regulation",
            "jurisdiction": "GL",
            "content": "Inhale 4, hold 4, exhale 4, hold 4.",
            "score": 0.88,
            "eligibility": "approved",
        }
        # References cognitive_reframing:01_thought_court which was NOT retrieved
        draft_with_unretrieved_doc = (
            "Here is Box Breathing [doc:cognitive_reframing:01_thought_court_catastrophizing]. Inhale 4, hold 4, exhale 4, hold 4."
        )
        state = create_verification_test_state(
            sanitized_text="How to breathe",
            draft_response=draft_with_unretrieved_doc,
            dialog_goal="somatic_grounding",
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_ungrounded")
        self.assertTrue(any("INVALID_CITATION_DETECTED" in f for f in res["verification_flags"]))

    # ------------------------------------------------------------------------
    # Test 5: Restricted-content citation rejected without authorization
    # ------------------------------------------------------------------------
    def test_05_restricted_content_citation_rejected_without_authorization(self):
        """Restricted clinical screening chunks are rejected if trusted authorization is missing."""
        restricted_chunk: KnowledgeChunk = {
            "chunk_id": "clinical_rubrics:phq9",
            "document_id": "clinical_rubrics:02_phq9_mood_screening_overview",
            "title": "PHQ-9 Overview",
            "category": "clinical_rubrics",
            "jurisdiction": "GL",
            "content": "PHQ-9 9-item scale. Scores 0-27.",
            "score": 0.92,
            "eligibility": "restricted",
        }
        draft = f"Here is the PHQ-9 overview: scores range from 0 to 27.\n\n{MANDATORY_SCREENING_DISCLAIMER}"
        state = create_verification_test_state(
            sanitized_text="What are PHQ-9 cutoffs",
            draft_response=draft,
            dialog_goal="screening",
            retrieved_chunks=[restricted_chunk],
            retrieval_authorization=None,  # Unauthorized!
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_unsafe")
        self.assertIn("UNAUTHORIZED_RESTRICTED_CONTENT_USED", res["verification_flags"])
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 6: Excluded content never accepted
    # ------------------------------------------------------------------------
    def test_06_excluded_content_never_accepted(self):
        """Chunks with eligibility=excluded are strictly rejected even with authorization."""
        excluded_chunk: KnowledgeChunk = {
            "chunk_id": "excluded_chunk_01",
            "document_id": "excluded_pending_review:unverified_tool",
            "title": "Unverified Clinical Tool",
            "category": "clinical_rubrics",
            "jurisdiction": "GL",
            "content": "Unverified clinical tool.",
            "score": 0.95,
            "eligibility": "excluded",  # type: ignore
        }
        auth = RetrievalAuthorization(authorized_by="admin", reason="test")
        draft = "Here is an unverified clinical tool."
        state = create_verification_test_state(
            sanitized_text="Tell me about this tool",
            draft_response=draft,
            retrieved_chunks=[excluded_chunk],
            retrieval_authorization=auth,
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_unsafe")
        self.assertIn("EXCLUDED_CONTENT_USED", res["verification_flags"])
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 7: Diagnostic claim rejected
    # ------------------------------------------------------------------------
    def test_07_diagnostic_claim_rejected(self):
        """Clinical diagnostic assertions are detected and rejected."""
        chunk: KnowledgeChunk = {
            "chunk_id": "c1", "document_id": "d1", "title": "Depression Support",
            "category": "psychoeducation", "jurisdiction": "GL",
            "content": "Depression psychoeducation.", "score": 0.85, "eligibility": "approved",
        }
        draft = "Based on your description, you have major depressive disorder and must seek treatment."
        state = create_verification_test_state(
            sanitized_text="I feel very sad",
            draft_response=draft,
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_unsafe")
        self.assertIn("DIAGNOSTIC_CLAIM_DETECTED", res["verification_flags"])
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 8: Medication dosage advice rejected
    # ------------------------------------------------------------------------
    def test_08_medication_dosage_advice_rejected(self):
        """Medication prescriptions and dosage instructions are detected and rejected."""
        chunk: KnowledgeChunk = {
            "chunk_id": "c1", "document_id": "d1", "title": "Insomnia Care",
            "category": "psychoeducation", "jurisdiction": "GL",
            "content": "Insomnia psychoeducation.", "score": 0.85, "eligibility": "approved",
        }
        draft = "To help with sleep, you should take 10mg of ambien before bed."
        state = create_verification_test_state(
            sanitized_text="I cannot sleep",
            draft_response=draft,
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_unsafe")
        self.assertIn("MEDICATION_PRESCRIPTION_DETECTED", res["verification_flags"])
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 9: Fabricated hotline number rejected
    # ------------------------------------------------------------------------
    def test_09_fabricated_hotline_number_rejected(self):
        """Unverified/fabricated phone numbers not in hotlines.json are rejected."""
        chunk: KnowledgeChunk = {
            "chunk_id": "c1", "document_id": "d1", "title": "Crisis Resources",
            "category": "psychoeducation", "jurisdiction": "GL",
            "content": "Crisis resources info.", "score": 0.85, "eligibility": "approved",
        }
        draft = "If you need immediate support, please call our special helpline at 555-123-4567 right now."
        state = create_verification_test_state(
            sanitized_text="Who can I talk to",
            draft_response=draft,
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_unsafe")
        self.assertIn("FABRICATED_HOTLINE_DETECTED", res["verification_flags"])
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 10: False emergency dispatch claim rejected
    # ------------------------------------------------------------------------
    def test_10_false_emergency_dispatch_claim_rejected(self):
        """Claims that emergency services have been dispatched are rejected."""
        chunk: KnowledgeChunk = {
            "chunk_id": "c1", "document_id": "d1", "title": "Support",
            "category": "psychoeducation", "jurisdiction": "GL",
            "content": "General support info.", "score": 0.85, "eligibility": "approved",
        }
        draft = "Stay where you are. We have dispatched emergency services and help is on the way."
        state = create_verification_test_state(
            sanitized_text="I am scared",
            draft_response=draft,
            retrieved_chunks=[chunk],
        )

        res = verification_node(state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "rejected_unsafe")
        self.assertIn("FALSE_DISPATCH_CLAIM_DETECTED", res["verification_flags"])
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

    # ------------------------------------------------------------------------
    # Test 11: Missing screening disclaimer causes rejection or safe correction
    # ------------------------------------------------------------------------
    def test_11_missing_screening_disclaimer_rejection_and_safe_correction(self):
        """Missing mandatory screening disclaimer triggers rejection by default, or safe correction if requested."""
        auth = RetrievalAuthorization(authorized_by="clinician", reason="screening")
        chunk: KnowledgeChunk = {
            "chunk_id": "rubrics:phq9",
            "document_id": "clinical_rubrics:02_phq9_mood_screening_overview",
            "title": "PHQ-9 Overview",
            "category": "clinical_rubrics",
            "jurisdiction": "GL",
            "content": "PHQ-9 9-item scale. Scores 0-27.",
            "score": 0.90,
            "eligibility": "restricted",
        }
        draft_without_disclaimer = "Here is the PHQ-9 scale: scores range from 0 to 27."

        # Case A: Default (auto_correct=False) -> Rejection
        state_a = create_verification_test_state(
            sanitized_text="What is PHQ-9",
            draft_response=draft_without_disclaimer,
            dialog_goal="screening",
            retrieved_chunks=[chunk],
            retrieval_authorization=auth,
        )
        res_a = verification_node(state_a, auto_correct_disclaimer=False)
        self.assertFalse(res_a["verification_passed"])
        self.assertEqual(res_a["verification_status"], "rejected_unsafe")
        self.assertIn("MISSING_SCREENING_DISCLAIMER", res_a["verification_flags"])
        self.assertEqual(res_a["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)

        # Case B: auto_correct_disclaimer=True -> Safe correction applied
        state_b = create_verification_test_state(
            sanitized_text="What is PHQ-9",
            draft_response=draft_without_disclaimer,
            dialog_goal="screening",
            retrieved_chunks=[chunk],
            retrieval_authorization=auth,
        )
        res_b = verification_node(state_b, auto_correct_disclaimer=True)
        self.assertTrue(res_b["verification_passed"])
        self.assertEqual(res_b["verification_status"], "verified")
        self.assertIn(MANDATORY_SCREENING_DISCLAIMER, res_b["final_response"])
        self.assertIn("DISCLAIMER_SAFE_CORRECTION_APPLIED", res_b["verification_flags"])

    # ------------------------------------------------------------------------
    # Test 12: Empty retrieval produces an explicitly ungrounded safe fallback
    # ------------------------------------------------------------------------
    def test_12_empty_retrieval_produces_safe_fallback(self):
        """Empty retrieval draft is verified as fallback_used with grounding_status=fallback."""
        state = create_verification_test_state(
            sanitized_text="How to sort a list in Python",
            draft_response=NEUTRAL_SUPPORTIVE_FALLBACK,
            dialog_goal="general",
            retrieved_chunks=[],
            retrieval_status="empty",
        )
        res = verification_node(state)
        self.assertTrue(res["verification_passed"])
        self.assertEqual(res["verification_status"], "fallback_used")
        self.assertEqual(res["grounding_status"], "fallback")
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        # Verify no unsolicited breathing commands
        self.assertNotIn("breathing", res["final_response"].lower())

    # ------------------------------------------------------------------------
    # Test 13: Verification exception fails safely
    # ------------------------------------------------------------------------
    def test_13_verification_exception_fails_safely(self):
        """Unexpected internal exceptions are caught with controlled error code and zero tracebacks in state."""
        # Pass malformed state to induce exception safely inside regex scanner
        malformed_state = create_verification_test_state(
            sanitized_text="Test",
            draft_response="Sample response",
        )
        # Inject bad types to simulate catastrophic runtime failure
        malformed_state["draft_response"] = 12345  # type: ignore

        res = verification_node(malformed_state)
        self.assertFalse(res["verification_passed"])
        self.assertEqual(res["verification_status"], "verification_error")
        self.assertEqual(res["error_code"], "VERIFICATION_FAILURE")
        self.assertEqual(res["error_component"], "verification_node")
        self.assertEqual(res["final_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(res["execution_trace"][-1]["status"], "error")

    # ------------------------------------------------------------------------
    # Test 14: Original sanitized message remains unchanged
    # ------------------------------------------------------------------------
    def test_14_original_sanitized_message_remains_unchanged(self):
        """Sanitized user text and messages are never modified by verification_node."""
        original_text = "I feel so overwhelmed by my final exams at university"
        state = create_verification_test_state(
            sanitized_text=original_text,
            draft_response=NEUTRAL_SUPPORTIVE_FALLBACK,
        )
        res = verification_node(state)
        # Verification update must not overwrite or contain sanitized_text or messages
        self.assertNotIn("sanitized_text", res)
        self.assertNotIn("messages", res)
        self.assertEqual(state["sanitized_text"], original_text)

    # ------------------------------------------------------------------------
    # Test 15: Emergency bypass skips response generation and verification
    # ------------------------------------------------------------------------
    def test_15_emergency_bypass_skips_verification(self):
        """Emergency bypass halts verification immediately, preserving static emergency guidance."""
        emergency_text = "Call 112 or Tele-MANAS 14416 immediately. Support is available 24/7."
        state = create_verification_test_state(
            sanitized_text="I am going to end my life right now with pills goodbye",
            overall_safety_route="emergency_bypass",
            deterministic_danger_flag=True,
        )
        state["final_response"] = emergency_text

        res = verification_node(state)
        self.assertEqual(res["verification_status"], "not_run")
        self.assertNotIn("final_response", res)  # Preserves emergency guidance untouched!
        self.assertEqual(state["final_response"], emergency_text)
        self.assertEqual(res["execution_trace"][-1]["metadata_code"], "EMERGENCY_BYPASS_SKIPPED")

    # ------------------------------------------------------------------------
    # Test 16: Existing safety decisions and classifier outputs remain unchanged
    # ------------------------------------------------------------------------
    def test_16_safety_decisions_and_classifier_outputs_preserved(self):
        """Verification node never overrides safety route, classifier probability, or follow-up status."""
        state = create_verification_test_state(
            sanitized_text="Exam anxiety",
            draft_response=NEUTRAL_SUPPORTIVE_FALLBACK,
            overall_safety_route="standard_conversation",
            roberta_prob=0.1234,
            roberta_crisis_flag=False,
            phq9_item9_endorsed=False,
            phq9_followup_status="not_started",
            phq9_risk_level="none",
        )
        res = verification_node(state)
        for preserved_key in (
            "overall_safety_route", "roberta_prob", "roberta_crisis_flag",
            "phq9_item9_endorsed", "phq9_followup_status", "phq9_risk_level"
        ):
            self.assertNotIn(preserved_key, res)
            self.assertIn(preserved_key, state)

    # ------------------------------------------------------------------------
    # Test 17: Jurisdiction metadata is preserved
    # ------------------------------------------------------------------------
    def test_17_jurisdiction_metadata_preserved(self):
        """Jurisdiction filter metadata is not modified or stripped by verification."""
        state = create_verification_test_state(
            sanitized_text="Anxiety support in India",
            draft_response=NEUTRAL_SUPPORTIVE_FALLBACK,
            jurisdiction_filter="IN",
        )
        res = verification_node(state)
        self.assertNotIn("jurisdiction_filter", res)
        self.assertEqual(state["jurisdiction_filter"], "IN")

    # ------------------------------------------------------------------------
    # Test 18: Complete graph integration works with verification gate active
    # ------------------------------------------------------------------------
    def test_18_complete_graph_integration_with_verification(self):
        """End-to-end graph execution runs through planning, response_node, and verification_node."""
        # 1. Standard conversation with valid generation passes verification
        app = build_mindbridge_graph()
        initial_state = create_verification_test_state(
            sanitized_text="How to do box breathing exercises for stress",
            jurisdiction_filter="GL",
        )
        final_state = app.invoke(initial_state)

        self.assertEqual(final_state["overall_safety_route"], "standard_conversation")
        self.assertTrue(final_state["verification_passed"])
        self.assertIn(final_state["verification_status"], ("verified", "fallback_used"))
        self.assertTrue(len(final_state["final_response"]) > 0)

        # 2. End-to-end graph execution catches an unsafe custom generator output
        def unsafe_generator(s: MindBridgeState) -> str:
            return "Based on your message, you have major depressive disorder. Take 50mg of sertraline daily."

        app_unsafe = build_mindbridge_graph(generator=unsafe_generator)
        final_state_unsafe = app_unsafe.invoke(initial_state)

        # The unsafe output must be suppressed by the safety gates and replaced with safe fallback!
        self.assertNotIn("you have major depressive disorder", final_state_unsafe["final_response"])
        self.assertNotIn("50mg", final_state_unsafe["final_response"])
        self.assertIn("I am here with you", final_state_unsafe["final_response"])


if __name__ == "__main__":
    unittest.main()
