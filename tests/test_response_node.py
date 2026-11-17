"""
Unit & Integration Tests for MindBridge 2.0 Response Generation Node
===================================================================

Verifies:
1. Grounded synthesis for somatic grounding (box breathing, physiological sigh)
2. Grounded synthesis for cognitive reframing (framed strictly as educational worksheets, not clinical psychotherapy)
3. Grounded synthesis for psychoeducation (insomnia stimulus control, exam anxiety)
4. Non-diagnostic framing and suppression of clinical/diagnostic assertions & prescription dosing
5. Mandatory screening disclaimer injection for clinical rubrics (PHQ-9, GAD-7, PSS-10)
6. Safe neutral supportive fallback on empty retrieval / OOD without unsolicited breathing commands
7. Universal emergency bypass preservation (draft_response=None, halts open generation)
8. Multi-turn safety follow-up and de-escalation route preservation
9. Pluggable custom generator execution and sanitization
10. Failure containment and controlled error code logging
11. Telemetry execution trace verification
"""

from datetime import datetime, timezone
import unittest
from typing import Any, Dict, List, Optional

from backend.orchestrator.response_node import (
    response_node,
    synthesize_grounded_response,
    _extract_protocol_steps_or_summary,
    _sanitize_against_clinical_claims,
)
from backend.orchestrator.state import (
    KnowledgeChunk,
    MindBridgeState,
    TraceEntry,
)
from backend.safety.fallbacks import (
    MANDATORY_SCREENING_DISCLAIMER,
    NEUTRAL_SUPPORTIVE_FALLBACK,
    contains_prohibited_clinical_claims,
)


def create_test_response_state(
    sanitized_text: str = "",
    dialog_goal: str = "general",
    retrieved_chunks: Optional[List[KnowledgeChunk]] = None,
    retrieval_status: str = "success",
    overall_safety_route: str = "standard_conversation",
    deterministic_danger_flag: bool = False,
    jurisdiction_filter: str = "GL",
) -> MindBridgeState:
    """Helper constructing conforming MindBridgeState for testing response_node."""
    chunks = retrieved_chunks if retrieved_chunks is not None else []
    return {
        "session_id": "test-response-session-uuid",
        "user_id_hash": "test-response-user-hash",
        "messages": [{"role": "user", "content": sanitized_text}] if sanitized_text else [],
        "sanitized_text": sanitized_text,
        "deterministic_danger_flag": deterministic_danger_flag,
        "emergency_trigger_id": None,
        "roberta_prob": 0.05,
        "roberta_crisis_flag": False,
        "classifier_status": "completed",
        "overall_safety_route": overall_safety_route,  # type: ignore
        "phq9_item9_endorsed": False,
        "phq9_followup_status": "not_started",  # type: ignore
        "phq9_risk_level": "none",
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": dialog_goal,  # type: ignore
        "retrieval_queries": ["test search query"],
        "retrieved_chunks": chunks,
        "retrieval_status": retrieval_status,  # type: ignore
        "jurisdiction_filter": jurisdiction_filter,  # type: ignore
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class TestResponseNode(unittest.TestCase):
    """Test suite for grounded response generation node."""

    def test_01_somatic_grounding_synthesis(self):
        """Generates gentle, grounded instructions for somatic regulation exercises."""
        chunk: KnowledgeChunk = {
            "chunk_id": "somatic:box_breathing",
            "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
            "title": "Box Breathing (4-4-4-4 Parasympathetic Reset)",
            "category": "somatic_regulation",
            "jurisdiction": "GL",
            "content": (
                "## 2. Summary & Operational Objective\n"
                "A real-time 4-4-4-4 physiological reset.\n\n"
                "## 3. Step-by-Step Instructions & Protocol Details\n"
                "1. Inhale gently through the nose for 4 seconds.\n"
                "2. Hold the breath calmly for 4 seconds.\n"
                "3. Exhale slowly through the mouth for 4 seconds.\n"
                "4. Rest in stillness for 4 seconds.\n"
                "Repeat for 2 to 4 cycles."
            ),
            "score": 0.89,
            "eligibility": "approved",
        }
        state = create_test_response_state(
            sanitized_text="I am having a panic attack and need to calm down right now",
            dialog_goal="somatic_grounding",
            retrieved_chunks=[chunk],
            retrieval_status="success",
        )
        updates = response_node(state)

        draft = updates.get("draft_response")
        self.assertIsNotNone(draft)
        self.assertIn("Box Breathing", draft)
        self.assertIn("Inhale gently through the nose for 4 seconds", draft)
        self.assertIn("Exhale slowly through the mouth for 4 seconds", draft)
        self.assertIn("at your own pace", draft)
        self.assertIn("in control", draft)
        self.assertNotIn(MANDATORY_SCREENING_DISCLAIMER, draft)
        self.assertEqual(updates["execution_trace"][-1]["status"], "success")

    def test_02_cognitive_reframing_educational_worksheet_framing(self):
        """Frames cognitive restructuring strictly as an educational self-reflection worksheet, NOT psychotherapy."""
        chunk: KnowledgeChunk = {
            "chunk_id": "cognitive:thought_court",
            "document_id": "cognitive_reframing:01_thought_court_catastrophizing",
            "title": "Cognitive Restructuring for Catastrophic Loops (Thought Court)",
            "category": "cognitive_reframing",
            "jurisdiction": "GL",
            "content": (
                "## 2. Summary & Operational Objective\n"
                "Learn the clinical Thought Court framework to weigh factual empirical evidence.\n\n"
                "## 3. Step-by-Step Instructions & Protocol Details\n"
                "1. State the intrusive thought as a factual claim.\n"
                "2. Identify the distortion: Catastrophizing or Fortune-Telling.\n"
                "3. Prosecuting evidence: What concrete proof confirms this outcome?\n"
                "4. Defense evidence: What facts prove I have survived before?\n"
                "5. Balanced Verdict: Formulate an evidence-grounded alternative."
            ),
            "score": 0.86,
            "eligibility": "approved",
        }
        state = create_test_response_state(
            sanitized_text="I keep catastrophizing about losing my job and failing at everything",
            dialog_goal="cognitive_reframing",
            retrieved_chunks=[chunk],
            retrieval_status="success",
        )
        updates = response_node(state)

        draft = updates.get("draft_response")
        self.assertIsNotNone(draft)
        self.assertIn("Thought Court", draft)
        self.assertIn("educational self-reflection worksheet", draft.lower())
        self.assertIn("not clinical psychotherapy", draft.lower())
        self.assertIn("State the intrusive thought", draft)
        self.assertIn("Balanced Verdict", draft)

    def test_03_psychoeducation_insomnia_stimulus_control(self):
        """Generates evidence-informed psychoeducational guidance for insomnia."""
        chunk: KnowledgeChunk = {
            "chunk_id": "psychoed:insomnia",
            "document_id": "psychoeducation:04_stimulus_control_insomnia_cbti",
            "title": "Stimulus Control Therapy for Insomnia (CBT-I)",
            "category": "psychoeducation",
            "jurisdiction": "GL",
            "content": (
                "## 2. Summary & Operational Objective\n"
                "Extinguish negative conditioned arousal between bed and waking anxiety.\n\n"
                "## 3. Step-by-Step Instructions & Protocol Details\n"
                "1. Use bed only for sleep (no screens, work, or active reading).\n"
                "2. 20-Minute Rule: If awake after ~20 minutes, leave bedroom calmly.\n"
                "3. Dim lighting: Sit quietly until drowsy; return to bed only when sleepy.\n"
                "4. Fixed wake time: Wake at identical time daily regardless of sleep duration."
            ),
            "score": 0.84,
            "eligibility": "approved",
        }
        state = create_test_response_state(
            sanitized_text="I have been lying awake in bed for hours unable to sleep",
            dialog_goal="general",
            retrieved_chunks=[chunk],
            retrieval_status="success",
        )
        updates = response_node(state)

        draft = updates.get("draft_response")
        self.assertIsNotNone(draft)
        self.assertIn("Stimulus Control Therapy", draft)
        self.assertIn("20-Minute Rule", draft)
        self.assertIn("Fixed wake time", draft)

    def test_04_mandatory_screening_disclaimer_appended(self):
        """Appends mandatory non-diagnostic disclaimer when clinical screening rubrics are retrieved."""
        chunk: KnowledgeChunk = {
            "chunk_id": "rubrics:phq9",
            "document_id": "clinical_rubrics:02_phq9_mood_screening_overview",
            "title": "PHQ-9 (Patient Health Questionnaire 9-Item Scale) Overview",
            "category": "clinical_rubrics",
            "jurisdiction": "GL",
            "content": (
                "## 2. Summary & Operational Objective\n"
                "A 9-item standardized self-report inventory evaluating depressive symptoms.\n\n"
                "## 3. Step-by-Step Instructions & Protocol Details\n"
                "Scores range from 0 to 27:\n"
                "- 0-4: Minimal or absent depressive symptoms\n"
                "- 5-9: Mild depressive symptoms\n"
                "- 10-14: Moderate depressive symptoms\n"
                "- 15-19: Moderately severe depressive symptoms\n"
                "- 20-27: Severe depressive symptoms"
            ),
            "score": 0.91,
            "eligibility": "restricted",
        }
        state = create_test_response_state(
            sanitized_text="Explain the PHQ-9 scoring scale and cutoff numbers",
            dialog_goal="screening",
            retrieved_chunks=[chunk],
            retrieval_status="success",
        )
        updates = response_node(state)

        draft = updates.get("draft_response")
        self.assertIsNotNone(draft)
        self.assertIn("PHQ-9", draft)
        self.assertIn("Minimal or absent", draft)
        self.assertIn(MANDATORY_SCREENING_DISCLAIMER, draft)
        self.assertIn("CRITICAL: This screening rubric is an informational educational tool", draft)

    def test_05_empty_retrieval_emits_neutral_supportive_fallback(self):
        """Empty retrieval emits vetted neutral fallback and strictly prohibits unconditional breathing exercises."""
        state = create_test_response_state(
            sanitized_text="How to sort a dictionary in Python",
            dialog_goal="general",
            retrieved_chunks=[],
            retrieval_status="empty",
        )
        updates = response_node(state)

        draft = updates.get("draft_response")
        self.assertEqual(draft, NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates.get("final_response"), NEUTRAL_SUPPORTIVE_FALLBACK)
        # Verify strict prohibition of unsolicited breathing exercises on empty context
        self.assertNotIn("breathing", draft.lower())
        self.assertNotIn("inhale", draft.lower())
        self.assertNotIn("exhale", draft.lower())
        self.assertEqual(updates["execution_trace"][-1]["status"], "fallback")
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "NO_CONTEXT_SUPPORTIVE_FALLBACK")

    def test_06_emergency_bypass_halts_open_generation(self):
        """Universal Emergency Bypass suppresses draft response and halts open generation."""
        state = create_test_response_state(
            sanitized_text="I am taking all my pills tonight goodbye",
            overall_safety_route="emergency_bypass",
            deterministic_danger_flag=True,
            retrieval_status="not_attempted",
        )
        updates = response_node(state)

        self.assertIsNone(updates.get("draft_response"))
        self.assertEqual(updates["execution_trace"][-1]["status"], "skipped")
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "EMERGENCY_BYPASS_SKIPPED")

    def test_07_active_followup_and_deescalation_preserved(self):
        """Pending safety follow-up and de-escalation routes preserve safety state."""
        for route in ("pending_safety_follow_up", "supportive_deescalation"):
            state = create_test_response_state(
                sanitized_text="I just feel completely drained",
                overall_safety_route=route,
            )
            updates = response_node(state)
            self.assertIsNone(updates.get("draft_response"))
            self.assertEqual(updates["execution_trace"][-1]["status"], "skipped")
            self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "SAFETY_STATE_PRESERVED")

    def test_08_sanitization_removes_diagnostic_assertions_and_prescriptions(self):
        """Sanitizes inadvertent diagnostic assertions and prescription dosing."""
        raw_draft = (
            "Based on your statements, you have major depressive disorder. "
            "You should take 50mg of sertraline daily to treat it."
        )
        sanitized = _sanitize_against_clinical_claims(raw_draft)
        self.assertNotIn("you have major depressive disorder", sanitized.lower())
        self.assertNotIn("50mg", sanitized.lower())
        self.assertNotIn("sertraline", sanitized.lower())
        self.assertFalse(contains_prohibited_clinical_claims(sanitized))

    def test_09_prohibits_automated_emergency_dispatch_phrasing(self):
        """Prohibits phrasing implying automated dialing or emergency services dispatch."""
        dispatch_draft = "Please remain calm. We have dispatched emergency services and are calling 911 for you now."
        sanitized = _sanitize_against_clinical_claims(dispatch_draft)
        self.assertNotIn("we have dispatched", sanitized.lower())
        self.assertNotIn("calling 911 for you", sanitized.lower())
        self.assertIn("please contact local emergency services immediately", sanitized.lower())

    def test_10_custom_generator_integration_and_guardrails(self):
        """Supports pluggable custom generator while enforcing clinical guardrails."""
        def safe_generator(s: MindBridgeState) -> str:
            return "Here is a safe psychoeducational summary for you to reflect upon."

        state = create_test_response_state(
            sanitized_text="Tell me about stress",
            retrieved_chunks=[{
                "chunk_id": "c1", "document_id": "d1", "title": "Stress",
                "category": "psychoeducation", "jurisdiction": "GL",
                "content": "Summary of stress management.", "score": 0.8,
                "eligibility": "approved",
            }],
        )
        updates = response_node(state, generator=safe_generator)
        self.assertEqual(updates["draft_response"], "Here is a safe psychoeducational summary for you to reflect upon.")

        # Test generator producing prohibited claim: falls back safely
        def unsafe_generator(s: MindBridgeState) -> str:
            return "You are diagnosed with bipolar disorder and must start taking lexapro."

        updates_unsafe = response_node(state, generator=unsafe_generator)
        self.assertEqual(updates_unsafe["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates_unsafe["execution_trace"][-1]["metadata_code"], "PROHIBITED_CLAIM_SUPPRESSED")

    def test_11_failure_containment_on_unexpected_exception(self):
        """Internal generator exception is caught gracefully with controlled error code and zero tracebacks in state."""
        def crashing_generator(s: MindBridgeState) -> str:
            raise RuntimeError("Unexpected tokenizer CUDA crash")

        state = create_test_response_state(
            sanitized_text="How to breathe",
            retrieved_chunks=[{
                "chunk_id": "c1", "document_id": "d1", "title": "Breathing",
                "category": "somatic_regulation", "jurisdiction": "GL",
                "content": "Instructions", "score": 0.8, "eligibility": "approved",
            }],
        )
        updates = response_node(state, generator=crashing_generator)
        self.assertEqual(updates["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates["error_code"], "LLM_GENERATION_FAILURE")
        self.assertEqual(updates["error_component"], "response_node")
        self.assertEqual(updates["execution_trace"][-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
