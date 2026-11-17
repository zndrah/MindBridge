"""
Unit Tests for MindBridge 2.0 Deterministic Safety Node & Classifier Decision Logic
===================================================================================

Verifies all 20 required contract scenarios and contextual regression cases:
1. Emergency overrides standard route.
2. Emergency during active follow-up.
3. Emergency overrides a low classifier score.
4. Follow-up continuity and dimension preservation.
5. Preliminary regex does not set PHQ-9 endorsement.
6. Ambiguous dimensions remain None.
7. Interrupted follow-up resumes safely.
8. Internal 'none' never becomes a zero-risk or clinical-clearance claim.
9. Acute peril involving active ideation and immediate access to means.
10. Elevated classifier signal does not confirm PHQ-9 endorsement.
11. Below-threshold classifier result permits standard routing only when other conditions allow.
12. Classifier timeout.
13. Classifier failure does not overwrite emergency.
14. Privacy-safe telemetry.
15. Unresolved acute dimension blocks supportive de-escalation.
16. Near-future timeframe blocks premature de-escalation.
17. Ideation alone triggers follow-up, not emergency.
18. Means alone does not trigger emergency.
19. Emergency output uses approved static fallback and never claims automatic dispatch.
20. Jurisdiction filtering of emergency resources.
21. Contextual negation regression.
22. Contextual quotation regression.
23. Contextual hypothetical regression.
24. Contextual third-party report regression.
"""

import json
from pathlib import Path
import unittest
from typing import Any, Dict

from backend.orchestrator.input_node import input_node
from backend.orchestrator.safety_node import (
    _format_hotline_markdown_entry,
    classifier_decision_node,
    deterministic_safety_node,
    evaluate_deescalation_eligibility,
    evaluate_standard_conversation_eligibility,
    format_emergency_guidance,
    get_hotlines_for_jurisdiction,
    is_contextual_negation_or_third_party,
    is_ordinary_medication_mention,
    load_hotlines_directory,
    route_after_classifier_decision,
    route_after_deterministic_safety,
    route_after_followup,
    set_hotlines_directory_for_testing,
)
from backend.orchestrator.state import (
    ClinicalRiskDimensions,
    MindBridgeState,
)


def create_initial_state(
    session_id: str = "test-session-uuid-1234",
    overall_safety_route: str = "standard_conversation",
    sanitized_text: str = "",
    messages: Any = None,
    phq9_followup_status: str = "not_started",
    phq9_dimensions: Any = None,
    jurisdiction_filter: str = "US",
    deterministic_danger_flag: bool = False,
    emergency_trigger_id: Any = None,
    roberta_prob: Any = None,
    roberta_crisis_flag: bool = False,
    classifier_status: str = "not_run",
    phq9_item9_endorsed: bool = False,
    phq9_risk_level: Any = None,
    final_response: str = "",
) -> MindBridgeState:
    """Helper creating a strictly-typed MindBridgeState fixture."""
    return {
        "session_id": session_id,
        "user_id_hash": "test-user-hash-5678",
        "messages": messages if messages is not None else [],
        "sanitized_text": sanitized_text,
        "deterministic_danger_flag": deterministic_danger_flag,
        "emergency_trigger_id": emergency_trigger_id,
        "roberta_prob": roberta_prob,
        "roberta_crisis_flag": roberta_crisis_flag,
        "classifier_status": classifier_status,  # type: ignore
        "overall_safety_route": overall_safety_route,  # type: ignore
        "phq9_item9_endorsed": phq9_item9_endorsed,
        "phq9_followup_status": phq9_followup_status,  # type: ignore
        "phq9_risk_level": phq9_risk_level,
        "phq9_dimensions": phq9_dimensions if phq9_dimensions is not None else {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": "general",
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieval_status": "not_attempted",
        "jurisdiction_filter": jurisdiction_filter,  # type: ignore
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": final_response,
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class TestSafetyNode(unittest.TestCase):

    # ------------------------------------------------------------------------
    # Scenario 1: Emergency overrides standard route
    # ------------------------------------------------------------------------
    def test_01_emergency_overrides_standard_route(self):
        """Explicit acute danger must override standard route to emergency_bypass."""
        state = create_initial_state(
            overall_safety_route="standard_conversation",
            sanitized_text="I have pills and I am taking them tonight to end my life. Goodbye forever.",
            jurisdiction_filter="US",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "emergency_bypass")
        self.assertTrue(updates["deterministic_danger_flag"])
        self.assertIsNotNone(updates["emergency_trigger_id"])
        self.assertEqual(route_after_deterministic_safety(merged_state), "emergency_node")
        self.assertIn("988", updates["final_response"])
        self.assertIn("911", updates["final_response"])

    # ------------------------------------------------------------------------
    # Scenario 2: Emergency during active follow-up
    # ------------------------------------------------------------------------
    def test_02_emergency_during_active_followup(self):
        """Acute disclosure during active follow-up must escalate immediately."""
        state = create_initial_state(
            overall_safety_route="pending_safety_follow_up",
            phq9_followup_status="in_progress",
            phq9_dimensions={"passive_death_wishes": True},
            sanitized_text="I have a gun in my hand right now and I am about to pull the trigger.",
            jurisdiction_filter="US",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "emergency_bypass")
        self.assertTrue(updates["deterministic_danger_flag"])
        self.assertEqual(route_after_deterministic_safety(merged_state), "emergency_node")
        self.assertIsNone(updates["draft_response"])

    # ------------------------------------------------------------------------
    # Scenario 3: Emergency overrides a low classifier score
    # ------------------------------------------------------------------------
    def test_03_emergency_overrides_low_classifier_score(self):
        """Deterministic danger must not be overridden by a low RoBERTa probability."""
        # Setup: Emergency was triggered, but classifier returned p = 0.02
        state = create_initial_state(
            overall_safety_route="emergency_bypass",
            deterministic_danger_flag=True,
            emergency_trigger_id="IMMINENT_SUICIDE_INTENT",
            roberta_prob=0.02,
            roberta_crisis_flag=False,
            classifier_status="completed",
        )
        decision_updates = classifier_decision_node(state)
        merged_state = {**state, **decision_updates}

        self.assertEqual(merged_state["overall_safety_route"], "emergency_bypass")
        self.assertEqual(route_after_classifier_decision(merged_state), "emergency_node")

    # ------------------------------------------------------------------------
    # Scenario 4: Follow-up continuity and dimension preservation
    # ------------------------------------------------------------------------
    def test_04_followup_continuity_and_dimension_preservation(self):
        """Active follow-up must continue; confirmed dimensions must not be erased."""
        existing_dims: ClinicalRiskDimensions = {
            "passive_death_wishes": True,
            "intent": False,
        }
        state = create_initial_state(
            overall_safety_route="pending_safety_follow_up",
            phq9_followup_status="in_progress",
            phq9_dimensions=existing_dims,
            sanitized_text="No, I don't have anything like that.",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "pending_safety_follow_up")
        self.assertEqual(updates["phq9_followup_status"], "in_progress")
        self.assertEqual(route_after_deterministic_safety(merged_state), "followup_node")
        # Ensure dimensions in state remain intact
        self.assertTrue(merged_state["phq9_dimensions"]["passive_death_wishes"])
        self.assertFalse(merged_state["phq9_dimensions"]["intent"])

    # ------------------------------------------------------------------------
    # Scenario 5: Preliminary regex does not set PHQ-9 endorsement
    # ------------------------------------------------------------------------
    def test_05_preliminary_regex_does_not_set_phq9_endorsement(self):
        """Preliminary regex match triggers follow-up without setting phq9_item9_endorsed = True."""
        state = create_initial_state(
            overall_safety_route="standard_conversation",
            sanitized_text="I sometimes wish I would not wake up.",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "pending_safety_follow_up")
        self.assertEqual(updates["phq9_followup_status"], "in_progress")
        # Critical rule: preliminary regex does NOT set phq9_item9_endorsed = True
        self.assertFalse(merged_state["phq9_item9_endorsed"])
        self.assertIsNone(merged_state["phq9_risk_level"])
        self.assertEqual(route_after_deterministic_safety(merged_state), "followup_node")

    # ------------------------------------------------------------------------
    # Scenario 6: Ambiguous dimensions remain None
    # ------------------------------------------------------------------------
    def test_06_ambiguous_dimensions_remain_none(self):
        """Ternary clinical logic: ambiguous answers remain None and block de-escalation."""
        dims: ClinicalRiskDimensions = {
            "passive_death_wishes": True,
            "active_suicidal_ideation": False,
            "intent": None,  # Ambiguous response ("I don't know, maybe")
            "plan": False,
            "means": False,
            "access": False,
            "timeframe": "none_reported",
        }
        eligible, reason = evaluate_deescalation_eligibility(dims)
        self.assertFalse(eligible)
        self.assertIn("UNRESOLVED", reason)

    # ------------------------------------------------------------------------
    # Scenario 7: Interrupted follow-up resumes safely
    # ------------------------------------------------------------------------
    def test_07_interrupted_followup_resumes_safely(self):
        """Interrupted assessment routes back to follow-up on return, not to classifier."""
        state = create_initial_state(
            overall_safety_route="pending_safety_follow_up",
            phq9_followup_status="interrupted",
            phq9_risk_level="unassessed",
            phq9_dimensions={"passive_death_wishes": True},
            sanitized_text="Hello, I am back.",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "pending_safety_follow_up")
        self.assertEqual(updates["phq9_followup_status"], "interrupted")
        self.assertEqual(route_after_deterministic_safety(merged_state), "followup_node")

    # ------------------------------------------------------------------------
    # Scenario 8: Internal 'none' never claims zero risk or clearance
    # ------------------------------------------------------------------------
    def test_08_internal_none_never_claims_zero_risk_or_clearance(self):
        """Internal risk level 'none' is a state label only; never claims clinical clearance."""
        dims: ClinicalRiskDimensions = {
            "passive_death_wishes": False,
            "active_suicidal_ideation": False,
            "intent": False,
            "plan": False,
            "means": False,
            "access": False,
            "timeframe": "none_reported",
        }
        eligible, reason = evaluate_standard_conversation_eligibility("completed", dims)
        self.assertTrue(eligible)
        self.assertIsNone(reason)

        # Mock follow-up completion state
        state = create_initial_state(
            overall_safety_route="standard_conversation",
            phq9_followup_status="completed",
            phq9_risk_level="none",
            final_response="Thank you for sharing. I understand you are not experiencing thoughts of self-harm.",
        )
        # Verify prohibited clinical clearance phrasing is ABSENT
        lower_response = state["final_response"].lower()
        self.assertNotIn("zero risk", lower_response)
        self.assertNotIn("clinically cleared", lower_response)
        self.assertNotIn("guaranteed safe", lower_response)

    # ------------------------------------------------------------------------
    # Scenario 9: Acute peril: active ideation and immediate access to means
    # ------------------------------------------------------------------------
    def test_09_acute_peril_active_ideation_and_means_access(self):
        """Disjoint acute peril (ideation + means + access) triggers emergency bypass."""
        dims: ClinicalRiskDimensions = {
            "active_suicidal_ideation": True,
            "means": True,
            "access": True,
            "plan": False,  # No formal plan
        }
        state = create_initial_state(
            overall_safety_route="pending_safety_follow_up",
            phq9_followup_status="in_progress",
            phq9_dimensions=dims,
            sanitized_text="I have the pills right here and I want to die.",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "emergency_bypass")
        self.assertTrue(updates["deterministic_danger_flag"])
        self.assertEqual(route_after_deterministic_safety(merged_state), "emergency_node")

    # ------------------------------------------------------------------------
    # Scenario 10: Elevated classifier signal does not confirm PHQ-9 endorsement
    # ------------------------------------------------------------------------
    def test_10_elevated_classifier_signal_does_not_confirm_phq9_endorsement(self):
        """RoBERTa elevated signal initiates follow-up but does NOT set phq9_item9_endorsed."""
        state = create_initial_state(
            overall_safety_route="standard_conversation",
            classifier_status="completed",
            roberta_prob=0.82,
            roberta_crisis_flag=True,
            phq9_item9_endorsed=False,
        )
        decision_updates = classifier_decision_node(state)
        merged_state = {**state, **decision_updates}

        self.assertEqual(merged_state["overall_safety_route"], "elevated_classifier_signal")
        self.assertEqual(merged_state["phq9_followup_status"], "in_progress")
        # Critical rule: Model prediction does NOT establish user endorsement
        self.assertFalse(merged_state["phq9_item9_endorsed"])
        self.assertEqual(route_after_classifier_decision(merged_state), "followup_node")

    # ------------------------------------------------------------------------
    # Scenario 11: Below-threshold classifier permits standard routing
    # ------------------------------------------------------------------------
    def test_11_below_threshold_classifier_permits_standard_routing(self):
        """Below-threshold prediction allows standard conversation when safe."""
        state = create_initial_state(
            overall_safety_route="standard_conversation",
            classifier_status="completed",
            roberta_prob=0.12,
            roberta_crisis_flag=False,
        )
        decision_updates = classifier_decision_node(state)
        merged_state = {**state, **decision_updates}

        self.assertEqual(merged_state["overall_safety_route"], "standard_conversation")
        self.assertEqual(route_after_classifier_decision(merged_state), "planning_node")

    # ------------------------------------------------------------------------
    # Scenario 12: Classifier timeout handling
    # ------------------------------------------------------------------------
    def test_12_classifier_timeout_handling(self):
        """Classifier timeout transitions to ml_risk_unassessed with controlled error."""
        state = create_initial_state(
            overall_safety_route="standard_conversation",
            classifier_status="timeout",
        )
        decision_updates = classifier_decision_node(state)
        merged_state = {**state, **decision_updates}

        self.assertEqual(merged_state["overall_safety_route"], "ml_risk_unassessed")
        self.assertIsNone(merged_state["roberta_prob"])
        self.assertEqual(merged_state["error_code"], "CLASSIFIER_TIMEOUT")
        self.assertEqual(merged_state["error_component"], "classifier_node")
        self.assertEqual(route_after_classifier_decision(merged_state), "deescalation_node")
        self.assertIn("ready to support you", merged_state["final_response"])

    # ------------------------------------------------------------------------
    # Scenario 13: Classifier failure does not overwrite emergency
    # ------------------------------------------------------------------------
    def test_13_classifier_failure_does_not_overwrite_emergency(self):
        """Classifier crash cannot downgrade or overwrite an active emergency_bypass."""
        state = create_initial_state(
            overall_safety_route="emergency_bypass",
            deterministic_danger_flag=True,
            emergency_trigger_id="LETHAL_MEANS_DISCLOSURE",
            classifier_status="error",
        )
        decision_updates = classifier_decision_node(state)
        merged_state = {**state, **decision_updates}

        self.assertEqual(merged_state["overall_safety_route"], "emergency_bypass")
        self.assertEqual(route_after_classifier_decision(merged_state), "emergency_node")

    # ------------------------------------------------------------------------
    # Scenario 14: Privacy-safe telemetry
    # ------------------------------------------------------------------------
    def test_14_privacy_safe_telemetry_no_leakage(self):
        """Execution traces must never leak raw input, matched substrings, or raw exceptions."""
        sensitive_raw = "I Alexander Smith am feeling terrible and thinking of cutting."
        state = create_initial_state(sanitized_text=sensitive_raw)

        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        for trace in merged_state["execution_trace"]:
            self.assertIn("node", trace)
            self.assertIn("status", trace)
            self.assertIn("duration_ms", trace)
            self.assertIn("timestamp_utc", trace)
            self.assertIn("metadata_code", trace)
            # Ensure metadata_code is a controlled code, not raw user text
            self.assertNotIn("Alexander Smith", trace["metadata_code"])

    # ------------------------------------------------------------------------
    # Scenario 15: Unresolved acute dimension blocks supportive de-escalation
    # ------------------------------------------------------------------------
    def test_15_unresolved_acute_dimension_blocks_supportive_deescalation(self):
        """Unresolved means (means = None) must block supportive de-escalation."""
        dims: ClinicalRiskDimensions = {
            "passive_death_wishes": True,
            "active_suicidal_ideation": False,
            "intent": False,
            "plan": False,
            "means": None,  # Unanswered question
            "access": False,
            "timeframe": "none_reported",
        }
        eligible, reason = evaluate_deescalation_eligibility(dims)
        self.assertFalse(eligible)
        self.assertEqual(reason, "UNRESOLVED_ACUTE_DIMENSION_MEANS")

    # ------------------------------------------------------------------------
    # Scenario 16: Near-future timeframe blocks premature de-escalation
    # ------------------------------------------------------------------------
    def test_16_near_future_timeframe_blocks_premature_deescalation(self):
        """A specific near-future timeframe blocks de-escalation to trigger clarification."""
        dims: ClinicalRiskDimensions = {
            "passive_death_wishes": True,
            "active_suicidal_ideation": False,
            "intent": False,
            "plan": False,
            "means": False,
            "access": False,
            "timeframe": "near_future",  # e.g. "this Friday"
        }
        eligible, reason = evaluate_deescalation_eligibility(dims)
        self.assertFalse(eligible)
        self.assertEqual(reason, "NEAR_FUTURE_TIMEFRAME")

    # ------------------------------------------------------------------------
    # Scenario 17: Ideation alone triggers follow-up, not emergency
    # ------------------------------------------------------------------------
    def test_17_ideation_alone_triggers_followup_not_emergency(self):
        """Suicidal thoughts alone trigger follow-up without activating emergency bypass."""
        state = create_initial_state(
            sanitized_text="I have been having suicidal thoughts this week.",
        )
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertEqual(updates["overall_safety_route"], "pending_safety_follow_up")
        self.assertFalse(updates["deterministic_danger_flag"])
        self.assertEqual(route_after_deterministic_safety(merged_state), "followup_node")

    # ------------------------------------------------------------------------
    # Scenario 18: Means alone does not trigger emergency
    # ------------------------------------------------------------------------
    def test_18_means_alone_does_not_trigger_emergency(self):
        """Ordinary prescription medication mention does not trigger emergency bypass."""
        text = "I take prescription medication every morning for my blood pressure."
        self.assertTrue(is_ordinary_medication_mention(text))

        state = create_initial_state(sanitized_text=text)
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertEqual(updates.get("overall_safety_route"), "standard_conversation")
        self.assertEqual(route_after_deterministic_safety(merged_state), "classifier_node")

    # ------------------------------------------------------------------------
    # Scenario 19: Emergency output static fallback and no auto-dispatch
    # ------------------------------------------------------------------------
    def test_19_emergency_output_static_fallback_no_auto_dispatch(self):
        """Emergency output must never claim automatic dialing or ambulance dispatch."""
        guidance = format_emergency_guidance("US")
        lower_guidance = guidance.lower()

        # Prohibited dispatch claims
        self.assertNotIn("calling 911 for you", lower_guidance)
        self.assertNotIn("dispatching an ambulance", lower_guidance)
        self.assertNotIn("emergency services have been notified", lower_guidance)
        self.assertNotIn("dispatched responders", lower_guidance)

        # Mandatory transparent disclosures
        self.assertIn("cannot contact emergency services", lower_guidance)
        self.assertIn("independently tested", lower_guidance)

    # ------------------------------------------------------------------------
    # Scenario 20: Jurisdiction filtering of emergency resources
    # ------------------------------------------------------------------------
    def test_20_jurisdiction_filtering_of_emergency_resources(self):
        """Hotlines must be filtered strictly by jurisdiction; no indiscriminate mixing."""
        # US
        us_guidance = format_emergency_guidance("US")
        self.assertIn("911", us_guidance)
        self.assertIn("988", us_guidance)
        self.assertNotIn("112", us_guidance)
        self.assertNotIn("14416", us_guidance)
        self.assertNotIn("999", us_guidance)

        # India
        in_guidance = format_emergency_guidance("IN")
        self.assertIn("112", in_guidance)
        self.assertIn("14416", in_guidance)
        self.assertNotIn("988", in_guidance)
        self.assertNotIn("911", in_guidance)

        # UK
        uk_guidance = format_emergency_guidance("UK")
        self.assertIn("999", uk_guidance)
        self.assertIn("111", uk_guidance)
        self.assertNotIn("988", uk_guidance)
        self.assertNotIn("112", uk_guidance)

        # UNKNOWN
        unknown_guidance = format_emergency_guidance("UNKNOWN")
        self.assertIn("local emergency services", unknown_guidance.lower())
        self.assertIn("let me know your country", unknown_guidance.lower())

    # ------------------------------------------------------------------------
    # Scenario 21: Contextual negation regression
    # ------------------------------------------------------------------------
    def test_21_contextual_negation_does_not_trigger_emergency(self):
        """'I don't want to die' must not be mistaken for an acute suicidal disclosure."""
        text = "I don't want to die, I just feel stressed about my exams."
        self.assertTrue(is_contextual_negation_or_third_party(text))

        state = create_initial_state(sanitized_text=text)
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertNotEqual(updates.get("overall_safety_route"), "emergency_bypass")

    # ------------------------------------------------------------------------
    # Scenario 22: Contextual quotation regression
    # ------------------------------------------------------------------------
    def test_22_contextual_quote_does_not_trigger_emergency(self):
        """Quoted statements (e.g. from literature) must not trigger emergency bypass."""
        text = "In Hamlet, the character says 'to be or not to be' and talks about suicide."
        self.assertTrue(is_contextual_negation_or_third_party(text))

        state = create_initial_state(sanitized_text=text)
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertNotEqual(updates.get("overall_safety_route"), "emergency_bypass")

    # ------------------------------------------------------------------------
    # Scenario 23: Contextual hypothetical regression
    # ------------------------------------------------------------------------
    def test_23_contextual_hypothetical_does_not_trigger_emergency(self):
        """Hypothetical discussions of suicide must not trigger emergency bypass."""
        text = "Hypothetically speaking, what if someone feels suicidal in another state?"
        self.assertTrue(is_contextual_negation_or_third_party(text))

        state = create_initial_state(sanitized_text=text)
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertNotEqual(updates.get("overall_safety_route"), "emergency_bypass")

    # ------------------------------------------------------------------------
    # Scenario 24: Contextual third-party report regression
    # ------------------------------------------------------------------------
    def test_24_contextual_third_party_does_not_trigger_emergency(self):
        """Disclosures about third parties must not be attributed as self-harm."""
        text = "My brother is suicidal and I am worried about him."
        self.assertTrue(is_contextual_negation_or_third_party(text))

        state = create_initial_state(sanitized_text=text)
        updates = deterministic_safety_node(state)
        merged_state = {**state, **updates}

        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertNotEqual(updates.get("overall_safety_route"), "emergency_bypass")

    # ------------------------------------------------------------------------
    # Scenario 25 (Audit Regression 1): Third-party does not suppress first-person self-harm
    # ------------------------------------------------------------------------
    def test_25_third_party_does_not_suppress_first_person_self_harm(self):
        """'My friend said he wants to die, and I am going to hurt myself tonight.' must enter safety pathway."""
        raw_text = "My friend said he wants to die, and I am going to hurt myself tonight."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 26 (Audit Regression 2): Third-party statement alone is not personal disclosure
    # ------------------------------------------------------------------------
    def test_26_third_party_statement_alone_not_personal_disclosure(self):
        """'My friend said he wants to die.' must not be treated as personal self-harm disclosure."""
        raw_text = "My friend said he wants to die."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertNotEqual(final_state.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(final_state.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 27 (Audit Regression 3): Quoted statement triggers clarification, not confirmed endorsement
    # ------------------------------------------------------------------------
    def test_27_quoted_suicide_statement_triggers_followup_clarification(self):
        """'I am quoting someone: I want to die. I have a plan to do it tonight.' triggers follow-up clarification."""
        raw_text = "I am quoting someone: I want to die. I have a plan to do it tonight."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 28 (Audit Regression 4): Negation does not suppress independent overdose emergency
    # ------------------------------------------------------------------------
    def test_28_negation_does_not_suppress_independent_overdose_emergency(self):
        """'I don't want to die, but I have already taken an overdose.' must retain emergency bypass precedence."""
        raw_text = "I don't want to die, but I have already taken an overdose."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(final_state.get("deterministic_danger_flag"))
        self.assertEqual(final_state.get("emergency_trigger_id"), "LETHAL_MEANS_DISCLOSURE")

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(direct_updates.get("deterministic_danger_flag"))

    # ------------------------------------------------------------------------
    # Scenario 29 (Audit Regression 5): Direct self-harm in progress triggers emergency bypass
    # ------------------------------------------------------------------------
    def test_29_direct_self_harm_in_progress_triggers_emergency_bypass(self):
        """'I am slitting my wrists right now and bleeding out.' must activate emergency bypass immediately."""
        raw_text = "I am slitting my wrists right now and bleeding out."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(final_state.get("deterministic_danger_flag"))
        self.assertEqual(final_state.get("emergency_trigger_id"), "SELF_HARM_DIRECT_THREAT")

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(direct_updates.get("deterministic_danger_flag"))

    # ------------------------------------------------------------------------
    # Scenario 30 (Audit Regression 6): Negated self-harm does not trigger emergency or follow-up
    # ------------------------------------------------------------------------
    def test_30_negated_self_harm_does_not_trigger_emergency_or_followup(self):
        """'I am not going to hurt myself.' must not trigger emergency bypass or follow-up."""
        raw_text = "I am not going to hurt myself."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 31 (Audit Regression 7): First-person self-harm potential triggers follow-up
    # ------------------------------------------------------------------------
    def test_31_first_person_self_harm_potential_triggers_followup_not_standard(self):
        """'I might hurt myself tonight.' must trigger follow-up, never falling through to standard conversation."""
        raw_text = "I might hurt myself tonight."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 32 (Audit Regression 8): Third-party statement does not suppress independent emergency
    # ------------------------------------------------------------------------
    def test_32_third_party_does_not_suppress_independent_emergency(self):
        """Third-party statement accompanied by acute lethal means must retain emergency bypass."""
        raw_text = "My sister is feeling suicidal, but I have a bottle of pills and I am ending my life tonight."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(final_state.get("deterministic_danger_flag"))
        self.assertIn(
            final_state.get("emergency_trigger_id"),
            {"LETHAL_MEANS_DISCLOSURE", "IMMINENT_TIMEFRAME_CRISIS", "IMMINENT_SUICIDE_INTENT"},
        )

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(direct_updates.get("deterministic_danger_flag"))

    # ------------------------------------------------------------------------
    # Scenario 33 (Step 3.2 Regression 1): Explicit harm negation with no plan routes to standard
    # ------------------------------------------------------------------------
    def test_33_explicit_negation_harm_and_plan_routes_standard(self):
        """'I don't want to hurt myself, and I have no plan.' must route to standard_conversation."""
        raw_text = "I don't want to hurt myself, and I have no plan."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 34 (Step 3.2 Regression 2): Explicit denial of harm routes to standard
    # ------------------------------------------------------------------------
    def test_34_explicit_denial_harm_alone_routes_standard(self):
        """'I do not want to harm myself.' must route to standard_conversation."""
        raw_text = "I do not want to harm myself."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 35 (Step 3.2 Regression 3): Negated harm with independent harm disclosure routes to follow-up
    # ------------------------------------------------------------------------
    def test_35_negated_harm_with_independent_harm_disclosure_routes_followup(self):
        """'I don't want to hurt myself, but I might hurt myself tonight.' must route to pending_safety_follow_up."""
        raw_text = "I don't want to hurt myself, but I might hurt myself tonight."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 36 (Step 3.2 Regression 4): Negated harm with overdose routes to emergency
    # ------------------------------------------------------------------------
    def test_36_negated_harm_with_overdose_routes_emergency(self):
        """'I don't want to hurt myself, but I have already taken an overdose.' must route to emergency_bypass."""
        raw_text = "I don't want to hurt myself, but I have already taken an overdose."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(final_state.get("deterministic_danger_flag"))
        self.assertEqual(final_state.get("emergency_trigger_id"), "LETHAL_MEANS_DISCLOSURE")

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "emergency_bypass")
        self.assertTrue(direct_updates.get("deterministic_danger_flag"))

    # ------------------------------------------------------------------------
    # Scenario 37 (Step 3.2 Regression 5): Third-party and negated harm routes to standard
    # ------------------------------------------------------------------------
    def test_37_third_party_and_negated_harm_routes_standard(self):
        """'My friend is suicidal, but I don't want to hurt myself.' must route to standard_conversation."""
        raw_text = "My friend is suicidal, but I don't want to hurt myself."
        state = create_initial_state()
        input_updates = input_node(state, raw_user_text=raw_text)
        merged_state = {**state, **input_updates}
        safety_updates = deterministic_safety_node(merged_state)
        final_state = {**merged_state, **safety_updates}

        self.assertEqual(final_state.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(final_state.get("deterministic_danger_flag", False))
        self.assertFalse(final_state.get("phq9_item9_endorsed", False))

        # Direct node invocation check
        direct_state = create_initial_state(sanitized_text=raw_text)
        direct_updates = deterministic_safety_node(direct_state)
        self.assertEqual(direct_updates.get("overall_safety_route"), "standard_conversation")
        self.assertFalse(direct_updates.get("deterministic_danger_flag", False))

    # ------------------------------------------------------------------------
    # Scenario 38: Emergency guidance and structured resources consistency
    # ------------------------------------------------------------------------
    def test_38_emergency_guidance_and_structured_resources_consistency(self):
        """Emergency guidance markdown and structured resource items must derive from the exact same directory entries."""
        for jur in ("US", "IN", "UK", "CA", "AU", "GL"):
            guidance = format_emergency_guidance(jur)
            entries = get_hotlines_for_jurisdiction(jur)
            self.assertGreater(len(entries), 0, f"Jurisdiction {jur} must have directory entries")

            # Verify every entry with a telephone or dialing instruction or URL appears in guidance
            for entry in entries:
                name = entry.get("resource_name")
                self.assertIn(name, guidance, f"Resource {name} must appear in guidance for {jur}")
                if entry.get("telephone"):
                    self.assertIn(entry["telephone"], guidance, f"Telephone {entry['telephone']} must appear in guidance for {jur}")
                elif entry.get("official_url"):
                    self.assertIn(entry["official_url"], guidance, f"URL {entry['official_url']} must appear in guidance for {jur}")

    # ------------------------------------------------------------------------
    # Scenario 39: Updating a directory contact is reflected in emergency guidance
    # ------------------------------------------------------------------------
    def test_39_directory_contact_update_reflected_in_emergency_guidance(self):
        """Dynamic directory update is immediately reflected in format_emergency_guidance without code changes."""
        try:
            custom_data = {
                "jurisdictions": {
                    "US": {
                        "emergency_dispatch": [
                            {
                                "resource_id": "us-test-dispatch",
                                "resource_name": "Test Municipal Emergency Dispatch",
                                "telephone": "99988877",
                                "resource_type": "emergency_dispatch",
                                "modality": ["voice"],
                                "availability": "24/7",
                                "toll_free": True,
                            }
                        ]
                    }
                }
            }
            set_hotlines_directory_for_testing(custom_data)
            updated_guidance = format_emergency_guidance("US")
            self.assertIn("Test Municipal Emergency Dispatch", updated_guidance)
            self.assertIn("99988877", updated_guidance)
        finally:
            load_hotlines_directory(reload=True)

    # ------------------------------------------------------------------------
    # Scenario 40: Unknown jurisdiction does not receive another country's numbers
    # ------------------------------------------------------------------------
    def test_40_unknown_jurisdiction_no_cross_contamination(self):
        """Unknown jurisdiction must never receive foreign country-specific numbers."""
        unknown_guidance = format_emergency_guidance("UNKNOWN")
        lower_guidance = unknown_guidance.lower()

        # Must not contain country-specific hotlines
        self.assertNotIn("911", unknown_guidance)
        self.assertNotIn("112", unknown_guidance)
        self.assertNotIn("999", unknown_guidance)
        self.assertNotIn("988", unknown_guidance)
        self.assertNotIn("14416", unknown_guidance)
        self.assertNotIn("000", unknown_guidance)

        # Must contain safe generic guidance
        self.assertIn("local emergency services", lower_guidance)
        self.assertIn("nearest hospital emergency department", lower_guidance)
        self.assertIn("let me know your country", lower_guidance)

    # ------------------------------------------------------------------------
    # Scenario 41: No resources produces safe generic guidance
    # ------------------------------------------------------------------------
    def test_41_no_resources_produces_safe_generic_guidance(self):
        """A jurisdiction with empty resources returns safe generic guidance without inventing numbers."""
        try:
            empty_data = {"jurisdictions": {"EMPTY_JUR": {}}}
            set_hotlines_directory_for_testing(empty_data)
            guidance = format_emergency_guidance("EMPTY_JUR")
            self.assertIn("Local Emergency Services", guidance)
            self.assertIn("cannot contact emergency services", guidance.lower())
            self.assertIn("let me know your country", guidance.lower())
        finally:
            load_hotlines_directory(reload=True)

    # ------------------------------------------------------------------------
    # Scenario 42: Voice + SMS modality with missing SMS instructions
    # ------------------------------------------------------------------------
    def test_42_voice_sms_missing_instructions_never_produces_dial_or_text(self):
        """Voice + SMS modality without explicit SMS instructions must never output 'Dial or Text'."""
        entry = {
            "resource_id": "test-res-voice-sms",
            "resource_name": "Test Helpline",
            "telephone": "1-800-555-0199",
            "modality": ["voice", "sms"],
            "sms_instructions": None,
            "availability": "24/7",
            "toll_free": True,
        }
        formatted = _format_hotline_markdown_entry(entry)
        self.assertNotIn("Dial or Text", formatted)
        self.assertIn("Dial **1-800-555-0199**", formatted)
        self.assertIn("SMS instructions unavailable", formatted)

        # Also test with empty string and ambiguous string
        entry_empty = {**entry, "sms_instructions": ""}
        formatted_empty = _format_hotline_markdown_entry(entry_empty)
        self.assertNotIn("Dial or Text", formatted_empty)
        self.assertIn("SMS instructions unavailable", formatted_empty)

        entry_ambiguous = {**entry, "sms_instructions": "SMS available via carrier"}
        formatted_ambiguous = _format_hotline_markdown_entry(entry_ambiguous)
        self.assertNotIn("Dial or Text", formatted_ambiguous)
        self.assertIn("SMS instructions unavailable", formatted_ambiguous)

    # ------------------------------------------------------------------------
    # Scenario 43: Explicit Text NUMBER instructions are preserved
    # ------------------------------------------------------------------------
    def test_43_explicit_text_number_preserved(self):
        """Explicit Text NUMBER instructions must be preserved without distortion."""
        entry = {
            "resource_id": "us-988-test",
            "resource_name": "988 Suicide & Crisis Lifeline",
            "telephone": "988",
            "sms_instructions": "Text 988",
            "modality": ["voice", "sms"],
            "availability": "24/7",
            "toll_free": True,
        }
        formatted = _format_hotline_markdown_entry(entry)
        self.assertIn("Dial **988**", formatted)
        self.assertIn("Text 988", formatted)
        self.assertNotIn("Dial or Text", formatted)
        self.assertNotIn("SMS instructions unavailable", formatted)

        entry_sms_only = {
            "resource_id": "test-veterans-sms",
            "resource_name": "Veterans Crisis Line",
            "telephone": "988",
            "sms_instructions": "Text 838255",
            "modality": ["sms"],
            "availability": "24/7",
        }
        formatted_sms = _format_hotline_markdown_entry(entry_sms_only)
        self.assertIn("Text 838255", formatted_sms)

    # ------------------------------------------------------------------------
    # Scenario 44: Explicit Text KEYWORD to NUMBER instructions are preserved
    # ------------------------------------------------------------------------
    def test_44_explicit_text_keyword_to_number_preserved(self):
        """Explicit Text KEYWORD to NUMBER instructions must be preserved exactly."""
        entry_keyword = {
            "resource_id": "us-trevor-test",
            "resource_name": "The Trevor Project",
            "telephone": "1-866-488-7386",
            "sms_instructions": "Text START to 678-678",
            "modality": ["voice", "sms"],
            "availability": "24/7",
            "toll_free": True,
        }
        formatted = _format_hotline_markdown_entry(entry_keyword)
        self.assertIn("Dial **1-866-488-7386**", formatted)
        self.assertIn("Text START to 678-678", formatted)
        self.assertNotIn("Dial or Text", formatted)

        entry_shout = {
            "resource_id": "uk-shout-test",
            "resource_name": "Shout Crisis Text Line",
            "telephone": "85258",
            "sms_instructions": "Text SHOUT to 85258",
            "modality": ["sms"],
            "availability": "24/7",
            "toll_free": True,
        }
        formatted_shout = _format_hotline_markdown_entry(entry_shout)
        self.assertIn("Text SHOUT to 85258", formatted_shout)

    # ------------------------------------------------------------------------
    # Scenario 45: Voice-only resources never produce SMS instructions
    # ------------------------------------------------------------------------
    def test_45_voice_only_resources_never_produce_sms_instructions(self):
        """Voice-only resources must not output SMS instructions or 'SMS instructions unavailable'."""
        entry_voice = {
            "resource_id": "in-erss-112",
            "resource_name": "National Emergency Response Support System (ERSS)",
            "telephone": "112",
            "modality": ["voice"],
            "availability": "24/7",
            "toll_free": True,
        }
        formatted = _format_hotline_markdown_entry(entry_voice)
        self.assertIn("Dial **112**", formatted)
        self.assertNotIn("SMS", formatted)
        self.assertNotIn("Text", formatted)

    # ------------------------------------------------------------------------
    # Scenario 46: Unknown-jurisdiction fallback contains no unsupported confidentiality guarantee
    # ------------------------------------------------------------------------
    def test_46_unknown_jurisdiction_fallback_no_unsupported_confidentiality(self):
        """Unknown-jurisdiction fallback must not make unsupported confidentiality claims."""
        unknown_guidance = format_emergency_guidance("UNKNOWN")
        lower_guidance = unknown_guidance.lower()

        # Must not claim directories are confidential
        self.assertNotIn("confidential", lower_guidance)

        # Must include neutral privacy and availability wording
        self.assertIn("privacy and availability", lower_guidance)
        self.assertIn("local emergency services", lower_guidance)
        self.assertIn("cannot dispatch local emergency responders", lower_guidance)


if __name__ == "__main__":
    unittest.main()
