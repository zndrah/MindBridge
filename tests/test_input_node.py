"""
Unit tests for MindBridge 2.0 Input Processing Node (`input_node`).

Phase 5 Step 2 Safety & Privacy Corrections Verification:
1. Valid ordinary message.
2. Empty and whitespace input.
3. Oversized input (non-crisis).
4. Structured PII masking (Email, Phone, SSN, Aadhaar, URL) with emergency hotline preservation.
5. Privacy test accuracy: Structured PII masking independent of general anonymization;
   acknowledging that arbitrary names are not regex-anonymized.
6. Exception handling: Raw exception messages and stack traces never leak into returned state.
7. Additional Test 1: Passive death wishes ("I sometimes wish I would not wake up")
   must NOT automatically activate emergency_bypass; sets pending_safety_follow_up.
8. Additional Test 2: Non-imminent suicidal ideation ("I have been having suicidal thoughts")
   must NOT automatically activate emergency_bypass; sets pending_safety_follow_up.
9. Additional Test 3: Explicit immediate danger with intent & timeframe activates emergency_bypass.
10. Additional Test 4: Existing unsafe message history fails validation, returns controlled error,
    and untrusted content is not copied into returned state.
11. Additional Test 5: Long message (>2000 chars) with explicit immediate danger near the end
    is detected prior to truncation.
12. Unicode, emoji, and multilingual handling.
"""

import sys
import unittest

sys.path.insert(0, ".")

from backend.orchestrator.input_node import (
    HARD_MAX_INPUT_LENGTH,
    MAX_INPUT_LENGTH,
    input_node,
    mask_pii,
    normalize_text,
    pre_truncation_safety_scan,
    validate_sanitized_messages,
)
from backend.orchestrator.state import MindBridgeState


def create_initial_state() -> MindBridgeState:
    """Helper to create a valid baseline MindBridgeState."""
    return {
        "session_id": "test_session_101",
        "user_id_hash": "hash_user_abc",
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
        "phq9_dimensions": {
            "passive_death_wishes": None,
            "active_suicidal_ideation": None,
            "intent": None,
            "plan": None,
            "means": None,
            "access": None,
            "timeframe": None,
        },
        "phq9_supplementary_context": {
            "protective_factors": None,
            "emotional_agitation": None,
        },
        "phq9_current_dimension": None,
        "dialog_goal": None,
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieval_status": "not_attempted",
        "jurisdiction_filter": "UNKNOWN",
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class TestInputNode(unittest.TestCase):

    def test_01_valid_ordinary_message(self):
        """Test processing a normal, well-formed psychoeducation query."""
        state = create_initial_state()
        raw_text = "I have been feeling stressed at work lately and having trouble sleeping."

        updates = input_node(state, raw_user_text=raw_text)

        # Assert expected fields
        self.assertIn("sanitized_text", updates)
        self.assertIn("messages", updates)
        self.assertIn("execution_trace", updates)

        self.assertEqual(updates["sanitized_text"], raw_text)
        self.assertEqual(len(updates["messages"]), 1)
        self.assertEqual(updates["messages"][0]["role"], "user")
        self.assertEqual(updates["messages"][0]["content"], raw_text)

        # Emergency flag should NOT be set
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertIsNone(updates.get("emergency_trigger_id"))
        self.assertNotIn("overall_safety_route", updates)

        # Execution trace
        self.assertEqual(len(updates["execution_trace"]), 1)
        trace = updates["execution_trace"][0]
        self.assertEqual(trace["node"], "input_node")
        self.assertEqual(trace["status"], "success")
        self.assertEqual(trace["metadata_code"], "NORMAL_INPUT")

    def test_02_empty_and_whitespace_input(self):
        """Test handling empty string, whitespace, and None input."""
        state = create_initial_state()

        # Test empty string
        updates_empty = input_node(state, raw_user_text="")
        self.assertEqual(updates_empty["sanitized_text"], "")
        self.assertEqual(updates_empty["error_code"], "INPUT_PARSING_FAILURE")
        self.assertEqual(updates_empty["error_component"], "input_node")
        self.assertIn("empty", updates_empty["final_response"].lower())
        self.assertEqual(updates_empty["execution_trace"][-1]["metadata_code"], "EMPTY_INPUT")

        # Test whitespace only
        updates_ws = input_node(state, raw_user_text="   \n\t   ")
        self.assertEqual(updates_ws["sanitized_text"], "")
        self.assertEqual(updates_ws["error_code"], "INPUT_PARSING_FAILURE")

        # Test None input
        updates_none = input_node(state, raw_user_text=None)
        self.assertEqual(updates_none["sanitized_text"], "")
        self.assertEqual(updates_none["error_code"], "INPUT_PARSING_FAILURE")
        self.assertEqual(updates_none["execution_trace"][-1]["metadata_code"], "INVALID_INPUT_TYPE")

    def test_03_oversized_input_non_crisis(self):
        """Test handling an oversized non-crisis input exceeding MAX_INPUT_LENGTH."""
        state = create_initial_state()
        # Create a non-crisis string of 3,500 characters
        base_sentence = "This is a detailed description of my daily mindfulness routine and walking habits. "
        raw_text = (base_sentence * 50)[:3500]

        updates = input_node(state, raw_user_text=raw_text)

        # Output text should be safely bounded to <= MAX_INPUT_LENGTH
        self.assertLessEqual(len(updates["sanitized_text"]), MAX_INPUT_LENGTH)
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "OVERSIZED_TRUNCATED")

    def test_04_structured_pii_masking_with_hotline_preservation(self):
        """
        CORRECTION 3: Explicitly test that structured PII (email, phone, SSN, Aadhaar, URL)
        is masked while emergency hotlines and short codes are preserved.
        """
        state = create_initial_state()
        raw_text = (
            "Contact me at alex.smith@example.org or visit https://myportal.example.com/info. "
            "My US phone is +1-555-234-5678 and India phone is 9876543210. "
            "My SSN is 123-45-6789 and Aadhaar is 1234 5678 9012. "
            "Call 911 or 988 or 112 or Tele-MANAS 14416 or text 741741 or 85258."
        )

        updates = input_node(state, raw_user_text=raw_text)
        sanitized = updates["sanitized_text"]

        # Structured PII MUST be masked
        self.assertNotIn("alex.smith@example.org", sanitized)
        self.assertIn("[REDACTED_EMAIL]", sanitized)

        self.assertNotIn("https://myportal.example.com/info", sanitized)
        self.assertIn("[REDACTED_URL]", sanitized)

        self.assertNotIn("555-234-5678", sanitized)
        self.assertNotIn("9876543210", sanitized)
        self.assertIn("[REDACTED_PHONE]", sanitized)

        self.assertNotIn("123-45-6789", sanitized)
        self.assertNotIn("1234 5678 9012", sanitized)
        self.assertIn("[REDACTED_ID]", sanitized)

        # Emergency lifelines and short codes MUST be preserved
        self.assertIn("911", sanitized)
        self.assertIn("988", sanitized)
        self.assertIn("112", sanitized)
        self.assertIn("14416", sanitized)
        self.assertIn("741741", sanitized)
        self.assertIn("85258", sanitized)

        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "PII_MASKED")

    def test_05_privacy_boundary_and_no_anonymization_overclaim(self):
        """
        CORRECTION 3: Verify structured PII masking independently from general anonymization.
        Confirm that raw input keys never enter state, and verify that the system does NOT
        falsely claim arbitrary names or text markers are automatically anonymized by regexes.
        """
        state = create_initial_state()
        # Ordinary name and custom marker - regex scrubbers do not perform full NER anonymization
        raw_text = "Dr. Robert attended the meeting with patient Alice regarding fatigue."

        updates = input_node(state, raw_user_text=raw_text)

        # 1. State must NEVER contain raw input keys
        self.assertNotIn("raw_user_text", updates)
        self.assertNotIn("latest_user_text", updates)

        # 2. Honest boundary verification: Arbitrary names like 'Dr. Robert' or 'Alice'
        # are NOT stripped by regex scrubbers (which target structured numeric/pattern PII).
        # We verify that regex scrubbing does not destructively distort normal clinical words.
        self.assertIn("fatigue", updates["sanitized_text"])

        # 3. Traces must never contain raw user text
        for trace_entry in updates["execution_trace"]:
            self.assertNotIn("Robert", trace_entry.get("metadata_code", ""))

    def test_06_verify_exception_handling_does_not_leak_raw_input(self):
        """
        CORRECTION 3: Verify that handled internal exceptions do not leak raw input
        or raw exception messages (str(e)) into returned state.
        """
        state = create_initial_state()
        sensitive_text = "Private medical details with SSN 123-45-6789."

        # Monkey-patch normalize_text to simulate an internal runtime failure
        import backend.orchestrator.input_node as mod
        orig_normalize = mod.normalize_text

        try:
            def broken_normalize(text):
                raise RuntimeError(f"Internal crash with raw payload: {text}")

            mod.normalize_text = broken_normalize

            updates = input_node(state, raw_user_text=sensitive_text)

            # Error code must be controlled
            self.assertEqual(updates.get("error_code"), "INPUT_PARSING_FAILURE")
            self.assertEqual(updates.get("error_component"), "input_node")

            # Must NOT leak raw exception message or sensitive text in any output
            for k, v in updates.items():
                if isinstance(v, str):
                    self.assertNotIn(sensitive_text, v)
                    self.assertNotIn("Internal crash with raw payload", v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            for sub_k, sub_v in item.items():
                                if isinstance(sub_v, str):
                                    self.assertNotIn(sensitive_text, sub_v)
                                    self.assertNotIn("Internal crash with raw payload", sub_v)

        finally:
            mod.normalize_text = orig_normalize

    def test_07_additional_1_passive_death_wishes_no_emergency_bypass(self):
        """
        ADDITIONAL TEST 1 (CORRECTION 1):
        'I sometimes wish I would not wake up' must NOT automatically activate emergency_bypass.
        Instead, it must flag pending_safety_follow_up with passive_death_wishes = True.
        """
        state = create_initial_state()
        raw_text = "I sometimes wish I would not wake up."

        updates = input_node(state, raw_user_text=raw_text)

        # Must NOT activate emergency_bypass
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertIsNone(updates.get("emergency_trigger_id"))
        self.assertNotEqual(updates.get("overall_safety_route"), "emergency_bypass")

        # Must activate structured safety follow-up as a preliminary screening trigger
        # Preliminary regex signals do NOT establish confirmed phq9_item9_endorsed or confirmed dimensions
        self.assertEqual(updates.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(updates.get("phq9_item9_endorsed", False))
        self.assertEqual(updates.get("phq9_followup_status"), "in_progress")
        self.assertNotIn("phq9_dimensions", updates)
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "SAFETY_FOLLOWUP_FLAGGED")

    def test_08_additional_2_suicidal_thoughts_no_emergency_bypass(self):
        """
        ADDITIONAL TEST 2 (CORRECTION 1):
        'I have been having suicidal thoughts' must NOT automatically activate emergency_bypass.
        Instead, it must flag pending_safety_follow_up with active_suicidal_ideation = True.
        """
        state = create_initial_state()
        raw_text = "I have been having suicidal thoughts."

        updates = input_node(state, raw_user_text=raw_text)

        # Must NOT activate emergency_bypass
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertIsNone(updates.get("emergency_trigger_id"))
        self.assertNotEqual(updates.get("overall_safety_route"), "emergency_bypass")

        # Must activate structured safety follow-up as a preliminary screening trigger
        # Preliminary regex signals do NOT establish confirmed phq9_item9_endorsed or confirmed dimensions
        self.assertEqual(updates.get("overall_safety_route"), "pending_safety_follow_up")
        self.assertFalse(updates.get("phq9_item9_endorsed", False))
        self.assertEqual(updates.get("phq9_followup_status"), "in_progress")
        self.assertNotIn("phq9_dimensions", updates)
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "SAFETY_FOLLOWUP_FLAGGED")

    def test_09_additional_3_explicit_immediate_danger_triggers_emergency_bypass(self):
        """
        ADDITIONAL TEST 3 (CORRECTION 1):
        Explicit immediate danger with active intent and immediate timeframe MUST activate emergency_bypass.
        """
        state = create_initial_state()
        raw_text = "I have a bottle of pills and I am ending my life tonight."

        updates = input_node(state, raw_user_text=raw_text)

        # MUST activate emergency_bypass
        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertIn(
            updates.get("emergency_trigger_id"),
            {"LETHAL_MEANS_DISCLOSURE", "IMMINENT_TIMEFRAME_CRISIS", "IMMINENT_SUICIDE_INTENT"},
        )
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "EMERGENCY_DETECTED")

    def test_10_additional_4_existing_unsafe_message_history_quarantined(self):
        """
        ADDITIONAL TEST 4 (CORRECTION 2):
        Never blindly copy existing state messages.
        If existing history fails validation (e.g., extra unsafe arbitrary fields,
        invalid role, non-string content), return a controlled error without copying untrusted content.
        """
        state = create_initial_state()
        # Poison state with unsafe message history containing arbitrary fields and invalid types
        state["messages"] = [
            {"role": "user", "content": "Hello", "unsafe_raw_dump": "LEAK_XYZ"},  # Extra key
            {"role": "admin", "content": "Unauthorized role"},                     # Invalid role
            {"role": "user", "content": 12345},                                    # Non-string content
        ]  # type: ignore

        raw_text = "I need some guidance."
        updates = input_node(state, raw_user_text=raw_text)

        # 1. Must return a controlled error code
        self.assertEqual(updates.get("error_code"), "STATE_TRANSITION_ERROR")
        self.assertEqual(updates.get("error_component"), "input_node")

        # 2. Must NOT copy the untrusted history
        for msg in updates.get("messages", []):
            self.assertNotIn("unsafe_raw_dump", msg)
            self.assertNotIn("LEAK_XYZ", str(msg))
            self.assertNotEqual(msg.get("role"), "admin")

        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "INVALID_MESSAGE_HISTORY")
        self.assertEqual(updates["execution_trace"][-1]["status"], "error")

    def test_11_additional_5_long_message_with_emergency_near_end(self):
        """
        ADDITIONAL TEST 5 (CORRECTION 1):
        A long message (>2,000 chars) with explicit immediate danger near the end must still
        be detected before any length reduction or truncation occurs.
        """
        state = create_initial_state()
        filler = "Walking through the park today I was reflecting on various books I have read. " * 35  # ~2,700 chars
        disclosure = "I am going to kill myself tonight. Goodbye forever."
        raw_text = filler + " " + disclosure

        self.assertGreater(len(raw_text), 2500)

        updates = input_node(state, raw_user_text=raw_text)

        # Pre-truncation scan MUST intercept explicit immediate danger regardless of length
        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertIn(
            updates.get("emergency_trigger_id"),
            {"IMMINENT_SUICIDE_INTENT", "IMMINENT_TIMEFRAME_CRISIS"},
        )
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "EMERGENCY_DETECTED")

    def test_12_unicode_emojis_and_multilingual_handling(self):
        """Verify that Unicode normalization handles emojis, accents, and non-ASCII cleanly without error."""
        state = create_initial_state()
        raw_text = "I am feeling so overwhelmed today 😔💔... café and naïve conversations don't help."

        updates = input_node(state, raw_user_text=raw_text)

        self.assertIn("😔💔", updates["sanitized_text"])
        self.assertIn("café", updates["sanitized_text"])
        self.assertEqual(updates["execution_trace"][-1]["status"], "success")
        self.assertIsNone(updates.get("error_code"))

    def test_13_crisis_disclosure_near_beginning_over_10k_chars(self):
        """Case 1: Crisis disclosure near the beginning of a message longer than 10,000 characters."""
        state = create_initial_state()
        raw_text = "I am going to kill myself tonight. " + ("Everyday thoughts on philosophy and life. " * 300)
        self.assertGreater(len(raw_text), 10000)

        updates = input_node(state, raw_user_text=raw_text)

        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(updates.get("emergency_trigger_id"), "IMMINENT_SUICIDE_INTENT")
        self.assertIsNone(updates.get("error_code"))

    def test_14_crisis_disclosure_immediately_before_10k_chars(self):
        """Case 2: Crisis disclosure immediately before character 10,000."""
        state = create_initial_state()
        filler = "The weather was calm and peaceful today. " * 240  # ~9600 chars
        padding = " " * (9900 - len(filler))
        crisis = " I have pills and I will take all of them to end my life."
        tail = " Extra trailing text after crisis disclosure."
        raw_text = (filler + padding) + crisis + tail

        crisis_pos = raw_text.find("pills")
        self.assertGreater(crisis_pos, 9850)
        self.assertLess(crisis_pos, 10000)

        updates = input_node(state, raw_user_text=raw_text)

        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(updates.get("emergency_trigger_id"), "LETHAL_MEANS_DISCLOSURE")

    def test_15_crisis_disclosure_immediately_after_10k_chars(self):
        """Case 3: Crisis disclosure immediately after character 10,000."""
        state = create_initial_state()
        filler = "The weather was calm and peaceful today. " * 255  # ~10200 chars
        crisis = " I am going to hang myself tonight."
        raw_text = filler[:10010] + crisis + (" " * 500)

        crisis_pos = raw_text.find("hang")
        self.assertGreater(crisis_pos, 10000)

        updates = input_node(state, raw_user_text=raw_text)

        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(updates.get("emergency_trigger_id"), "IMMINENT_SUICIDE_INTENT")

    def test_16_crisis_disclosure_at_very_end_of_20k_chars(self):
        """Case 4: Crisis disclosure at the very end of a 20,000-character message."""
        state = create_initial_state()
        filler = "General everyday reflection on daily habits. " * 440  # ~19800 chars
        crisis = " I am going to kill myself tonight."
        raw_text = filler + crisis

        self.assertGreater(len(raw_text), 19500)
        self.assertGreater(raw_text.find("kill"), 19000)

        updates = input_node(state, raw_user_text=raw_text)

        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(updates.get("emergency_trigger_id"), "IMMINENT_SUICIDE_INTENT")

    def test_17_normal_message_longer_than_hard_limit_exceeded(self):
        """Case 5: A normal message longer than HARD_MAX_INPUT_LENGTH is safely bounded and rejected."""
        state = create_initial_state()
        filler = "I had a very long productive day studying biology in the university library. " * 250  # ~18.7k chars
        self.assertGreater(len(filler), HARD_MAX_INPUT_LENGTH)

        updates = input_node(state, raw_user_text=filler)

        # Must NOT be flagged as emergency or follow-up, but routed to ml_risk_unassessed
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertEqual(updates.get("overall_safety_route"), "ml_risk_unassessed")

        # Must record controlled parsing error and rejection notice to prevent silent conversion
        self.assertEqual(updates.get("error_code"), "INPUT_PARSING_FAILURE")
        self.assertEqual(updates.get("error_component"), "input_node")
        self.assertEqual(updates.get("sanitized_text"), "")
        self.assertIn("exceeds the maximum allowed length", updates.get("final_response", ""))
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "HARD_LIMIT_TRUNCATED")

    def test_18_malformed_unicode_and_zero_width_spaces_crisis_detection(self):
        """Case 6: Extremely long input containing malformed Unicode or obfuscated whitespace."""
        state = create_initial_state()

        # 6a. Fullwidth Unicode crisis
        text_fullwidth = ("\u200b\xa0 \t\r\n" * 500) + " ｉ ｗｉｌｌ ｋｉｌｌ ｍｙｓｅｌｆ ｔｏｎｉｇｈｔ " + ("\x00\ufffd " * 500)
        updates_6a = input_node(state, raw_user_text=text_fullwidth)
        self.assertTrue(updates_6a.get("deterministic_danger_flag"))
        self.assertEqual(updates_6a.get("overall_safety_route"), "emergency_bypass")

        # 6b. Zero-width space inside keywords
        text_zwsp = ("hello world " * 500) + " i will kill\u200bmyself tonight " + ("more text " * 200)
        updates_6b = input_node(state, raw_user_text=text_zwsp)
        self.assertTrue(updates_6b.get("deterministic_danger_flag"))
        self.assertEqual(updates_6b.get("overall_safety_route"), "emergency_bypass")

        # 6c. Zero-width space between letters
        text_intra = ("hello world " * 500) + " i will k\u200bi\u200bl\u200bl myself tonight " + ("more text " * 200)
        updates_6c = input_node(state, raw_user_text=text_intra)
        self.assertTrue(updates_6c.get("deterministic_danger_flag"))
        self.assertEqual(updates_6c.get("overall_safety_route"), "emergency_bypass")

    def test_19_error_handling_preserves_preexisting_emergency_state(self):
        """Verifies that an unexpected exception during input processing preserves pre-existing emergency state."""
        import backend.orchestrator.input_node as mod

        state = create_initial_state()
        state["overall_safety_route"] = "emergency_bypass"
        state["deterministic_danger_flag"] = True
        state["emergency_trigger_id"] = "LETHAL_MEANS_DISCLOSURE"

        orig_norm = mod.normalize_text
        try:
            def broken_norm(t):
                raise RuntimeError("simulated error")
            mod.normalize_text = broken_norm

            updates = input_node(state, raw_user_text="ordinary text that crashes normalize")
            self.assertTrue(updates.get("deterministic_danger_flag"))
            self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
            self.assertEqual(updates.get("emergency_trigger_id"), "LETHAL_MEANS_DISCLOSURE")
            self.assertEqual(updates.get("error_code"), "INPUT_PARSING_FAILURE")
        finally:
            mod.normalize_text = orig_norm

    def test_20_adversarial_30k_crisis_at_char_100(self):
        """Case 1: 30,000-character message with crisis disclosure at character 100."""
        state = create_initial_state()
        crisis = "I am going to kill myself tonight. "
        raw_text = ("a" * 100) + crisis + ("b" * (30000 - 100 - len(crisis)))
        self.assertEqual(len(raw_text), 30000)

        updates = input_node(state, raw_user_text=raw_text)
        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(updates.get("emergency_trigger_id"), "IMMINENT_SUICIDE_INTENT")

    def test_21_adversarial_30k_crisis_at_char_24950(self):
        """Case 2: 30,000-character message with crisis disclosure immediately before character 25,000."""
        state = create_initial_state()
        filler = "The weather was calm and peaceful today. " * 607  # ~24887 chars
        crisis = " I am going to kill myself tonight."
        raw_text = (filler + " " * (24900 - len(filler))) + crisis
        raw_text += " " * (30000 - len(raw_text))

        crisis_pos = raw_text.find("kill")
        self.assertGreater(crisis_pos, 24850)
        self.assertLess(crisis_pos, 25000)

        updates = input_node(state, raw_user_text=raw_text)
        self.assertTrue(updates.get("deterministic_danger_flag"))
        self.assertEqual(updates.get("overall_safety_route"), "emergency_bypass")
        self.assertEqual(updates.get("emergency_trigger_id"), "IMMINENT_SUICIDE_INTENT")

    def test_22_adversarial_30k_crisis_at_char_25100_rejected_unassessed(self):
        """Case 3: 30,000-character message with crisis disclosure at character 25,100 (in unscanned tail)."""
        state = create_initial_state()
        filler = "Everyday reflections on ordinary life. " * 750  # ~29250 chars
        crisis = " I am going to end my life tonight."
        raw_text = filler[:25100] + crisis + (" " * (30000 - 25100 - len(crisis)))

        self.assertGreater(raw_text.find("end my life"), 25000)

        updates = input_node(state, raw_user_text=raw_text)

        # Must NOT be marked as confirmed non-crisis (never standard_conversation)
        self.assertEqual(updates.get("overall_safety_route"), "ml_risk_unassessed")
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertEqual(updates.get("error_code"), "INPUT_PARSING_FAILURE")
        self.assertEqual(updates.get("sanitized_text"), "")
        self.assertIn("exceeds the maximum allowed length", updates.get("final_response", ""))

    def test_23_adversarial_30k_crisis_at_char_29900_rejected_unassessed(self):
        """Case 4: 30,000-character message with crisis disclosure at character 29,900."""
        state = create_initial_state()
        filler = "General everyday reflection on daily habits. " * 670  # ~29480 chars
        crisis = " I am going to kill myself tonight."
        raw_text = filler[:29900] + crisis + (" " * (30000 - 29900 - len(crisis)))

        self.assertGreater(raw_text.find("kill"), 29800)

        updates = input_node(state, raw_user_text=raw_text)

        # Safely rejected as unassessed; never silently allowed into LLM generation
        self.assertEqual(updates.get("overall_safety_route"), "ml_risk_unassessed")
        self.assertFalse(updates.get("deterministic_danger_flag", False))
        self.assertEqual(updates.get("error_code"), "INPUT_PARSING_FAILURE")
        self.assertEqual(updates.get("sanitized_text"), "")

    def test_24_adversarial_api_request_size_limit_rejection(self):
        """Case 5: Message exceeding the API request size limit is rejected at schema validation."""
        from pydantic import ValidationError
        from backend.api.schemas import ChatRequest

        # Message of 10,001 characters
        with self.assertRaises(ValidationError):
            ChatRequest(message="a" * 10001)

        # Message of 30,000 characters
        with self.assertRaises(ValidationError):
            ChatRequest(message="a" * 30000)

        # Message within 10,000 characters is accepted
        req = ChatRequest(message="a" * 10000)
        self.assertEqual(len(req.message), 10000)

    def test_25_adversarial_unicode_and_zero_width_near_25k_boundary(self):
        """Case 6: Unicode and zero-width obfuscation near the 25,000-character boundary."""
        state = create_initial_state()
        filler = ("\u200b\xa0 word " * 2480)[:24800]

        # 6a. Fullwidth Unicode crisis near 25k boundary
        crisis_fw = " ｉ ｗｉｌｌ ｋｉｌｌ ｍｙｓｅｌｆ ｔｏｎｉｇｈｔ "
        text_fw = filler + crisis_fw + ("\x00\ufffd " * 800)
        updates_fw = input_node(state, raw_user_text=text_fw)
        self.assertTrue(updates_fw.get("deterministic_danger_flag"))
        self.assertEqual(updates_fw.get("overall_safety_route"), "emergency_bypass")

        # 6b. Zero-width space inside keywords near 25k boundary
        crisis_zwsp = " i will kill\u200bmyself tonight "
        text_zwsp = filler + crisis_zwsp + ("tail " * 800)
        updates_zwsp = input_node(state, raw_user_text=text_zwsp)
        self.assertTrue(updates_zwsp.get("deterministic_danger_flag"))
        self.assertEqual(updates_zwsp.get("overall_safety_route"), "emergency_bypass")

        # 6c. Zero-width space intra-character near 25k boundary
        crisis_intra = " i will k\u200bi\u200bl\u200bl myself tonight "
        text_intra = filler + crisis_intra + ("tail " * 800)
        updates_intra = input_node(state, raw_user_text=text_intra)
        self.assertTrue(updates_intra.get("deterministic_danger_flag"))
        self.assertEqual(updates_intra.get("overall_safety_route"), "emergency_bypass")

    def test_26_oversized_unscanned_input_never_enters_ordinary_generation(self):
        """Ensures that an oversized input (>10k chars) with no crisis in scanned prefix halts without LLM generation."""
        from backend.orchestrator.graph import build_mindbridge_graph

        graph = build_mindbridge_graph()
        filler = "Ordinary reflection on studying and books. " * 750  # ~31.5k chars
        crisis_at_end = " I am going to end my life tonight."
        text = filler[:26000] + crisis_at_end + (" " * (30000 - 26000 - len(crisis_at_end)))

        state = create_initial_state()
        state["_ephemeral_raw_text"] = text

        final = graph.invoke(state)

        # Must halt safely without generating ordinary LLM response or vector search
        self.assertEqual(final.get("overall_safety_route"), "ml_risk_unassessed")
        self.assertFalse(final.get("deterministic_danger_flag", False))
        self.assertEqual(final.get("error_code"), "INPUT_PARSING_FAILURE")
        self.assertIn("exceeds the maximum allowed length", final.get("final_response", ""))

        # Verify planning_node and response_node were never executed
        nodes = [t["node"] for t in final.get("execution_trace", [])]
        self.assertNotIn("planning_node", nodes)
        self.assertNotIn("response_node", nodes)
        self.assertNotIn("verification_node", nodes)


if __name__ == "__main__":
    unittest.main()
