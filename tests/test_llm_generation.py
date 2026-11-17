"""
Hermetic Unit & Integration Tests for Gemini Developer API Grounded Generation
=============================================================================

Verifies:
1. Generator initialization with valid mocked configuration
2. Missing API key containment
3. Missing model configuration containment
4. Correct grounded prompt construction
5. Retrieved context passed into the prompt
6. Community content (CounselChat) is explicitly framed as unverified peer perspective
7. No fabricated citation IDs in prompt structure
8. Mock successful Gemini response
9. API authentication failure containment
10. HTTP 429 / quota failure containment
11. Network timeout containment
12. Empty or malformed response containment
13. Local deterministic grounded fallback activation upon generator failure
14. Verification node audits generated output before presentation
15. Universal emergency bypass never invokes Gemini or RAG
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

from backend.generation.llm import (
    GeminiGenerator,
    GenerationResult,
    build_grounding_prompt,
    DEFAULT_SYSTEM_INSTRUCTION,
)
from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.response_node import response_node
from backend.orchestrator.state import KnowledgeChunk, MindBridgeState
from backend.orchestrator.verification_node import verification_node


def create_mock_chunk(
    chunk_id: str = "mock_chunk_01",
    title: str = "Box Breathing Exercise",
    category: str = "somatic_regulation",
    content: str = "Inhale for 4 counts, hold for 4 counts, exhale for 4 counts, hold for 4 counts.",
    source_type: Optional[str] = None,
) -> KnowledgeChunk:
    chunk: KnowledgeChunk = {
        "chunk_id": chunk_id,
        "document_id": f"{category}:{chunk_id}",
        "title": title,
        "category": category,
        "jurisdiction": "GL",
        "content": content,
        "score": 0.89,
        "eligibility": "approved",
    }
    if source_type:
        chunk["source"] = source_type
    return chunk


def create_test_state(
    user_text: str = "I am feeling overwhelmed with stress today",
    chunks: Optional[List[KnowledgeChunk]] = None,
    overall_safety_route: str = "standard_conversation",
) -> MindBridgeState:
    return {
        "session_id": "test-session-1234",
        "user_id_hash": "test-user-hash",
        "messages": [{"role": "user", "content": user_text}],
        "sanitized_text": user_text,
        "_ephemeral_raw_text": user_text,
        "deterministic_danger_flag": False,
        "emergency_trigger_id": None,
        "roberta_prob": None,
        "roberta_crisis_flag": False,
        "classifier_status": "not_run",
        "overall_safety_route": overall_safety_route,
        "phq9_item9_endorsed": False,
        "phq9_followup_status": "not_started",
        "phq9_risk_level": None,
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": "somatic_grounding",
        "retrieval_queries": ["box breathing for stress"],
        "retrieved_chunks": chunks or [create_mock_chunk()],
        "retrieval_status": "success",
        "jurisdiction_filter": "GL",
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class TestGeminiGenerator(unittest.TestCase):
    """Hermetic test suite for Gemini Developer API integration."""

    # 1. Generator initialization with a valid mocked configuration
    def test_01_generator_initialization_with_valid_mocked_config(self):
        mock_client = MagicMock()
        gen = GeminiGenerator(
            api_key="mock_test_key_12345",
            model_name="gemini-2.5-flash",
            timeout=8.0,
            enable_gemini=True,
            client=mock_client,
        )
        self.assertTrue(gen.is_available())
        self.assertEqual(gen.model_name, "gemini-2.5-flash")
        self.assertEqual(gen.timeout, 8.0)
        self.assertTrue(gen.enable_gemini)

    # 2. Missing API key containment
    def test_02_missing_api_key_containment(self):
        gen = GeminiGenerator(
            api_key="",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
        )
        res = gen.generate_response(user_message="Hello")
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "MISSING_API_KEY")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 3. Missing model configuration containment
    def test_03_missing_model_configuration_containment(self):
        mock_client = MagicMock()
        gen = GeminiGenerator(
            api_key="mock_test_key_12345",
            model_name="",
            enable_gemini=True,
            client=mock_client,
        )
        res = gen.generate_response(user_message="Hello")
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "MISSING_MODEL")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 4. Correct prompt construction
    def test_04_correct_prompt_construction(self):
        sys_inst, prompt = build_grounding_prompt(
            user_message="I have trouble falling asleep",
            retrieved_chunks=[create_mock_chunk(title="Sleep Hygiene", content="Maintain consistent sleep schedule.")],
            conversation_history=[{"role": "user", "content": "I feel tired"}],
        )
        self.assertIn("NON-CLINICAL ROLE", sys_inst)
        self.assertIn("NO DIAGNOSES", sys_inst)
        self.assertIn("NO MEDICATIONS OR DOSAGES", sys_inst)
        self.assertIn("I have trouble falling asleep", prompt)
        self.assertIn("Sleep Hygiene", prompt)
        self.assertIn("Maintain consistent sleep schedule", prompt)
        self.assertIn("User: I feel tired", prompt)

    # 5. Retrieved context passed into the prompt
    def test_05_retrieved_context_passed_into_prompt(self):
        chunk1 = create_mock_chunk(chunk_id="chk_somatic_01", title="Box Breathing", content="4-4-4-4 rhythm.")
        chunk2 = create_mock_chunk(chunk_id="chk_psycho_02", title="Circadian Rhythm", content="Morning sunlight exposure.")
        _, prompt = build_grounding_prompt(
            user_message="Need coping mechanisms",
            retrieved_chunks=[chunk1, chunk2],
        )
        self.assertIn("chk_somatic_01", prompt)
        self.assertIn("Box Breathing", prompt)
        self.assertIn("chk_psycho_02", prompt)
        self.assertIn("Circadian Rhythm", prompt)

    # 6. Community content is explicitly framed as unverified
    def test_06_community_content_framed_as_unverified(self):
        counselchat_chunk = create_mock_chunk(
            chunk_id="counselchat_chunk_0042",
            title="Peer advice on stress",
            category="counselchat",
            content="When I feel overwhelmed, I like to write in my journal.",
        )
        clinical_chunk = create_mock_chunk(
            chunk_id="somatic_01",
            title="Box Breathing",
            category="somatic_regulation",
            content="Breathe in for 4 seconds.",
        )
        _, prompt = build_grounding_prompt(
            user_message="How do others handle stress?",
            retrieved_chunks=[counselchat_chunk, clinical_chunk],
        )
        # Verify CounselChat is explicitly labeled as unverified community Q&A
        self.assertIn("Unverified Community Q&A (Peer Perspective", prompt)
        # Verify clinical protocol is labeled approved
        self.assertIn("Approved Educational / Clinical Protocol", prompt)

    # 7. No fabricated citation IDs
    def test_07_no_fabricated_citation_ids(self):
        # Empty context test
        _, prompt = build_grounding_prompt(
            user_message="Hello assistant",
            retrieved_chunks=[],
        )
        self.assertIn("[No specific reference context retrieved", prompt)
        self.assertNotIn("Reference Item 1", prompt)
        self.assertIn("use ONLY the exact titles or Chunk IDs from the Reference Items above", prompt)

    # 8. Mock successful Gemini response
    def test_08_mock_successful_gemini_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I hear that you are feeling stressed. Taking a moment for Box Breathing can help regulate your nervous system."
        mock_client.models.generate_content.return_value = mock_response

        gen = GeminiGenerator(
            api_key="valid_mock_key",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
            client=mock_client,
        )
        result = gen.generate_response(user_message="I feel anxious")
        self.assertTrue(result.success)
        self.assertIn("Box Breathing", result.text)
        self.assertEqual(result.model, "gemini-2.5-flash")

        # Test graph callable hook
        state = create_test_state()
        call_output = gen(state)
        self.assertEqual(call_output, mock_response.text)

    # 9. API authentication failure
    def test_09_api_authentication_failure(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API_KEY_INVALID: 401 Unauthorized")

        gen = GeminiGenerator(
            api_key="invalid_mock_key",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "AUTH_FAILURE")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 10. HTTP 429 / quota failure
    def test_10_quota_failure(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

        gen = GeminiGenerator(
            api_key="valid_mock_key",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "QUOTA_EXCEEDED")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 11. Network timeout
    def test_11_network_timeout(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("Request timed out after 10.0 seconds")

        gen = GeminiGenerator(
            api_key="valid_mock_key",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "TIMEOUT")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 12. Empty response
    def test_12_empty_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_client.models.generate_content.return_value = mock_response

        gen = GeminiGenerator(
            api_key="valid_mock_key",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "EMPTY_RESPONSE")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 13. Local fallback activation in response_node
    def test_13_local_fallback_activation_in_response_node(self):
        # Generator that always fails (simulating Gemini unavailable)
        failing_generator = MagicMock(return_value=None)
        state = create_test_state(
            user_text="How can I practice box breathing?",
            chunks=[create_mock_chunk(title="Box Breathing", content="Step 1: Inhale for 4 counts.")],
        )
        result = response_node(state, generator=failing_generator)

        # Must fall back to local deterministic grounded synthesis
        self.assertIsNotNone(result.get("draft_response"))
        self.assertIn("Box Breathing", result["draft_response"])
        self.assertIn("Inhale for 4 counts", result["draft_response"])
        self.assertEqual(result["execution_trace"][-1]["metadata_code"], "LOCAL_GROUNDED_FALLBACK_ON_LLM_FAILURE")

    # 14. Verification node still processes generated output
    def test_14_verification_node_processes_generated_output(self):
        # Case A: Generated output contains diagnostic violation
        unverified_state = create_test_state()
        unverified_state["draft_response"] = "Based on your symptoms, you have Major Depressive Disorder and should take 50mg of sertraline daily."
        verified_state = verification_node(unverified_state)

        # Prohibited claim must be rejected and replaced with safe neutral fallback
        self.assertFalse(verified_state["verification_passed"])
        self.assertIn("DIAGNOSTIC_CLAIM_DETECTED", verified_state["verification_flags"])
        self.assertNotIn("sertraline", verified_state["final_response"])
        self.assertIn("I am here with you", verified_state["final_response"])

        # Case B: Generated output is safe and grounded
        safe_state = create_test_state()
        safe_state["draft_response"] = "Here is a gentle grounding exercise: Box Breathing involves inhaling for 4 seconds, holding for 4, and exhaling for 4."
        verified_safe = verification_node(safe_state)
        self.assertTrue(verified_safe["verification_passed"])
        self.assertEqual(verified_safe["final_response"], safe_state["draft_response"])

    # 15. Emergency bypass never invokes Gemini
    def test_15_emergency_bypass_never_invokes_gemini(self):
        mock_generator = MagicMock(return_value="I am a chatbot response.")
        mock_retriever = MagicMock()

        app = build_mindbridge_graph(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        emergency_text = "I have a bottle of pills and I am swallowing all the pills tonight to end it all. Goodbye."
        emergency_state: MindBridgeState = {
            "session_id": "test-emergency-session",
            "user_id_hash": "test-user-hash",
            "messages": [{"role": "user", "content": emergency_text}],
            "sanitized_text": emergency_text,
            "_ephemeral_raw_text": emergency_text,
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
            "jurisdiction_filter": "US",
            "draft_response": None,
            "verification_passed": False,
            "verification_flags": [],
            "verification_status": "not_run",
            "final_response": "",
            "error_code": None,
            "error_component": None,
            "execution_trace": [],
        }

        final_state = app.invoke(emergency_state)

        # 1. Emergency bypass engaged
        self.assertEqual(final_state["overall_safety_route"], "emergency_bypass")
        self.assertTrue(final_state["deterministic_danger_flag"])
        self.assertIn("988", final_state["final_response"])

        # 2. CRITICAL: Gemini generator was NEVER called
        mock_generator.assert_not_called()

        # 3. CRITICAL: FAISS retriever was NEVER called
        mock_retriever.retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
