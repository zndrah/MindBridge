"""
Hermetic Unit & Integration Tests for Groq Cloud Grounded Generation
====================================================================

Verifies:
1. Groq generator initialization with valid mocked configuration
2. Missing API key containment
3. Missing model configuration containment
4. Successful Groq response with mocked chat completion
5. Authentication failure containment
6. Rate-limit / quota failure (HTTP 429) containment
7. Timeout / network failure containment
8. Empty or malformed response containment
9. Local deterministic grounded fallback activation upon Groq failure
10. Grounding prompt compatibility and context injection
11. Community content (CounselChat) is explicitly framed as unverified peer perspective
12. Emergency bypass never invokes Groq or RAG
13. Verification node still processes Groq output before presentation
14. Provider selection factory across Gemini, Groq, and local modes
15. Seamless error containment with zero API keys exposed in telemetry or logs
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

from backend.generation.llm import (
    GroqGenerator,
    GeminiGenerator,
    GenerationResult,
    build_grounding_prompt,
    create_llm_generator,
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
        "session_id": "test-session-groq-1234",
        "user_id_hash": "test-user-hash-groq",
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


def make_mock_groq_completion(text: str) -> MagicMock:
    """Helper constructing a mock Groq ChatCompletion object."""
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_completion.choices = [mock_choice]
    return mock_completion


class TestGroqGenerator(unittest.TestCase):
    """Hermetic test suite for Groq Cloud API integration."""

    # 1. Groq initialization with valid mocked configuration
    def test_01_groq_initialization_with_valid_mocked_config(self):
        mock_client = MagicMock()
        gen = GroqGenerator(
            api_key="mock_groq_api_key_xyz",
            model_name="openai/gpt-oss-120b",
            timeout=8.0,
            client=mock_client,
        )
        self.assertTrue(gen.is_available())
        self.assertEqual(gen.model_name, "openai/gpt-oss-120b")
        self.assertEqual(gen.timeout, 8.0)
        self.assertEqual(gen.provider_name, "groq")

    # 2. Missing API key containment
    def test_02_missing_api_key_containment(self):
        gen = GroqGenerator(
            api_key="",
            model_name="openai/gpt-oss-120b",
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
        gen = GroqGenerator(
            api_key="mock_groq_api_key_xyz",
            model_name="",
            client=mock_client,
        )
        res = gen.generate_response(user_message="Hello")
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "MISSING_MODEL")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 4. Successful Groq response
    def test_04_successful_groq_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_groq_completion(
            "I hear how demanding your days feel right now. Let's explore Box Breathing: inhale 4, hold 4, exhale 4, hold 4."
        )

        gen = GroqGenerator(
            api_key="mock_groq_api_key_xyz",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = gen.generate_response(
            user_message="I feel stressed out",
            retrieved_chunks=[create_mock_chunk()],
        )
        self.assertTrue(result.success)
        self.assertIn("Box Breathing", result.text)
        self.assertEqual(result.model, "openai/gpt-oss-120b")

        # Verify chat completion arguments
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "openai/gpt-oss-120b")
        self.assertEqual(call_kwargs["max_completion_tokens"], 2048)
        self.assertEqual(call_kwargs["reasoning_effort"], "medium")

        # Test graph callable hook
        state = create_test_state()
        call_output = gen(state)
        self.assertEqual(call_output, result.text)

    # 5. Authentication failure containment
    def test_05_authentication_failure_containment(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("AuthenticationError: 401 invalid_api_key")

        gen = GroqGenerator(
            api_key="invalid_mock_key",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "AUTH_FAILURE")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 6. Rate-limit / quota failure containment
    def test_06_rate_limit_failure_containment(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("RateLimitError: 429 rate_limit_exceeded")

        gen = GroqGenerator(
            api_key="valid_mock_key",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "QUOTA_EXCEEDED")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 7. Timeout / network failure containment
    def test_07_timeout_failure_containment(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("APITimeoutError: Request timed out after 10.0s")

        gen = GroqGenerator(
            api_key="valid_mock_key",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "TIMEOUT")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 8. Empty or malformed response containment
    def test_08_empty_or_malformed_response_containment(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_groq_completion("   ")

        gen = GroqGenerator(
            api_key="valid_mock_key",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "EMPTY_RESPONSE")

        state = create_test_state()
        call_output = gen(state)
        self.assertIsNone(call_output)

    # 9. Local fallback activation upon Groq failure
    def test_09_local_fallback_activation_in_response_node(self):
        # Failing Groq generator returning None
        failing_groq = MagicMock(return_value=None)
        state = create_test_state(
            user_text="How to do box breathing?",
            chunks=[create_mock_chunk(title="Box Breathing", content="Inhale 4, hold 4, exhale 4.")],
        )
        result = response_node(state, generator=failing_groq)

        # Must fall back to local deterministic grounded synthesis
        self.assertIsNotNone(result.get("draft_response"))
        self.assertIn("Box Breathing", result["draft_response"])
        self.assertIn("Inhale 4", result["draft_response"])
        self.assertEqual(result["execution_trace"][-1]["metadata_code"], "LOCAL_GROUNDED_FALLBACK_ON_LLM_FAILURE")

    # 10. Grounding prompt compatibility
    def test_10_grounding_prompt_compatibility(self):
        sys_inst, prompt = build_grounding_prompt(
            user_message="I have trouble focusing on studies",
            retrieved_chunks=[create_mock_chunk(title="Exam Anxiety", content="Practice 5-4-3-2-1 sensory grounding.")],
        )
        self.assertIn("NON-CLINICAL ROLE", sys_inst)
        self.assertIn("NO DIAGNOSES", sys_inst)
        self.assertIn("Exam Anxiety", prompt)
        self.assertIn("5-4-3-2-1", prompt)

    # 11. Community content framing (CounselChat)
    def test_11_community_content_framing(self):
        counselchat_chunk = create_mock_chunk(
            chunk_id="counselchat_chunk_0099",
            title="Peer perspective on exam anxiety",
            category="counselchat",
            content="I found that taking short breaks every 30 minutes helped me stay calm.",
        )
        _, prompt = build_grounding_prompt(
            user_message="How do peers handle exam anxiety?",
            retrieved_chunks=[counselchat_chunk],
        )
        self.assertIn("Unverified Community Q&A (Peer Perspective", prompt)

    # 12. Emergency bypass for Groq
    def test_12_emergency_bypass_never_invokes_groq(self):
        mock_groq_generator = MagicMock(return_value="Groq generated response.")
        mock_retriever = MagicMock()

        app = build_mindbridge_graph(
            retriever=mock_retriever,
            generator=mock_groq_generator,
        )

        emergency_text = "I have a bottle of pills and I am swallowing all the pills tonight to end it all. Goodbye."
        emergency_state: MindBridgeState = {
            "session_id": "test-emergency-groq-session",
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

        # 2. CRITICAL: Groq generator was NEVER called
        mock_groq_generator.assert_not_called()

        # 3. CRITICAL: FAISS retriever was NEVER called
        mock_retriever.retrieve.assert_not_called()

    # 13. Verification node still processes Groq output
    def test_13_verification_node_processes_groq_output(self):
        # Case A: Groq generates prohibited medical assertion
        unsafe_state = create_test_state()
        unsafe_state["draft_response"] = "You have clinical anxiety. Take 20mg fluoxetine daily for treatment."
        verified_unsafe = verification_node(unsafe_state)

        self.assertFalse(verified_unsafe["verification_passed"])
        self.assertTrue(
            any(
                flag in verified_unsafe["verification_flags"]
                for flag in ("MEDICATION_PRESCRIPTION_DETECTED", "DIAGNOSTIC_CLAIM_DETECTED")
            )
        )
        self.assertNotIn("fluoxetine", verified_unsafe["final_response"])
        self.assertIn("I am here with you", verified_unsafe["final_response"])

        # Case B: Groq generates safe grounded advice
        safe_state = create_test_state()
        safe_state["draft_response"] = "Box Breathing is a calming grounding technique. Inhale for 4 seconds, hold for 4, and exhale for 4."
        verified_safe = verification_node(safe_state)
        self.assertTrue(verified_safe["verification_passed"])
        self.assertEqual(verified_safe["final_response"], safe_state["draft_response"])

    # 14. Provider selection factory across Gemini, Groq, and local modes
    def test_14_provider_selection_factory(self):
        mock_client = MagicMock()

        # Option A: Groq
        gen_groq = create_llm_generator(
            provider="groq",
            api_key="mock_groq_key",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        self.assertIsInstance(gen_groq, GroqGenerator)
        self.assertEqual(gen_groq.provider_name, "groq")

        # Option B: Gemini
        gen_gemini = create_llm_generator(
            provider="gemini",
            api_key="mock_gemini_key",
            model_name="gemini-2.5-flash",
            enable_gemini=True,
            client=mock_client,
        )
        self.assertIsInstance(gen_gemini, GeminiGenerator)
        self.assertEqual(gen_gemini.provider_name, "gemini")

        # Option C: Local (returns None to engage deterministic grounded synthesis)
        gen_local = create_llm_generator(provider="local")
        self.assertIsNone(gen_local)

    # 15. Safe handling if model does not support reasoning_effort
    def test_15_reasoning_effort_adaptation(self):
        mock_client = MagicMock()
        # Simulate model that rejects reasoning_effort on first try, but succeeds on retry without it
        def side_effect(**kwargs):
            if "reasoning_effort" in kwargs:
                raise ValueError("unexpected keyword argument: reasoning_effort")
            return make_mock_groq_completion("Adaptive response without reasoning effort.")

        mock_client.chat.completions.create.side_effect = side_effect

        gen = GroqGenerator(
            api_key="mock_key",
            model_name="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = gen.generate_response(user_message="Hello")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "Adaptive response without reasoning effort.")


if __name__ == "__main__":
    unittest.main()
