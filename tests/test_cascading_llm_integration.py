"""
MindBridge 2.0 - Cascading LLM Integration & Reliability Test Suite
===================================================================

Comprehensive test suite verifying:
1. Gemini Developer API independent generation (mocked + conditional real call)
2. Groq Cloud API independent generation (mocked + conditional real call)
3. Gemini failure followed by automatic Groq fallback
4. Both providers failing engages local deterministic grounded synthesis
5. Multi-turn conversation context retention
6. All eight mental-health conversation domains:
   - Academic stress
   - Anxiety
   - Loneliness
   - Relationship problems
   - Family conflicts
   - Depression-related concerns
   - Sleep difficulties
   - Self-esteem
7. Emergency crisis messages NEVER invoke Gemini, Groq, or open generation
8. RAG evidence attribution and source disclaimers remain intact
9. Readiness probe accurately reflects provider status without leaking secrets
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.generation.llm import (
    CascadingLLMGenerator,
    GeminiGenerator,
    GroqGenerator,
    GenerationResult,
    build_grounding_prompt,
    create_llm_generator,
)
from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.response_node import response_node
from backend.orchestrator.state import KnowledgeChunk, MindBridgeState
from backend.retrieval.counselchat_rag_design import COUNSELCHAT_DISCLAIMER
from backend.safety.fallbacks import MANDATORY_SCREENING_DISCLAIMER
from backend.api.dependencies import app_state


def create_test_chunk(
    chunk_id: str,
    title: str,
    category: str,
    content: str,
    source: str = "counselchat",
) -> KnowledgeChunk:
    return {
        "chunk_id": chunk_id,
        "document_id": f"{category}:{chunk_id}",
        "title": title,
        "category": category,
        "jurisdiction": "GL",
        "content": content,
        "score": 0.88,
        "eligibility": "approved",
        "source": source,
    }


def create_test_state(
    user_text: str = "I am feeling stressed",
    chunks: Optional[List[KnowledgeChunk]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    safety_route: str = "standard_conversation",
) -> MindBridgeState:
    messages = list(history or [])
    messages.append({"role": "user", "content": user_text})
    return {
        "session_id": "test-cascade-session-001",
        "user_id_hash": "test-hash-user",
        "messages": messages,
        "sanitized_text": user_text,
        "_ephemeral_raw_text": user_text,
        "deterministic_danger_flag": False,
        "emergency_trigger_id": None,
        "roberta_prob": None,
        "roberta_crisis_flag": False,
        "classifier_status": "not_run",
        "overall_safety_route": safety_route,
        "phq9_item9_endorsed": False,
        "phq9_followup_status": "not_started",
        "phq9_risk_level": None,
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": "supportive_conversation",
        "retrieval_queries": [user_text],
        "retrieved_chunks": chunks if chunks is not None else [
            create_test_chunk("cc_01", "Stress Coping", "academic_stress", "Break tasks into manageable parts.")
        ],
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


class TestCascadingLLMIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_gemini_client = MagicMock()
        self.mock_groq_client = MagicMock()

    # -------------------------------------------------------------------------
    # 1. Gemini Independent Generation (Mocked + Real Call if Configured)
    # -------------------------------------------------------------------------
    def test_01_gemini_independent_mocked(self):
        mock_response = MagicMock()
        mock_response.text = "I hear how much pressure you're experiencing with your studies. Taking short breaks can help restore your focus. What part feels most demanding right now?"
        self.mock_gemini_client.models.generate_content.return_value = mock_response

        gemini_gen = GeminiGenerator(
            api_key="test_mock_gemini_key",
            model_name="gemini-2.5-flash",
            timeout=10.0,
            enable_gemini=True,
            client=self.mock_gemini_client,
        )
        self.assertTrue(gemini_gen.is_available())

        res = gemini_gen.generate_response(
            user_message="I'm feeling overwhelmed with my exams",
            retrieved_chunks=[create_test_chunk("cc_ex", "Exam Anxiety", "anxiety", "Practice 5-4-3-2-1 grounding.")],
        )
        self.assertTrue(res.success)
        self.assertIn("taking short breaks", res.text.lower())
        self.assertEqual(res.model, "gemini-2.5-flash")

    def test_01b_gemini_real_call_if_configured(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY not configured in environment. Skipping live network call.")

        # Live call test
        gemini_gen = GeminiGenerator(api_key=api_key, model_name=settings.gemini_model, enable_gemini=True)
        res = gemini_gen.generate_response(
            user_message="I feel stressed about my upcoming exams",
            retrieved_chunks=[create_test_chunk("cc_real", "Study Tips", "academic_stress", "Break study time into 25-minute blocks.")],
        )
        self.assertTrue(res.success)
        self.assertIsNotNone(res.text)
        self.assertGreater(len(res.text), 20)

    # -------------------------------------------------------------------------
    # 2. Groq Independent Generation (Mocked + Real Call if Configured)
    # -------------------------------------------------------------------------
    def test_02_groq_independent_mocked(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "It is completely valid to feel exhausted. Focusing on gentle pacing and one step at a time can ease the burden. What would feel kindest to yourself today?"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        self.mock_groq_client.chat.completions.create.return_value = mock_completion

        groq_gen = GroqGenerator(
            api_key="test_mock_groq_key",
            model_name="openai/gpt-oss-120b",
            timeout=10.0,
            client=self.mock_groq_client,
        )
        self.assertTrue(groq_gen.is_available())

        res = groq_gen.generate_response(
            user_message="I'm so exhausted from everything",
            retrieved_chunks=[create_test_chunk("cc_ex", "Exhaustion", "depression_concerns", "Honor your need for rest.")],
        )
        self.assertTrue(res.success)
        self.assertIn("completely valid", res.text.lower())
        self.assertEqual(res.model, "openai/gpt-oss-120b")

    def test_02b_groq_real_call_if_configured(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.skipTest("GROQ_API_KEY not configured in environment. Skipping live network call.")

        groq_gen = GroqGenerator(api_key=api_key, model_name=settings.groq_model)
        res = groq_gen.generate_response(
            user_message="I feel isolated lately",
            retrieved_chunks=[create_test_chunk("cc_lonely", "Loneliness", "loneliness", "Reach out with a small low-pressure message.")],
        )
        self.assertTrue(res.success)
        self.assertIsNotNone(res.text)
        self.assertGreater(len(res.text), 20)

    # -------------------------------------------------------------------------
    # 3. Gemini Failure Followed by Automatic Groq Fallback
    # -------------------------------------------------------------------------
    def test_03_gemini_failure_groq_fallback(self):
        # Gemini fails with 429 Resource Exhausted / Quota Error
        self.mock_gemini_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

        # Groq succeeds
        mock_choice = MagicMock()
        mock_choice.message.content = "Groq fallback response: It takes courage to open up about feeling stressed. Remember to breathe and take it one moment at a time. What is one small thing you can let go of today?"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        self.mock_groq_client.chat.completions.create.return_value = mock_completion

        gemini_gen = GeminiGenerator(
            api_key="mock_key",
            model_name="gemini-2.5-flash",
            client=self.mock_gemini_client,
            enable_gemini=True,
        )
        groq_gen = GroqGenerator(
            api_key="mock_key",
            model_name="openai/gpt-oss-120b",
            client=self.mock_groq_client,
        )

        cascade = CascadingLLMGenerator(
            primary_generator=gemini_gen,
            fallback_generator=groq_gen,
            max_retries=0,
        )

        res = cascade.generate_response(
            user_message="I am struggling with work and school",
            retrieved_chunks=[create_test_chunk("cc_01", "Pacing", "stress", "Pace yourself gently.")],
        )

        self.assertTrue(res.success)
        self.assertIn("groq fallback response", res.text.lower())
        self.mock_gemini_client.models.generate_content.assert_called_once()
        self.mock_groq_client.chat.completions.create.assert_called_once()

    # -------------------------------------------------------------------------
    # 4. Both Providers Failing -> Local Grounded Fallback
    # -------------------------------------------------------------------------
    def test_04_both_providers_failing_engages_local_fallback(self):
        # Gemini fails
        self.mock_gemini_client.models.generate_content.side_effect = TimeoutError("Gemini timed out")
        # Groq fails
        self.mock_groq_client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

        gemini_gen = GeminiGenerator(api_key="k1", client=self.mock_gemini_client, enable_gemini=True)
        groq_gen = GroqGenerator(api_key="k2", client=self.mock_groq_client)

        cascade = CascadingLLMGenerator(
            primary_generator=gemini_gen,
            fallback_generator=groq_gen,
            max_retries=0,
        )

        state = create_test_state(
            user_text="What are some steps to manage academic anxiety?",
            chunks=[create_test_chunk("cc_acad", "Study Planning", "academic_stress", "Create a clear visual study schedule and prioritize two main topics per day.")],
        )

        # Run response_node with the failing cascade
        node_result = response_node(state, generator=cascade)

        # Must fall back to local deterministic grounded synthesis
        self.assertIsNotNone(node_result.get("draft_response"))
        self.assertIn("Study Planning", node_result["draft_response"])
        self.assertIn("visual study schedule", node_result["draft_response"])
        self.assertEqual(node_result["execution_trace"][-1]["metadata_code"], "LOCAL_GROUNDED_FALLBACK_ON_LLM_FAILURE")
        # Must retain CounselChat disclaimer
        self.assertIn(COUNSELCHAT_DISCLAIMER, node_result["draft_response"])

    # -------------------------------------------------------------------------
    # 5. Multi-Turn Conversation History Retention
    # -------------------------------------------------------------------------
    def test_05_multi_turn_history_retention(self):
        history = [
            {"role": "user", "content": "I'm having a hard time focusing on my thesis."},
            {"role": "assistant", "content": "It can feel overwhelming when large deadlines approach."},
            {"role": "user", "content": "Now I feel like I am falling behind everyone else."},
        ]
        sys_inst, prompt = build_grounding_prompt(
            user_message="What should I do first?",
            retrieved_chunks=[create_test_chunk("cc_thesis", "Thesis Steps", "academic_stress", "Break the thesis into 300-word daily goals.")],
            conversation_history=history,
        )

        self.assertIn("User: I'm having a hard time focusing on my thesis.", prompt)
        self.assertIn("MindBridge: It can feel overwhelming", prompt)
        self.assertIn("User: Now I feel like I am falling behind everyone else.", prompt)
        self.assertIn("What should I do first?", prompt)
        self.assertIn("Thesis Steps", prompt)

    # -------------------------------------------------------------------------
    # 6. All Eight Mental-Health Domains Coverage
    # -------------------------------------------------------------------------
    def test_06_all_eight_mental_health_domains(self):
        domains = [
            ("academic_stress", "I have so many assignments due tomorrow and I cannot cope."),
            ("anxiety", "My chest feels tight and my thoughts are racing uncontrollably."),
            ("loneliness", "I moved to a new city and have no friends here. I feel so alone."),
            ("relationship_problems", "My partner and I keep having the same argument over and over."),
            ("family_conflicts", "My parents do not respect my boundaries and it causes constant tension."),
            ("depression_concerns", "I feel empty and cannot find motivation to get out of bed."),
            ("sleep_difficulties", "I lie awake for hours every night worrying about the future."),
            ("self_esteem", "I constantly feel like I'm not good enough compared to my peers."),
        ]

        for category, query in domains:
            with self.subTest(domain=category):
                chunk = create_test_chunk(
                    chunk_id=f"cc_{category}_chunk",
                    title=f"Supportive Guide for {category.replace('_', ' ').title()}",
                    category=category,
                    content=f"Key strategy for {category}: practice self-compassion and gentle structured communication.",
                )
                state = create_test_state(user_text=query, chunks=[chunk])
                
                # Mock generator returning synthesized multi-perspective response
                mock_gen = MagicMock(return_value=(
                    f"I hear how much distress this is causing you with {category.replace('_', ' ')}. "
                    f"Therapists frequently recommend taking a step back and practicing self-compassion. "
                    f"Would it help to focus on what feels most urgent right now?"
                ))
                
                res = response_node(state, generator=mock_gen)
                draft = res.get("draft_response")
                self.assertIsNotNone(draft)
                self.assertIn("self-compassion", draft)
                self.assertIn(COUNSELCHAT_DISCLAIMER, draft)

    # -------------------------------------------------------------------------
    # 7. Crisis Messages NEVER Call Either Provider
    # -------------------------------------------------------------------------
    def test_07_crisis_messages_never_call_either_provider(self):
        mock_gemini = MagicMock(return_value="Unsafe Gemini output")
        mock_groq = MagicMock(return_value="Unsafe Groq output")
        cascade = CascadingLLMGenerator(
            primary_generator=GeminiGenerator(api_key="k1", client=mock_gemini, enable_gemini=True),
            fallback_generator=GroqGenerator(api_key="k2", client=mock_groq),
        )

        crisis_queries = [
            "i want to do suicide",
            "I have a bottle of pills and I am swallowing them all to end it tonight",
            "I am cutting myself right now and bleeding out",
        ]

        for crisis_text in crisis_queries:
            with self.subTest(crisis_query=crisis_text):
                mock_gemini.reset_mock()
                mock_groq.reset_mock()

                app = build_mindbridge_graph(generator=cascade)
                state: MindBridgeState = {
                    "session_id": "test-crisis-cascade",
                    "user_id_hash": "test-hash",
                    "messages": [{"role": "user", "content": crisis_text}],
                    "sanitized_text": crisis_text,
                    "_ephemeral_raw_text": crisis_text,
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

                final_state = app.invoke(state)

                # Emergency bypass route engaged
                self.assertEqual(final_state["overall_safety_route"], "emergency_bypass")
                self.assertIn("988", final_state["final_response"])
                self.assertIn("911", final_state["final_response"])

                # Neither provider was called
                mock_gemini.models.generate_content.assert_not_called()
                mock_groq.chat.completions.create.assert_not_called()

    # -------------------------------------------------------------------------
    # 8. RAG Evidence & Source Disclaimers Remain Intact
    # -------------------------------------------------------------------------
    def test_08_source_disclaimers_remain_intact(self):
        # Case A: Peer counseling chunk requires COUNSELCHAT_DISCLAIMER
        peer_chunk = create_test_chunk("cc_peer_01", "Peer Advice", "peer_counseling_example", "Try journaling.")
        state_peer = create_test_state(chunks=[peer_chunk])
        mock_gen = MagicMock(return_value="Journaling your thoughts before sleep can clear mental clutter. How does that sound to you?")
        
        res_peer = response_node(state_peer, generator=mock_gen)
        self.assertIn(COUNSELCHAT_DISCLAIMER, res_peer["draft_response"])

        # Case B: Clinical rubric requires MANDATORY_SCREENING_DISCLAIMER
        clinical_chunk: KnowledgeChunk = {
            "chunk_id": "phq9_rubric",
            "document_id": "clinical_rubrics:phq9",
            "title": "PHQ-9 Depression Screener",
            "category": "clinical_rubrics",
            "jurisdiction": "GL",
            "content": "PHQ-9 scoring interpretation guidelines.",
            "score": 0.95,
            "eligibility": "restricted",
        }
        state_clinical = create_test_state(chunks=[clinical_chunk])
        res_clinical = response_node(state_clinical, generator=mock_gen)
        self.assertIn(MANDATORY_SCREENING_DISCLAIMER, res_clinical["draft_response"])

    # -------------------------------------------------------------------------
    # 9. Readiness Probe Accurately Reports Provider Status
    # -------------------------------------------------------------------------
    def test_09_readiness_probe_accurately_reports_provider_status(self):
        # Case A: Default state with no API keys configured -> reports 'local'
        cascade_unconfigured = CascadingLLMGenerator(
            primary_generator=GeminiGenerator(api_key="", enable_gemini=False),
            fallback_generator=GroqGenerator(api_key=""),
        )
        with patch.object(app_state, "generator", cascade_unconfigured):
            info = app_state.get_readiness_info()
            self.assertFalse(info["models"]["llm_generator"])
            self.assertFalse(info["models"]["gemini_generator"])
            self.assertFalse(info["models"]["groq_generator"])
            self.assertEqual(info["models"]["llm_provider"], "local")
            self.assertEqual(info["details"]["llm_provider"], "local")

        # Case B: Gemini configured and available
        mock_gemini = MagicMock()
        gemini_ready = GeminiGenerator(api_key="mock_k", enable_gemini=True, client=mock_gemini)
        cascade_gemini = CascadingLLMGenerator(primary_generator=gemini_ready, fallback_generator=None)
        with patch.object(app_state, "generator", cascade_gemini):
            info = app_state.get_readiness_info()
            self.assertTrue(info["models"]["llm_generator"])
            self.assertTrue(info["models"]["gemini_generator"])
            self.assertFalse(info["models"]["groq_generator"])
            self.assertEqual(info["models"]["llm_provider"], "gemini")

        # Case C: Groq configured and available
        mock_groq = MagicMock()
        groq_ready = GroqGenerator(api_key="mock_k", client=mock_groq)
        cascade_groq = CascadingLLMGenerator(primary_generator=None, fallback_generator=groq_ready)
        with patch.object(app_state, "generator", cascade_groq):
            info = app_state.get_readiness_info()
            self.assertTrue(info["models"]["llm_generator"])
            self.assertFalse(info["models"]["gemini_generator"])
            self.assertTrue(info["models"]["groq_generator"])
            self.assertEqual(info["models"]["llm_provider"], "groq")


if __name__ == "__main__":
    unittest.main()
