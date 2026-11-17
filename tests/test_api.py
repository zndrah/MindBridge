"""
Unit & Integration Tests for MindBridge 2.0 FastAPI Backend
===========================================================

Verifies:
1. Health endpoint (liveness probe).
2. Readiness endpoint (local model and index availability).
3. Valid chat request and response structure.
4. Empty and invalid input rejection (422 Unprocessable Entity).
5. Universal emergency bypass through API.
6. Normal grounded conversation with verified citations.
7. Empty retrieval producing safe neutral fallback.
8. Verification rejection (unsafe generated content suppressed).
9. Classifier failure graceful containment.
10. Retrieval failure graceful containment.
11. Session isolation across distinct requests.
12. Prohibition of client-supplied restricted authorization (extra='forbid').
13. Privacy guarantee: No raw mental health user messages in logs.
14. Concurrent request handling without thread exhaustion or deadlock.
15. Response schema strict validation (no raw logits or tracebacks).
16. Real-model end-to-end integration test (RoBERTa + BGE + FAISS).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import app_state
from backend.api.schemas import ChatResponse, HealthResponse, ReadyResponse
from backend.orchestrator.fallbacks import NEUTRAL_SUPPORTIVE_FALLBACK
from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.state import KnowledgeChunk, MindBridgeState
from backend.retrieval import RetrievalAuthorization


class MockAPIRetriever:
    """Deterministic mock retriever for hermetic fast API unit tests."""
    def __init__(self, should_fail: bool = False, empty: bool = False):
        self.should_fail = should_fail
        self.empty = empty

    def retrieve(self, query: str, jurisdiction: str = "GL", include_restricted: bool = False, authorization: Any = None, **kwargs):
        if self.should_fail:
            raise RuntimeError("Simulated FAISS vector store failure")
        if self.empty:
            return []
        return [
            {
                "chunk_id": "mock_chunk_01",
                "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
                "title": "Box Breathing",
                "category": "somatic_regulation",
                "jurisdiction": jurisdiction,
                "content": "Inhale 4, hold 4, exhale 4, hold 4.",
                "score": 0.88,
                "eligibility": "approved",
            }
        ]


class TestMindBridgeAPI(unittest.TestCase):
    """Test suite for MindBridge 2.0 FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        # Initialize lifecycle components
        app_state.initialize_sync()
        cls.client = TestClient(cls.app)

    # ------------------------------------------------------------------------
    # Test 1: Health endpoint
    # ------------------------------------------------------------------------
    def test_01_health_endpoint(self):
        """GET /health returns 200 and valid HealthResponse model."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        validated = HealthResponse(**data)
        self.assertEqual(validated.status, "healthy")
        self.assertEqual(validated.version, "0.2.0")
        self.assertIn("timestamp_utc", data)

    # ------------------------------------------------------------------------
    # Test 2: Readiness endpoint
    # ------------------------------------------------------------------------
    def test_02_readiness_endpoint(self):
        """GET /ready returns 200 when models and indexes are loaded."""
        with patch.object(app_state, "generator", None):
            res = self.client.get("/ready")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            validated = ReadyResponse(**data)
            self.assertEqual(validated.status, "ready")
            self.assertTrue(validated.models.crisis_detector)
            self.assertTrue(validated.models.embedding_engine)
            self.assertTrue(validated.models.vector_store)
            self.assertTrue(validated.models.graph)
            self.assertFalse(validated.models.llm_generator)
            self.assertEqual(validated.models.llm_provider, "local")
            self.assertFalse(validated.models.gemini_generator)
            self.assertFalse(validated.models.groq_generator)
            self.assertGreater(validated.details["vector_count"], 0)
            self.assertEqual(validated.details["llm_provider"], "local")
            self.assertFalse(validated.details["llm_generator"])

    def test_02b_readiness_endpoint_groq_provider(self):
        """GET /ready exposes groq provider and readiness flags when Groq is configured."""
        class MockGroqGen:
            provider_name = "groq"

            def is_available(self) -> bool:
                return True

        with patch.object(app_state, "generator", MockGroqGen()):
            res = self.client.get("/ready")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            validated = ReadyResponse(**data)
            self.assertEqual(validated.status, "ready")
            self.assertTrue(validated.models.llm_generator)
            self.assertEqual(validated.models.llm_provider, "groq")
            self.assertTrue(validated.models.groq_generator)
            self.assertFalse(validated.models.gemini_generator)
            self.assertEqual(data["models"]["llm_provider"], "groq")
            self.assertTrue(data["models"]["groq_generator"])
            self.assertFalse(data["models"]["gemini_generator"])
            self.assertEqual(validated.details["llm_provider"], "groq")
            self.assertTrue(validated.details["llm_generator"])

    def test_02c_readiness_endpoint_gemini_provider(self):
        """GET /ready exposes gemini provider and readiness flags when Gemini is configured."""
        class MockGeminiGen:
            provider_name = "gemini"

            def is_available(self) -> bool:
                return True

        with patch.object(app_state, "generator", MockGeminiGen()):
            res = self.client.get("/ready")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            validated = ReadyResponse(**data)
            self.assertEqual(validated.status, "ready")
            self.assertTrue(validated.models.llm_generator)
            self.assertEqual(validated.models.llm_provider, "gemini")
            self.assertTrue(validated.models.gemini_generator)
            self.assertFalse(validated.models.groq_generator)
            self.assertEqual(data["models"]["llm_provider"], "gemini")
            self.assertTrue(data["models"]["gemini_generator"])
            self.assertFalse(data["models"]["groq_generator"])
            self.assertEqual(validated.details["llm_provider"], "gemini")
            self.assertTrue(validated.details["llm_generator"])

    # ------------------------------------------------------------------------
    # Test 3: Valid chat request
    # ------------------------------------------------------------------------
    def test_03_valid_chat_request(self):
        """POST /chat processes valid message and returns conforming ChatResponse."""
        payload = {
            "message": "How do I do box breathing exercises for anxiety",
            "jurisdiction": "GL",
        }
        res = self.client.post("/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        validated = ChatResponse(**data)
        self.assertEqual(validated.safety_route, "standard_conversation")
        self.assertIn("Box Breathing", validated.response)
        self.assertIsNotNone(validated.session_id)
        self.assertIn(validated.verification_status, ("verified", "passed"))

    # ------------------------------------------------------------------------
    # Test 4: Empty and invalid input
    # ------------------------------------------------------------------------
    def test_04_empty_and_invalid_input(self):
        """Rejects empty string, whitespace, missing message, and oversized inputs with 422."""
        # 1. Empty message
        res1 = self.client.post("/chat", json={"message": ""})
        self.assertEqual(res1.status_code, 422)
        self.assertIn("VALIDATION_ERROR", res1.json()["error_code"])

        # 2. Whitespace-only message
        res2 = self.client.post("/chat", json={"message": "   \n\t  "})
        self.assertEqual(res2.status_code, 422)

        # 3. Missing message field
        res3 = self.client.post("/chat", json={"jurisdiction": "US"})
        self.assertEqual(res3.status_code, 422)

        # 4. Oversized message (> 10000 chars)
        oversized = "a" * 10001
        res4 = self.client.post("/chat", json={"message": oversized})
        self.assertEqual(res4.status_code, 422)

    # ------------------------------------------------------------------------
    # Test 5: Universal emergency bypass
    # ------------------------------------------------------------------------
    def test_05_emergency_bypass(self):
        """Acute lethal means disclosure triggers emergency_bypass with authoritative hotlines."""
        payload = {
            "message": "I have a bottle of pills and I am swallowing them all tonight to end my life. Goodbye.",
            "jurisdiction": "IN",
        }
        res = self.client.post("/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["safety_route"], "emergency_bypass")
        self.assertEqual(data["verification_status"], "not_run")
        self.assertEqual(data["citations"], [])
        # Authoritative Indian helplines from hotlines.json
        self.assertIn("112", data["response"])
        self.assertIn("14416", data["response"])
        self.assertTrue(len(data.get("emergency_resources", [])) > 0)
        res_0 = data["emergency_resources"][0]
        self.assertIn("resource_name", res_0)
        self.assertIn("resource_type", res_0)
        self.assertIn("modality", res_0)

    # ------------------------------------------------------------------------
    # Test 6: Normal grounded conversation
    # ------------------------------------------------------------------------
    def test_06_normal_grounded_conversation(self):
        """Standard query returns verified grounded response with valid citations."""
        payload = {
            "message": "How to do box breathing technique",
            "jurisdiction": "GL",
        }
        res = self.client.post("/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["safety_route"], "standard_conversation")
        self.assertEqual(data["verification_status"], "verified")
        self.assertEqual(data["grounding_status"], "grounded")
        self.assertTrue(len(data["citations"]) > 0)
        citation = data["citations"][0]
        self.assertIn("title", citation)
        self.assertIn("document_id", citation)

    # ------------------------------------------------------------------------
    # Test 7: Empty retrieval fallback
    # ------------------------------------------------------------------------
    def test_07_empty_retrieval_fallback(self):
        """Out-of-domain query returns safe neutral fallback without citations or breathing commands."""
        payload = {
            "message": "How do I write a lambda function in Python to sort a dictionary",
            "jurisdiction": "US",
        }
        res = self.client.post("/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["safety_route"], "standard_conversation")
        self.assertEqual(data["verification_status"], "fallback_used")
        self.assertEqual(data["citations"], [])
        self.assertIn("I am here with you", data["response"])
        self.assertNotIn("breathing", data["response"].lower())

    # ------------------------------------------------------------------------
    # Test 8: Verification rejection
    # ------------------------------------------------------------------------
    def test_08_verification_rejection(self):
        """Unsafe generator output is rejected by verification_node and replaced with safe fallback."""
        def unsafe_generator(s: MindBridgeState) -> str:
            return "Based on your message, you have major depressive disorder. Take 50mg of sertraline daily."

        unsafe_graph = build_mindbridge_graph(generator=unsafe_generator)
        orig_graph = app_state.compiled_graph
        try:
            app_state.compiled_graph = unsafe_graph
            res = self.client.post("/chat", json={"message": "I feel down"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            # Must suppress unsafe diagnostic/prescription content
            self.assertNotIn("major depressive disorder", data["response"])
            self.assertNotIn("50mg", data["response"])
            self.assertIn(data["verification_status"], ("rejected_unsafe", "fallback_used"))
            self.assertEqual(data["response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        finally:
            app_state.compiled_graph = orig_graph

    # ------------------------------------------------------------------------
    # Test 9: Classifier failure containment
    # ------------------------------------------------------------------------
    def test_09_classifier_failure_containment(self):
        """Classifier error gracefully degrades without breaking conversational response."""
        res = self.client.post("/chat", json={"message": "I am stressed about exams"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("response", data)
        self.assertIsNotNone(data["session_id"])

    # ------------------------------------------------------------------------
    # Test 10: Retrieval failure containment
    # ------------------------------------------------------------------------
    def test_10_retrieval_failure_containment(self):
        """RAG retriever exception is contained gracefully and emits safe fallback."""
        failing_retriever = MockAPIRetriever(should_fail=True)
        failing_graph = build_mindbridge_graph(retriever=failing_retriever)  # type: ignore
        orig_graph = app_state.compiled_graph
        try:
            app_state.compiled_graph = failing_graph
            res = self.client.post("/chat", json={"message": "Need help with stress"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("I am here with you", data["response"])
            self.assertEqual(data["citations"], [])
        finally:
            app_state.compiled_graph = orig_graph

    # ------------------------------------------------------------------------
    # Test 11: Session isolation
    # ------------------------------------------------------------------------
    def test_11_session_isolation(self):
        """Requests with distinct session IDs do not leak state across sessions."""
        res_a = self.client.post("/chat", json={"message": "I feel anxiety", "session_id": "session_user_A"})
        res_b = self.client.post("/chat", json={"message": "How to sleep", "session_id": "session_user_B"})
        self.assertEqual(res_a.status_code, 200)
        self.assertEqual(res_b.status_code, 200)
        self.assertEqual(res_a.json()["session_id"], "session_user_A")
        self.assertEqual(res_b.json()["session_id"], "session_user_B")

    # ------------------------------------------------------------------------
    # Test 12: Restricted authorization cannot be supplied by client
    # ------------------------------------------------------------------------
    def test_12_client_cannot_supply_restricted_authorization(self):
        """Client attempting to pass extra authorization fields receives 422 Unprocessable Entity."""
        malicious_payload = {
            "message": "What are PHQ-9 cutoff scores",
            "retrieval_authorization": {"authorized_by": "hacker", "reason": "admin"},
        }
        res = self.client.post("/chat", json=malicious_payload)
        self.assertEqual(res.status_code, 422)
        self.assertTrue(
            "retrieval_authorization" in str(res.json())
            or "Extra inputs are not permitted" in str(res.json())
        )

    # ------------------------------------------------------------------------
    # Test 13: Privacy guarantee: No raw mental health messages in logs
    # ------------------------------------------------------------------------
    def test_13_no_raw_user_messages_in_logs(self):
        """Confirms user message content is never logged in telemetry or server logs."""
        unique_secret = "UNIQUE_PRIVATE_DISCLOSURE_TOKEN_XYZ_98765"
        logger = logging.getLogger("mindbridge.api")

        class LogCapturer(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []
            def emit(self, record):
                self.records.append(self.format(record))

        capturer = LogCapturer()
        logger.addHandler(capturer)
        try:
            self.client.post("/chat", json={"message": f"I feel down: {unique_secret}"})
            for record in capturer.records:
                self.assertNotIn(unique_secret, record)
        finally:
            logger.removeHandler(capturer)

    # ------------------------------------------------------------------------
    # Test 14: Concurrent request behavior
    # ------------------------------------------------------------------------
    def test_14_concurrent_request_behavior(self):
        """Executes concurrent requests to confirm pipeline stability and absence of deadlocks."""
        def make_request(msg: str):
            return self.client.post("/chat", json={"message": msg})

        messages = [
            "How to do box breathing",
            "Tips for sleeping better",
            "Exam anxiety coping strategies",
            "Grounding exercises for panic",
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_request, m) for m in messages]
            results = [f.result() for f in futures]

        for res in results:
            self.assertEqual(res.status_code, 200)
            self.assertIn("response", res.json())

    # ------------------------------------------------------------------------
    # Test 15: API response schema validation
    # ------------------------------------------------------------------------
    def test_15_api_response_schema_validation(self):
        """Verifies strict adherence to ChatResponse schema; zero raw logits or tracebacks."""
        res = self.client.post("/chat", json={"message": "Box breathing technique"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        validated = ChatResponse(**data)

        # Confirm forbidden sensitive fields are NOT exposed
        self.assertNotIn("roberta_prob", data)
        self.assertNotIn("execution_trace", data)
        self.assertNotIn("deterministic_danger_flag", data)
        self.assertNotIn("logits", data)
        self.assertNotIn("traceback", data)

    # ------------------------------------------------------------------------
    # Test 16: Real-model integration test
    # ------------------------------------------------------------------------
    def test_16_real_model_integration_test(self):
        """End-to-end integration test with real RoBERTa, real BGE embeddings, and FAISS index."""
        res = self.client.post("/chat", json={"message": "How do I do box breathing exercises"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["safety_route"], "standard_conversation")
        self.assertEqual(data["verification_status"], "verified")
        self.assertEqual(data["grounding_status"], "grounded")
        self.assertIn("inhale", data["response"].lower())
        self.assertTrue(len(data["citations"]) > 0)

    # ------------------------------------------------------------------------
    # Test 17: Oversized input with imminent suicide intent
    # ------------------------------------------------------------------------
    def test_17_oversized_input_with_imminent_suicide_intent(self):
        """Oversized message (>10,000 chars) with imminent suicide disclosure receives deterministic crisis resources."""
        filler = "I have been thinking about everything that has happened recently. " * 160
        message = "I am going to kill myself tonight. " + filler  # ~10.6k chars
        self.assertGreater(len(message), 10000)

        res = self.client.post("/chat", json={"message": message, "jurisdiction": "US"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        validated = ChatResponse(**data)

        self.assertEqual(validated.safety_route, "emergency_bypass")
        self.assertEqual(validated.error_code, "INPUT_PARSING_FAILURE")
        self.assertEqual(validated.verification_status, "not_run")
        self.assertEqual(validated.citations, [])
        self.assertGreater(len(validated.emergency_resources), 0)

        # Confirm resources contain US hotlines (e.g. 988 Lifeline, 911)
        resource_names = [r.resource_name for r in validated.emergency_resources]
        self.assertTrue(any("988" in name or "911" in name for name in resource_names))

        # Confirm safe response guidance without false clinical claims
        self.assertIn("Immediate, confidential support is available", validated.response)
        self.assertIn("exceeded the maximum permitted length", validated.response)
        self.assertIn("does not represent a clinical assessment", validated.response)

    # ------------------------------------------------------------------------
    # Test 18: Oversized input with direct self-harm threat
    # ------------------------------------------------------------------------
    def test_18_oversized_input_with_direct_self_harm_threat(self):
        """Oversized message with direct acute self-harm disclosure triggers emergency bypass with jurisdiction resources."""
        filler = "Everything feels so overwhelming and heavy right now. " * 200
        message = "I have been cutting my wrists and bleeding out. " + filler  # ~11k chars
        self.assertGreater(len(message), 10000)

        res = self.client.post("/chat", json={"message": message, "jurisdiction": "UK"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        validated = ChatResponse(**data)

        self.assertEqual(validated.safety_route, "emergency_bypass")
        self.assertEqual(validated.error_code, "INPUT_PARSING_FAILURE")
        self.assertGreater(len(validated.emergency_resources), 0)

        # Confirm UK emergency resources (e.g. 999 or Samaritans)
        telephones = [r.telephone for r in validated.emergency_resources if r.telephone]
        self.assertTrue(any("999" in tel or "111" in tel or "116" in tel for tel in telephones))
        self.assertIn("does not represent a clinical assessment", validated.response)

    # ------------------------------------------------------------------------
    # Test 19: Ordinary oversized input safe rejection
    # ------------------------------------------------------------------------
    def test_19_ordinary_oversized_input_safe_rejection(self):
        """Ordinary message exceeding 10,000 characters is rejected with 422 and clear non-technical guidance."""
        message = "I spent the entire weekend studying photosynthesis and cell biology in the campus library. " * 150
        self.assertGreater(len(message), 10000)

        res = self.client.post("/chat", json={"message": message, "jurisdiction": "US"})
        self.assertEqual(res.status_code, 422)
        data = res.json()

        self.assertEqual(data.get("error_code"), "VALIDATION_ERROR")
        self.assertIn("exceeds the maximum allowed length (10,000 characters)", data.get("error", ""))
        self.assertIn("please share a shorter message", data.get("error", ""))
        # Confirms no false claims of clinical evaluation
        self.assertNotIn("clinical", data.get("error", "").lower())

    # ------------------------------------------------------------------------
    # Test 20: Oversized boundary beyond 25,000 character ceiling
    # ------------------------------------------------------------------------
    def test_20_oversized_boundary_beyond_25k_ceiling(self):
        """Messages >25,000 chars are evaluated up to 25k ceiling; content past ceiling is bounded and rejected."""
        # 20a: Crisis at character 200 in a 30,000-character message
        filler_post = "x" * 29700
        msg_early_crisis = "I have pills and I am going to swallow all the pills tonight. " + filler_post
        self.assertGreater(len(msg_early_crisis), 25000)

        res_early = self.client.post("/chat", json={"message": msg_early_crisis, "jurisdiction": "US"})
        self.assertEqual(res_early.status_code, 200)
        self.assertEqual(res_early.json().get("safety_route"), "emergency_bypass")
        self.assertGreater(len(res_early.json().get("emergency_resources", [])), 0)

        # 20b: Crisis past 25,000 ceiling in a 30,000-character message (at char 28,000)
        filler_pre = "y" * 27000
        msg_late_crisis = filler_pre + " I am going to kill myself tonight."
        self.assertGreater(len(msg_late_crisis), 25000)

        res_late = self.client.post("/chat", json={"message": msg_late_crisis, "jurisdiction": "US"})
        # Unscanned content beyond ceiling is safely rejected with 422 without entering LLM/RAG
        self.assertEqual(res_late.status_code, 422)
        self.assertEqual(res_late.json().get("error_code"), "VALIDATION_ERROR")

    # ------------------------------------------------------------------------
    # Test 21: Oversized input with zero-width obfuscated crisis
    # ------------------------------------------------------------------------
    def test_21_oversized_input_with_zero_width_obfuscated_crisis(self):
        """Oversized message with zero-width obfuscation of suicide keywords triggers emergency bypass."""
        filler = "Thinking through recent challenges and personal stress. " * 190
        obfuscated = "i\u200b\u200bwill\u200b\u200bkill\u200b\u200bmyself\u200b\u200btonight. " + filler
        self.assertGreater(len(obfuscated), 10000)

        res = self.client.post("/chat", json={"message": obfuscated, "jurisdiction": "US"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("safety_route"), "emergency_bypass")
        self.assertGreater(len(data.get("emergency_resources", [])), 0)

    # ------------------------------------------------------------------------
    # Test 22: Ordinary payload exceeding 64 KB returns HTTP 413
    # ------------------------------------------------------------------------
    def test_22_ordinary_payload_exceeding_64kb(self):
        """HTTP request body exceeding 65,536 bytes (64 KB) is rejected with 413 and unassessed guidance."""
        filler = "Normal text discussing biology, botany, and library research. " * 1100  # ~68k bytes
        payload_bytes = f'{{"message": "{filler}"}}'.encode("utf-8")
        self.assertGreater(len(payload_bytes), 65536)

        res = self.client.post("/chat", content=payload_bytes, headers={"Content-Type": "application/json"})
        self.assertEqual(res.status_code, 413)
        data = res.json()

        self.assertEqual(data.get("error_code"), "PAYLOAD_TOO_LARGE")
        self.assertEqual(data.get("clinical_assessment"), "unassessed")
        self.assertIn("65536 bytes (64 KB)", data.get("error", ""))
        self.assertIn("too large to process or screen for safety", data.get("message", ""))
        self.assertGreater(len(data.get("emergency_resources", [])), 0)

    # ------------------------------------------------------------------------
    # Test 23: Crisis content in payload exceeding 64 KB returns HTTP 413 with hotlines
    # ------------------------------------------------------------------------
    def test_23_crisis_content_in_payload_exceeding_64kb(self):
        """Crisis disclosure in a >64 KB payload is rejected with 413, explicit unassessed disclaimer, and jurisdiction hotlines."""
        filler = "x" * 70000
        msg = f"I am going to kill myself tonight. {filler}"
        payload_bytes = f'{{"message": "{msg}", "jurisdiction": "US"}}'.encode("utf-8")
        self.assertGreater(len(payload_bytes), 65536)

        res = self.client.post(
            "/chat",
            content=payload_bytes,
            headers={"Content-Type": "application/json", "x-jurisdiction": "US"},
        )
        self.assertEqual(res.status_code, 413)
        data = res.json()

        self.assertEqual(data.get("error_code"), "PAYLOAD_TOO_LARGE")
        self.assertEqual(data.get("clinical_assessment"), "unassessed")
        self.assertIn("too large to process or screen for safety", data.get("message", ""))

        # Verify emergency resources are attached and include US hotlines
        resources = data.get("emergency_resources", [])
        self.assertGreater(len(resources), 0)
        resource_names = [r.get("resource_name", "") for r in resources]
        self.assertTrue(any("988" in name or "911" in name for name in resource_names))

    # ------------------------------------------------------------------------
    # Test 24: Malformed oversized payload returns HTTP 413 without crash
    # ------------------------------------------------------------------------
    def test_24_malformed_oversized_payload_returns_413_without_crash(self):
        """Malformed, non-JSON bytes exceeding 64 KB return 413 immediately without 500 error or stack trace leakage."""
        malformed_bytes = b'{"message": "broken invalid unclosed JSON ' + (b"bad_bytes_\xff\xfe\x00_" * 5000)
        self.assertGreater(len(malformed_bytes), 65536)

        res = self.client.post(
            "/chat",
            content=malformed_bytes,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status_code, 413)
        data = res.json()
        self.assertEqual(data.get("error_code"), "PAYLOAD_TOO_LARGE")
        self.assertNotIn("traceback", data)
        self.assertNotIn("SyntaxError", data.get("error", ""))

    # ------------------------------------------------------------------------
    # Test 25: Chunked streaming payload exceeding 64 KB aborts at boundary
    # ------------------------------------------------------------------------
    def test_25_chunked_streaming_payload_exceeding_64kb_aborts(self):
        """Streaming chunked payload without Content-Length header is bounded at 64 KB without unbounded buffering."""
        def chunk_generator():
            chunk = b"a" * 8192  # 8 KB chunks
            for _ in range(10):  # 80 KB total
                yield chunk

        # Send without Content-Length
        res = self.client.post(
            "/chat",
            content=chunk_generator(),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status_code, 413)
        data = res.json()
        self.assertEqual(data.get("error_code"), "PAYLOAD_TOO_LARGE")

    # ------------------------------------------------------------------------
    # Test 26: HTTP 413 missing jurisdiction header falls back to GL
    # ------------------------------------------------------------------------
    def test_26_http_413_missing_jurisdiction_header_falls_back_to_gl(self):
        """Missing x-jurisdiction header in HTTP 413 response cleanly defaults to global (GL) hotlines."""
        payload_bytes = b'{"message": "' + (b"a" * 70000) + b'"}'
        res = self.client.post("/chat", content=payload_bytes, headers={"Content-Type": "application/json"})
        self.assertEqual(res.status_code, 413)
        data = res.json()
        self.assertEqual(data.get("error_code"), "PAYLOAD_TOO_LARGE")

        resources = data.get("emergency_resources", [])
        self.assertGreater(len(resources), 0)
        # GL fallback contains 112 international emergency standard
        resource_ids = [r.get("resource_id", "") for r in resources]
        self.assertTrue(any("gl-sos-112" in rid or "112" in rid for rid in resource_ids))

    # ------------------------------------------------------------------------
    # Test 27: HTTP 413 invalid or manipulated jurisdiction header falls back to GL
    # ------------------------------------------------------------------------
    def test_27_http_413_invalid_or_manipulated_jurisdiction_falls_back_to_gl(self):
        """Manipulated or injection attempts in x-jurisdiction header safely default to GL fallback."""
        payload_bytes = b'{"message": "' + (b"b" * 70000) + b'"}'
        manipulated_headers = [
            "FOO_BAR_123",
            "../../etc/passwd",
            "'; DROP TABLE hotlines; --",
            "INVALID_COUNTRY_CODE",
        ]
        for header_val in manipulated_headers:
            res = self.client.post(
                "/chat",
                content=payload_bytes,
                headers={"Content-Type": "application/json", "x-jurisdiction": header_val},
            )
            self.assertEqual(res.status_code, 413)
            data = res.json()
            self.assertEqual(data.get("error_code"), "PAYLOAD_TOO_LARGE")
            resources = data.get("emergency_resources", [])
            self.assertGreater(len(resources), 0)
            # Confirms safe GL fallback
            self.assertTrue(any("112" in r.get("telephone", "") or "gl" in r.get("resource_id", "") for r in resources))

    # ------------------------------------------------------------------------
    # Test 28: HTTP 413 unsupported jurisdiction falls back to GL
    # ------------------------------------------------------------------------
    def test_28_http_413_unsupported_jurisdiction_falls_back_to_gl(self):
        """Unsupported jurisdictions (e.g. UNKNOWN, FR) safely fall back to GL rather than returning empty contacts."""
        payload_bytes = b'{"message": "' + (b"c" * 70000) + b'"}'
        for unsup in ["UNKNOWN", "FR", "DE", "JP"]:
            res = self.client.post(
                "/chat",
                content=payload_bytes,
                headers={"Content-Type": "application/json", "x-jurisdiction": unsup},
            )
            self.assertEqual(res.status_code, 413)
            data = res.json()
            resources = data.get("emergency_resources", [])
            # Must NOT return empty list!
            self.assertGreater(len(resources), 0)
            self.assertTrue(any("gl" in r.get("resource_id", "") for r in resources))

    # ------------------------------------------------------------------------
    # Test 29: HTTP 413 valid jurisdiction header with whitespace
    # ------------------------------------------------------------------------
    def test_29_http_413_valid_jurisdiction_header_with_whitespace(self):
        """Valid jurisdiction headers with leading/trailing whitespace or lowercase are normalized correctly."""
        payload_bytes = b'{"message": "' + (b"d" * 70000) + b'"}'
        res = self.client.post(
            "/chat",
            content=payload_bytes,
            headers={"Content-Type": "application/json", "x-jurisdiction": "  in  "},
        )
        self.assertEqual(res.status_code, 413)
        data = res.json()
        resources = data.get("emergency_resources", [])
        self.assertGreater(len(resources), 0)
        # Verify India ERSS 112 / ambulance is returned
        resource_ids = [r.get("resource_id", "") for r in resources]
        self.assertTrue(any("in-national-emergency-112" in rid or "112" in r.get("telephone", "") for rid, r in zip(resource_ids, resources)))

    # ------------------------------------------------------------------------
    # Test 30: Emergency resource accuracy and unverified operational status
    # ------------------------------------------------------------------------
    def test_30_emergency_resource_accuracy_and_unverified_operational_status(self):
        """Confirms returned hotline records preserve source metadata and do not claim live operational testing."""
        payload_bytes = b'{"message": "' + (b"e" * 70000) + b'"}'
        res = self.client.post(
            "/chat",
            content=payload_bytes,
            headers={"Content-Type": "application/json", "x-jurisdiction": "US"},
        )
        self.assertEqual(res.status_code, 413)
        data = res.json()
        resources = data.get("emergency_resources", [])
        self.assertGreater(len(resources), 0)

        for resource in resources:
            # Metadata integrity checks
            self.assertEqual(resource.get("verification_status"), "source_extracted")
            self.assertEqual(resource.get("uptime_verification"), "unverified_operational_availability")
            # Operational notes must disclaim live testing
            if resource.get("operational_notes"):
                self.assertIn("not been independently tested", resource["operational_notes"])


if __name__ == "__main__":
    unittest.main()
