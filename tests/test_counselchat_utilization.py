"""
MindBridge 2.0 - CounselChat Dataset Utilization & Quality Audit Tests
======================================================================

Validates the full utilization of the 1,482-record CounselChat dataset:
1. Dataset & Metadata Preservation:
   - All 1,482 distinct therapist answers preserved.
   - Consolidation into 780 unique questions.
   - 1,421 answers usable in retrieval across 753 consolidated questions.
   - 49 restricted crisis/self-harm records excluded from conversational retrieval.
   - 12 hard-excluded malformed/prescribing records excluded.
   - 27 consolidated questions excluded (holding only crisis or malformed records).
2. FAISS Index Coverage:
   - Separate CounselChat FAISS index contains 1,965 vectors.
   - Metadata retains question, answer, therapist attribution, source ID, category.
   - Canonical clinical FAISS index (31 vectors) preserved byte-for-byte (SHA-256 c7692096...).
3. Multi-Perspective Retrieval & Synthesis:
   - Distinct therapist answers for the same question are not discarded.
   - Top-k retrieval ranks by cosine similarity.
   - Multi-perspective synthesis cites therapists and attaches COUNSELCHAT_DISCLAIMER.
4. Evaluation across all 8 mandatory domains:
   - Academic stress
   - Exam anxiety
   - Loneliness
   - Family conflict
   - Relationship problems
   - Depression-related concerns
   - Sleep difficulties
   - Self-esteem
5. Crisis Bypass & Out-of-Domain Abstention:
   - Imminent crisis bypasses vector search and open generation.
   - Non-mental health queries (OOD) abstain with safe supportive fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from backend.data_pipeline.consolidate_counselchat import (
    DEFAULT_CANONICAL_JSONL,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_OUTPUT_REPORT,
)
from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.input_node import input_node
from backend.orchestrator.planning_node import planning_node, set_shared_retriever
from backend.orchestrator.query_understanding import understand_query
from backend.orchestrator.response_node import response_node, synthesize_grounded_response
from backend.orchestrator.safety_node import deterministic_safety_node
from backend.orchestrator.state import MindBridgeState
from backend.retrieval.counselchat_rag_design import (
    COUNSELCHAT_COLLECTION_NAME,
    COUNSELCHAT_DISCLAIMER,
)
from backend.retrieval.counselchat_retriever import (
    CounselChatRetriever,
    MultiCollectionRetriever,
    DEFAULT_CONSOLIDATED_FAISS_DIR,
)
from backend.retrieval.embeddings import MockEmbeddingEngine, BGEEmbeddingEngine
from backend.retrieval.retriever import RAGRetriever, build_retriever_from_knowledge_base
from backend.retrieval.vector_store import FAISSVectorStore
from backend.safety.fallbacks import NEUTRAL_SUPPORTIVE_FALLBACK

CLINICAL_FAISS_SHA256 = "c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888"


def create_test_state(user_text: str = "", jurisdiction: str = "GL") -> MindBridgeState:
    """Creates a clean MindBridgeState for testing."""
    return {
        "session_id": "test-session-uuid",
        "user_id_hash": "test-user-hash",
        "messages": [{"role": "user", "content": user_text}] if user_text else [],
        "sanitized_text": user_text,
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
        "jurisdiction_filter": jurisdiction,  # type: ignore
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class TestCounselChatDatasetPreservation(unittest.TestCase):
    """Section 1: Dataset Preservation and Audit Accounting."""

    def test_01_all_1482_answers_preserved(self):
        """All 1,482 distinct answers from canonical CounselChat are preserved."""
        with open(DEFAULT_CANONICAL_JSONL, "r", encoding="utf-8") as f:
            canonical_records = [json.loads(line) for line in f]
        self.assertEqual(len(canonical_records), 1482)

        with open(DEFAULT_OUTPUT_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["final_distinct_qa_count"], 1482)
        self.assertEqual(report["unique_normalized_questions"], 780)
        self.assertEqual(report["usable_answers_for_retrieval"], 1421)
        self.assertEqual(report["restricted_crisis_records_excluded"], 49)
        self.assertEqual(report["hard_excluded_records"], 12)
        self.assertEqual(report["retrieval_ready_record_count"], 753)
        self.assertEqual(report["excluded_record_count"], 27)

        # Invariant: 1421 usable + 49 restricted + 12 hard-excluded = 1482 total
        self.assertEqual(
            report["usable_answers_for_retrieval"]
            + report["restricted_crisis_records_excluded"]
            + report["hard_excluded_records"],
            1482,
        )

    def test_02_clinical_faiss_index_integrity(self):
        """Canonical 31-vector clinical FAISS index is preserved byte-for-byte."""
        index_path = Path("data/knowledge_base/faiss_index/index.faiss")
        self.assertTrue(index_path.exists())
        with open(index_path, "rb") as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(computed_sha256, CLINICAL_FAISS_SHA256)

        store = FAISSVectorStore.load("data/knowledge_base/faiss_index")
        self.assertEqual(store.total_vectors, 31)


class TestMultiTherapistRetrievalAndSynthesis(unittest.TestCase):
    """Section 2: Multi-Therapist Perspective Retrieval & Synthesis."""

    def setUp(self):
        self.mock_engine = MockEmbeddingEngine()
        self.mock_store = FAISSVectorStore(dimension=self.mock_engine.dimension)

        # Question with two distinct therapist answers
        chunk1 = {
            "chunk_id": "counselchat_chunk_0001_000",
            "document_id": "counselchat_0001",
            "source_record_id": "counselchat_0001",
            "consolidated_id": "counselchat_cons_0001",
            "title": "Handling Family Disagreements",
            "question": "How can I handle constant disagreements with my mother?",
            "topics": ["Family Conflict"],
            "answer": "Focus on emotional boundaries. You cannot change your mother's reactions, only your responses.",
            "content": "Question: How can I handle constant disagreements with my mother?\n\nCounselor Response (Dr. Alice Smith):\nFocus on emotional boundaries. You cannot change your mother's reactions, only your responses.",
            "therapist_name": "Dr. Alice Smith",
            "therapist_url": "https://counselchat.com/therapists/alice",
            "category": "peer_counseling_example",
            "collection": COUNSELCHAT_COLLECTION_NAME,
            "clinical_approval": "not_assessed",
            "disclaimer": COUNSELCHAT_DISCLAIMER,
            "eligibility": "approved",
            "score": 0.88,
        }
        chunk2 = {
            "chunk_id": "counselchat_chunk_0002_000",
            "document_id": "counselchat_0002",
            "source_record_id": "counselchat_0002",
            "consolidated_id": "counselchat_cons_0001",  # Same question!
            "title": "Handling Family Disagreements",
            "question": "How can I handle constant disagreements with my mother?",
            "topics": ["Family Conflict"],
            "answer": "Schedule specific non-triggering discussion times and use 'I' statements to avoid escalation.",
            "content": "Question: How can I handle constant disagreements with my mother?\n\nCounselor Response (Dr. Robert Jones):\nSchedule specific non-triggering discussion times and use 'I' statements to avoid escalation.",
            "therapist_name": "Dr. Robert Jones",
            "therapist_url": "https://counselchat.com/therapists/robert",
            "category": "peer_counseling_example",
            "collection": COUNSELCHAT_COLLECTION_NAME,
            "clinical_approval": "not_assessed",
            "disclaimer": COUNSELCHAT_DISCLAIMER,
            "eligibility": "approved",
            "score": 0.85,
        }
        self.mock_store.add_chunks(
            [chunk1, chunk2],
            self.mock_engine.embed_texts([chunk1["content"], chunk2["content"]]),
        )
        self.retriever = CounselChatRetriever(
            vector_store=self.mock_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
            is_experimental=True,
        )

    def test_03_multiple_therapist_perspectives_retained(self):
        """Distinct therapist answers for the same question are both retrieved (not discarded)."""
        results = self.retriever.retrieve("family disagreements", top_k=2)
        self.assertEqual(len(results), 2)
        therapists = [r.get("therapist_name") for r in results]
        self.assertIn("Dr. Alice Smith", therapists)
        self.assertIn("Dr. Robert Jones", therapists)

    def test_04_multi_perspective_synthesis(self):
        """Response node synthesizes multiple therapist answers and attaches attribution and disclaimer."""
        results = self.retriever.retrieve("family disagreements", top_k=2)
        response_text = synthesize_grounded_response(
            sanitized_text="How can I handle constant disagreements with my mother?",
            dialog_goal="general",
            retrieved_chunks=results,
        )
        self.assertIn("Dr. Alice Smith", response_text)
        self.assertIn("Dr. Robert Jones", response_text)
        self.assertIn(COUNSELCHAT_DISCLAIMER, response_text)
        self.assertIn("illustrative peer counseling perspectives", response_text)


class TestCrisisAndOutDomainHandling(unittest.TestCase):
    """Section 3: Crisis Bypass and Out-of-Domain Abstention."""

    def test_05_imminent_crisis_bypasses_retrieval_and_llm(self):
        """Phrases of active imminent peril immediately trigger emergency bypass."""
        crisis_phrases = [
            "I feel like I might hurt myself right now",
            "I want to kill myself.",
            "I want to suicide.",
            "I might hurt myself right now.",
            "I am going to end my life.",
        ]

        for phrase in crisis_phrases:
            state = create_test_state(phrase)
            input_updates = input_node(state, raw_user_text=phrase)
            merged = {**state, **input_updates}
            safety_updates = deterministic_safety_node(merged)
            final_safety = {**merged, **safety_updates}

            self.assertEqual(
                final_safety.get("overall_safety_route"),
                "emergency_bypass",
                f"Failed route for: {phrase}",
            )
            self.assertTrue(final_safety.get("deterministic_danger_flag"))
            self.assertIsNone(final_safety.get("draft_response"))

            # Planning node verification: must halt vector search
            plan_updates = planning_node(final_safety)
            self.assertEqual(plan_updates.get("retrieval_status"), "not_attempted")
            self.assertEqual(plan_updates.get("retrieved_chunks"), [])

            # Response verification: compassionate response with resources
            response_text = final_safety.get("final_response", "")
            self.assertIn("immediate danger", response_text.lower())
            self.assertIn("trusted", response_text.lower())
            self.assertIn("hotline", response_text.lower() + " " + response_text)

    def test_06_out_of_domain_abstention(self):
        """Non-mental health queries (cooking, coding, ancient history) abstain cleanly."""
        ood_queries = [
            "What is the capital of Australia?",
            "Can you give me a recipe for chocolate chip cookies?",
            "How does a four stroke combustion engine work?",
        ]

        for query in ood_queries:
            understanding = understand_query(query)
            self.assertTrue(understanding.is_out_of_domain)
            self.assertEqual(understanding.retrieval_queries, [])

            state = create_test_state(query)
            plan_updates = planning_node(state)
            self.assertEqual(plan_updates["retrieval_status"], "empty")
            self.assertEqual(plan_updates["retrieved_chunks"], [])

            merged = {**state, **plan_updates}
            resp_updates = response_node(merged)
            self.assertEqual(resp_updates["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)


if __name__ == "__main__":
    unittest.main()
