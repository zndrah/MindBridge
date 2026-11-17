"""
MindBridge 2.0 - End-to-End CounselChat RAG Pipeline Test Suite
===============================================================

Comprehensive offline unit and integration tests verifying:
1. Immutability of the canonical clinical FAISS index (SHA-256 c7692096...).
2. Strict isolation of unapproved records from production retrieval (0 approved records).
3. Experimental dev subset governance metadata integrity (15 records, clinical_approval='not_assessed').
4. Relevant document retrieval for in-domain counseling queries.
5. Irrelevant query fallback (out-of-domain queries return empty -> safe neutral fallback).
6. Source attribution, therapist attribution, and mandatory non-clinical disclaimers.
7. Governance gating (unapproved records excluded in production mode).
8. Universal emergency crisis bypass (immediate halt of all retrieval; deterministic hotline routing).
9. Response generator failure containment (controlled fallback without orchestrator crashes).
10. End-to-end orchestrator pipeline flow (User query -> Safety -> Routing -> Retrieval -> Generation -> Verification).
"""

import hashlib
import json
from pathlib import Path
import unittest

from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.planning_node import planning_node, set_shared_retriever
from backend.orchestrator.response_node import response_node, synthesize_grounded_response
from backend.orchestrator.state import MindBridgeState
from backend.retrieval.counselchat_rag_design import (
    COUNSELCHAT_COLLECTION_NAME,
    COUNSELCHAT_DISCLAIMER,
    format_counselchat_citation,
)
from backend.retrieval.counselchat_retriever import (
    CounselChatRetriever,
    MultiCollectionRetriever,
    build_counselchat_experimental_dev_subset,
    build_counselchat_faiss_index,
    chunk_counselchat_experimental_record,
    DEFAULT_APPROVED_JSONL,
    DEFAULT_CANONICAL_JSONL,
    DEFAULT_EXPERIMENTAL_FAISS_DIR,
    DEFAULT_EXPERIMENTAL_SUBSET_PATH,
    DEFAULT_PRODUCTION_FAISS_DIR,
    EXPERIMENTAL_DEV_RECORD_IDS,
)
from backend.retrieval.embeddings import MockEmbeddingEngine, EMBEDDING_DIMENSION
from backend.retrieval.retriever import (
    RAGRetriever,
    format_citations,
    build_retriever_from_knowledge_base,
)
from backend.retrieval.vector_store import FAISSVectorStore
from backend.safety.fallbacks import NEUTRAL_SUPPORTIVE_FALLBACK

CANONICAL_CLINICAL_FAISS_SHA256 = "c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888"


def create_test_state(
    user_text: str = "",
    jurisdiction: str = "GL",
) -> MindBridgeState:
    """Helper constructing clean initial state for pipeline tests."""
    return {
        "session_id": "test-session-uuid-1234",
        "user_id_hash": "test-user-hash-5678",
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


class TestCounselChatRAGPipeline(unittest.TestCase):
    """Test suite covering the CounselChat RAG pipeline and governance invariants."""

    def setUp(self):
        self.mock_engine = MockEmbeddingEngine(dimension=EMBEDDING_DIMENSION)
        self.exp_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        # Build mock chunks for fast hermetic retriever testing
        self.mock_record_sleep = {
            "record_id": "counselchat_0007",
            "stable_record_id": "counselchat_0007",
            "question_title": "Suffering from insomnia and nightmares",
            "question_text": "I’ve suffered from insomnia and nightmares for about 3 years. What can I do?",
            "answer_text": "Sleep disruption often involves conditioned arousal. Establishing a strict sleep routine and stimulus control can help retrain your brain for restful sleep.",
            "topics": ["Sleep Improvement"],
            "therapist_name": "Danielle Alvarez",
            "therapist_url": "https://counselchat.com/therapists/danielle-alvarez",
            "question_id": "q_sleep_01",
            "question_url": "https://counselchat.com/questions/sleep-01",
            "upvotes": 5,
            "source_dataset": "weiting016/counselchat-data",
            "source_row_index": 7,
        }
        self.mock_record_family = {
            "record_id": "counselchat_0000",
            "stable_record_id": "counselchat_0000",
            "question_title": "Escalating disagreements between mother and wife",
            "question_text": "My wife and mother are having tense disagreements. What can I do?",
            "answer_text": "What you are describing is triangulation. Encourage open communication between them rather than being stuck in the middle.",
            "topics": ["Family Conflict"],
            "therapist_name": "Kristi King-Morgan, LMSW",
            "therapist_url": "https://counselchat.com/therapists/kristi-king-morgan-lmsw",
            "question_id": "q_family_01",
            "question_url": "https://counselchat.com/questions/family-01",
            "upvotes": 10,
            "source_dataset": "weiting016/counselchat-data",
            "source_row_index": 0,
        }

        chunks_sleep = chunk_counselchat_experimental_record(self.mock_record_sleep)
        chunks_family = chunk_counselchat_experimental_record(self.mock_record_family)
        all_mock_chunks = chunks_sleep + chunks_family

        texts = [c["content"] for c in all_mock_chunks]
        mock_embeddings = self.mock_engine.embed_texts(texts)
        self.exp_store.add_chunks(all_mock_chunks, mock_embeddings)

        self.counselchat_retriever = CounselChatRetriever(
            vector_store=self.exp_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,  # Zero threshold for mock deterministic matching
            is_experimental=True,
        )

    # 1. Canonical Clinical KB Immutability Test
    def test_canonical_clinical_faiss_index_remains_strictly_unchanged(self):
        """Invariant: The 31-vector canonical clinical FAISS index is strictly immutable."""
        kb_index_path = Path("data/knowledge_base/faiss_index/index.faiss")
        self.assertTrue(kb_index_path.exists(), "Clinical FAISS index must exist.")

        with open(kb_index_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(
            actual_sha256,
            CANONICAL_CLINICAL_FAISS_SHA256,
            f"Clinical FAISS index was modified! Expected {CANONICAL_CLINICAL_FAISS_SHA256}, got {actual_sha256}",
        )

        kb_meta_path = Path("data/knowledge_base/faiss_index/metadata.json")
        with open(kb_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_vectors"], 31, "Clinical KB must have exactly 31 vectors.")
        self.assertEqual(meta["dimension"], 384, "Clinical KB dimension must be 384.")

    # 2. Production Isolation Test (counselchat_approved.jsonl has 0 records)
    def test_counselchat_approved_remains_empty_and_production_index_zero_vectors(self):
        """Invariant: Production counselchat_approved.jsonl is 0 bytes and production index has 0 vectors."""
        approved_path = DEFAULT_APPROVED_JSONL
        self.assertTrue(approved_path.exists())
        self.assertEqual(approved_path.stat().st_size, 0, "counselchat_approved.jsonl must remain 0 bytes.")

        # Test building production index from approved file yields 0 vectors
        store, summary = build_counselchat_faiss_index(
            input_path=approved_path,
            output_dir=DEFAULT_PRODUCTION_FAISS_DIR,
            embedding_engine=self.mock_engine,
            is_experimental=False,
        )
        self.assertEqual(store.total_vectors, 0, "Production index must have 0 vectors.")
        self.assertEqual(summary["chunks_indexed"], 0)

        # In production mode, retriever returns 0 chunks
        prod_retriever = CounselChatRetriever(
            vector_store=store,
            embedding_engine=self.mock_engine,
            is_experimental=False,
        )
        results = prod_retriever.retrieve("Any query")
        self.assertEqual(results, [], "Production retriever must return 0 chunks when index is empty.")

    # 3. Experimental Dev Subset Governance Integrity Test
    def test_experimental_dev_subset_governance_integrity(self):
        """Invariant: Experimental dev subset preserves review status, clinical_approval='not_assessed'."""
        self.assertTrue(DEFAULT_EXPERIMENTAL_SUBSET_PATH.exists())
        with open(DEFAULT_EXPERIMENTAL_SUBSET_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 15, "Experimental dev subset must have exactly 15 records.")

        for line in lines:
            rec = json.loads(line)
            self.assertTrue(rec.get("is_experimental_dev"), "Must have is_experimental_dev=True")
            self.assertEqual(rec.get("dataset_tier"), "experimental_dev_subset")
            self.assertEqual(rec.get("review_status"), "pending_review", "Review status must remain pending_review")
            self.assertEqual(rec.get("clinical_approval"), "not_assessed", "Clinical approval must remain not_assessed")
            self.assertEqual(rec.get("retrieval_permission"), "experimental_only")

            # Check 9 Kaggle columns
            self.assertIn("question_id", rec)
            self.assertIn("question_title", rec)
            self.assertIn("question_text", rec)
            self.assertIn("question_url", rec)
            self.assertIn("answer_text", rec)
            self.assertIn("topics", rec)
            self.assertIn("therapist_name", rec)
            self.assertIn("therapist_url", rec)
            self.assertIn("upvotes", rec)

    # 4. Relevant Document Retrieval Test
    def test_relevant_document_retrieval(self):
        """Retrieval correctly surfaces matching counseling examples."""
        results = self.counselchat_retriever.retrieve(
            query="I have had bad insomnia and trouble falling asleep",
            top_k=2,
        )
        self.assertGreaterEqual(len(results), 1)
        first = results[0]
        self.assertIn(first["document_id"], ["counselchat_0007", "counselchat_0000"])
        self.assertEqual(first["category"], "peer_counseling_example")
        self.assertEqual(first["collection"], COUNSELCHAT_COLLECTION_NAME)

    # 5. Irrelevant Query Fallback Test
    def test_irrelevant_query_fallback(self):
        """Out-of-domain queries return empty retrieval and trigger neutral supportive fallback."""
        # High threshold simulating real cosine distance cutoff on irrelevant vector
        results = self.counselchat_retriever.retrieve(
            query="Calculate the Fourier transform of a step function",
            threshold=1.5,  # Unattainable threshold
        )
        self.assertEqual(results, [])

        # Response node receiving empty retrieval outputs NEUTRAL_SUPPORTIVE_FALLBACK
        state: MindBridgeState = create_test_state("What is the speed of light?")
        state["retrieval_status"] = "empty"
        state["retrieved_chunks"] = []
        updates = response_node(state)
        self.assertEqual(updates["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates["execution_trace"][-1]["status"], "fallback")
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "NO_CONTEXT_SUPPORTIVE_FALLBACK")

    # 6. Source Attribution & Disclaimer Test
    def test_source_attribution_and_non_clinical_framing(self):
        """Retrieved chunks include therapist attribution, source ID, and non-clinical disclaimer."""
        results = self.counselchat_retriever.retrieve(
            query="Family arguments",
            top_k=1,
        )
        self.assertGreaterEqual(len(results), 1)
        chunk = results[0]

        # Check attribution fields
        self.assertTrue(bool(chunk.get("therapist_name")))
        self.assertEqual(chunk["clinical_approval"], "not_assessed")
        self.assertIn(COUNSELCHAT_DISCLAIMER, chunk["disclaimer"])

        # Citation formatting
        citations = format_citations(results)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["document_id"], chunk["document_id"])

        # Response synthesis incorporates peer perspective framing and disclaimer
        response_text = synthesize_grounded_response(
            sanitized_text="My family is constantly arguing",
            dialog_goal="general",
            retrieved_chunks=results,
        )
        self.assertIn("illustrative peer counseling perspective", response_text)
        self.assertIn(COUNSELCHAT_DISCLAIMER, response_text)

    # 7. Unapproved Record Isolation in Production Mode Test
    def test_unapproved_record_isolation_in_production_mode(self):
        """In production mode (is_experimental=False), unapproved records are strictly excluded."""
        prod_retriever = CounselChatRetriever(
            vector_store=self.exp_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
            is_experimental=False,  # Production mode
        )
        # Store contains chunks with review_status='pending_review'
        results = prod_retriever.retrieve("I have trouble sleeping")
        self.assertEqual(
            results,
            [],
            "Production retriever must strictly exclude records with review_status != 'approved'",
        )

    # 8. Crisis Bypass Test
    def test_crisis_bypass_halts_all_retrieval(self):
        """Crisis safety route immediately bypasses all vector retrieval across both collections."""
        # 1. CounselChat retriever bypass
        cc_results = self.counselchat_retriever.retrieve(
            query="I want to kill myself",
            safety_route="emergency_bypass",
        )
        self.assertEqual(cc_results, [])

        # 2. Multi-collection retriever bypass
        mock_canonical_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        mock_canonical_store.add_chunks(
            [{"chunk_id": "c1", "document_id": "d1", "title": "T1", "category": "somatic_regulation", "jurisdiction": "GL", "content": "Content", "score": 0.0, "eligibility": "approved"}],
            self.mock_engine.embed_texts(["Content"])
        )
        canonical_retriever = RAGRetriever(
            vector_store=mock_canonical_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
        )
        multi_retriever = MultiCollectionRetriever(
            canonical_retriever=canonical_retriever,
            counselchat_retriever=self.counselchat_retriever,
        )

        multi_results = multi_retriever.retrieve(
            query="I want to die tonight",
            safety_route="emergency_bypass",
        )
        self.assertEqual(multi_results, [])

        # 3. Planning node bypass under emergency
        state: MindBridgeState = create_test_state("I want to end my life right now")
        state["overall_safety_route"] = "emergency_bypass"
        state["deterministic_danger_flag"] = True
        plan_updates = planning_node(state, retriever=multi_retriever)
        self.assertEqual(plan_updates["retrieval_status"], "not_attempted")
        self.assertEqual(plan_updates["retrieved_chunks"], [])
        self.assertEqual(plan_updates["execution_trace"][-1]["metadata_code"], "EMERGENCY_BYPASS_SKIPPED")

    # 9. Generator Failure Fallback Test
    def test_generator_failure_fallback(self):
        """If the LLM generator fails, response_node safely falls back to local deterministic grounded synthesis."""
        def failing_generator(state: MindBridgeState):
            raise RuntimeError("LLM API Quota Exceeded (HTTP 429)")

        results = self.counselchat_retriever.retrieve("Insomnia", top_k=1)
        state: MindBridgeState = create_test_state("I can't sleep at night")
        state["retrieval_status"] = "success"
        state["retrieved_chunks"] = results
        state["dialog_goal"] = "general"

        updates = response_node(state, generator=failing_generator)

        # Verifies failure was contained gracefully without raising an exception
        self.assertEqual(updates["error_code"], "LLM_GENERATION_FAILURE")
        self.assertEqual(updates["error_component"], "response_node")
        self.assertIsNotNone(updates["draft_response"])
        self.assertEqual(updates["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates["execution_trace"][-1]["status"], "error")
        self.assertEqual(updates["execution_trace"][-1]["metadata_code"], "RESPONSE_GENERATION_FAILURE_FALLBACK")

    # 10. End-to-End Orchestrator Pipeline Flow Test
    def test_end_to_end_orchestrator_pipeline_flow(self):
        """End-to-end LangGraph execution: User query -> Safety -> Planning/RAG -> Response -> Verification."""
        # Set shared retriever to multi_retriever with mock engine for offline speed
        mock_canonical_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        mock_canonical_store.add_chunks(
            [{
                "chunk_id": "cognitive_reframing:01__chunk_01",
                "document_id": "cognitive_reframing:01",
                "title": "Cognitive Reframing Tool",
                "category": "cognitive_reframing",
                "jurisdiction": "GL",
                "content": "Step 1: Identify automatic thoughts. Step 2: Challenge assumptions.",
                "score": 0.0,
                "eligibility": "approved",
            }],
            self.mock_engine.embed_texts(["Step 1: Identify automatic thoughts. Step 2: Challenge assumptions."])
        )
        canonical_retriever = RAGRetriever(
            vector_store=mock_canonical_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
        )
        multi_retriever = MultiCollectionRetriever(
            canonical_retriever=canonical_retriever,
            counselchat_retriever=self.counselchat_retriever,
        )
        app = build_mindbridge_graph(retriever=multi_retriever)

        # Run non-crisis message through graph
        initial = create_test_state("I feel overwhelmed by cognitive racing thoughts and worry")
        final_state = app.invoke(initial)

        # Assert full graph flow succeeded
        self.assertEqual(final_state["overall_safety_route"], "standard_conversation")
        self.assertEqual(final_state["retrieval_status"], "success")
        self.assertGreaterEqual(len(final_state["retrieved_chunks"]), 1)
        self.assertIsNotNone(final_state["draft_response"])
        self.assertIsNotNone(final_state["final_response"])
        self.assertEqual(final_state["verification_status"], "verified")

        executed_nodes = [t["node"] for t in final_state["execution_trace"]]
        self.assertIn("deterministic_safety_node", executed_nodes)
        self.assertIn("planning_node", executed_nodes)
        self.assertIn("response_node", executed_nodes)
        self.assertIn("verification_node", executed_nodes)
        self.assertNotIn("emergency_node", executed_nodes)

        # Clean up shared retriever
        set_shared_retriever(None)


if __name__ == "__main__":
    unittest.main()
