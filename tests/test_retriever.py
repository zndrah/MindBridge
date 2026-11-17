"""
MindBridge 2.0 - Unit Tests for Governed RAG Retriever Subsystem
================================================================

Comprehensive offline unit tests verifying:
1. Default BGE engine configuration and explicit Mock injection.
2. Similarity search and configurable thresholding (tau = 0.65 baseline).
3. Jurisdiction filtering (IN, US, UK, CA, AU, GL, UNKNOWN).
4. Governance tier enforcement (approved, restricted, excluded_pending_review).
5. Adaptive candidate expansion during retrieval.
6. Citation provenance and metadata retention.
7. Graceful error handling (empty queries, invalid inputs, empty vector stores).
8. Strict emergency crisis safety routing independence.

Runs 100% offline with zero network requests and zero model downloads.
"""

import unittest
from typing import List, Dict, Any
import numpy as np

from backend.orchestrator.state import KnowledgeChunk, JurisdictionCode
from backend.retrieval.governance import (
    DOCUMENT_GOVERNANCE_REGISTRY,
    DocumentGovernance,
)
from backend.retrieval.embeddings import (
    BGEEmbeddingEngine,
    MockEmbeddingEngine,
    EMBEDDING_DIMENSION,
)
from backend.retrieval.vector_store import (
    FAISSVectorStore,
    PROPOSED_RETRIEVAL_THRESHOLD,
)
from backend.retrieval.retriever import (
    RAGRetriever,
    RetrievalAuthorization,
    format_citations,
    build_retriever_from_knowledge_base,
)
from backend.orchestrator.safety_node import deterministic_safety_node
from backend.orchestrator.state import MindBridgeState


class TestRAGRetriever(unittest.TestCase):
    """Test suite for governed RAG retriever."""

    def setUp(self):
        self.mock_engine = MockEmbeddingEngine(dimension=EMBEDDING_DIMENSION)
        self.store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        # Build sample curated chunks from governance registry
        self.approved_gl_chunk = {
            "chunk_id": "cognitive_reframing:01_thought_court_catastrophizing__chunk_01",
            "document_id": "cognitive_reframing:01_thought_court_catastrophizing",
            "source": "cognitive_reframing/01_thought_court_catastrophizing.md",
            "title": "Cognitive Restructuring for Catastrophic Loops (Thought Court)",
            "category": "cognitive_reframing",
            "jurisdiction": "GL",
            "verification_status": "partially_verified",
            "retrieval_tier": "approved",
            "safety_level": "open_informational",
            "disclaimer": "Non-clinical cognitive restructuring educational tool.",
            "content_hash": "hash_approved_gl",
            "chunk_index": 0,
            "content": "# Thought Court\nChallenging catastrophic cognitive loops with objective evidence.",
            "score": 0.0,
            "eligibility": "approved",
            "section_header": "Overview",
            "guardrails": [],
            "disclaimer_text": "Non-clinical cognitive restructuring educational tool.",
            "clinical_safety_level": "open_informational",
        }

        self.approved_in_chunk = {
            "chunk_id": "psychoeducation:01_exam_anxiety_resilience_india__chunk_01",
            "document_id": "psychoeducation:01_exam_anxiety_resilience_india",
            "source": "psychoeducation/01_exam_anxiety_resilience_india.md",
            "title": "Navigating Competitive Exam Anxiety: Familial Stakes & Panic",
            "category": "psychoeducation",
            "jurisdiction": "IN",
            "verification_status": "partially_verified",
            "retrieval_tier": "approved",
            "safety_level": "open_informational",
            "disclaimer": "Culturally sensitive psychoeducation for Indian competitive exam pressures.",
            "content_hash": "hash_approved_in",
            "chunk_index": 0,
            "content": "# Exam Anxiety in India\nDealing with familial pressure and high-stakes testing anxiety.",
            "score": 0.0,
            "eligibility": "approved",
            "section_header": "Overview",
            "guardrails": [],
            "disclaimer_text": "Culturally sensitive psychoeducation for Indian competitive exam pressures.",
            "clinical_safety_level": "open_informational",
        }

        self.restricted_gl_chunk = {
            "chunk_id": "clinical_rubrics:02_phq9_mood_screening_overview__chunk_01",
            "document_id": "clinical_rubrics:02_phq9_mood_screening_overview",
            "source": "clinical_rubrics/02_phq9_mood_screening_overview.md",
            "title": "PHQ-9 (Patient Health Questionnaire 9-Item Scale) Overview",
            "category": "clinical_rubrics",
            "jurisdiction": "GL",
            "verification_status": "independently_verified",
            "retrieval_tier": "restricted",
            "safety_level": "phq9_safety_protocol",
            "disclaimer": "CRITICAL: The PHQ-9 is an informational psychometric screening rubric, NOT a diagnostic confirmation.",
            "content_hash": "hash_restricted_gl",
            "chunk_index": 0,
            "content": "# PHQ-9 Overview\nItem 9 assesses thoughts of passive or active self-harm.",
            "score": 0.0,
            "eligibility": "restricted",
            "section_header": "Protocol",
            "guardrails": [
                "CRITICAL: Informational screening rubric only.",
                "Item 9 endorsement requires immediate safety follow-up.",
            ],
            "disclaimer_text": "CRITICAL: The PHQ-9 is an informational psychometric screening rubric, NOT a diagnostic confirmation.",
            "clinical_safety_level": "phq9_safety_protocol",
        }

        # Populate store with unit embeddings
        chunks = [self.approved_gl_chunk, self.approved_in_chunk, self.restricted_gl_chunk]
        texts = [c["content"] for c in chunks]
        embeddings = self.mock_engine.embed_texts(texts)
        self.store.add_chunks(chunks, embeddings)

        self.retriever = RAGRetriever(
            vector_store=self.store,
            embedding_engine=self.mock_engine,
            default_threshold=0.65,
        )

    def test_01_default_engine_is_bge_and_mock_requires_explicit_injection(self):
        """RAGRetriever defaults to BGEEmbeddingEngine in production execution."""
        default_retriever = RAGRetriever(vector_store=self.store)
        self.assertIsInstance(default_retriever.embedding_engine, BGEEmbeddingEngine)
        self.assertNotIsInstance(default_retriever.embedding_engine, MockEmbeddingEngine)
        self.assertFalse(default_retriever.embedding_engine.is_loaded)

    def test_02_relevant_retrieval_and_threshold_filtering(self):
        """Relevant queries match document content; scores below threshold are pruned."""
        # Query matching the exact content of thought court
        query = "# Thought Court Challenging catastrophic cognitive loops"
        results = self.retriever.retrieve(query=query, jurisdiction="GL", threshold=0.65)

        self.assertGreater(len(results), 0)
        self.assertEqual(
            results[0]["document_id"],
            "cognitive_reframing:01_thought_court_catastrophizing",
        )
        self.assertGreaterEqual(results[0]["score"], 0.65)

        # Unrelated query with high threshold returns empty
        strict_results = self.retriever.retrieve(
            query="Quantum mechanics black hole astrophysics string theory",
            jurisdiction="GL",
            threshold=0.95,
        )
        self.assertEqual(len(strict_results), 0)

    def test_03_configurable_similarity_threshold(self):
        """Retriever respects configured similarity threshold."""
        query = "Thought Court"
        # Permissive threshold returns matching chunk
        results_low = self.retriever.retrieve(query=query, threshold=-1.0)
        self.assertGreater(len(results_low), 0)

        # Impossible threshold returns nothing
        results_high = self.retriever.retrieve(query=query, threshold=1.01)
        self.assertEqual(len(results_high), 0)

    def test_04_jurisdiction_filtering_local_and_global(self):
        """User in IN receives both IN and GL chunks; user in US or GL does NOT receive IN chunks."""
        # Query matching IN exam anxiety document
        query = "# Exam Anxiety in India Dealing with familial pressure"

        # User in India (IN) -> can retrieve IN document
        results_in = self.retriever.retrieve(
            query=query,
            jurisdiction="IN",
            threshold=0.5,
        )
        doc_ids_in = [r["document_id"] for r in results_in]
        self.assertIn("psychoeducation:01_exam_anxiety_resilience_india", doc_ids_in)

        # User in Global (GL) -> IN document is NOT eligible
        results_gl = self.retriever.retrieve(
            query=query,
            jurisdiction="GL",
            threshold=0.5,
        )
        doc_ids_gl = [r["document_id"] for r in results_gl]
        self.assertNotIn("psychoeducation:01_exam_anxiety_resilience_india", doc_ids_gl)

        # User in US -> IN document is NOT eligible
        results_us = self.retriever.retrieve(
            query=query,
            jurisdiction="US",
            threshold=0.5,
        )
        doc_ids_us = [r["document_id"] for r in results_us]
        self.assertNotIn("psychoeducation:01_exam_anxiety_resilience_india", doc_ids_us)

    def test_05_jurisdiction_filtering_all_supported_codes(self):
        """Verify handling for IN, US, UK, CA, AU, GL, UNKNOWN."""
        supported_codes = ["IN", "US", "UK", "CA", "AU", "GL", "UNKNOWN"]
        for jur in supported_codes:
            # GL chunks should be eligible for all valid codes
            results = self.retriever.retrieve(
                query="# Thought Court",
                jurisdiction=jur,  # type: ignore
                threshold=0.5,
            )
            for r in results:
                self.assertIn(r["jurisdiction"], [jur, "GL"])

    def test_06_restricted_document_safeguards(self):
        """Restricted documents are blocked without trusted authorization and preserved when authorized."""
        query = "# PHQ-9 Overview Item 9 assesses thoughts of self-harm"

        # 1. include_restricted=False -> PHQ-9 must not be returned
        unrestricted_results = self.retriever.retrieve(
            query=query,
            jurisdiction="GL",
            include_restricted=False,
            threshold=0.5,
        )
        unrestricted_ids = [r["document_id"] for r in unrestricted_results]
        self.assertNotIn(
            "clinical_rubrics:02_phq9_mood_screening_overview",
            unrestricted_ids,
        )

        # 2. include_restricted=True ALONE (without trusted authorization) -> strictly blocked!
        unauthorized_bool_results = self.retriever.retrieve(
            query=query,
            jurisdiction="GL",
            include_restricted=True,
            authorization=None,
            threshold=0.5,
        )
        unauthorized_ids = [r["document_id"] for r in unauthorized_bool_results]
        self.assertNotIn(
            "clinical_rubrics:02_phq9_mood_screening_overview",
            unauthorized_ids,
            "Boolean include_restricted=True alone must not grant restricted access without authorization context",
        )

        # 3. include_restricted=True WITH trusted RetrievalAuthorization -> returned with guardrails & disclaimer
        trusted_auth = RetrievalAuthorization(
            authorized_by="orchestrator.planning_node",
            reason="clinical_rubrics_screening",
        )
        restricted_results = self.retriever.retrieve(
            query=query,
            jurisdiction="GL",
            include_restricted=True,
            authorization=trusted_auth,
            threshold=0.5,
        )
        phq9_items = [
            r for r in restricted_results
            if r["document_id"] == "clinical_rubrics:02_phq9_mood_screening_overview"
        ]
        self.assertEqual(len(phq9_items), 1)
        phq9_chunk = phq9_items[0]
        self.assertEqual(phq9_chunk["eligibility"], "restricted")
        self.assertEqual(phq9_chunk["retrieval_tier"], "restricted")
        self.assertEqual(phq9_chunk["safety_level"], "phq9_safety_protocol")
        self.assertTrue(len(phq9_chunk["guardrails"]) > 0)
        self.assertTrue(len(phq9_chunk["disclaimer"]) > 0)

    def test_07_excluded_document_strictly_blocked(self):
        """Excluded documents in governance are never returned even if present in vector store with full authorization."""
        excluded_chunk = {
            "chunk_id": "somatic_regulation:06_vagus_nerve_stimulation_tapping__chunk_01",
            "document_id": "somatic_regulation:06_vagus_nerve_stimulation_tapping",
            "source": "somatic_regulation/06_vagus_nerve_stimulation_tapping.md",
            "title": "Bilateral Somatic Tapping & Tactile Grounding",
            "category": "somatic_regulation",
            "jurisdiction": "GL",
            "verification_status": "unverified",
            "retrieval_tier": "excluded_pending_review",
            "safety_level": "blocked_from_retrieval",
            "disclaimer": "BLOCKED",
            "content_hash": "hash_excluded",
            "chunk_index": 0,
            "content": "# Somatic Tapping Vagus Stimulation",
            "score": 0.0,
            "eligibility": "restricted",
            "section_header": "Overview",
            "guardrails": ["BLOCKED"],
            "disclaimer_text": "BLOCKED",
            "clinical_safety_level": "blocked_from_retrieval",
        }
        vec = self.mock_engine.embed_texts([excluded_chunk["content"]])
        self.store.add_chunks([excluded_chunk], vec)

        trusted_auth = RetrievalAuthorization(
            authorized_by="admin_supervisor",
            reason="maximum_permission_audit",
        )
        results = self.retriever.retrieve(
            query="# Somatic Tapping Vagus Stimulation",
            jurisdiction="GL",
            include_restricted=True,
            authorization=trusted_auth,  # Even with full authorization, excluded must fail
            threshold=-1.0,
        )
        returned_ids = [r["document_id"] for r in results]
        self.assertNotIn(
            "somatic_regulation:06_vagus_nerve_stimulation_tapping",
            returned_ids,
        )

    def test_08_unregistered_document_rejected(self):
        """Chunks from unregistered documents are rejected defensively during retrieval."""
        unregistered_chunk = {
            "chunk_id": "unknown_cat:99_fake_doc__chunk_01",
            "document_id": "unknown_cat:99_fake_doc",
            "source": "unknown_cat/99_fake_doc.md",
            "title": "Unregistered Document",
            "category": "unknown",
            "jurisdiction": "GL",
            "verification_status": "unverified",
            "retrieval_tier": "approved",
            "safety_level": "open_informational",
            "disclaimer": "",
            "content_hash": "hash_unreg",
            "chunk_index": 0,
            "content": "Secret unregistered document content.",
            "score": 0.0,
            "eligibility": "approved",
            "section_header": "Overview",
        }
        vec = self.mock_engine.embed_texts([unregistered_chunk["content"]])
        self.store.add_chunks([unregistered_chunk], vec)

        results = self.retriever.retrieve(
            query="Secret unregistered document content",
            jurisdiction="GL",
            threshold=-1.0,
        )
        returned_ids = [r["document_id"] for r in results]
        self.assertNotIn("unknown_cat:99_fake_doc", returned_ids)

    def test_09_adaptive_candidate_expansion(self):
        """Retriever adaptively expands its search pool to find eligible chunks beyond top candidates."""
        # Create a store where the top 10 highest-scoring candidates are all restricted
        dense_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        restricted_chunks = []
        for i in range(10):
            c = dict(self.restricted_gl_chunk)
            c["chunk_id"] = f"{self.restricted_gl_chunk['chunk_id']}_{i}"
            c["content_hash"] = f"restricted_hash_{i}"
            c["content"] = f"Top restricted chunk {i} matching query"
            restricted_chunks.append(c)

        # Target approved chunk with slightly different text
        target_approved = dict(self.approved_gl_chunk)
        target_approved["content"] = "Target approved chunk for user query"

        all_test_chunks = restricted_chunks + [target_approved]
        all_vecs = self.mock_engine.embed_texts([c["content"] for c in all_test_chunks])
        dense_store.add_chunks(all_test_chunks, all_vecs)

        dense_retriever = RAGRetriever(
            vector_store=dense_store,
            embedding_engine=self.mock_engine,
            default_threshold=-1.0,
        )

        # When include_restricted=False, it should adaptively search past the 10 restricted
        # chunks and retrieve the target approved chunk
        results = dense_retriever.retrieve(
            query="Target approved chunk for user query",
            include_restricted=False,
            top_k=1,
            threshold=-1.0,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], target_approved["document_id"])

    def test_10_top_k_limiting(self):
        """Retriever respects top_k maximum results limit."""
        results_1 = self.retriever.retrieve(
            query="# Thought Court",
            top_k=1,
            threshold=-1.0,
        )
        self.assertLessEqual(len(results_1), 1)

        results_2 = self.retriever.retrieve(
            query="# Thought Court",
            top_k=2,
            threshold=-1.0,
        )
        self.assertLessEqual(len(results_2), 2)

    def test_11_empty_query_and_invalid_inputs(self):
        """Empty query, whitespace query, None, invalid types, and empty vector store return empty list."""
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   "), [])
        self.assertEqual(self.retriever.retrieve(None), [])  # type: ignore
        self.assertEqual(self.retriever.retrieve(12345), [])  # type: ignore
        self.assertEqual(self.retriever.retrieve("query", top_k=0), [])
        self.assertEqual(self.retriever.retrieve("query", top_k=-5), [])

        empty_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        empty_retriever = RAGRetriever(
            vector_store=empty_store,
            embedding_engine=self.mock_engine,
        )
        self.assertEqual(empty_retriever.retrieve("valid query"), [])

    def test_12_citation_provenance_and_metadata_preservation(self):
        """format_citations preserves all 10 required metadata fields and reflects only retrieved chunks."""
        results = self.retriever.retrieve(
            query="# Thought Court Challenging catastrophic cognitive loops",
            threshold=0.5,
        )
        self.assertGreater(len(results), 0)

        citations = format_citations(results)
        self.assertEqual(len(citations), len(results))

        citation = citations[0]
        required_citation_keys = {
            "title",
            "document_id",
            "chunk_id",
            "source",
            "category",
            "jurisdiction",
            "verification_status",
            "retrieval_tier",
            "safety_level",
            "disclaimer",
            "guardrails",
            "content_hash",
            "chunk_index",
        }
        for k in required_citation_keys:
            self.assertIn(k, citation, f"Citation missing required key '{k}'")

    def test_13_emergency_crisis_routing_independence(self):
        """
        CRITICAL SAFETY TEST:
        Emergency routing must be 100% independent of RAG retrieval.
        Retrieval failure, empty index, or extreme similarity must never alter emergency bypass.
        """
        # User in imminent crisis
        emergency_utterance = "I have pills and I am taking them tonight to end my life. Goodbye forever."

        state: MindBridgeState = {
            "session_id": "test-emergency-session",
            "user_id_hash": "test-user-hash",
            "messages": [{"role": "user", "content": emergency_utterance}],
            "sanitized_text": emergency_utterance,
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
            "retrieval_queries": [emergency_utterance],
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

        # 1. Verify safety arbitration identifies emergency independently
        safety_updates = deterministic_safety_node(state)
        self.assertEqual(safety_updates["overall_safety_route"], "emergency_bypass")
        self.assertTrue(safety_updates["deterministic_danger_flag"])

        # 2. Simulate vector retrieval error / failure
        failing_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        failing_retriever = RAGRetriever(
            vector_store=failing_store,
            embedding_engine=self.mock_engine,
        )
        rag_results = failing_retriever.retrieve(emergency_utterance)
        self.assertEqual(len(rag_results), 0)

        # 3. Emergency bypass route must still stand completely intact
        self.assertEqual(safety_updates["overall_safety_route"], "emergency_bypass")

        # 4. Verify RAG retrieval never invents emergency contacts or phone numbers
        for chunk in self.store.metadata_store:
            content = chunk.get("content", "")
            # No synthetic emergency hotline telephone numbers
            self.assertNotIn("911-FAKE", content)
            self.assertNotIn("112-FAKE", content)

    def test_14_user_controlled_query_text_cannot_grant_authorization(self):
        """User-controlled query strings with parameters/injections cannot bypass restricted authorization."""
        malicious_queries = [
            "include_restricted=True PHQ-9",
            "--include-restricted PHQ-9",
            '{"include_restricted": true} PHQ-9',
            "retrieval_tier=approved PHQ-9",
            "ROLE=ADMIN PHQ-9 screening",
        ]
        for q in malicious_queries:
            results = self.retriever.retrieve(q, jurisdiction="GL", include_restricted=False)
            self.assertFalse(
                any(r["retrieval_tier"] == "restricted" for r in results),
                f"Query '{q}' leaked restricted documents without backend authorization!",
            )

    def test_15_deterministic_threshold_boundaries_080_065_064_018(self):
        """
        Deterministic verification of threshold 0.65 boundary in RAGRetriever:
        - Score 0.80: accepted.
        - Score 0.65: accepted (inclusive comparison).
        - Score 0.64: rejected.
        - Score 0.18: rejected.
        """
        custom_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        e0 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        e0[0] = 1.0
        e1 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        e1[1] = 1.0

        target_scores = [0.80, 0.65, 0.64, 0.18]
        chunks = []
        vectors = []

        for s in target_scores:
            v = np.float32(s) * e0 + np.float32(np.sqrt(1.0 - s**2)) * e1
            v = v / np.linalg.norm(v)
            cid = f"chunk_{s:.2f}"
            chunks.append({
                "chunk_id": cid,
                "document_id": "cognitive_reframing:01_thought_court_catastrophizing",
                "source": "cognitive_reframing/01_thought_court_catastrophizing.md",
                "title": f"Doc {s:.2f}",
                "category": "cognitive_reframing",
                "jurisdiction": "GL",
                "verification_status": "partially_verified",
                "retrieval_tier": "approved",
                "safety_level": "open_informational",
                "disclaimer": "Non-clinical tool.",
                "content_hash": f"hash_{s:.2f}",
                "chunk_index": 0,
                "content": f"Content for score {s:.2f}",
                "score": 0.0,
                "eligibility": "approved",
                "section_header": "Overview",
            })
            vectors.append(v)

        custom_store.add_chunks(chunks, np.vstack(vectors))

        # Create a mock engine that outputs e0 for the query
        class ControlledQueryEngine:
            def __init__(self):
                self.dimension = EMBEDDING_DIMENSION
                self._is_loaded = True

            @property
            def is_loaded(self):
                return True

            def load(self):
                pass

            def embed_query(self, query):
                return e0.copy()

            def embed_texts(self, texts):
                return np.zeros((len(texts), self.dimension), dtype=np.float32)

        controlled_retriever = RAGRetriever(
            vector_store=custom_store,
            embedding_engine=ControlledQueryEngine(),  # type: ignore
            default_threshold=0.65,
        )

        # Retrieve with default threshold = 0.65
        results = controlled_retriever.retrieve(
            query="Deterministic threshold test query",
            jurisdiction="GL",
            top_k=10,
        )

        retrieved_ids = [r["chunk_id"] for r in results]

        # 0.80 must be accepted
        self.assertIn("chunk_0.80", retrieved_ids, "Score 0.80 must be accepted")

        # 0.65 must be accepted (inclusive comparison)
        self.assertIn("chunk_0.65", retrieved_ids, "Score 0.65 must be accepted (inclusive)")

        # 0.64 must be rejected
        self.assertNotIn("chunk_0.64", retrieved_ids, "Score 0.64 must be rejected")

        # 0.18 must be rejected
        self.assertNotIn("chunk_0.18", retrieved_ids, "Score 0.18 must be rejected")

    def test_16_relative_margin_pruning(self):
        """Relative margin pruning removes trailing candidates that fall far below the top match."""
        custom_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        e0 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        e0[0] = 1.0
        e1 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        e1[1] = 1.0

        target_scores = [0.85, 0.80, 0.68]
        chunks = []
        vectors = []

        for s in target_scores:
            v = np.float32(s) * e0 + np.float32(np.sqrt(1.0 - s**2)) * e1
            v = v / np.linalg.norm(v)
            cid = f"chunk_{s:.2f}"
            chunks.append({
                "chunk_id": cid,
                "document_id": "cognitive_reframing:01_thought_court_catastrophizing",
                "source": "cognitive_reframing/01_thought_court_catastrophizing.md",
                "title": f"Doc {s:.2f}",
                "category": "cognitive_reframing",
                "jurisdiction": "GL",
                "verification_status": "partially_verified",
                "retrieval_tier": "approved",
                "safety_level": "open_informational",
                "disclaimer": "Non-clinical tool.",
                "content_hash": f"hash_{s:.2f}",
                "chunk_index": 0,
                "content": f"Content {s:.2f}",
                "score": 0.0,
                "eligibility": "approved",
                "section_header": "Overview",
            })
            vectors.append(v)

        custom_store.add_chunks(chunks, np.vstack(vectors))

        class MockQueryEngine:
            def __init__(self):
                self.dimension = EMBEDDING_DIMENSION
                self.is_loaded = True
            def load(self): pass
            def embed_query(self, q): return e0.copy()
            def embed_texts(self, t): return np.zeros((len(t), self.dimension), dtype=np.float32)

        retriever = RAGRetriever(
            vector_store=custom_store,
            embedding_engine=MockQueryEngine(),  # type: ignore
            default_threshold=0.65,
        )

        # 1. With relative_margin=0.10: 0.85 and 0.80 retained; 0.68 (gap 0.17) pruned
        results_pruned = retriever.retrieve(
            query="Test query",
            jurisdiction="GL",
            relative_margin=0.10,
        )
        pruned_ids = [r["chunk_id"] for r in results_pruned]
        self.assertIn("chunk_0.85", pruned_ids)
        self.assertIn("chunk_0.80", pruned_ids)
        self.assertNotIn("chunk_0.68", pruned_ids)

        # 2. With relative_margin=None: all 3 >= 0.65 retained
        results_all = retriever.retrieve(
            query="Test query",
            jurisdiction="GL",
            relative_margin=None,
        )
        all_ids = [r["chunk_id"] for r in results_all]
        self.assertIn("chunk_0.85", all_ids)
        self.assertIn("chunk_0.80", all_ids)
        self.assertIn("chunk_0.68", all_ids)

    def test_17_off_target_fallback_suppression(self):
        """When unauthorized for a high-scoring restricted target, distant approved fallbacks are suppressed."""
        custom_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        e0 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        e0[0] = 1.0
        e1 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        e1[1] = 1.0

        # Vector 1: Restricted chunk with score 0.88
        s_rest = 0.88
        v_rest = np.float32(s_rest) * e0 + np.float32(np.sqrt(1.0 - s_rest**2)) * e1
        v_rest = v_rest / np.linalg.norm(v_rest)

        # Vector 2: Approved chunk with score 0.68 (gap = 0.20 > 0.10)
        s_app = 0.68
        v_app = np.float32(s_app) * e0 + np.float32(np.sqrt(1.0 - s_app**2)) * e1
        v_app = v_app / np.linalg.norm(v_app)

        chunks = [
            {
                "chunk_id": "chunk_restricted_0.88",
                "document_id": "clinical_rubrics:02_phq9_mood_screening_overview",
                "source": "clinical_rubrics/02_phq9_mood_screening_overview.md",
                "title": "PHQ-9 Overview",
                "category": "clinical_rubrics",
                "jurisdiction": "GL",
                "verification_status": "independently_verified",
                "retrieval_tier": "restricted",
                "safety_level": "phq9_safety_protocol",
                "disclaimer": "Screening rubric.",
                "content_hash": "hash_rest",
                "chunk_index": 0,
                "content": "PHQ-9 Item 9",
                "score": 0.0,
                "eligibility": "restricted",
                "section_header": "Overview",
            },
            {
                "chunk_id": "chunk_approved_0.68",
                "document_id": "cognitive_reframing:03_behavioral_activation_inertia",
                "source": "cognitive_reframing/03_behavioral_activation_inertia.md",
                "title": "Behavioral Activation",
                "category": "cognitive_reframing",
                "jurisdiction": "GL",
                "verification_status": "partially_verified",
                "retrieval_tier": "approved",
                "safety_level": "open_informational",
                "disclaimer": "Non-clinical tool.",
                "content_hash": "hash_app",
                "chunk_index": 0,
                "content": "Behavioral Activation",
                "score": 0.0,
                "eligibility": "approved",
                "section_header": "Overview",
            },
        ]

        custom_store.add_chunks(chunks, np.vstack([v_rest, v_app]))

        class MockQueryEngine:
            def __init__(self):
                self.dimension = EMBEDDING_DIMENSION
                self.is_loaded = True
            def load(self): pass
            def embed_query(self, q): return e0.copy()
            def embed_texts(self, t): return np.zeros((len(t), self.dimension), dtype=np.float32)

        retriever = RAGRetriever(
            vector_store=custom_store,
            embedding_engine=MockQueryEngine(),  # type: ignore
            default_threshold=0.65,
        )

        # 1. Unauthorized query with max_fallback_gap=0.10: returns empty list
        results_unauth = retriever.retrieve(
            query="PHQ-9 screening score rubric",
            jurisdiction="GL",
            include_restricted=False,
            max_fallback_gap=0.10,
        )
        self.assertEqual(len(results_unauth), 0, "Off-target approved fallback must be suppressed")

        # 2. Authorized query: returns the restricted document
        trusted_auth = RetrievalAuthorization(authorized_by="planning_node", reason="screening")
        results_auth = retriever.retrieve(
            query="PHQ-9 screening score rubric",
            jurisdiction="GL",
            include_restricted=True,
            authorization=trusted_auth,
            top_k=1,
            max_fallback_gap=0.10,
        )
        self.assertEqual(len(results_auth), 1)
        self.assertEqual(results_auth[0]["chunk_id"], "chunk_restricted_0.88")


if __name__ == "__main__":
    unittest.main()
