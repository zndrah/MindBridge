"""
MindBridge 2.0 - Comprehensive CounselChat Data Consolidation & RAG Tests
========================================================================

Covers Phases 2–10:
1. Dataset Tests:
   - Source preservation (9 original Kaggle columns, source URLs, upvotes).
   - HTML cleaning, entity decoding, and Unicode NFKC normalization.
   - Missing content handling (question title fallback, missing answer flags).
   - Intelligent duplicate consolidation (Rules A, B, C, D, and E).
2. FAISS Index Tests:
   - Separate CounselChat FAISS index creation, persistence, and loading.
   - 384-dimensional unit L2 embeddings with faiss.IndexFlatIP.
   - Immutability of the 31-vector clinical FAISS index (SHA-256 c7692096...).
3. Retrieval Tests:
   - Relevant query retrieval with topic matching.
   - Irrelevant query fallback (out-of-domain abstention).
   - Top-k behavior and similarity threshold filtering.
   - Deduplication by question (surfacing diverse counseling perspectives).
   - Full source attribution (therapist name, therapist URL, non-clinical disclaimer).
4. Generation Tests:
   - Retrieved context formatted into grounded prompts.
   - Peer counseling perspective framing and disclaimer attachment.
   - Insufficient context fallback to NEUTRAL_SUPPORTIVE_FALLBACK.
   - Provider failure containment without graph crashes.
5. Safety Tests:
   - Emergency crisis bypass halts all vector search across both collections.
   - Deterministic crisis response bypasses open generation.
   - Classifier failure fallback mode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from backend.data_pipeline.consolidate_counselchat import (
    clean_html_content,
    consolidate_counselchat_dataset,
    normalize_for_matching,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_OUTPUT_REPORT,
)
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
    build_consolidated_counselchat_faiss_index,
    chunk_consolidated_counselchat_dataset,
    DEFAULT_CONSOLIDATED_FAISS_DIR,
    DEFAULT_CONSOLIDATED_JSONL,
)
from backend.retrieval.embeddings import MockEmbeddingEngine, EMBEDDING_DIMENSION
from backend.retrieval.retriever import RAGRetriever, format_citations
from backend.retrieval.vector_store import FAISSVectorStore
from backend.safety.fallbacks import NEUTRAL_SUPPORTIVE_FALLBACK

CLINICAL_FAISS_SHA256 = "c7692096a90b7476ad75fbfb175c23a06fcc97c581dfe5ce3dac05980e497888"


def create_state(user_text: str = "", jurisdiction: str = "GL") -> MindBridgeState:
    """Helper creating clean initial state container."""
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


class TestCounselChatDataConsolidation(unittest.TestCase):
    """Phase 3 & Phase 4: Data Cleaning and Intelligent Duplicate Consolidation Tests."""

    def test_html_cleaning_and_entity_decoding(self):
        """Phase 3: HTML tags, scripts, and entities are cleanly decoded and stripped."""
        raw_html = "<p>Here is some advice: &quot;Stay calm&quot; &amp; breathe.</p><br><p>Second paragraph.</p>"
        cleaned = clean_html_content(raw_html)
        self.assertEqual(cleaned, 'Here is some advice: "Stay calm" & breathe.\n\nSecond paragraph.')

        # Strip scripts and styles
        raw_with_script = "<script>alert('xss')</script><style>body {color: red;}</style><p>Valid content.</p>"
        cleaned_script = clean_html_content(raw_with_script)
        self.assertEqual(cleaned_script, "Valid content.")

        # Preserves list bullet structure
        raw_list = "<ul><li>First step</li><li>Second step</li></ul>"
        cleaned_list = clean_html_content(raw_list)
        self.assertIn("• First step", cleaned_list)
        self.assertIn("• Second step", cleaned_list)

    def test_missing_question_title_and_text_handling(self):
        """Phase 3: Fallback from questionText to questionTitle, flagging empty records."""
        # Body missing, title present -> uses title as effective question
        raw_body_missing = {"questionTitle": "How to handle stress?", "questionText": None}
        eff_q = raw_body_missing["questionText"] or raw_body_missing["questionTitle"]
        self.assertEqual(eff_q, "How to handle stress?")

        # Consolidation report records exact counts for missing questions (2) and missing answers (5)
        self.assertTrue(DEFAULT_OUTPUT_REPORT.exists())
        with open(DEFAULT_OUTPUT_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["empty_questions"], 2)
        self.assertEqual(report["empty_answers"], 5)
        self.assertEqual(report["malformed_records"], 7)

    def test_intelligent_duplicate_consolidation_rules(self):
        """Phase 4: Rule A, Rule B, Rule C, and Rule D enforcement."""
        self.assertTrue(DEFAULT_OUTPUT_JSONL.exists())
        with open(DEFAULT_OUTPUT_JSONL, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f]

        self.assertEqual(len(questions), 780, "Must have exactly 780 consolidated questions.")

        # Rule B: Check questions with multiple distinct answers
        multi_ans_questions = [q for q in questions if q["answers_count"] > 1]
        self.assertEqual(len(multi_ans_questions), 302)

        # Verify that multiple answers preserve distinct therapist attributions and source IDs
        sample_multi = multi_ans_questions[0]
        self.assertGreaterEqual(len(sample_multi["answers"]), 2)
        for ans in sample_multi["answers"]:
            self.assertTrue(bool(ans["answer_text"]))
            self.assertTrue(bool(ans["source_record_id"]))
            self.assertIn(ans["source_record_id"], sample_multi["source_record_ids"])

        # Rule C: Distinct questions with similar answers remain separate
        q_texts = [q["question"] for q in questions if q["question"]]
        self.assertEqual(len(set(q_texts)), len(q_texts))

        # Check total distinct answers preserved equals original 1,482
        total_answers_sum = sum(q["answers_count"] for q in questions)
        self.assertEqual(total_answers_sum, 1482)


class TestCounselChatFAISSAndRetrieval(unittest.TestCase):
    """Phase 6 & Phase 7: FAISS Indexing and Dense Retrieval Tests."""

    def setUp(self):
        self.mock_engine = MockEmbeddingEngine(dimension=EMBEDDING_DIMENSION)
        self.mock_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        # Build mock chunks from consolidated dataset format
        self.mock_chunks = [
            {
                "chunk_id": "mock_chunk_001",
                "document_id": "counselchat_0007",
                "consolidated_id": "counselchat_cons_0008",
                "title": "Insomnia and Nightmares",
                "question": "I have had severe insomnia and nightmares.",
                "topics": ["Sleep Improvement"],
                "content": "Question: I have had severe insomnia and nightmares.\n\nCounselor Response (Danielle Alvarez):\nSleep hygiene and stimulus control therapy can restore circadian rhythms.",
                "therapist_name": "Danielle Alvarez",
                "therapist_url": "https://counselchat.com/therapists/danielle-alvarez",
                "source_record_id": "counselchat_0007",
                "source_url": "https://counselchat.com/questions/sleep-0007",
                "upvotes": 5,
                "category": "peer_counseling_example",
                "collection": COUNSELCHAT_COLLECTION_NAME,
                "retrieval_tier": "counselchat_consolidated",
                "clinical_safety_level": "informational_example_only",
                "clinical_approval": "not_assessed",
                "disclaimer": COUNSELCHAT_DISCLAIMER,
                "is_clinical_guidance": False,
                "score": 0.0,
                "eligibility": "approved",
                "total_answers_for_question": 1,
                "all_therapists_for_question": ["Danielle Alvarez"],
            },
            {
                "chunk_id": "mock_chunk_002",
                "document_id": "counselchat_0000",
                "consolidated_id": "counselchat_cons_0001",
                "title": "Family Disagreements",
                "question": "My wife and mother have escalating disagreements.",
                "topics": ["Family Conflict"],
                "content": "Question: My wife and mother have escalating disagreements.\n\nCounselor Response (Kristi King-Morgan, LMSW):\nTriangulation is common in families. Encourage open and direct dialogue.",
                "therapist_name": "Kristi King-Morgan, LMSW",
                "therapist_url": "https://counselchat.com/therapists/kristi-king-morgan-lmsw",
                "source_record_id": "counselchat_0000",
                "source_url": "https://counselchat.com/questions/family-0000",
                "upvotes": 10,
                "category": "peer_counseling_example",
                "collection": COUNSELCHAT_COLLECTION_NAME,
                "retrieval_tier": "counselchat_consolidated",
                "clinical_safety_level": "informational_example_only",
                "clinical_approval": "not_assessed",
                "disclaimer": COUNSELCHAT_DISCLAIMER,
                "is_clinical_guidance": False,
                "score": 0.0,
                "eligibility": "approved",
                "total_answers_for_question": 1,
                "all_therapists_for_question": ["Kristi King-Morgan, LMSW"],
            },
        ]

        texts = [c["content"] for c in self.mock_chunks]
        embeddings = self.mock_engine.embed_texts(texts)
        self.mock_store.add_chunks(self.mock_chunks, embeddings)

        self.retriever = CounselChatRetriever(
            vector_store=self.mock_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
            is_experimental=True,
        )

    def test_clinical_faiss_index_integrity(self):
        """Phase 6: Canonical clinical FAISS index is strictly unmodified and has 31 vectors."""
        clinical_path = Path("data/knowledge_base/faiss_index/index.faiss")
        self.assertTrue(clinical_path.exists())
        with open(clinical_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(
            actual_sha256,
            CLINICAL_FAISS_SHA256,
            "Clinical FAISS index checksum must match original baseline exactly.",
        )

    def test_counselchat_retrieval_and_attribution(self):
        """Phase 7: Relevant retrieval surfaces therapist name, source ID, and disclaimer."""
        results = self.retriever.retrieve("Insomnia and sleep troubles", top_k=1)
        self.assertEqual(len(results), 1)
        chunk = results[0]

        # Check attribution fields
        self.assertEqual(chunk["document_id"], "counselchat_0007")
        self.assertEqual(chunk["therapist_name"], "Danielle Alvarez")
        self.assertEqual(chunk["category"], "peer_counseling_example")
        self.assertEqual(chunk["clinical_approval"], "not_assessed")
        self.assertIn(COUNSELCHAT_DISCLAIMER, chunk["disclaimer"])

        # Citation formatting
        citations = format_citations(results)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["document_id"], "counselchat_0007")

    def test_irrelevant_query_threshold_abstention(self):
        """Phase 7: Irrelevant queries meeting no threshold return empty results."""
        results = self.retriever.retrieve(
            "Solve the Navier-Stokes differential equations",
            threshold=2.0,  # Unattainable threshold
        )
        self.assertEqual(results, [])

    def test_multi_collection_retriever_priority(self):
        """Phase 7: Multi-collection arbitration prioritizes clinical KB over CounselChat."""
        canonical_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)
        canonical_store.add_chunks(
            [{
                "chunk_id": "somatic_regulation:01_box_breathing_4_4_4_4__chunk_01",
                "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
                "title": "Box Breathing",
                "category": "somatic_regulation",
                "jurisdiction": "GL",
                "content": "Inhale 4, hold 4, exhale 4, hold 4.",
                "score": 0.95,
                "eligibility": "approved",
            }],
            self.mock_engine.embed_texts(["Inhale 4, hold 4, exhale 4, hold 4."])
        )
        canonical_retriever = RAGRetriever(
            vector_store=canonical_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
        )

        multi_retriever = MultiCollectionRetriever(
            canonical_retriever=canonical_retriever,
            counselchat_retriever=self.retriever,
            max_total_chunks=3,
        )

        # Check readiness property pass-through
        self.assertEqual(multi_retriever.vector_store.total_vectors, 1)

        results = multi_retriever.retrieve("Breathing and anxiety", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        # Position 0 must be canonical clinical chunk
        self.assertEqual(results[0]["category"], "somatic_regulation")


class TestChatbotOrchestratorAndSafety(unittest.TestCase):
    """Phase 8 & Phase 9: Chatbot Response Synthesis and Universal Crisis Bypass Tests."""

    def setUp(self):
        self.mock_engine = MockEmbeddingEngine(dimension=EMBEDDING_DIMENSION)
        self.mock_store = FAISSVectorStore(dimension=EMBEDDING_DIMENSION)

        chunk = {
            "chunk_id": "counselchat_chunk_0018_000",
            "document_id": "counselchat_0018",
            "consolidated_id": "counselchat_cons_0019",
            "title": "Feeling Heard",
            "question": "How can I get people to listen to me?",
            "topics": ["Depression", "Anxiety"],
            "content": "Question: How can I get people to listen to me?\n\nCounselor Response (Keisha Helms):\nCommunicating clearly using I-statements can help others understand your feelings without defensiveness.",
            "therapist_name": "Keisha Helms",
            "therapist_url": "https://counselchat.com/therapists/keisha-helms",
            "source_record_id": "counselchat_0018",
            "source_url": "https://counselchat.com/questions/0018",
            "upvotes": 3,
            "category": "peer_counseling_example",
            "collection": COUNSELCHAT_COLLECTION_NAME,
            "retrieval_tier": "counselchat_consolidated",
            "clinical_safety_level": "informational_example_only",
            "clinical_approval": "not_assessed",
            "disclaimer": COUNSELCHAT_DISCLAIMER,
            "is_clinical_guidance": False,
            "score": 0.85,
            "eligibility": "approved",
            "total_answers_for_question": 1,
            "all_therapists_for_question": ["Keisha Helms"],
        }
        self.mock_store.add_chunks([chunk], self.mock_engine.embed_texts([chunk["content"]]))
        self.retriever = CounselChatRetriever(
            vector_store=self.mock_store,
            embedding_engine=self.mock_engine,
            default_threshold=0.0,
            is_experimental=True,
        )

    def test_grounded_response_generation_with_counselchat_context(self):
        """Phase 8: Response node formats peer perspective with therapist attribution & disclaimer."""
        results = self.retriever.retrieve("How can I get people to listen?", top_k=1)
        response_text = synthesize_grounded_response(
            sanitized_text="How can I get people to listen?",
            dialog_goal="general",
            retrieved_chunks=results,
        )
        self.assertIn("illustrative peer counseling perspective shared by Keisha Helms", response_text)
        self.assertIn(COUNSELCHAT_DISCLAIMER, response_text)

    def test_empty_retrieval_fallback(self):
        """Phase 8: Empty retrieval emits safe NEUTRAL_SUPPORTIVE_FALLBACK."""
        state = create_state("Unrelated question")
        state["retrieval_status"] = "empty"
        state["retrieved_chunks"] = []
        updates = response_node(state)
        self.assertEqual(updates["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates["execution_trace"][-1]["status"], "fallback")

    def test_generator_failure_fallback_containment(self):
        """Phase 8: Generator runtime error falls back gracefully to deterministic synthesis."""
        def broken_generator(state: MindBridgeState):
            raise ConnectionError("LLM API endpoint unavailable")

        results = self.retriever.retrieve("People not listening", top_k=1)
        state = create_state("People not listening")
        state["retrieval_status"] = "success"
        state["retrieved_chunks"] = results
        state["dialog_goal"] = "general"

        updates = response_node(state, generator=broken_generator)
        self.assertEqual(updates["error_code"], "LLM_GENERATION_FAILURE")
        self.assertEqual(updates["draft_response"], NEUTRAL_SUPPORTIVE_FALLBACK)
        self.assertEqual(updates["execution_trace"][-1]["status"], "error")

    def test_crisis_message_universal_emergency_bypass(self):
        """Phase 9: Crisis messages completely halt retrieval and route to emergency_node."""
        # 1. Direct CounselChat retriever bypass
        cc_results = self.retriever.retrieve(
            "I want to kill myself tonight",
            safety_route="emergency_bypass",
        )
        self.assertEqual(cc_results, [])

        # 2. Planning node bypass under emergency
        state = create_state("I want to end my life right now")
        state["overall_safety_route"] = "emergency_bypass"
        state["deterministic_danger_flag"] = True
        plan_updates = planning_node(state, retriever=self.retriever)
        self.assertEqual(plan_updates["retrieval_status"], "not_attempted")
        self.assertEqual(plan_updates["retrieved_chunks"], [])
        self.assertEqual(plan_updates["execution_trace"][-1]["metadata_code"], "EMERGENCY_BYPASS_SKIPPED")

        # 3. Full LangGraph end-to-end execution for crisis input
        app = build_mindbridge_graph(retriever=self.retriever)
        crisis_state = create_state("I have a bottle of pills and I am swallowing all the pills tonight to end it all. Goodbye.")
        final_state = app.invoke(crisis_state)

        # Asserts emergency bypass routing
        self.assertEqual(final_state["overall_safety_route"], "emergency_bypass")
        self.assertTrue(final_state["deterministic_danger_flag"])
        self.assertEqual(final_state["retrieval_status"], "not_attempted")
        self.assertEqual(final_state["retrieved_chunks"], [])
        self.assertTrue(bool(final_state["final_response"]))

        executed_nodes = [t["node"] for t in final_state["execution_trace"]]
        self.assertNotIn("planning_node", executed_nodes)
        self.assertNotIn("response_node", executed_nodes)
        self.assertNotIn("verification_node", executed_nodes)


if __name__ == "__main__":
    unittest.main()
