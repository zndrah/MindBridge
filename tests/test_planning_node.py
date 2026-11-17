"""
Unit Tests for MindBridge 2.0 Query Understanding & Planning Node
==================================================================

Verifies:
1. Informal language to canonical retrieval-query mapping
2. Concept extraction for anxiety, panic, depression, sleep, exam stress, and restructuring
3. Original user message preservation throughout the state transition
4. Jurisdiction preservation and routing
5. Authorization context creation for clinical screening rubrics
6. Refusal to convert crisis or self-harm disclosures into search queries
7. Universal emergency bypass behavior
8. Graceful failure containment when retrieval throws an exception
9. Out-of-domain query handling with safe empty results
"""

from datetime import datetime, timezone
import unittest
from typing import Any, Dict, List, Optional

from backend.orchestrator.planning_node import planning_node
from backend.orchestrator.query_understanding import (
    understand_query,
    QueryUnderstandingResult,
)
from backend.orchestrator.state import (
    KnowledgeChunk,
    MindBridgeState,
)
from backend.retrieval import (
    FAISSVectorStore,
    MockEmbeddingEngine,
    RAGRetriever,
    RetrievalAuthorization,
)


def create_test_state(
    sanitized_text: str = "",
    overall_safety_route: str = "standard_conversation",
    deterministic_danger_flag: bool = False,
    jurisdiction_filter: str = "GL",
    phq9_item9_endorsed: bool = False,
    phq9_followup_status: str = "not_started",
) -> MindBridgeState:
    """Helper constructing conforming MindBridgeState for testing."""
    return {
        "session_id": "test-session-uuid",
        "user_id_hash": "test-user-hash",
        "messages": [{"role": "user", "content": sanitized_text}] if sanitized_text else [],
        "sanitized_text": sanitized_text,
        "deterministic_danger_flag": deterministic_danger_flag,
        "emergency_trigger_id": None,
        "roberta_prob": 0.05,
        "roberta_crisis_flag": False,
        "classifier_status": "completed",
        "overall_safety_route": overall_safety_route,  # type: ignore
        "phq9_item9_endorsed": phq9_item9_endorsed,
        "phq9_followup_status": phq9_followup_status,  # type: ignore
        "phq9_risk_level": "none",
        "phq9_dimensions": {},
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
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }


class MockHermeticRetriever:
    """Deterministic, fast mock retriever for isolated unit tests."""
    def __init__(self, canned_chunks: Optional[List[KnowledgeChunk]] = None, should_fail: bool = False):
        self.canned_chunks = canned_chunks or []
        self.should_fail = should_fail
        self.last_query: Optional[str] = None
        self.last_jurisdiction: Optional[str] = None
        self.last_include_restricted: Optional[bool] = None
        self.last_auth: Optional[RetrievalAuthorization] = None

    def retrieve(
        self,
        query: str,
        jurisdiction: str = "GL",
        include_restricted: bool = False,
        authorization: Optional[RetrievalAuthorization] = None,
        top_k: int = 5,
        **kwargs,
    ) -> List[KnowledgeChunk]:
        if self.should_fail:
            raise RuntimeError("Simulated FAISS vector store failure")
        self.last_query = query
        self.last_jurisdiction = jurisdiction
        self.last_include_restricted = include_restricted
        self.last_auth = authorization
        return self.canned_chunks


class TestQueryUnderstanding(unittest.TestCase):
    """Unit tests for deterministic query understanding rules."""

    def test_01_informal_low_motivation_mapping(self):
        """Maps informal depressive inertia and bed-inactivity expressions to behavioral activation."""
        queries = [
            "I feel empty and do not have the energy to get out of bed.",
            "I have zero motivation and stay in bed all day doing nothing",
            "How to overcome depressive inertia when everything feels pointless",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertEqual(res.dialog_goal, "cognitive_reframing")
            self.assertFalse(res.is_crisis_related)
            self.assertFalse(res.is_out_of_domain)
            self.assertIn("behavioral_activation", res.identified_concepts)
            self.assertTrue(len(res.retrieval_queries) > 0)
            self.assertIn("Behavioral activation", res.retrieval_queries[0])

    def test_02_anxiety_and_acute_panic_mapping(self):
        """Maps acute panic and somatic hyperventilation expressions to rapid de-escalation."""
        queries = [
            "My heart is pounding and I feel like I cannot breathe what should I do right now",
            "I am having a sudden intense panic attack and freaking out",
            "I need immediate physical de-escalation to calm down fast",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertEqual(res.dialog_goal, "somatic_grounding")
            self.assertIn("acute_panic", res.identified_concepts)
            self.assertTrue(len(res.retrieval_queries) > 0)
            self.assertIn("Breathing techniques and rapid somatic de-escalation", res.retrieval_queries[0])

    def test_03_sleep_and_insomnia_mapping(self):
        """Maps insomnia and tossing/turning expressions to CBT-I stimulus control."""
        queries = [
            "What is the 20 minute rule in stimulus control therapy for insomnia",
            "I keep lying awake in bed tossing and turning for hours unable to sleep",
            "How to fix irregular sleep wake cycles with circadian hygiene",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertIn("insomnia", res.identified_concepts)
            self.assertIn("stimulus_control", res.identified_concepts)
            self.assertIn("Stimulus control therapy", res.retrieval_queries[0])

    def test_04_competitive_exam_stress_mapping(self):
        """Maps competitive exam stress (JEE/NEET/Kota) to exam anxiety psychoeducation."""
        queries = [
            "I am having panic attacks about failing my JEE and NEET entrance exams",
            "My parents have spent their savings on my coaching classes and I feel crushing academic pressure",
            "Mock test score anxiety and fear of disappointing my family",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertIn("competitive_exam_stress", res.identified_concepts)
            self.assertIn("Navigating competitive exam anxiety", res.retrieval_queries[0])

    def test_05_cognitive_restructuring_and_catastrophizing(self):
        """Maps catastrophizing and worst-case scenario thoughts to Thought Court."""
        queries = [
            "How to use the Thought Court exercise to challenge catastrophic predictions",
            "I keep assuming the worst possible scenario will definitely happen at work",
            "Negative thought loop challenging catastrophic cognitive distortions",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertEqual(res.dialog_goal, "cognitive_reframing")
            self.assertIn("catastrophizing", res.identified_concepts)
            self.assertIn("Thought Court", res.retrieval_queries[0])

    def test_06_worry_postponement_racing_thoughts(self):
        """Maps nighttime rumination and racing thoughts to CBT worry postponement log."""
        queries = [
            "How does the CBT worry postponement log work for racing nighttime thoughts",
            "Writing down worries during the day instead of ruminating all night in bed",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertEqual(res.dialog_goal, "cognitive_reframing")
            self.assertIn("worry_postponement", res.identified_concepts)
            self.assertIn("worry postponement", res.retrieval_queries[0].lower())

    def test_07_screening_rubrics_require_restricted_access(self):
        """Identifies clinical screening rubrics and flags requires_restricted_access=True."""
        queries = [
            "What are the cutoff scores and severity tiers for the PHQ-9 depression scale",
            "GAD-7 anxiety and PSS-10 stress screening rubric overview",
            "NIMHANS digital mental health triage framework clinical red lines",
        ]
        for q in queries:
            res = understand_query(q)
            self.assertEqual(res.dialog_goal, "screening")
            self.assertTrue(res.requires_restricted_access)
            self.assertIn("clinical_screening_rubric", res.identified_concepts)

    def test_08_out_of_domain_queries(self):
        """Correctly identifies out-of-domain queries and emits empty query list."""
        ood_queries = [
            "What is the recipe for baking chocolate chip cookies with unsalted butter",
            "How to sort a dictionary by values in Python using lambda functions",
            "Who was the first emperor of the Roman Empire after Julius Caesar",
            "What is the current stock price and market capitalization of Apple",
            "How does a four stroke internal combustion engine generate power",
        ]
        for q in ood_queries:
            res = understand_query(q)
            self.assertTrue(res.is_out_of_domain)
            self.assertEqual(len(res.retrieval_queries), 0)

    def test_09_crisis_disclosure_refusal_defense_in_depth(self):
        """Refuses to convert explicit self-harm disclosures into ordinary mental-health search queries."""
        crisis_utterances = [
            "I have pills and I am ending my life tonight goodbye",
            "Writing my suicide note right now",
            "I have a knife to my throat",
            "Swallowing all the pills in the bottle",
        ]
        for c in crisis_utterances:
            res = understand_query(c)
            self.assertTrue(res.is_crisis_related)
            self.assertEqual(len(res.retrieval_queries), 0)
            self.assertEqual(len(res.identified_concepts), 0)

    def test_10_empty_and_whitespace_input(self):
        """Empty or whitespace input returns safe out-of-domain empty query."""
        self.assertEqual(understand_query("").retrieval_queries, [])
        self.assertEqual(understand_query("   \t\n  ").retrieval_queries, [])
        self.assertEqual(understand_query(None).retrieval_queries, [])  # type: ignore


class TestPlanningNode(unittest.TestCase):
    """Unit tests for the planning_node function."""

    def test_01_emergency_bypass_skips_retrieval(self):
        """Emergency bypass state halts planning and prevents retrieval without executing queries."""
        state = create_test_state(
            sanitized_text="How to do box breathing",
            overall_safety_route="emergency_bypass",
            deterministic_danger_flag=True,
        )
        mock_retriever = MockHermeticRetriever()
        updates = planning_node(state, retriever=mock_retriever)  # type: ignore

        self.assertEqual(updates["retrieval_status"], "not_attempted")
        self.assertEqual(updates["retrieval_queries"], [])
        self.assertEqual(updates["retrieved_chunks"], [])
        self.assertIsNone(mock_retriever.last_query)
        self.assertEqual(updates["execution_trace"][-1]["status"], "skipped")

    def test_02_original_message_preservation(self):
        """Original user message is strictly preserved and not overwritten in state."""
        original = "I feel totally empty and cannot find any energy to get out of bed"
        state = create_test_state(sanitized_text=original)
        mock_retriever = MockHermeticRetriever(canned_chunks=[{
            "chunk_id": "chunk_01",
            "document_id": "cognitive_reframing:03_behavioral_activation_inertia",
            "title": "Behavioral Activation",
            "category": "cognitive_reframing",
            "jurisdiction": "GL",
            "content": "Behavioral Activation text",
            "score": 0.85,
            "eligibility": "approved",
        }])

        updates = planning_node(state, retriever=mock_retriever)  # type: ignore
        # State updates do not overwrite sanitized_text
        self.assertNotIn("sanitized_text", updates)
        # Original text remains available in state
        self.assertEqual(state["sanitized_text"], original)
        # Separate retrieval query was generated
        self.assertTrue(len(updates["retrieval_queries"]) > 0)
        self.assertIn("Behavioral activation", updates["retrieval_queries"][0])

    def test_03_jurisdiction_preservation(self):
        """User jurisdiction is preserved and passed to RAGRetriever."""
        state_in = create_test_state(
            sanitized_text="I am stressed about JEE competitive entrance exams",
            jurisdiction_filter="IN",
        )
        mock_retriever = MockHermeticRetriever()
        updates = planning_node(state_in, retriever=mock_retriever)  # type: ignore

        self.assertEqual(updates["jurisdiction_filter"], "IN")
        self.assertEqual(mock_retriever.last_jurisdiction, "IN")

    def test_04_restricted_screening_authorization_context(self):
        """Screening rubric queries do NOT automatically authorize restricted access without trusted authorization."""
        # 1. Ordinary query without trusted authorization must have include_restricted=False
        state = create_test_state(
            sanitized_text="What are the PHQ-9 depression screening cutoff scores",
            overall_safety_route="standard_conversation",
        )
        mock_retriever = MockHermeticRetriever()
        updates = planning_node(state, retriever=mock_retriever)  # type: ignore

        self.assertEqual(updates["dialog_goal"], "screening")
        self.assertFalse(mock_retriever.last_include_restricted)
        self.assertIsNone(mock_retriever.last_auth)

        # 2. Query WITH explicit trusted authorization context successfully enables restricted retrieval
        trusted_auth = RetrievalAuthorization(
            authorized_by="clinical_supervisor_gate",
            reason="supervised_screening_evaluation",
        )
        state_with_auth = dict(state)
        state_with_auth["retrieval_authorization"] = trusted_auth
        mock_retriever_2 = MockHermeticRetriever()
        updates_2 = planning_node(state_with_auth, retriever=mock_retriever_2)  # type: ignore

        self.assertEqual(updates_2["dialog_goal"], "screening")
        self.assertTrue(mock_retriever_2.last_include_restricted)
        self.assertEqual(mock_retriever_2.last_auth, trusted_auth)

    def test_05_out_of_domain_returns_empty_retrieval(self):
        """Out-of-domain queries return empty retrieval status without calling retriever."""
        state = create_test_state(sanitized_text="How to bake chocolate chip cookies with unsalted butter")
        mock_retriever = MockHermeticRetriever()
        updates = planning_node(state, retriever=mock_retriever)  # type: ignore

        self.assertEqual(updates["retrieval_status"], "empty")
        self.assertEqual(updates["retrieval_queries"], [])
        self.assertEqual(updates["retrieved_chunks"], [])
        self.assertIsNone(mock_retriever.last_query)

    def test_06_retriever_failure_handled_gracefully(self):
        """Retriever exceptions do not crash planning; sets controlled error status."""
        state = create_test_state(sanitized_text="I feel panic and my heart is pounding")
        failing_retriever = MockHermeticRetriever(should_fail=True)
        updates = planning_node(state, retriever=failing_retriever)  # type: ignore

        self.assertEqual(updates["retrieval_status"], "error")
        self.assertEqual(updates["retrieved_chunks"], [])
        self.assertEqual(updates["error_code"], "RETRIEVAL_FAILURE")
        self.assertEqual(updates["error_component"], "planning_node")
        self.assertEqual(updates["execution_trace"][-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
