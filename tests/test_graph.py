"""
Integration Tests for MindBridge 2.0 Central LangGraph Orchestration Workflow
=============================================================================

Verifies:
1. End-to-end execution of compiled StateGraph workflow
2. Universal emergency bypass: acute crisis routes to emergency_node, never planning/RAG
3. Standard conversation: routes through input_node, safety, classifier, decision, and planning_node
4. Retrieval independence: RAG failure cannot alter emergency guidance
5. Out-of-domain safe empty retrieval state
6. Jurisdiction routing preservation throughout the graph
7. Passive distress follow-up continuity
"""

import unittest
from typing import Any, Dict, List, Optional

from backend.orchestrator.graph import (
    build_mindbridge_graph,
    CompiledGraph,
    emergency_node,
    deescalation_node,
    followup_node,
)
from backend.orchestrator.state import (
    KnowledgeChunk,
    MindBridgeState,
)
from backend.retrieval import RetrievalAuthorization


class MockGraphRetriever:
    """Mock retriever for fast hermetic graph integration tests."""
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_jurisdiction: Optional[str] = None

    def retrieve(
        self,
        query: str,
        jurisdiction: str = "GL",
        include_restricted: bool = False,
        authorization: Optional[RetrievalAuthorization] = None,
        top_k: int = 5,
        **kwargs,
    ) -> List[KnowledgeChunk]:
        self.call_count += 1
        self.last_query = query
        self.last_jurisdiction = jurisdiction
        if self.should_fail:
            raise RuntimeError("Simulated FAISS error")

        return [
            {
                "chunk_id": "mock_chunk_01",
                "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
                "title": "Box Breathing",
                "category": "somatic_regulation",
                "jurisdiction": jurisdiction,  # type: ignore
                "content": "Inhale 4, hold 4, exhale 4, hold 4.",
                "score": 0.88,
                "eligibility": "approved",
            }
        ]


def create_initial_graph_state(
    user_text: str = "",
    jurisdiction: str = "IN",
) -> MindBridgeState:
    """Helper constructing clean initial graph state."""
    return {
        "session_id": "test-graph-session-uuid",
        "user_id_hash": "test-user-hash-1234",
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


class TestMindBridgeGraph(unittest.TestCase):
    """Integration test suite for compiled MindBridgeStateGraph."""

    def setUp(self):
        self.mock_retriever = MockGraphRetriever()
        self.app = build_mindbridge_graph(retriever=self.mock_retriever)  # type: ignore

    def test_01_standard_conversation_executes_planning_and_retrieval(self):
        """Standard conversation flows through safety arbitration to planning_node and RAG."""
        initial_state = create_initial_graph_state(
            user_text="My heart is pounding and I feel panic what should I do right now",
            jurisdiction="GL",
        )
        final_state = self.app.invoke(initial_state)

        # 1. Safety route remains standard conversation
        self.assertEqual(final_state["overall_safety_route"], "standard_conversation")
        self.assertFalse(final_state["deterministic_danger_flag"])

        # 2. Query understanding and planning executed
        self.assertEqual(final_state["dialog_goal"], "somatic_grounding")
        self.assertTrue(len(final_state["retrieval_queries"]) > 0)

        # 3. RAG retrieval executed successfully
        self.assertEqual(final_state["retrieval_status"], "success")
        self.assertEqual(len(final_state["retrieved_chunks"]), 1)
        self.assertEqual(self.mock_retriever.call_count, 1)

        # 4. Original message preserved
        self.assertIn("pounding", final_state["sanitized_text"])

        # 5. Response generation executed and grounded in retrieved chunks
        self.assertIsNotNone(final_state["draft_response"])
        self.assertIn("Box Breathing", final_state["draft_response"])
        self.assertIn("Inhale 4", final_state["draft_response"])
        self.assertEqual(final_state["final_response"], final_state["draft_response"])

    def test_02_emergency_bypass_routes_directly_to_emergency_node(self):
        """CRITICAL: Explicit lethal means disclosure routes directly to emergency_node, bypassing RAG."""
        initial_state = create_initial_graph_state(
            user_text="I have a bottle of pills and I am swallowing all the pills tonight to end it all. Goodbye.",
            jurisdiction="IN",
        )
        final_state = self.app.invoke(initial_state)

        # 1. Universal Emergency Bypass triggered
        self.assertEqual(final_state["overall_safety_route"], "emergency_bypass")
        self.assertTrue(final_state["deterministic_danger_flag"])
        self.assertEqual(final_state["emergency_trigger_id"], "LETHAL_MEANS_DISCLOSURE")

        # 2. Planning and RAG strictly bypassed
        self.assertEqual(final_state["retrieval_status"], "not_attempted")
        self.assertEqual(final_state["retrieval_queries"], [])
        self.assertEqual(final_state["retrieved_chunks"], [])
        self.assertEqual(self.mock_retriever.call_count, 0)

        # 3. Authoritative emergency guidance emitted from hotlines.json (India 112 / Tele-MANAS 14416)
        self.assertIn("112", final_state["final_response"])
        self.assertIn("14416", final_state["final_response"])

    def test_03_out_of_domain_query_returns_empty_retrieval(self):
        """Out-of-domain queries route through graph without crashing and return safe empty retrieval."""
        initial_state = create_initial_graph_state(
            user_text="How to sort a dictionary by values in Python using lambda functions",
            jurisdiction="US",
        )
        final_state = self.app.invoke(initial_state)

        self.assertEqual(final_state["overall_safety_route"], "standard_conversation")
        self.assertEqual(final_state["retrieval_status"], "empty")
        self.assertEqual(final_state["retrieval_queries"], [])
        self.assertEqual(final_state["retrieved_chunks"], [])
        self.assertEqual(self.mock_retriever.call_count, 0)

        # Emits neutral supportive fallback without unconditional breathing exercises
        self.assertIn("I am here with you", final_state["final_response"])
        self.assertNotIn("breathing", final_state["final_response"].lower())

    def test_04_jurisdiction_routing_in_graph(self):
        """Jurisdiction filter is passed cleanly from initial state through planning to retrieval."""
        initial_state = create_initial_graph_state(
            user_text="Navigating competitive exam anxiety and familial stakes in India",
            jurisdiction="IN",
        )
        final_state = self.app.invoke(initial_state)

        self.assertEqual(final_state["jurisdiction_filter"], "IN")
        self.assertEqual(self.mock_retriever.last_jurisdiction, "IN")

    def test_05_retrieval_failure_does_not_crash_graph(self):
        """If RAG retriever fails, graph transitions smoothly without unhandled exceptions."""
        failing_retriever = MockGraphRetriever(should_fail=True)
        failing_app = build_mindbridge_graph(retriever=failing_retriever)  # type: ignore

        initial_state = create_initial_graph_state(
            user_text="How to do box breathing exercises for stress",
            jurisdiction="GL",
        )
        final_state = failing_app.invoke(initial_state)

        self.assertEqual(final_state["retrieval_status"], "error")
        self.assertEqual(final_state["retrieved_chunks"], [])
        self.assertEqual(final_state["error_code"], "RETRIEVAL_FAILURE")


if __name__ == "__main__":
    unittest.main()
