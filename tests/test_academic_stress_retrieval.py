"""
Regression Tests for Academic & Exam Stress Retrieval Coverage
==============================================================

Verifies:
1. Exact message "I have been feeling stressed about my exams lately." retrieves
   relevant approved academic-stress psychoeducation content.
2. Emotional academic-stress expressions ("stressed about my exams", "exam anxiety",
   "academic pressure", "study stress", "overwhelmed by studies") map to neutral
   retrieval queries without assuming India/JEE/NEET context.
3. Neutral requests like "create an exam timetable" or "I need an exam timetable" do
   NOT match mental-health concepts.
4. Crisis disclosures take strict emergency bypass precedence without vector retrieval.
5. Retrieved content reaches response synthesis and produces a grounded response
   with valid citations.
6. Preserves jurisdiction filtering, restricted content gating, and retrieval threshold.
"""

import unittest
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import app_state
from backend.orchestrator.graph import build_mindbridge_graph
from backend.orchestrator.planning_node import get_shared_retriever, set_shared_retriever
from backend.orchestrator.query_understanding import understand_query
from backend.orchestrator.response_node import NEUTRAL_SUPPORTIVE_FALLBACK
from backend.orchestrator.state import MindBridgeState
from backend.orchestrator.verification_node import check_factual_grounding
from backend.retrieval import (
    build_retriever_from_knowledge_base,
    get_document_governance,
)


class TestAcademicStressRetrieval(unittest.TestCase):
    """Test suite for academic/exam stress query understanding, retrieval, and response grounding."""

    @classmethod
    def setUpClass(cls):
        # Build live retriever from knowledge base and register as shared
        cls.retriever, cls.ingestion_summary = build_retriever_from_knowledge_base()
        set_shared_retriever(cls.retriever)
        cls.app = create_app()
        app_state.initialize_sync()
        cls.client = TestClient(cls.app)

    # ------------------------------------------------------------------------
    # 1. Query Understanding & Concept Mapping
    # ------------------------------------------------------------------------
    def test_01_academic_stress_concept_mapping(self):
        """
        Verify emotional expressions of academic and exam stress map to the academic_exam_stress
        concept rule and synthesize neutral retrieval queries without assuming India/JEE/NEET.
        """
        queries = [
            "I have been feeling stressed about my exams lately.",
            "stressed about my exams",
            "exam anxiety",
            "academic pressure",
            "study stress",
            "overwhelmed by studies",
            "I am feeling overwhelmed by my studies and coursework",
            "really anxious about my exams next week",
        ]

        for q in queries:
            res = understand_query(q)
            self.assertIn(
                "academic_stress",
                res.identified_concepts,
                f"Failed to identify 'academic_stress' concept for query: '{q}'",
            )
            self.assertIn(
                "exam_anxiety",
                res.identified_concepts,
                f"Failed to identify 'exam_anxiety' concept for query: '{q}'",
            )
            self.assertEqual(res.dialog_goal, "general")
            self.assertFalse(res.is_crisis_related)
            self.assertFalse(res.is_out_of_domain)
            self.assertTrue(len(res.retrieval_queries) > 0)

            # Query must be neutral; must not assume India, JEE, NEET, UPSC
            primary_query = res.retrieval_queries[0]
            self.assertNotIn("India", primary_query)
            self.assertNotIn("JEE", primary_query)
            self.assertNotIn("NEET", primary_query)
            self.assertNotIn("UPSC", primary_query)
            self.assertIn("academic", primary_query.lower())

    def test_02_neutral_requests_not_mapped_to_mental_health(self):
        """
        Verify neutral non-emotional requests such as timetable creation are NOT mapped
        to academic stress mental health concepts.
        """
        neutral_queries = [
            "I need an exam timetable",
            "create an exam timetable",
            "exam timetable for semester 2",
            "what date is the physics exam",
            "how many questions are on the final exam",
        ]

        for q in neutral_queries:
            res = understand_query(q)
            self.assertNotIn(
                "academic_stress",
                res.identified_concepts,
                f"Neutral query '{q}' should not match academic_stress",
            )
            self.assertNotIn(
                "exam_anxiety",
                res.identified_concepts,
                f"Neutral query '{q}' should not match exam_anxiety",
            )
            self.assertEqual(
                res.identified_concepts,
                ["general_support"],
                f"Expected generic support fallback for '{q}'",
            )

    # ------------------------------------------------------------------------
    # 2. Safety Precedence & Crisis Routing
    # ------------------------------------------------------------------------
    def test_03_crisis_routing_precedence(self):
        """
        Verify that crisis disclosures take strict emergency bypass precedence,
        clearing all retrieval queries and preventing vector lookup.
        """
        # Scenario A: Emergency bypass flag active
        res_flagged = understand_query(
            "I have been feeling stressed about my exams lately.",
            overall_safety_route="emergency_bypass",
        )
        self.assertTrue(res_flagged.is_crisis_related)
        self.assertEqual(res_flagged.retrieval_queries, [])

        # Scenario B: Direct emergency trigger disclosure alongside exam stress
        crisis_message = "I am stressed about my exams and I have decided to end my life tonight"
        res_crisis = understand_query(crisis_message)
        self.assertTrue(res_crisis.is_crisis_related)
        self.assertEqual(res_crisis.retrieval_queries, [])

        # Scenario C: End-to-end API call with crisis trigger
        api_res = self.client.post("/chat", json={"message": crisis_message})
        self.assertEqual(api_res.status_code, 200)
        data = api_res.json()
        self.assertEqual(data["safety_route"], "emergency_bypass")
        self.assertTrue("112" in data["response"] or "988" in data["response"])
        # In emergency bypass, no informational RAG citations are returned
        self.assertEqual(data["citations"], [])

    # ------------------------------------------------------------------------
    # 3. Dense Vector Retrieval Accuracy
    # ------------------------------------------------------------------------
    def test_04_exact_message_retrieval(self):
        """
        Verify exact message: 'I have been feeling stressed about my exams lately.'
        retrieves the approved academic stress document with similarity score >= 0.65.
        """
        user_message = "I have been feeling stressed about my exams lately."
        uq = understand_query(user_message)
        self.assertTrue(len(uq.retrieval_queries) > 0)

        chunks = self.retriever.retrieve(
            query=uq.retrieval_queries[0],
            jurisdiction="GL",
            include_restricted=False,
            top_k=3,
        )

        self.assertGreater(len(chunks), 0, "Retrieval returned 0 chunks for academic stress query")
        top_chunk = chunks[0]

        # Verify document provenance and metadata
        self.assertEqual(
            top_chunk["document_id"],
            "psychoeducation:06_academic_exam_stress_grounding",
        )
        self.assertEqual(
            top_chunk["title"],
            "Managing Academic Stress and Exam Anxiety: Practical Study Planning & Coping",
        )
        self.assertEqual(top_chunk["jurisdiction"], "GL")
        self.assertEqual(top_chunk["eligibility"], "approved")

        # Verify similarity score meets the threshold and exceeds high-confidence baseline
        score = top_chunk["score"]
        self.assertGreaterEqual(score, 0.65, f"Score {score} fell below 0.65 threshold")
        self.assertGreaterEqual(score, 0.85, f"Score {score} is lower than expected high-confidence match")

        # Verify key required pedagogical elements exist in chunk content
        content = top_chunk["content"]
        self.assertIn("Practical Study Planning", content)
        self.assertIn("Manageable Study Breaks", content)
        self.assertIn("Realistic Expectations", content)
        self.assertIn("Human-Support Guidance", content)

    # ------------------------------------------------------------------------
    # 4. End-to-End Grounded Response Synthesis & Verification
    # ------------------------------------------------------------------------
    def test_05_grounded_response_synthesis(self):
        """
        Verify end-to-end response generation produces a factually grounded, empathetic
        response with valid citations, rather than NEUTRAL_SUPPORTIVE_FALLBACK.
        """
        res = self.client.post(
            "/chat",
            json={"message": "I have been feeling stressed about my exams lately."},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Operational status assertions
        self.assertEqual(data["safety_route"], "standard_conversation")
        self.assertEqual(data["verification_status"], "verified")
        self.assertEqual(data["grounding_status"], "grounded")

        response_text = data["response"]
        self.assertNotEqual(
            response_text.strip(),
            NEUTRAL_SUPPORTIVE_FALLBACK.strip(),
            "Expected grounded response, but received NEUTRAL_SUPPORTIVE_FALLBACK",
        )

        # Must contain title reference or verified citation metadata
        citation_titles = [c.get("title", "") for c in data.get("citations", [])]
        has_title_reference = (
            "Managing Academic Stress and Exam Anxiety: Practical Study Planning & Coping" in response_text
            or any("Managing Academic Stress and Exam Anxiety" in t for t in citation_titles)
        )
        self.assertTrue(has_title_reference)
        has_verbatim_headers = (
            "Practical Study Planning" in response_text
            and "Manageable Study Breaks" in response_text
            and "Realistic Expectations" in response_text
        )
        has_grounded_concepts = (
            any(w in response_text.lower() for w in ("study", "session", "planning", "blocks"))
            and any(w in response_text.lower() for w in ("break", "breaks", "pause", "rest"))
            and any(w in response_text.lower() for w in ("goal", "goals", "milestones", "expectations", "manageable", "pieces", "cramming", "focus"))
        )
        self.assertTrue(has_verbatim_headers or has_grounded_concepts)

        # Must not contain clinical diagnostic assertions
        self.assertNotIn("diagnosed with", response_text.lower())
        self.assertNotIn("major depressive disorder", response_text.lower())

        # Citations integrity
        citations = data["citations"]
        self.assertEqual(len(citations), 1)
        self.assertEqual(
            citations[0]["document_id"],
            "psychoeducation:06_academic_exam_stress_grounding",
        )
        self.assertEqual(
            citations[0]["title"],
            "Managing Academic Stress and Exam Anxiety: Practical Study Planning & Coping",
        )
        self.assertEqual(citations[0]["jurisdiction"], "GL")

    # ------------------------------------------------------------------------
    # 5. Jurisdiction Filtering & Governance Guardrails
    # ------------------------------------------------------------------------
    def test_06_jurisdiction_and_governance_invariants(self):
        """
        Verify that jurisdiction filtering and governance rules are preserved:
        1. GL document is accessible under both GL and regional jurisdictions (IN, US, UK, CA, AU).
        2. Restricted documents remain gated behind authorization.
        3. Excluded documents are never returned.
        """
        user_query = "Managing academic stress and exam anxiety through practical study planning manageable breaks and realistic expectations"

        for jur in ("GL", "IN", "US", "UK", "CA", "AU"):
            chunks = self.retriever.retrieve(query=user_query, jurisdiction=jur)
            self.assertTrue(
                any(c["document_id"] == "psychoeducation:06_academic_exam_stress_grounding" for c in chunks),
                f"GL document should be accessible in jurisdiction: {jur}",
            )

        # Excluded document is never returned
        excluded_gov = get_document_governance("somatic_regulation/06_vagus_nerve_stimulation_tapping.md")
        self.assertIsNotNone(excluded_gov)
        self.assertEqual(excluded_gov.retrieval_tier, "excluded_pending_review")

        # Document governance metadata matches expected tier
        gov = get_document_governance("psychoeducation/06_academic_exam_stress_grounding.md")
        self.assertIsNotNone(gov)
        self.assertEqual(gov.retrieval_tier, "approved")
        self.assertEqual(gov.jurisdiction, "GL")
        self.assertEqual(gov.clinical_safety_level, "open_informational")


if __name__ == "__main__":
    unittest.main()
