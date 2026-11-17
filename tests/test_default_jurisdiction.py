"""
MindBridge 2.0 - Default Jurisdiction Regression Test Suite (`test_default_jurisdiction.py`)
=============================================================================================

Validates the migration of the application default jurisdiction from Global (GL) / US to India (IN):
1. Fresh chat sessions inherit India (IN) as their default jurisdiction.
2. Crisis response in a default session displays India-specific emergency resources (ERSS 112, Tele-MANAS, KIRAN).
3. Manually selecting the US displays US-specific resources (988, 911).
4. Subsequent turns preserve an explicitly selected jurisdiction and do not overwrite it with the default.
5. Changing the region preserves safety state continuity (e.g. pending_safety_follow_up -> emergency_bypass).
6. GET /ready reports jurisdiction_default = "IN".
"""

import unittest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.dependencies import app_state
from backend.config import settings
from backend.safety.session_store import session_safety_store


class TestDefaultJurisdiction(unittest.TestCase):
    """Test suite for India (IN) default jurisdiction behavior and persistence."""

    @classmethod
    def setUpClass(cls):
        app_state.initialize_sync()
        cls.client = TestClient(app)

    def setUp(self):
        session_safety_store.reset_all()

    def tearDown(self):
        session_safety_store.reset_all()

    # ------------------------------------------------------------------------
    # 1. Readiness endpoint reports India default
    # ------------------------------------------------------------------------
    def test_01_readiness_reports_india_default(self):
        """GET /ready reports jurisdiction_default = 'IN'."""
        res = self.client.get("/ready")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["details"]["jurisdiction_default"], "IN")

    # ------------------------------------------------------------------------
    # 2. Fresh chat session defaults to India (IN)
    # ------------------------------------------------------------------------
    def test_02_fresh_chat_session_defaults_to_india(self):
        """A new chat session with no jurisdiction parameter inherits India (IN)."""
        sid = "test-session-fresh-india"
        res = self.client.post(
            "/chat",
            json={
                "message": "I am having a hard time sleeping due to exam stress.",
                "session_id": sid,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["jurisdiction"], "IN")
        self.assertEqual(data["safety_route"], "standard_conversation")

    # ------------------------------------------------------------------------
    # 3. Crisis response in default session displays India-specific resources
    # ------------------------------------------------------------------------
    def test_03_default_session_crisis_displays_india_resources(self):
        """Crisis disclosure in default session returns India statutory lifelines."""
        sid = "test-session-crisis-india"
        res = self.client.post(
            "/chat",
            json={
                "message": "I want to kill myself.",
                "session_id": sid,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["safety_route"], "emergency_bypass")
        self.assertEqual(data["jurisdiction"], "IN")

        # Response text contains India-specific helplines
        self.assertIn("112", data["response"])
        self.assertIn("14416", data["response"])  # Tele-MANAS
        self.assertIn("1800-599-0019", data["response"])  # KIRAN

        # Emergency resources list contains India directory items
        resource_ids = [r["resource_id"] for r in data["emergency_resources"]]
        self.assertTrue(any(rid.startswith("in-") for rid in resource_ids))
        self.assertIn("in-national-emergency-112", resource_ids)
        self.assertIn("in-tele-manas", resource_ids)
        self.assertIn("in-kiran", resource_ids)

        # US resources must NOT be present in default India response
        self.assertNotIn("us-988-lifeline", resource_ids)
        self.assertNotIn("us-emergency-911", resource_ids)

    # ------------------------------------------------------------------------
    # 4. Manually selecting US displays US-specific resources
    # ------------------------------------------------------------------------
    def test_04_manual_us_selection_displays_us_resources(self):
        """Manually selecting 'US' returns US hotlines (988, 911), not India."""
        sid = "test-session-manual-us"
        res = self.client.post(
            "/chat",
            json={
                "message": "I want to kill myself.",
                "session_id": sid,
                "jurisdiction": "US",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["safety_route"], "emergency_bypass")
        self.assertEqual(data["jurisdiction"], "US")

        # Response text contains 988 and 911
        self.assertIn("988", data["response"])
        self.assertIn("911", data["response"])

        # Emergency resources contains US directory entries
        resource_ids = [r["resource_id"] for r in data["emergency_resources"]]
        self.assertIn("us-988-lifeline", resource_ids)
        self.assertIn("us-emergency-911", resource_ids)

        # India resources must NOT be present in US response
        self.assertNotIn("in-national-emergency-112", resource_ids)
        self.assertNotIn("in-tele-manas", resource_ids)

    # ------------------------------------------------------------------------
    # 5. Subsequent turn preserves manually selected jurisdiction
    # ------------------------------------------------------------------------
    def test_05_subsequent_turn_preserves_manually_selected_jurisdiction(self):
        """Subsequent turns in a session preserve manual jurisdiction and do NOT overwrite with default."""
        sid = "test-session-preserve-us"

        # Turn 1: User explicitly selects US
        res1 = self.client.post(
            "/chat",
            json={
                "message": "I am feeling really anxious today.",
                "session_id": sid,
                "jurisdiction": "US",
            },
        )
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["jurisdiction"], "US")

        # Turn 2: User sends a message WITHOUT passing jurisdiction
        res2 = self.client.post(
            "/chat",
            json={
                "message": "I want to kill myself.",
                "session_id": sid,
                # Omitting jurisdiction
            },
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["safety_route"], "emergency_bypass")
        # Must retain US, not revert to India default
        self.assertEqual(data2["jurisdiction"], "US")
        resource_ids = [r["resource_id"] for r in data2["emergency_resources"]]
        self.assertIn("us-988-lifeline", resource_ids)
        self.assertNotIn("in-tele-manas", resource_ids)

    # ------------------------------------------------------------------------
    # 6. Region change preserves safety state continuity
    # ------------------------------------------------------------------------
    def test_06_region_change_preserves_safety_state(self):
        """Changing the region across turns does not break conversational crisis escalation."""
        sid = "test-session-region-switch"

        # Turn 1: User starts in India (default) with passive ideation
        res1 = self.client.post(
            "/chat",
            json={
                "message": "i want to die i cannot live in this world !!",
                "session_id": sid,
            },
        )
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["safety_route"], "pending_safety_follow_up")
        self.assertEqual(res1.json()["jurisdiction"], "IN")

        # Turn 2: User switches to US and confirms distress
        history2 = [
            {"role": "user", "content": "i want to die i cannot live in this world !!"},
            {"role": "assistant", "content": res1.json()["response"]},
        ]
        res2 = self.client.post(
            "/chat",
            json={
                "message": "yes i have so many thoughts of ending my life",
                "session_id": sid,
                "jurisdiction": "US",
                "conversation_history": history2,
            },
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        # Crisis escalation must succeed
        self.assertEqual(data2["safety_route"], "emergency_bypass")
        # Jurisdiction updated to US
        self.assertEqual(data2["jurisdiction"], "US")
        resource_ids = [r["resource_id"] for r in data2["emergency_resources"]]
        self.assertIn("us-988-lifeline", resource_ids)

    # ------------------------------------------------------------------------
    # 7. Safety directory resources endpoint defaults to India
    # ------------------------------------------------------------------------
    def test_07_safety_resources_endpoint_defaults_to_india(self):
        """GET /safety/resources without params returns complete India statutory directory."""
        res = self.client.get("/safety/resources")
        self.assertEqual(res.status_code, 200)
        items = res.json()
        self.assertIsInstance(items, list)
        self.assertGreaterEqual(len(items), 6)

        # Check key India statutory and institutional helplines
        names = [item["resource_name"] for item in items]
        phones = [item["telephone"] for item in items]
        resource_ids = [item["resource_id"] for item in items]

        # 1. ERSS Emergency: 112
        self.assertIn("in-national-emergency-112", resource_ids)
        self.assertIn("112", phones)

        # 2. National Ambulance: 102
        self.assertIn("in-ambulance-102", resource_ids)
        self.assertIn("102", phones)

        # 3. Tele-MANAS: 14416 / 1800-891-4416
        self.assertIn("in-tele-manas", resource_ids)
        tele_manas = next(it for it in items if it["resource_id"] == "in-tele-manas")
        self.assertEqual(tele_manas["telephone"], "14416")
        self.assertEqual(tele_manas["alternate_telephone"], "1800-891-4416")

        # 4. KIRAN: 1800-599-0019
        self.assertIn("in-kiran", resource_ids)
        self.assertIn("1800-599-0019", phones)

        # 5. Vandrevala Foundation: +91 9999 666 555
        self.assertIn("in-vandrevala", resource_ids)
        self.assertIn("+91 9999 666 555", phones)

        # 6. AASRA: +91 98204 66726
        self.assertIn("in-aasra", resource_ids)
        self.assertIn("+91 98204 66726", phones)

        # Verify operational notes and notices are preserved
        for item in items:
            self.assertTrue(item["resource_name"])
            self.assertIn(item["verification_status"], ("source_extracted", "verified"))
            self.assertEqual(item["uptime_verification"], "unverified_operational_availability")

    # ------------------------------------------------------------------------
    # 8. Safety directory resources endpoint for US and alias endpoint
    # ------------------------------------------------------------------------
    def test_08_safety_resources_endpoint_us_and_alias(self):
        """GET /safety/resources?jurisdiction=US returns US resources, and /emergency-resources alias works."""
        # Query US resources explicitly
        res_us = self.client.get("/safety/resources?jurisdiction=US")
        self.assertEqual(res_us.status_code, 200)
        us_items = res_us.json()
        us_ids = [item["resource_id"] for item in us_items]
        self.assertIn("us-emergency-911", us_ids)
        self.assertIn("us-988-lifeline", us_ids)
        self.assertNotIn("in-national-emergency-112", us_ids)

        # Query alias /emergency-resources for India
        res_alias = self.client.get("/emergency-resources?jurisdiction=IN")
        self.assertEqual(res_alias.status_code, 200)
        alias_items = res_alias.json()
        alias_ids = [item["resource_id"] for item in alias_items]
        self.assertIn("in-national-emergency-112", alias_ids)
        self.assertIn("in-tele-manas", alias_ids)


if __name__ == "__main__":
    unittest.main()
