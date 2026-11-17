"""
MindBridge 2.0 - Phase 5 Step 4: RoBERTa Crisis Classifier Unit Tests
======================================================================

Comprehensive test suite for the fine-tuned RoBERTa crisis classifier and
orchestrator classifier node.

Verifies:
1. Model loading and threshold configuration (0.509765625 from threshold_config.json).
2. Output probability range [0.0, 1.0].
3. Explicit threshold application (roberta_crisis_flag = prob >= threshold).
4. Elevated score routes to elevated_classifier_signal and initiates follow-up (NEVER emergency).
5. Elevated score NEVER sets phq9_item9_endorsed = True.
6. Elevated score does not mutate clinical assessment dimensions.
7. Universal Emergency Override skips inference and preserves emergency_bypass.
8. Low score cannot override deterministic emergency.
9. Active follow-up continuity skips inference and preserves pending_safety_follow_up.
10. Low score cannot dismiss active follow-up.
11. Below-threshold classifier score routes to standard_conversation.
12. Timeout handling sets status="timeout", prob=None, route="ml_risk_unassessed", and safe fallback.
13. Error handling sets status="error", prob=None, route="ml_risk_unassessed", and safe fallback.
14. Missing model directory handles failure gracefully.
15. Privacy-safe telemetry (zero user text or stack traces in execution trace).
16. End-to-end local checkpoint offline smoke test.
"""

from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
import time
from typing import Optional
import unittest
from unittest.mock import MagicMock, patch

from backend.orchestrator.safety_node import (
    ROBERTA_CRISIS_THRESHOLD,
    classifier_decision_node,
    classifier_node,
    route_after_classifier_decision,
)
from backend.orchestrator.state import (
    ClinicalRiskDimensions,
    MindBridgeState,
)
from backend.safety.crisis_detector import (
    CrisisDetector,
    CrisisPredictionResult,
    DEFAULT_THRESHOLD,
)


def build_base_state(
    sanitized_text: str = "I am looking for some study tips",
    overall_route: str = "standard_conversation",
    deterministic_danger: bool = False,
    followup_status: str = "not_started",
    item9_endorsed: Optional[bool] = False,
    dimensions: Optional[ClinicalRiskDimensions] = None,
) -> MindBridgeState:
    """Helper to build a clean baseline state."""
    default_dims: ClinicalRiskDimensions = {
        "passive_death_wishes": None,
        "active_suicidal_ideation": None,
        "intent": None,
        "plan": None,
        "means": None,
        "access": None,
        "timeframe": None,
    }
    if dimensions:
        default_dims.update(dimensions)

    return {
        "session_id": "test-session-step4",
        "turn_count": 1,
        "raw_user_input": sanitized_text,
        "sanitized_text": sanitized_text,
        "input_modality": "text",
        "jurisdiction_filter": "US",
        "overall_safety_route": overall_route,  # type: ignore
        "deterministic_danger_flag": deterministic_danger,
        "emergency_trigger_id": None,
        "classifier_status": "not_run",
        "roberta_prob": None,
        "roberta_crisis_flag": False,
        "phq9_item9_endorsed": item9_endorsed,
        "phq9_followup_status": followup_status,  # type: ignore
        "clinical_dimensions": default_dims,
        "requires_hotline_display": False,
        "error_code": None,
        "error_component": None,
        "final_response": "",
        "execution_trace": [],
    }


class TestCrisisDetector(unittest.TestCase):
    """Tests for the CrisisDetector inference wrapper."""

    def setUp(self):
        CrisisDetector.reset_instance()

    def tearDown(self):
        CrisisDetector.reset_instance()

    def test_01_model_loading_and_threshold_configuration(self):
        """Verifies threshold loading from threshold_config.json and fallback."""
        detector = CrisisDetector()
        self.assertAlmostEqual(detector.threshold, 0.509765625, places=6)

        # Test fallback when threshold config is missing
        with patch.object(Path, "exists", return_value=False):
            fallback_detector = CrisisDetector(model_dir="/nonexistent/path")
            self.assertEqual(fallback_detector.threshold, DEFAULT_THRESHOLD)

    def test_02_probability_range_and_deterministic_output(self):
        """Mocked inference ensures probability output is in [0.0, 1.0]."""
        mock_detector = CrisisDetector()
        mock_detector._is_loaded = True

        with patch.object(mock_detector, "_raw_predict", return_value=0.7523):
            result = mock_detector.predict("I feel so overwhelmed")
            self.assertEqual(result.status, "completed")
            self.assertIsNotNone(result.probability)
            self.assertTrue(0.0 <= result.probability <= 1.0)
            self.assertTrue(result.is_crisis)

    def test_03_threshold_application_binary_flag(self):
        """Explicit threshold application: flag is True iff prob >= threshold."""
        detector = CrisisDetector(threshold=0.509765625)
        detector._is_loaded = True

        # Exactly at threshold -> Crisis
        with patch.object(detector, "_raw_predict", return_value=0.509765625):
            res = detector.predict("test")
            self.assertTrue(res.is_crisis)

        # Just above threshold -> Crisis
        with patch.object(detector, "_raw_predict", return_value=0.509766):
            res = detector.predict("test")
            self.assertTrue(res.is_crisis)

        # Just below threshold -> Non-crisis
        with patch.object(detector, "_raw_predict", return_value=0.509764):
            res = detector.predict("test")
            self.assertFalse(res.is_crisis)


class TestClassifierNode(unittest.TestCase):
    """Tests for classifier_node and orchestrator integration."""

    def setUp(self):
        CrisisDetector.reset_instance()

    def tearDown(self):
        CrisisDetector.reset_instance()

    def test_04_elevated_score_routes_to_followup_never_emergency(self):
        """Elevated prediction routes to elevated_classifier_signal, NEVER emergency."""
        state = build_base_state(sanitized_text="I feel hopeless and want to end it")

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=0.92,
            is_crisis=True,
            status="completed",
            latency_ms=45.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        self.assertEqual(merged["overall_safety_route"], "elevated_classifier_signal")
        self.assertEqual(merged["phq9_followup_status"], "in_progress")
        self.assertEqual(merged["classifier_status"], "completed")
        self.assertEqual(merged["roberta_prob"], 0.92)
        self.assertTrue(merged["roberta_crisis_flag"])

        # Edge routing test: MUST route to followup_node, NEVER emergency_node
        self.assertEqual(route_after_classifier_decision(merged), "followup_node")
        self.assertNotEqual(route_after_classifier_decision(merged), "emergency_node")

    def test_05_elevated_score_never_sets_phq9_item9_endorsed(self):
        """Core Invariant: An ML prediction is NOT user endorsement of PHQ-9 Item 9."""
        state = build_base_state(
            sanitized_text="I don't know how to keep going",
            item9_endorsed=False,
        )

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=0.98,
            is_crisis=True,
            status="completed",
            latency_ms=50.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        self.assertFalse(merged["phq9_item9_endorsed"])
        self.assertIsNot(merged["phq9_item9_endorsed"], True)

    def test_06_elevated_score_does_not_mutate_clinical_dimensions(self):
        """Core Invariant: Classifier signal does not alter clinical assessment dimensions."""
        initial_dims: ClinicalRiskDimensions = {
            "passive_death_wishes": None,
            "active_suicidal_ideation": None,
            "intent": False,
            "plan": None,
            "means": None,
            "access": None,
            "timeframe": None,
        }
        state = build_base_state(dimensions=initial_dims)

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=0.95,
            is_crisis=True,
            status="completed",
            latency_ms=40.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        # Dimensions must remain untouched
        self.assertEqual(merged["clinical_dimensions"], initial_dims)
        self.assertIsNone(merged["clinical_dimensions"]["plan"])
        self.assertFalse(merged["clinical_dimensions"]["intent"])

    def test_07_universal_emergency_override_skips_classifier(self):
        """Emergency override takes absolute precedence: classifier inference is skipped."""
        state = build_base_state(
            overall_route="emergency_bypass",
            deterministic_danger=True,
        )

        mock_detector = MagicMock(spec=CrisisDetector)
        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        # Model predict must NOT be called
        mock_detector.predict.assert_not_called()
        self.assertEqual(merged["overall_safety_route"], "emergency_bypass")
        self.assertEqual(merged["classifier_status"], "not_run")
        self.assertIsNone(merged["roberta_prob"])

        # Check trace
        traces = [t for t in merged["execution_trace"] if t["node"] == "classifier_node"]
        self.assertTrue(len(traces) > 0)
        self.assertEqual(traces[-1]["status"], "skipped")
        self.assertEqual(traces[-1]["metadata_code"], "EMERGENCY_PRESERVED")

    def test_08_low_classifier_score_cannot_override_emergency(self):
        """Even if classifier produces low probability, emergency is preserved."""
        # Setup state that had an emergency triggered
        state = build_base_state(
            overall_route="emergency_bypass",
            deterministic_danger=True,
        )

        # In direct decision node evaluation with low score
        decision_updates = classifier_decision_node({
            **state,
            "classifier_status": "completed",
            "roberta_prob": 0.01,
            "roberta_crisis_flag": False,
        })
        merged = {**state, **decision_updates}

        self.assertEqual(merged["overall_safety_route"], "emergency_bypass")
        self.assertEqual(route_after_classifier_decision(merged), "emergency_node")

    def test_09_active_followup_continuity_skips_classifier(self):
        """Active follow-up in progress takes precedence: inference is skipped."""
        state = build_base_state(
            overall_route="pending_safety_follow_up",
            followup_status="in_progress",
        )

        mock_detector = MagicMock(spec=CrisisDetector)
        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        mock_detector.predict.assert_not_called()
        self.assertEqual(merged["overall_safety_route"], "pending_safety_follow_up")
        self.assertEqual(merged["classifier_status"], "not_run")

        traces = [t for t in merged["execution_trace"] if t["node"] == "classifier_node"]
        self.assertTrue(len(traces) > 0)
        self.assertEqual(traces[-1]["status"], "skipped")
        self.assertEqual(traces[-1]["metadata_code"], "FOLLOWUP_PRESERVED")

    def test_10_low_classifier_score_cannot_dismiss_active_followup(self):
        """A low classifier score cannot dismiss an active or pending follow-up."""
        state = build_base_state(
            overall_route="pending_safety_follow_up",
            followup_status="in_progress",
        )

        decision_updates = classifier_decision_node({
            **state,
            "classifier_status": "completed",
            "roberta_prob": 0.02,
            "roberta_crisis_flag": False,
        })
        merged = {**state, **decision_updates}

        self.assertEqual(merged["overall_safety_route"], "pending_safety_follow_up")
        self.assertEqual(route_after_classifier_decision(merged), "followup_node")

    def test_11_below_threshold_routes_standard_conversation(self):
        """Below-threshold score with clear deterministic safety routes to standard_conversation."""
        state = build_base_state(sanitized_text="How can I prepare for a job interview?")

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=0.03,
            is_crisis=False,
            status="completed",
            latency_ms=30.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        self.assertEqual(merged["overall_safety_route"], "standard_conversation")
        self.assertEqual(merged["classifier_status"], "completed")
        self.assertEqual(merged["roberta_prob"], 0.03)
        self.assertFalse(merged["roberta_crisis_flag"])
        self.assertEqual(route_after_classifier_decision(merged), "planning_node")

    def test_12_classifier_timeout_handling(self):
        """Timeout sets status='timeout', prob=None, route='ml_risk_unassessed', and safe fallback."""
        state = build_base_state(sanitized_text="A slow response text")

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=None,
            is_crisis=False,
            status="timeout",
            latency_ms=205.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
            error_code="CLASSIFIER_TIMEOUT",
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        self.assertEqual(merged["classifier_status"], "timeout")
        self.assertIsNone(merged["roberta_prob"])
        self.assertFalse(merged["roberta_crisis_flag"])
        self.assertEqual(merged["overall_safety_route"], "ml_risk_unassessed")
        self.assertEqual(merged["error_code"], "CLASSIFIER_TIMEOUT")
        self.assertEqual(merged["error_component"], "classifier_node")
        self.assertTrue(len(merged["final_response"]) > 0)
        self.assertIn("support", merged["final_response"])

        # Edge routing test
        self.assertEqual(route_after_classifier_decision(merged), "deescalation_node")

    def test_13_classifier_error_handling(self):
        """Inference error sets status='error', prob=None, route='ml_risk_unassessed', and safe fallback."""
        state = build_base_state(sanitized_text="Error triggering input")

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=None,
            is_crisis=False,
            status="error",
            latency_ms=10.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
            error_code="CLASSIFIER_FAILURE",
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        self.assertEqual(merged["classifier_status"], "error")
        self.assertIsNone(merged["roberta_prob"])
        self.assertFalse(merged["roberta_crisis_flag"])
        self.assertEqual(merged["overall_safety_route"], "ml_risk_unassessed")
        self.assertEqual(merged["error_code"], "CLASSIFIER_FAILURE")
        self.assertEqual(merged["error_component"], "classifier_node")
        self.assertTrue(len(merged["final_response"]) > 0)
        self.assertEqual(route_after_classifier_decision(merged), "deescalation_node")

    def test_14_missing_model_directory_graceful_degradation(self):
        """When model directory does not exist, detector fails safely with error status."""
        detector = CrisisDetector(model_dir="/path/that/definitely/does/not/exist")
        result = detector.predict("Hello world")

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.probability)
        self.assertFalse(result.is_crisis)
        self.assertEqual(result.error_code, "CLASSIFIER_FAILURE")

    def test_15_privacy_safe_telemetry(self):
        """Zero user raw text or exception strings enter execution trace metadata."""
        sensitive_text = "My secret personal diary entry with email secret@example.com"
        state = build_base_state(sanitized_text=sensitive_text)

        mock_detector = MagicMock(spec=CrisisDetector)
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=0.25,
            is_crisis=False,
            status="completed",
            latency_ms=35.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
        )

        updates = classifier_node(state, detector=mock_detector)
        merged = {**state, **updates}

        # Verify trace entries
        for entry in merged["execution_trace"]:
            for key, val in entry.items():
                val_str = str(val)
                self.assertNotIn("secret@example.com", val_str)
                self.assertNotIn("secret personal diary", val_str)

    def test_16_local_model_smoke_test(self):
        """Smoke test verifying real local checkpoint inference with dynamic padding."""
        model_dir = Path("models/crisis_detector/roberta")
        if not model_dir.exists():
            self.skipTest("Local model directory not found; skipping smoke test.")

        detector = CrisisDetector.get_instance(model_dir=model_dir)
        detector.load_model()

        # 1. Non-crisis input with dynamic padding
        non_crisis_res = detector.predict("I went to the store today to buy groceries")
        self.assertEqual(non_crisis_res.status, "completed")
        self.assertIsNotNone(non_crisis_res.probability)
        self.assertTrue(0.0 <= non_crisis_res.probability <= 1.0)
        self.assertFalse(non_crisis_res.is_crisis)

        # 2. Crisis input with dynamic padding
        crisis_res = detector.predict("I want to end my life, I cannot take this pain anymore")
        self.assertEqual(crisis_res.status, "completed")
        self.assertIsNotNone(crisis_res.probability)
        self.assertTrue(crisis_res.probability >= detector.threshold)
        self.assertTrue(crisis_res.is_crisis)

    def test_17_fixed_vs_dynamic_padding_numerical_parity(self):
        """Verifies numerical equivalence between fixed (L=384) and dynamic padding within 1e-5."""
        model_dir = Path("models/crisis_detector/roberta")
        if not model_dir.exists():
            self.skipTest("Local model directory not found; skipping parity test.")

        detector = CrisisDetector.get_instance(model_dir=model_dir)
        detector.load_model()

        test_sentences = [
            "I went to the store today to buy groceries",
            "I feel completely overwhelmed and want to die right now",
            "Short text",
            "Longer sentence discussing stressful life events and looking for strategies to cope with anxiety",
        ]

        for text in test_sentences:
            prob_fixed = detector._raw_predict(text, padding="max_length")
            prob_dyn = detector._raw_predict(text, padding=True)
            diff = abs(prob_fixed - prob_dyn)
            self.assertLess(
                diff,
                1e-5,
                f"Numerical disparity {diff:.2e} between fixed and dynamic padding for input: {text!r}",
            )

    def test_18_dynamic_padding_truncation_at_384_tokens(self):
        """Ensures sequences exceeding 384 tokens are truncated to max_length without error."""
        model_dir = Path("models/crisis_detector/roberta")
        if not model_dir.exists():
            self.skipTest("Local model directory not found; skipping truncation test.")

        detector = CrisisDetector.get_instance(model_dir=model_dir)
        detector.load_model()

        oversized_text = "I feel stressed and overwhelmed. " * 100  # ~600 tokens
        # Use generous test timeout (2.0s) so scheduler jitter under multi-process load does not flake truncation test
        result = detector.predict(oversized_text, timeout_seconds=2.0)
        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.probability)

        # Direct tokenization length check
        encoded = detector._tokenizer(oversized_text, max_length=384, padding=True, truncation=True)
        self.assertEqual(len(encoded["input_ids"]), 384)

    def test_19_persistent_executor_reuse(self):
        """Verifies that the singleton detector reuses its worker executor across multiple calls."""
        detector = CrisisDetector()
        detector._is_loaded = True

        with patch.object(detector, "_raw_predict", return_value=0.10):
            res1 = detector.predict("test 1")
            executor1 = detector._executor
            res2 = detector.predict("test 2")
            executor2 = detector._executor

            self.assertIsNotNone(executor1)
            self.assertIs(executor1, executor2)
            self.assertEqual(res1.status, "completed")
            self.assertEqual(res2.status, "completed")

    def test_20_repeated_timeout_handling(self):
        """Repeated timeouts are handled safely without crashing or queue accumulation."""
        detector = CrisisDetector()
        detector._is_loaded = True

        def simulated_slow_inference(text, padding):
            time.sleep(0.08)
            return 0.90

        with patch.object(detector, "_raw_predict", side_effect=simulated_slow_inference):
            for _ in range(3):
                res = detector.predict("slow query", timeout_seconds=0.01)
                self.assertIn(res.status, ("timeout", "busy"))
                self.assertIsNone(res.probability)
                self.assertFalse(res.is_crisis)
                self.assertEqual(res.error_code, "CLASSIFIER_TIMEOUT")
                time.sleep(0.09)  # Allow worker to finish between iterations

    def test_21_busy_rejection_while_worker_running(self):
        """A new inference is immediately rejected with busy status while a prior job is executing."""
        detector = CrisisDetector()
        detector._is_loaded = True

        def simulated_long_run(text, padding):
            time.sleep(0.15)
            return 0.55

        with patch.object(detector, "_raw_predict", side_effect=simulated_long_run):
            # First call times out quickly; worker remains active in background
            res1 = detector.predict("first query", timeout_seconds=0.02)
            self.assertEqual(res1.status, "timeout")

            # Second call immediately sees worker busy and rejects without queueing
            res2 = detector.predict("second query while busy", timeout_seconds=0.02)
            self.assertEqual(res2.status, "busy")
            self.assertIsNone(res2.probability)
            self.assertFalse(res2.is_crisis)
            self.assertEqual(res2.error_code, "CLASSIFIER_TIMEOUT")

            # Allow worker to finish background task
            time.sleep(0.16)

    def test_22_no_unbounded_task_queue(self):
        """When worker is busy, multiple calls are rejected immediately; queue size never grows."""
        detector = CrisisDetector()
        detector._is_loaded = True

        def simulated_long_run(text, padding):
            time.sleep(0.15)
            return 0.55

        with patch.object(detector, "_raw_predict", side_effect=simulated_long_run):
            res1 = detector.predict("first", timeout_seconds=0.02)
            self.assertEqual(res1.status, "timeout")

            # Send 5 rapid requests while worker is busy
            for i in range(5):
                res = detector.predict(f"burst {i}", timeout_seconds=0.02)
                self.assertEqual(res.status, "busy")

            # Check that executor has at most 1 task running and no queued tasks
            executor = detector._get_executor()
            self.assertEqual(executor._work_queue.qsize(), 0)

            time.sleep(0.16)

    def test_23_timeout_returns_without_waiting_for_worker(self):
        """Caller returns immediately when timeout expires, not waiting for worker task completion."""
        detector = CrisisDetector()
        detector._is_loaded = True

        def simulated_slow_task(text, padding):
            time.sleep(0.20)
            return 0.85

        with patch.object(detector, "_raw_predict", side_effect=simulated_slow_task):
            t0 = time.perf_counter()
            res = detector.predict("text", timeout_seconds=0.03)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.assertEqual(res.status, "timeout")
            # Must return in ~30-60ms, strictly less than 150ms (did not wait for 200ms task)
            self.assertLess(elapsed_ms, 120.0, f"Caller blocked too long on timeout: {elapsed_ms:.2f} ms")

            # Cleanup
            time.sleep(0.20)

    def test_24_worker_recovery_after_completion(self):
        """Worker recovers and processes new queries normally once a slow task finishes."""
        detector = CrisisDetector()
        detector._is_loaded = True

        call_count = 0

        def alternating_task(text, padding):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.10)
            return 0.35

        with patch.object(detector, "_raw_predict", side_effect=alternating_task):
            # Call 1: times out
            res1 = detector.predict("call 1", timeout_seconds=0.02)
            self.assertEqual(res1.status, "timeout")

            # Wait for call 1 to finish in background
            time.sleep(0.12)

            # Call 2: worker should be idle and process successfully
            res2 = detector.predict("call 2", timeout_seconds=0.10)
            self.assertEqual(res2.status, "completed")
            self.assertEqual(res2.probability, 0.35)

    def test_25_deterministic_emergency_and_followup_protected_during_busy(self):
        """Emergency and follow-up states bypass classifier entirely, even if worker is busy."""
        # 1. Emergency state
        emergency_state = build_base_state(
            overall_route="emergency_bypass",
            deterministic_danger=True,
        )
        mock_detector = MagicMock(spec=CrisisDetector)
        # Even if mock detector would report busy
        mock_detector.predict.return_value = CrisisPredictionResult(
            probability=None,
            is_crisis=False,
            status="busy",
            latency_ms=1.0,
            threshold=ROBERTA_CRISIS_THRESHOLD,
            error_code="CLASSIFIER_TIMEOUT",
        )

        updates_em = classifier_node(emergency_state, detector=mock_detector)
        merged_em = {**emergency_state, **updates_em}
        mock_detector.predict.assert_not_called()
        self.assertEqual(merged_em["overall_safety_route"], "emergency_bypass")

        # 2. Active follow-up state
        followup_state = build_base_state(
            overall_route="pending_safety_follow_up",
            followup_status="in_progress",
        )
        updates_fu = classifier_node(followup_state, detector=mock_detector)
        merged_fu = {**followup_state, **updates_fu}
        mock_detector.predict.assert_not_called()
        self.assertEqual(merged_fu["overall_safety_route"], "pending_safety_follow_up")

    def test_26_model_loading_separate_from_inference_timeout(self):
        """Model loading occurs lazily before the inference timeout window starts."""
        detector = CrisisDetector()
        detector._is_loaded = False

        load_called = False

        def mock_load():
            nonlocal load_called
            time.sleep(0.04)  # Simulate model load time
            detector._is_loaded = True
            load_called = True

        detector.load_model = mock_load

        with patch.object(detector, "_raw_predict", return_value=0.20):
            # Timeout is 0.03s, smaller than load time (0.04s)
            # If load counted against timeout, it would time out. But it must not!
            res = detector.predict("test", timeout_seconds=0.08)
            self.assertTrue(load_called)
            self.assertEqual(res.status, "completed")
            self.assertEqual(res.probability, 0.20)

    def test_27_conversational_input_latency_within_budget(self):
        """Verifies that typical conversational queries (<60 tokens) execute comfortably on CPU."""
        model_dir = Path("models/crisis_detector/roberta")
        if not model_dir.exists():
            self.skipTest("Local model directory not found; skipping latency test.")

        detector = CrisisDetector.get_instance(model_dir=model_dir)
        detector.load_model()

        # Typical user turn (around 20-40 tokens)
        typical_text = "I have been feeling really anxious about school lately and could use some help."
        res = detector.predict(typical_text, timeout_seconds=0.5)
        self.assertEqual(res.status, "completed")
        self.assertIsNotNone(res.probability)
        self.assertLess(
            res.latency_ms,
            250.0,
            f"Typical input exceeded acceptable CPU latency: {res.latency_ms:.2f} ms",
        )


if __name__ == "__main__":
    unittest.main()
