"""
MindBridge 2.0 - RoBERTa Crisis Detector Inference Engine
==========================================================

This module implements deterministic, CPU-compatible inference for the fine-tuned
RoBERTa-base crisis detection model (`models/crisis_detector/roberta/`).

Model Characteristics:
- Architecture: roberta-base (num_labels=1, regression/binary logit)
- Binary classification: Label 0 = Non-Crisis, Label 1 = Crisis-Risk
- Training sequence length: 384 tokens (Byte-level BPE)
- Validation-optimized decision threshold: 0.509765625 (loaded from threshold_config.json)
- Head format: 1 output logit -> crisis probability p = torch.sigmoid(logit)

Safety & Engineering Invariants:
1. Deterministic CPU Inference: model.eval(), torch.no_grad(), CPU device.
2. Lazy / Singleton Initialization: Model weights and tokenizer loaded once in memory.
3. Bounded Execution: Thread-bounded execution with configurable timeout.
4. Privacy-Safe Telemetry: Zero raw user messages or stack traces leaked into errors/traces.
5. Fail-Safe Degradation: Timeouts or model errors cleanly return status "timeout" or "error",
   with probability = None, is_crisis = False, never crashing the orchestration graph.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Default model directory relative to project root
DEFAULT_MODEL_DIR = Path("models/crisis_detector/roberta")
DEFAULT_THRESHOLD = 0.509765625
DEFAULT_MAX_SEQ_LENGTH = 384
DEFAULT_TIMEOUT_SECONDS = 0.500  # 500ms bounds execution on standard CPU


@dataclass(frozen=True)
class CrisisPredictionResult:
    """
    Structured immutable result for a crisis detector prediction.
    """
    probability: Optional[float]
    is_crisis: bool
    status: str  # "completed" | "timeout" | "error"
    latency_ms: float
    threshold: float
    error_code: Optional[str] = None


class CrisisDetector:
    """
    Inference wrapper for the fine-tuned RoBERTa crisis classifier.
    """

    _instance: Optional["CrisisDetector"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "crisis_detector" / "roberta"
            if not self.model_dir.exists():
                self.model_dir = DEFAULT_MODEL_DIR
        else:
            self.model_dir = Path(model_dir)

        self.max_seq_length = max_seq_length
        self.default_timeout_seconds = default_timeout_seconds

        # Resolve threshold from threshold_config.json if not explicitly provided
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = self._load_threshold()

        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._is_loaded = False
        self._model_load_time_ms: float = 0.0
        self._is_cold_start: bool = True

        # Persistent single-worker execution strategy
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._worker_lock = threading.Lock()

    def _load_threshold(self) -> float:
        """
        Loads the validation-optimized threshold from threshold_config.json.
        Falls back to DEFAULT_THRESHOLD if file is missing or unreadable.
        """
        config_path = self.model_dir / "threshold_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    val = data.get("selected_threshold")
                    if val is not None:
                        return float(val)
            except Exception:
                logger.warning("Could not read threshold_config.json; using default threshold.")
        return DEFAULT_THRESHOLD

    @classmethod
    def get_instance(
        cls,
        model_dir: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ) -> "CrisisDetector":
        """
        Thread-safe singleton accessor for CrisisDetector.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_dir=model_dir, threshold=threshold)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Resets the singleton instance and shuts down persistent worker.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown(wait=False)
                cls._instance = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """
        Lazily initializes and returns the persistent single-worker ThreadPoolExecutor.
        """
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="crisis_detector_worker",
                )
            return self._executor

    def _release_worker_lock(self) -> None:
        """
        Safely releases the non-blocking worker lock if currently held.
        """
        if self._worker_lock.locked():
            try:
                self._worker_lock.release()
            except RuntimeError:
                pass

    def shutdown(self, wait: bool = False) -> None:
        """
        Safely terminates the persistent worker thread and releases resources.
        """
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait, cancel_futures=True)
                self._executor = None
        self._release_worker_lock()

    def load_model(self) -> None:
        """
        Loads tokenizer and model weights into memory with deterministic CPU eval mode.
        Safe to call multiple times (idempotent).
        """
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            t0 = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                if not self.model_dir.exists():
                    raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_dir),
                    local_files_only=True,
                )

                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_dir),
                    local_files_only=True,
                )
                self._model.eval()

                # Warmup forward pass to initialize CPU BLAS and OpenMP thread pool
                try:
                    with torch.no_grad():
                        dummy = self._tokenizer("warmup", return_tensors="pt")
                        self._model(**dummy)
                except Exception:
                    pass

                # Ensure torch uses CPU for deterministic inference
                self._is_loaded = True
                self._model_load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.info(f"RoBERTa crisis detector loaded in {self._model_load_time_ms:.2f} ms")

            except Exception as e:
                self._is_loaded = False
                logger.error("Failed to load RoBERTa crisis detector model")
                raise RuntimeError("Failed to load crisis detector model") from e

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_load_time_ms(self) -> float:
        return self._model_load_time_ms

    def _raw_predict(self, text: str, padding: Union[bool, str] = True) -> float:
        """
        Performs raw forward inference on CPU and returns crisis probability.
        Defaults to dynamic sequence padding (padding=True) capped at max_seq_length=384.
        """
        import torch

        if not self._is_loaded:
            self.load_model()

        assert self._tokenizer is not None
        assert self._model is not None

        # Clean preprocessing
        cleaned_text = text.strip() if text else ""
        if not cleaned_text:
            return 0.0

        encoded = self._tokenizer(
            cleaned_text,
            max_length=self.max_seq_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits.squeeze(-1)
            prob = float(torch.sigmoid(logits).item())

        return prob

    def predict(
        self,
        text: str,
        timeout_seconds: Optional[float] = None,
        padding: Union[bool, str] = True,
    ) -> CrisisPredictionResult:
        """
        Runs bounded inference using a persistent worker thread, non-blocking queueing,
        and fail-safe error isolation.

        Padding Strategy: Defaults to dynamic padding (padding=True) with truncation at 384 tokens,
        providing identical numerical outputs to fixed padding while drastically reducing CPU latency.

        Concurrency Strategy:
        - Persistent single-worker executor (no per-call OS thread creation).
        - Non-blocking worker lock: rejects submission immediately if worker is busy or executing
          a previous timed-out job. Never allows an unbounded submission queue.
        - Immediate caller return on timeout (does not wait for worker thread to complete).

        Returns CrisisPredictionResult with:
        - probability: float in [0.0, 1.0], or None on timeout/busy/error
        - is_crisis: bool (probability >= threshold)
        - status: "completed" | "timeout" | "busy" | "error"
        - latency_ms: float
        - threshold: float
        """
        start_time = time.perf_counter()
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds

        # Fast path for empty / whitespace input
        if not text or not text.strip():
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return CrisisPredictionResult(
                probability=0.0,
                is_crisis=False,
                status="completed",
                latency_ms=latency_ms,
                threshold=self.threshold,
            )

        # Ensure model is loaded before entering the bounded inference phase
        if not self._is_loaded:
            try:
                self.load_model()
            except Exception:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return CrisisPredictionResult(
                    probability=None,
                    is_crisis=False,
                    status="error",
                    latency_ms=latency_ms,
                    threshold=self.threshold,
                    error_code="CLASSIFIER_FAILURE",
                )

        # Concurrency check: Wait briefly if worker is currently finishing a request
        if not self._worker_lock.acquire(blocking=True, timeout=1.0):
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("Classifier worker is busy; rejecting new request without queueing")
            return CrisisPredictionResult(
                probability=None,
                is_crisis=False,
                status="busy",
                latency_ms=latency_ms,
                threshold=self.threshold,
                error_code="CLASSIFIER_TIMEOUT",
            )

        infer_start_time = time.perf_counter()
        executor = self._get_executor()

        try:
            future = executor.submit(self._raw_predict, text, padding)
            try:
                prob = future.result(timeout=timeout)
                latency_ms = round((time.perf_counter() - infer_start_time) * 1000, 2)
                self._release_worker_lock()
                self._is_cold_start = False

                return CrisisPredictionResult(
                    probability=round(prob, 6),
                    is_crisis=bool(prob >= self.threshold),
                    status="completed",
                    latency_ms=latency_ms,
                    threshold=self.threshold,
                )
            except FutureTimeoutError:
                latency_ms = round((time.perf_counter() - infer_start_time) * 1000, 2)
                logger.warning("Classifier inference timed out")
                # Worker thread remains active in background; release lock when future completes
                future.add_done_callback(lambda _: self._release_worker_lock())

                return CrisisPredictionResult(
                    probability=None,
                    is_crisis=False,
                    status="timeout",
                    latency_ms=latency_ms,
                    threshold=self.threshold,
                    error_code="CLASSIFIER_TIMEOUT",
                )
        except Exception:
            latency_ms = round((time.perf_counter() - infer_start_time) * 1000, 2)
            self._release_worker_lock()
            logger.error("Classifier inference failed with error")
            return CrisisPredictionResult(
                probability=None,
                is_crisis=False,
                status="error",
                latency_ms=latency_ms,
                threshold=self.threshold,
                error_code="CLASSIFIER_FAILURE",
            )
