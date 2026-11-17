"""
MindBridge 2.0 - API Dependencies & Lifecycle State Container (`dependencies.py`)
==================================================================================

Manages shared singletons, concurrency semaphores, and readiness state
without duplicating internal model logic.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from backend.api.middleware import InMemoryRateLimiter
from backend.config import settings
from backend.orchestrator.graph import CompiledGraph, build_mindbridge_graph
from backend.orchestrator.planning_node import get_shared_retriever
from backend.retrieval import RAGRetriever
from backend.safety.crisis_detector import CrisisDetector


class ApplicationLifecycleState:
    """
    Thread-safe container managing application singletons across the lifespan.
    """
    def __init__(self):
        self.compiled_graph: Optional[CompiledGraph] = None
        self.retriever: Optional[RAGRetriever] = None
        self.crisis_detector: Optional[CrisisDetector] = None
        self.generator: Optional[Any] = None
        self.rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.rate_limit_per_minute)
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.is_ready: bool = False
        self._initialization_error: Optional[str] = None

    def initialize_sync(self) -> None:
        """
        Synchronously pre-warms local models and compiles graph during lifespan startup.
        Never downloads external weights automatically.
        """
        try:
            # 1. Initialize CrisisDetector singleton
            self.crisis_detector = CrisisDetector.get_instance(
                model_dir=settings.crisis_model_dir
            )
            # Pre-load RoBERTa weights if present locally
            if settings.crisis_model_dir.exists():
                self.crisis_detector.load_model()

            # 2. Initialize RAG knowledge retriever singleton
            self.retriever = get_shared_retriever()

            # 2.5 Initialize configured LLM generator (Gemini, Groq, or Local)
            try:
                from backend.generation.llm import create_llm_generator
                self.generator = create_llm_generator()
            except Exception:
                self.generator = None

            # 3. Compile deterministic MindBridge state graph
            self.compiled_graph = build_mindbridge_graph(
                retriever=self.retriever,
                generator=self.generator,
            )

            # 4. Mark operational readiness
            self.is_ready = True
            self._initialization_error = None
        except Exception as e:
            self.is_ready = False
            self._initialization_error = "Model or index initialization failed."

    def shutdown(self) -> None:
        """
        Gracefully releases models, background thread pools, and singletons.
        """
        if self.crisis_detector:
            CrisisDetector.reset_instance()
            self.crisis_detector = None
        self.rate_limiter.reset()
        self.compiled_graph = None
        self.retriever = None
        self.generator = None
        self.is_ready = False

    def get_readiness_info(self) -> Dict[str, Any]:
        """Evaluates readiness of all internal local subsystems."""
        has_detector = bool(self.crisis_detector and getattr(self.crisis_detector, "_is_loaded", False))
        has_retriever = bool(self.retriever and self.retriever.vector_store.total_vectors > 0)
        has_graph = self.compiled_graph is not None
        has_generator = bool(self.generator and hasattr(self.generator, "is_available") and self.generator.is_available())

        gemini_ready = False
        groq_ready = False
        active_provider = "local"

        if has_generator:
            if hasattr(self.generator, "primary_generator") and hasattr(self.generator, "fallback_generator"):
                gemini_ready = bool(self.generator.primary_generator and self.generator.primary_generator.is_available())
                groq_ready = bool(self.generator.fallback_generator and self.generator.fallback_generator.is_available())
                if gemini_ready:
                    active_provider = "gemini"
                elif groq_ready:
                    active_provider = "groq"
                else:
                    active_provider = getattr(self.generator, "provider_name", "local")
            else:
                prov_name = getattr(self.generator, "provider_name", "local")
                if prov_name == "gemini":
                    gemini_ready = True
                    active_provider = "gemini"
                elif prov_name == "groq":
                    groq_ready = True
                    active_provider = "groq"
                else:
                    active_provider = prov_name

        all_ready = has_detector and has_retriever and has_graph

        return {
            "is_ready": all_ready,
            "models": {
                "crisis_detector": has_detector,
                "embedding_engine": has_retriever,
                "vector_store": has_retriever,
                "graph": has_graph,
                "gemini_generator": gemini_ready,
                "groq_generator": groq_ready,
                "llm_generator": has_generator,
                "llm_provider": active_provider,
            },
            "details": {
                "crisis_model_path": str(settings.crisis_model_dir),
                "embedding_model_path": str(settings.embedding_model_dir),
                "vector_count": self.retriever.vector_store.total_vectors if self.retriever else 0,
                "jurisdiction_default": settings.default_jurisdiction,
                "llm_provider": active_provider,
                "llm_generator": has_generator,
                "gemini_available": gemini_ready,
                "groq_available": groq_ready,
            },
            "error": self._initialization_error,
        }


# Global application lifecycle state singleton
app_state = ApplicationLifecycleState()


def get_app_graph() -> CompiledGraph:
    """Dependency provider returning the compiled MindBridge state graph."""
    if app_state.compiled_graph is None:
        # Fallback compilation if accessed before lifespan (e.g. test harnesses)
        app_state.compiled_graph = build_mindbridge_graph(
            retriever=app_state.retriever,
            generator=app_state.generator,
        )
    return app_state.compiled_graph


def get_concurrency_semaphore() -> asyncio.Semaphore:
    """Returns or creates the concurrency-limiting semaphore."""
    if app_state.semaphore is None:
        app_state.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    return app_state.semaphore


def get_rate_limiter() -> InMemoryRateLimiter:
    """Dependency provider returning the active rate limiter."""
    return app_state.rate_limiter
