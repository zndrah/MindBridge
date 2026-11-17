"""
MindBridge 2.0 - Gemini Developer API Grounded Generation (`llm.py`)
=====================================================================

Provides grounded, non-clinical supportive conversational generation via
Google's Gemini Developer API (Google AI Studio) using the official `google-genai` SDK.

Key Safety & Architectural Principles:
1. Google AI Studio — Gemini Developer API:
   Uses the Gemini Developer API through Google AI Studio (NOT Vertex AI).
2. Configuration & Secrets:
   API key is read from GEMINI_API_KEY; model identifier from GEMINI_MODEL.
   Zero hardcoded API keys or credentials. Real keys are never logged.
3. Grounding Integrity:
   Constructs a grounded prompt from retrieved knowledge chunks. Distinguishes
   approved clinical/educational protocols from unverified community peer Q&A.
4. Non-Clinical Boundaries:
   Strictly prohibits diagnostic claims, medication recommendations, dosage advice,
   and claims of emergency responder dispatch.
5. Controlled Failure Containment:
   Catches all authentication, quota (429), network timeout, and malformed response
   errors gracefully. Returns structured failure results, allowing the orchestrator
   to engage the deterministic local grounded synthesis fallback.
6. Universal Emergency Bypass:
   Emergency bypass operates upstream in the orchestrator, completely bypassing
   RAG retrieval and LLM generation.
"""

from dataclasses import dataclass, field
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from google import genai
    from google.genai import types
    from google.genai import errors
    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore
    errors = None  # type: ignore
    _GENAI_AVAILABLE = False

try:
    from groq import Groq
    import groq
    _GROQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore
    groq = None  # type: ignore
    _GROQ_AVAILABLE = False

from backend.config import settings
from backend.orchestrator.state import KnowledgeChunk, MindBridgeState

logger = logging.getLogger("mindbridge.generation")

DEFAULT_SYSTEM_INSTRUCTION = """You are MindBridge, an empathetic, supportive, and non-clinical conversational first-response assistant.
Your mission is to provide compassionate listening, psychoeducation, and grounded wellness coping strategies strictly based on the reference context provided below.

CRITICAL SAFETY & GOVERNANCE BOUNDARIES:
1. NON-CLINICAL ROLE: You are an AI conversational first-response assistant, NOT a doctor, licensed therapist, psychiatrist, or medical professional. Never claim to be a clinician or provide formal healthcare evaluations.
2. NO DIAGNOSES: Never diagnose the user with any mental health condition, illness, or psychiatric disorder (e.g., do not say "You have major depression" or "You are suffering from GAD").
3. NO MEDICATIONS OR DOSAGES: Never prescribe, recommend, adjust, or evaluate psychiatric medications, pharmacological treatments, or dosages.
4. NO DISPATCH CLAIMS: Never claim or imply that you can dispatch emergency services, call 911/112/988 on the user's behalf, or send an ambulance.
5. GROUNDING & EVIDENCE FIDELITY: Ground your factual statements, psychoeducation, and coping suggestions strictly in the provided reference context. Do not invent facts, statistics, citations, or source IDs not present in the reference items. Verify that any claim attributed to a source matches its actual excerpt, not just an inference from its title.
6. SOURCE ATTRIBUTION:
   - Approved Educational / Clinical Protocols: Evidence-informed educational materials and self-reflection rubrics. Prioritize these for academic/study stress.
   - Community Q&A Peer Perspectives: Unverified peer counseling examples from community forums. Use them ONLY when the counselor's answer genuinely and directly supports the coping strategy, and treat them strictly as illustrative conversational peer examples, NEVER as clinical instructions, diagnostic rubrics, or medical standards.
7. CITATION INTEGRITY: When referencing techniques, refer strictly to the exact titles or Chunk IDs provided in the context that you actually use. Never invent phantom citations, and never cite reference items that you did not actually use.
8. CONCISENESS & FOCUS: Prefer concise, natural, and conversational responses (2 to 3 brief paragraphs). Avoid giving too many suggestions at once—prioritize 1 or 2 practical, high-impact coping strategies.
9. EMPATHY & TONE: Warm, validating, non-judgmental, calm, and respectful. Avoid repetitive motivational statements, empty cheerleading (e.g., "you've got this", "believe in yourself"), or generic platitudes.
10. GENTLE FOLLOW-UP: Conclude with exactly ONE thoughtful, gentle open-ended follow-up question to invite further reflection or conversation."""


@dataclass
class GenerationResult:
    """
    Controlled output container for LLM generation requests.
    Prevents unhandled exceptions from propagating to orchestrator state.
    """
    text: Optional[str] = None
    success: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    model: Optional[str] = None
    duration_ms: float = 0.0
    provider: Optional[str] = None
    fallback_triggered: bool = False
    local_fallback: bool = False


def build_grounding_prompt(
    user_message: str,
    retrieved_chunks: Optional[List[KnowledgeChunk]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Constructs the grounded prompt pair: (system_instruction, contents_prompt).
    
    Ensures:
    - Retrieved information is used as grounding context.
    - Community Q&A is explicitly labeled as unverified peer perspective.
    - Multiple therapist perspectives are synthesized collaboratively.
    - No citation IDs are fabricated.
    - Boundaries against diagnosis and prescriptions are reinforced.
    """
    system_instruction = DEFAULT_SYSTEM_INSTRUCTION

    # 1. Format Reference Context
    if not retrieved_chunks:
        context_block = "[No specific reference context retrieved. Provide general supportive listening and compassionate validation without clinical assertions or diagnostic claims.]"
    else:
        items: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks):
            cid = chunk.get("chunk_id", f"chunk_{idx+1}")
            title = chunk.get("title", "Resource")
            category = chunk.get("category", "General")
            content = (chunk.get("content") or "").strip()

            # Explicitly distinguish community peer Q&A from approved clinical/educational protocols
            is_community = (
                "counselchat" in str(cid).lower()
                or "counselchat" in str(category).lower()
                or str(chunk.get("source_type", "")).lower() == "counselchat"
                or str(chunk.get("source", "")).lower() == "counselchat"
                or str(chunk.get("collection", "")).lower() == "counselchat_examples"
            )
            if is_community:
                source_label = "Unverified Community Q&A (Peer Perspective — Non-Clinical, Illustrative Only)"
            else:
                source_label = "Approved Educational / Clinical Protocol"

            item_text = (
                f"--- Reference Item {idx+1} ---\n"
                f"Source Type: {source_label}\n"
                f"Chunk ID: {cid}\n"
                f"Title: {title}\n"
                f"Category: {category}\n"
                f"Excerpt:\n{content}"
            )
            items.append(item_text)
        context_block = "\n\n".join(items)

    # 2. Format Recent Conversation History (bounded to last 8 turns)
    history_lines: List[str] = []
    if conversation_history:
        for turn in conversation_history[-8:]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if content:
                label = "User" if role == "user" else "MindBridge"
                history_lines.append(f"{label}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "[No prior conversation turns]"

    # 3. Assemble User Content Block
    prompt_body = (
        f"=== AUTHORIZED REFERENCE CONTEXT ===\n"
        f"{context_block}\n\n"
        f"=== RECENT CONVERSATION HISTORY ===\n"
        f"{history_block}\n\n"
        f"=== CURRENT USER MESSAGE ===\n"
        f"{user_message.strip()}\n\n"
        f"=== INSTRUCTIONS FOR YOUR RESPONSE ===\n"
        f"1. Provide a warm, conversational, empathetic, and non-clinical supportive response addressing the user's specific message.\n"
        f"2. Keep your response concise, focused, and natural (typically 2 to 3 brief paragraphs). Avoid overwhelming the user with too many suggestions at once—prioritize 1 or 2 practical, bite-sized coping strategies.\n"
        f"3. Avoid repetitive generic motivational statements, empty cheerleading (such as 'you've got this', 'believe in yourself'), or platitudes. Ground your tone in compassionate validation and concrete, achievable steps.\n"
        f"4. Ground coping strategies and psychoeducation strictly in the Reference Items above.\n"
        f"5. Prioritize highly relevant psychoeducation documents. Use Community Q&A (CounselChat) items ONLY when the counselor's answer genuinely and directly supports the specific statement being made.\n"
        f"6. Citation Fidelity: When referencing techniques, verify that the title and answer content match the claim attributed to it. Never invent or infer evidence from a source title alone. Cite ONLY the Reference Items you actually use by Title or Chunk ID. Do not cite Reference Items that you did not use.\n"
        f"7. If a Reference Item is marked as 'Unverified Community Q&A', treat it solely as an illustrative peer perspective, never as a clinical directive.\n"
        f"8. Do NOT assign any psychiatric diagnoses or recommend medications.\n"
        f"9. Do NOT claim the ability to dispatch emergency responders.\n"
        f"10. Conclude with exactly ONE gentle, thoughtful, and relevant follow-up question to help the user continue the conversation.\n"
        f"Response:"
    )

    return system_instruction, prompt_body


class GeminiGenerator:
    """
    Google AI Studio Gemini Developer API Client for grounded first-response generation.
    """
    provider_name: str = "gemini"
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        enable_gemini: Optional[bool] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model_name = model_name if model_name is not None else settings.gemini_model
        self.timeout = timeout if timeout is not None else settings.gemini_request_timeout
        self.enable_gemini = enable_gemini if enable_gemini is not None else settings.enable_gemini

        self._client = client
        self._init_error: Optional[str] = None
        self.last_result: Optional[GenerationResult] = None

        if self._client is None and self.enable_gemini and self.api_key:
            if not _GENAI_AVAILABLE:  # pragma: no cover
                self._init_error = "google-genai package is not installed."
                logger.error("Failed to initialize Gemini client: %s", self._init_error)
            else:
                try:
                    timeout_val = float(self.timeout) if self.timeout else 10.0
                    timeout_ms = int(timeout_val * 1000) if timeout_val < 1000 else int(timeout_val)
                    http_opts = types.HttpOptions(timeout=timeout_ms) if types else None
                    self._client = genai.Client(
                        api_key=self.api_key,
                        http_options=http_opts,
                    )
                except Exception as e:
                    self._init_error = f"Client initialization error: {type(e).__name__}"
                    logger.error("Failed to initialize Gemini client: %s", self._init_error)

    def is_available(self) -> bool:
        """Evaluates whether the generator is configured and ready for requests."""
        return bool(self.enable_gemini and self.api_key and self.model_name and (self._client is not None))

    def generate_response(
        self,
        user_message: str,
        retrieved_chunks: Optional[List[KnowledgeChunk]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """
        Executes grounded generation via Gemini Developer API with comprehensive error containment.
        Never raises unhandled exceptions.
        """
        start_time = time.perf_counter()

        # 1. Configuration Gates
        if not self.enable_gemini:
            return GenerationResult(
                success=False,
                error_code="GEMINI_DISABLED",
                error_message="Gemini integration is disabled via configuration (ENABLE_GEMINI=false).",
            )

        if not self.api_key or not str(self.api_key).strip():
            return GenerationResult(
                success=False,
                error_code="MISSING_API_KEY",
                error_message="GEMINI_API_KEY environment variable is not configured.",
            )

        if not self.model_name or not str(self.model_name).strip():
            return GenerationResult(
                success=False,
                error_code="MISSING_MODEL",
                error_message="GEMINI_MODEL configuration is missing or empty.",
            )

        if self._client is None:
            return GenerationResult(
                success=False,
                error_code="CLIENT_UNAVAILABLE",
                error_message=self._init_error or "Gemini client could not be initialized.",
            )

        # 2. Build Grounded Prompt
        system_instruction, prompt_body = build_grounding_prompt(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )

        # 3. Execute Generation with Timeout & Exception Containment
        try:
            config = None
            if types is not None:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=600,
                )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt_body,
                config=config,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # 4. Validate Response Payload
            candidate_text: Optional[str] = None
            if hasattr(response, "text"):
                candidate_text = response.text

            if not candidate_text or not candidate_text.strip():
                return GenerationResult(
                    success=False,
                    error_code="EMPTY_RESPONSE",
                    error_message="Gemini API returned an empty or whitespace-only response.",
                    model=self.model_name,
                    duration_ms=round(duration_ms, 2),
                )

            return GenerationResult(
                text=candidate_text.strip(),
                success=True,
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            exc_str = str(exc)
            exc_type = type(exc).__name__

            # Error classification without exposing credentials in telemetry or logs
            error_code = "API_ERROR"
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower():
                error_code = "QUOTA_EXCEEDED"
            elif any(k in exc_str for k in ("401", "403", "API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                error_code = "AUTH_FAILURE"
            elif isinstance(exc, (TimeoutError, TimeoutException if "TimeoutException" in globals() else ())) or "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
                error_code = "TIMEOUT"

            logger.warning(
                "Gemini generation failure contained: code=%s, type=%s, duration_ms=%.1f",
                error_code,
                exc_type,
                duration_ms,
            )

            return GenerationResult(
                success=False,
                error_code=error_code,
                error_message=f"Gemini generation failed: {error_code}",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
            )

    def __call__(self, state: MindBridgeState) -> Optional[str]:
        """
        Orchestration graph callable hook.
        Invoked by `response_node(state, generator=generator)`.
        
        Returns:
            Grounded response string if successful.
            None on any failure, triggering seamless local fallback in response_node.
        """
        user_message = state.get("sanitized_text") or state.get("_ephemeral_raw_text") or ""
        retrieved_chunks = state.get("retrieved_chunks") or []
        conversation_history = state.get("messages") or []

        result = self.generate_response(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )

        if result.success and result.text:
            return result.text
        return None


class GroqGenerator:
    """
    Groq Cloud API Client for grounded first-response generation.
    Compatible with the GeminiGenerator interface.
    """
    provider_name: str = "groq"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model_name = model_name if model_name is not None else settings.groq_model
        self.timeout = timeout if timeout is not None else settings.groq_request_timeout

        self._client = client
        self._init_error: Optional[str] = None
        self.last_result: Optional[GenerationResult] = None

        if self._client is None and self.api_key:
            if not _GROQ_AVAILABLE:  # pragma: no cover
                self._init_error = "groq package is not installed."
                logger.error("Failed to initialize Groq client: %s", self._init_error)
            else:
                try:
                    self._client = Groq(
                        api_key=self.api_key,
                        timeout=float(self.timeout),
                    )
                except Exception as e:
                    self._init_error = f"Client initialization error: {type(e).__name__}"
                    logger.error("Failed to initialize Groq client: %s", self._init_error)

    def is_available(self) -> bool:
        """Evaluates whether the generator is configured and ready for requests."""
        return bool(self.api_key and self.model_name and (self._client is not None))

    def generate_response(
        self,
        user_message: str,
        retrieved_chunks: Optional[List[KnowledgeChunk]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """
        Executes grounded generation via Groq Chat Completions API with error containment.
        Never raises unhandled exceptions.
        """
        start_time = time.perf_counter()

        # 1. Configuration Gates
        if not self.api_key or not str(self.api_key).strip():
            return GenerationResult(
                success=False,
                error_code="MISSING_API_KEY",
                error_message="GROQ_API_KEY environment variable is not configured.",
                model=self.model_name,
            )

        if not self.model_name or not str(self.model_name).strip():
            return GenerationResult(
                success=False,
                error_code="MISSING_MODEL",
                error_message="GROQ_MODEL configuration is missing or empty.",
                model=self.model_name,
            )

        if self._client is None:
            return GenerationResult(
                success=False,
                error_code="CLIENT_UNAVAILABLE",
                error_message=self._init_error or "Groq client could not be initialized.",
                model=self.model_name,
            )

        # 2. Build Grounded Prompt (reusing common prompt builder)
        system_instruction, prompt_body = build_grounding_prompt(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_body},
        ]

        # 3. Call Chat Completions API (Non-streaming to preserve verification gate)
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 1.0,
                "max_completion_tokens": 2048,
                "top_p": 1.0,
            }
            if "gpt-oss" in str(self.model_name).lower():
                kwargs["reasoning_effort"] = "medium"

            try:
                completion = self._client.chat.completions.create(**kwargs)
            except Exception as param_err:
                # If reasoning_effort was rejected by a particular model, retry cleanly without it
                if "reasoning_effort" in kwargs and ("reasoning_effort" in str(param_err) or "unexpected" in str(param_err).lower()):
                    kwargs.pop("reasoning_effort", None)
                    completion = self._client.chat.completions.create(**kwargs)
                else:
                    raise param_err

            duration_ms = (time.perf_counter() - start_time) * 1000

            # 4. Extract and validate text
            candidate_text: Optional[str] = None
            if completion and hasattr(completion, "choices") and completion.choices:
                choice = completion.choices[0]
                if hasattr(choice, "message") and choice.message:
                    candidate_text = choice.message.content

            if not candidate_text or not candidate_text.strip():
                return GenerationResult(
                    success=False,
                    error_code="EMPTY_RESPONSE",
                    error_message="Groq API returned an empty or whitespace-only response.",
                    model=self.model_name,
                    duration_ms=round(duration_ms, 2),
                )

            return GenerationResult(
                text=candidate_text.strip(),
                success=True,
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            exc_str = str(exc)
            exc_type = type(exc).__name__

            error_code = "API_ERROR"
            if "RateLimitError" in exc_type or "429" in exc_str or "rate limit" in exc_str.lower() or "quota" in exc_str.lower():
                error_code = "QUOTA_EXCEEDED"
            elif "AuthenticationError" in exc_type or "PermissionDeniedError" in exc_type or any(k in exc_str for k in ("401", "403", "invalid_api_key", "invalid api key")):
                error_code = "AUTH_FAILURE"
            elif "APITimeoutError" in exc_type or isinstance(exc, TimeoutError) or "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
                error_code = "TIMEOUT"

            logger.warning(
                "Groq generation failure contained: code=%s, type=%s, duration_ms=%.1f",
                error_code,
                exc_type,
                duration_ms,
            )

            return GenerationResult(
                success=False,
                error_code=error_code,
                error_message=f"Groq generation failed: {error_code}",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
            )

    def __call__(self, state: MindBridgeState) -> Optional[str]:
        """
        Orchestration graph callable hook.
        Invoked by `response_node(state, generator=generator)`.
        
        Returns:
            Grounded response string if successful.
            None on any failure, triggering seamless local fallback in response_node.
        """
        user_message = state.get("sanitized_text") or state.get("_ephemeral_raw_text") or ""
        retrieved_chunks = state.get("retrieved_chunks") or []
        conversation_history = state.get("messages") or []

        result = self.generate_response(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )

        if result.success and result.text:
            return result.text
        return None


class CascadingLLMGenerator:
    """
    Cascading Multi-Provider LLM Generator.
    Executes primary LLM generation with automatic fallback:
    1. Primary: Gemini Developer API (Google AI Studio)
    2. Fallback: Groq Cloud API
    3. Final Fallback: Local deterministic grounded synthesis in response_node (returns None)
    """
    provider_name: str = "cascade"

    def __init__(
        self,
        primary_generator: Optional[GeminiGenerator] = None,
        fallback_generator: Optional[GroqGenerator] = None,
        max_retries: int = 1,
    ):
        self.primary_generator = primary_generator
        self.fallback_generator = fallback_generator
        self.max_retries = max_retries
        self.last_result: Optional[GenerationResult] = None

    @property
    def active_provider(self) -> str:
        """Returns the currently ready provider name: 'gemini', 'groq', or 'local'."""
        if self.primary_generator and self.primary_generator.is_available():
            return "gemini"
        elif self.fallback_generator and self.fallback_generator.is_available():
            return "groq"
        return "local"

    def is_available(self) -> bool:
        """Returns True if at least one remote provider is configured and available."""
        return bool(
            (self.primary_generator and self.primary_generator.is_available())
            or (self.fallback_generator and self.fallback_generator.is_available())
        )

    def get_provider_status(self) -> Dict[str, Any]:
        """Returns safe provider diagnostics without exposing API keys or secrets."""
        gemini_ready = bool(self.primary_generator and self.primary_generator.is_available())
        groq_ready = bool(self.fallback_generator and self.fallback_generator.is_available())
        return {
            "cascade_enabled": True,
            "active_provider": self.active_provider,
            "primary": {
                "name": "gemini",
                "available": gemini_ready,
                "model": self.primary_generator.model_name if self.primary_generator else None,
                "configured": bool(self.primary_generator and self.primary_generator.api_key),
            },
            "fallback": {
                "name": "groq",
                "available": groq_ready,
                "model": self.fallback_generator.model_name if self.fallback_generator else None,
                "configured": bool(self.fallback_generator and self.fallback_generator.api_key),
            },
        }

    def generate_response(
        self,
        user_message: str,
        retrieved_chunks: Optional[List[KnowledgeChunk]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """
        Executes generation with automatic multi-provider cascade:
        Tries primary (Gemini) -> if failure, logs & tries fallback (Groq) -> if failure, returns failure.
        """
        start_time = time.perf_counter()
        last_error_code = "NO_PROVIDERS_AVAILABLE"
        last_error_msg = "No remote LLM providers are configured or available."

        # 1. Attempt Primary: Gemini Developer API
        if self.primary_generator and self.primary_generator.is_available():
            res = self.primary_generator.generate_response(
                user_message=user_message,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history,
            )
            if res.success:
                res.provider = "gemini"
                res.fallback_triggered = False
                res.local_fallback = False
                self.last_result = res
                return res

            last_error_code = res.error_code or "GEMINI_ERROR"
            last_error_msg = res.error_message or "Gemini generation failed"
            logger.warning(
                "Primary provider (Gemini) failed (code=%s). Falling back to Groq Cloud API...",
                last_error_code,
            )

        # 2. Attempt Fallback: Groq Cloud API
        if self.fallback_generator and self.fallback_generator.is_available():
            res = self.fallback_generator.generate_response(
                user_message=user_message,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history,
            )
            if res.success:
                res.provider = "groq"
                res.fallback_triggered = True
                res.local_fallback = False
                self.last_result = res
                logger.info("Fallback provider (Groq) successfully generated response.")
                return res

            last_error_code = res.error_code or "GROQ_ERROR"
            last_error_msg = res.error_message or "Groq generation failed"
            logger.warning(
                "Fallback provider (Groq) failed (code=%s). Engaging local deterministic synthesis...",
                last_error_code,
            )

        # 3. Both failed (or neither available)
        duration_ms = (time.perf_counter() - start_time) * 1000
        fail_res = GenerationResult(
            success=False,
            error_code=last_error_code,
            error_message=last_error_msg,
            model="cascade_exhausted",
            duration_ms=round(duration_ms, 2),
            provider="local",
            fallback_triggered=True,
            local_fallback=True,
        )
        self.last_result = fail_res
        return fail_res

    def __call__(self, state: MindBridgeState) -> Optional[str]:
        user_message = state.get("sanitized_text") or state.get("_ephemeral_raw_text") or ""
        retrieved_chunks = state.get("retrieved_chunks") or []
        conversation_history = state.get("messages") or []

        result = self.generate_response(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )

        if result.success and result.text:
            return result.text
        return None


def create_llm_generator(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = None,
    enable_gemini: Optional[bool] = None,
    client: Optional[Any] = None,
    fallback_client: Optional[Any] = None,
) -> Optional[Union[GeminiGenerator, GroqGenerator, CascadingLLMGenerator]]:
    """
    Factory creating the configured LLM generator based on provider selection.
    
    Supported providers:
    - 'auto' / 'cascade': Cascading Multi-Provider (Gemini -> Groq -> Local deterministic fallback)
    - 'gemini': Google AI Studio Gemini Developer API directly
    - 'groq': Groq Cloud Chat Completions API directly
    - 'local': Local deterministic grounded fallback (returns None)
    """
    prov = (provider or settings.llm_provider or "auto").lower().strip()

    if prov == "local":
        # Deterministic grounded synthesis fallback
        return None
    elif prov == "groq":
        key = api_key if api_key is not None else settings.groq_api_key
        model = model_name if model_name is not None else settings.groq_model
        to = timeout if timeout is not None else settings.groq_request_timeout
        return GroqGenerator(api_key=key, model_name=model, timeout=to, client=client)
    elif prov == "gemini":
        key = api_key if api_key is not None else settings.gemini_api_key
        model = model_name if model_name is not None else settings.gemini_model
        to = timeout if timeout is not None else settings.gemini_request_timeout
        en = enable_gemini if enable_gemini is not None else settings.enable_gemini
        return GeminiGenerator(api_key=key, model_name=model, timeout=to, enable_gemini=en, client=client)
    elif prov in ("auto", "cascade"):
        gemini_gen = GeminiGenerator(
            api_key=api_key if api_key is not None else settings.gemini_api_key,
            model_name=model_name if model_name is not None else settings.gemini_model,
            timeout=timeout if timeout is not None else settings.gemini_request_timeout,
            enable_gemini=enable_gemini if enable_gemini is not None else settings.enable_gemini,
            client=client,
        )
        groq_gen = GroqGenerator(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            timeout=settings.groq_request_timeout,
            client=fallback_client,
        )
        return CascadingLLMGenerator(
            primary_generator=gemini_gen,
            fallback_generator=groq_gen,
            max_retries=settings.gemini_max_retries,
        )
    else:
        # Unknown provider: returns None, engaging local deterministic fallback
        return None
