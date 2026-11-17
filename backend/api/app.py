"""
MindBridge 2.0 - FastAPI Application Factory (`app.py`)
=======================================================

Assembles the FastAPI application with:
- Async lifespan model pre-warming and cleanup.
- CORS policy.
- Privacy-by-design logging and rate limiting middleware.
- Sanitized exception handlers preventing raw stack traces or text leakage.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator
import uuid
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.dependencies import app_state
from backend.api.middleware import (
    MaxBodySizeMiddleware,
    PrivacyLoggingMiddleware,
    RateLimitMiddleware,
)
from backend.api.routes import router
from backend.config import settings

logger = logging.getLogger("mindbridge.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager: pre-loads local models and compiles
    the orchestration graph once during application startup.
    Avoids reloading models per request and never downloads external weights.
    """
    logger.info("Initializing MindBridge 2.0 application lifespan...")

    # Configure PyTorch CPU inference threads for MindBridge staging optimization
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        intra_threads = torch.get_num_threads()
        inter_threads = torch.get_num_interop_threads()
        logger.info(
            "PyTorch CPU inference threads configured: intra_op=%d, inter_op=%d",
            intra_threads,
            inter_threads,
        )
    except Exception as exc:
        error_msg = type(exc).__name__
        logger.warning(
            "Failed to configure PyTorch CPU thread count (%s); proceeding with default threading",
            error_msg,
        )

    # Initialize singletons and pre-warm local models
    app_state.initialize_sync()

    if settings.web_concurrency > 1:
        logger.warning(
            "WEB_CONCURRENCY=%d configured with process-local InMemoryRateLimiter. "
            "Rate limits and concurrency semaphores are NOT globally shared across "
            "worker processes. Single-worker deployment (WEB_CONCURRENCY=1) is strictly "
            "required for staging until a shared distributed backend exists.",
            settings.web_concurrency,
        )

    for proxy in settings.trusted_proxies:
        if proxy.strip() in ("10.0.0.0/8", "0.0.0.0/0", "::/0"):
            logger.warning(
                "Overly broad TRUSTED_PROXIES entry detected: '%s'. Broad CIDRs allow "
                "untrusted network hosts to forge X-Forwarded-For headers and bypass rate limits.",
                proxy,
            )

    if app_state.is_ready:
        logger.info("All local models and indexes initialized successfully.")
    else:
        logger.warning("Application started in degraded mode: %s", app_state._initialization_error)

    yield

    logger.info("Shutting down MindBridge 2.0 application...")
    app_state.shutdown()


def create_app() -> FastAPI:
    """
    Factory creating and configuring the MindBridge 2.0 FastAPI application.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Safety-first, non-clinical mental health first-response and "
            "knowledge-grounded support system."
        ),
        lifespan=lifespan,
    )

    # 1. CORS Configuration (explicit origins only; no wildcard in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # 2. Maximum Request Body Size Middleware (rejects oversized payloads before buffering)
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_body_size=settings.max_body_size_bytes,
    )

    # 3. Rate Limiting Middleware (respecting trusted reverse proxies)
    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=app_state.rate_limiter,
        trusted_proxies=settings.trusted_proxies,
    )

    # 4. Privacy-by-Design Logging Middleware
    app.add_middleware(
        PrivacyLoggingMiddleware,
        trusted_proxies=settings.trusted_proxies,
    )

    # 4. Sanitized Error Handlers (Zero Raw Exception Leakage)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Format safe field-level errors without printing user input values
        safe_errors = []
        is_message_too_long = False

        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            safe_errors.append(f"{loc}: {err.get('msg', 'Invalid input')}")
            err_loc = err.get("loc", ())
            if "message" in err_loc and err.get("type") in ("string_too_long", "string_length"):
                is_message_too_long = True

        # Specialized safety handling for oversized conversational messages
        if is_message_too_long and request.url.path.endswith("/chat"):
            try:
                body_json = await request.json()
            except Exception:
                body_json = {}

            if isinstance(body_json, dict):
                raw_message = body_json.get("message")
                if isinstance(raw_message, str) and len(raw_message) > settings.max_input_length:
                    from backend.orchestrator.input_node import (
                        SAFETY_SCAN_CEILING,
                        pre_truncation_safety_scan,
                    )
                    from backend.api.schemas import EmergencyResourceItem, VALID_JURISDICTIONS
                    from backend.orchestrator.safety_node import get_hotlines_for_jurisdiction

                    # Bounded safety scan over scannable window (strictly capped at SAFETY_SCAN_CEILING)
                    bounded_text = raw_message[:SAFETY_SCAN_CEILING]
                    trigger_id, is_passive, is_ideation = pre_truncation_safety_scan(bounded_text)

                    # Extract jurisdiction safely
                    raw_jurisdiction = body_json.get("jurisdiction")
                    default_jur = getattr(settings, "default_jurisdiction", "IN") or "IN"
                    jurisdiction = (
                        raw_jurisdiction.upper()
                        if isinstance(raw_jurisdiction, str) and raw_jurisdiction.upper() in VALID_JURISDICTIONS
                        else default_jur
                    )

                    # 1. Imminent suicide or direct self-harm threat detected in oversized input
                    if trigger_id is not None:
                        session_id = body_json.get("session_id") or str(uuid.uuid4())
                        raw_hotlines = get_hotlines_for_jurisdiction(jurisdiction)
                        hotline_items = [EmergencyResourceItem(**h).model_dump() for h in raw_hotlines]

                        response_text = (
                            "If you are feeling overwhelmed, considering ending your life, or in immediate danger, "
                            "please know that you do not have to carry this alone. Immediate, confidential support is available 24/7.\n\n"
                            "Your message exceeded the maximum permitted length (10,000 characters) and could not undergo "
                            "conversational processing. Because your safety is our highest priority, immediate crisis "
                            "support resources are provided below.\n\n"
                            "Notice: This response is an automated safety precaution and does not represent a clinical assessment."
                        )

                        return JSONResponse(
                            status_code=200,
                            content={
                                "session_id": session_id,
                                "response": response_text,
                                "safety_route": "emergency_bypass",
                                "verification_status": "not_run",
                                "grounding_status": None,
                                "citations": [],
                                "emergency_resources": hotline_items,
                                "error_code": "INPUT_PARSING_FAILURE",
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                    # 2. Suicidal ideation or passive death wishes without imminent emergency trigger
                    if is_ideation or is_passive:
                        session_id = body_json.get("session_id") or str(uuid.uuid4())
                        raw_hotlines = get_hotlines_for_jurisdiction(jurisdiction)
                        hotline_items = [EmergencyResourceItem(**h).model_dump() for h in raw_hotlines]

                        response_text = (
                            "If you are experiencing thoughts of suicide or feeling overwhelmed, please know that support is available right now. "
                            "Your message exceeded the maximum permitted length (10,000 characters) and could not undergo conversational processing. "
                            "Free, confidential support resources are provided below.\n\n"
                            "Notice: This response is an automated safety precaution and does not represent a clinical assessment."
                        )

                        return JSONResponse(
                            status_code=200,
                            content={
                                "session_id": session_id,
                                "response": response_text,
                                "safety_route": "emergency_bypass",
                                "verification_status": "not_run",
                                "grounding_status": None,
                                "citations": [],
                                "emergency_resources": hotline_items,
                                "error_code": "INPUT_PARSING_FAILURE",
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                    # 3. Ordinary oversized message without any safety crisis disclosure
                    return JSONResponse(
                        status_code=422,
                        content={
                            "error": (
                                "Your message exceeds the maximum allowed length (10,000 characters). "
                                "To ensure support is safe and accurate, please share a shorter message about what is on your mind."
                            ),
                            "error_code": "VALIDATION_ERROR",
                            "details": safe_errors,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )

        return JSONResponse(
            status_code=422,
            content={
                "error": "Request validation failed.",
                "error_code": "VALIDATION_ERROR",
                "details": safe_errors,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        if "timestamp_utc" not in detail:
            detail["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return JSONResponse(status_code=exc.status_code, content=detail)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception encountered on path: %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "An internal server error occurred.",
                "error_code": "INTERNAL_SERVER_ERROR",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    # 5. Include API routes
    app.include_router(router)

    return app


# Default application instance for ASGI servers (e.g. uvicorn backend.api.app:app)
app = create_app()
