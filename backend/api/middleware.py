"""
MindBridge 2.0 - Privacy-by-Design Middleware & Rate Limiting (`middleware.py`)
================================================================================

Enforces strict boundary privacy guarantees:
1. PrivacyLoggingMiddleware: Prohibits logging raw mental health text or request bodies.
2. InMemoryRateLimiter: Sliding window rate limiting protecting CPU model inference.
"""

from collections import deque
from datetime import datetime, timezone
import ipaddress
import logging
import time
from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.config import settings

logger = logging.getLogger("mindbridge.api")


def is_trusted_proxy(direct_peer: str, trusted_proxies: List[str]) -> bool:
    """
    Safely verifies whether direct_peer is an authorized reverse proxy.
    Supports both exact IP matches (e.g., '127.0.0.1', '::1') and CIDR notation
    (e.g., '172.28.0.0/16').
    
    Security note: Broad networks like '10.0.0.0/8' or '0.0.0.0/0' must NOT be
    trusted, as any internal or external host on that network could forge forwarded
    headers and bypass rate limits.
    """
    if not direct_peer or direct_peer == "unknown":
        return False
    # Fast path: exact string match
    if direct_peer in trusted_proxies:
        return True
    try:
        peer_ip = ipaddress.ip_address(direct_peer)
    except ValueError:
        return False

    for proxy_entry in trusted_proxies:
        if "/" in proxy_entry:
            try:
                network = ipaddress.ip_network(proxy_entry, strict=False)
                if peer_ip in network:
                    return True
            except ValueError:
                continue
    return False


def get_client_ip(request: Request, trusted_proxies: Optional[List[str]] = None) -> str:
    """
    Safely resolves the client's IP address.
    
    If the direct connection originates from a trusted reverse proxy (e.g. 127.0.0.1,
    or a dedicated narrow ingress subnet), reads the original client IP from
    X-Forwarded-For (leftmost client hop) or X-Real-IP.
    If the direct connection is NOT from a trusted proxy, any forwarded headers are
    strictly ignored and discarded to prevent IP spoofing and rate limit evasion.
    """
    direct_peer = request.client.host if request.client else "unknown"
    if trusted_proxies is None:
        trusted_proxies = settings.trusted_proxies

    if is_trusted_proxy(direct_peer, trusted_proxies):
        # Check X-Forwarded-For header
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if hops:
                return hops[0]
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip and real_ip.strip():
            return real_ip.strip()

    return direct_peer


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Enforces a strict upper bound on incoming HTTP request body sizes (default: 64 KB).
    Protects the server from memory exhaustion and denial-of-service before request
    buffering into memory or Pydantic serialization.
    """
    def __init__(self, app, max_body_size: Optional[int] = None):
        super().__init__(app)
        self.max_body_size = max_body_size if max_body_size is not None else settings.max_body_size_bytes

    def _create_payload_too_large_response(self, request: Request) -> JSONResponse:
        from backend.orchestrator.safety_node import get_hotlines_for_jurisdiction
        from backend.api.schemas import EmergencyResourceItem

        # Supported regional jurisdictions with directory hotline records in hotlines.json
        SUPPORTED_REGIONAL_JURISDICTIONS = frozenset({"IN", "US", "UK", "CA", "AU"})

        raw_header = request.headers.get("x-jurisdiction")
        if raw_header and isinstance(raw_header, str):
            clean_jurisdiction = raw_header.strip().upper()
            jurisdiction = clean_jurisdiction if clean_jurisdiction in SUPPORTED_REGIONAL_JURISDICTIONS else "GL"
        else:
            jurisdiction = "GL"

        raw_hotlines = get_hotlines_for_jurisdiction(jurisdiction)
        if not raw_hotlines:
            raw_hotlines = get_hotlines_for_jurisdiction("GL")

        hotline_items = [EmergencyResourceItem(**h).model_dump() for h in raw_hotlines]

        return JSONResponse(
            status_code=413,
            content={
                "error": (
                    f"Request body exceeds the maximum permitted size of {self.max_body_size} bytes (64 KB). "
                    "To ensure support is safe, reliable, and responsive, please share a shorter message."
                ),
                "error_code": "PAYLOAD_TOO_LARGE",
                "message": (
                    "Notice: The message was too large to process or screen for safety. "
                    "If you or someone you know is in distress or immediate danger, free confidential crisis support is available 24/7."
                ),
                "emergency_resources": hotline_items,
                "clinical_assessment": "unassessed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        # Fast path check on Content-Length header
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_size:
                    return self._create_payload_too_large_response(request)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Invalid Content-Length header.",
                        "error_code": "INVALID_CONTENT_LENGTH",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )

        # For chunked or streaming requests without a Content-Length header
        if request.method in ("POST", "PUT", "PATCH"):
            total_bytes = 0
            chunks = []
            async for chunk in request.stream():
                total_bytes += len(chunk)
                if total_bytes > self.max_body_size:
                    return self._create_payload_too_large_response(request)
                chunks.append(chunk)
            request._body = b"".join(chunks)

        return await call_next(request)


class PrivacyLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware recording request metadata while strictly prohibiting logging
    of user messages, clinical disclosures, or response bodies.
    """
    def __init__(self, app, trusted_proxies: Optional[List[str]] = None):
        super().__init__(app)
        self.trusted_proxies = trusted_proxies

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        client_ip = get_client_ip(request, self.trusted_proxies)
        path = request.url.path
        method = request.method

        # Proceed with request without logging body payloads
        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # Telemetry log: ONLY records method, path, status code, IP, and latency
        logger.info(
            "API Request: method=%s path=%s status=%d client_ip=%s latency_ms=%.2f",
            method,
            path,
            response.status_code,
            client_ip,
            elapsed_ms,
        )
        return response


class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter per client IP.
    Protects CPU inference pipeline from unbounded concurrent request floods.
    """
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._clients: Dict[str, deque] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """Checks and records a request for the given client IP."""
        now = time.monotonic()
        if client_ip not in self._clients:
            self._clients[client_ip] = deque()

        timestamps = self._clients[client_ip]
        # Purge entries older than sliding window
        while timestamps and (now - timestamps[0]) > self.window_seconds:
            timestamps.popleft()

        if len(timestamps) >= self.requests_per_minute:
            return False

        timestamps.append(now)
        return True

    def reset(self) -> None:
        """Clears client rate limiting history (for test isolation)."""
        self._clients.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware enforcing per-IP rate limits before endpoint routing.
    Safely resolves client IP respecting trusted reverse proxies.
    """
    def __init__(
        self,
        app,
        rate_limiter: InMemoryRateLimiter,
        trusted_proxies: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.trusted_proxies = trusted_proxies

    async def dispatch(self, request: Request, call_next) -> Response:
        # Exclude liveness/readiness probes from aggressive rate limits
        if request.url.path in ("/health", "/ready", "/"):
            return await call_next(request)

        client_ip = get_client_ip(request, self.trusted_proxies)
        if not self.rate_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for IP: %s on path: %s", client_ip, request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Please wait before sending more requests.",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        return await call_next(request)
