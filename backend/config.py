"""
MindBridge 2.0 - Application Configuration (`config.py`)
========================================================

Environment-based runtime configuration with strict safety defaults.
Authoritative references:
- docs/PRD.md §5.2 (NFR-2: Privacy-by-Design & Input Safety)
- docs/ARCHITECTURE.md §3.1 (State Schema & Boundary Isolation)
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.orchestrator.state import JurisdictionCode

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicitly load environment variables from project-root .env file
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseModel):
    """
    Application configuration loaded from environment variables with safe defaults.
    """
    app_name: str = "MindBridge 2.0 API"
    app_version: str = "0.2.0"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Network binding
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    # CORS policy (never defaults to wildcard in production)
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ]
    )
    
    # Regional jurisdiction default
    default_jurisdiction: JurisdictionCode = Field(
        default_factory=lambda: os.getenv("DEFAULT_JURISDICTION", "IN")  # type: ignore
    )
    
    # Model locations
    crisis_model_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("CRISIS_MODEL_DIR", "models/crisis_detector/roberta")
    )
    embedding_model_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("EMBEDDING_MODEL_DIR", "models/embeddings/bge-small-en-v1.5")
    )
    
    # Concurrency and Rate Limiting
    rate_limit_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    max_concurrent_requests: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    )
    
    # Reverse proxy security (trusted proxy IPs allowed to forward client headers)
    # NOTE: Never configure broad subnets like 10.0.0.0/8, as that allows any internal
    # or compromised host on that network to forge forwarded headers and bypass rate limits.
    # Specify only exact reverse proxy IPs or narrow ingress subnets (e.g. 127.0.0.1, ::1).
    trusted_proxies: List[str] = Field(
        default_factory=lambda: [
            proxy.strip()
            for proxy in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",")
            if proxy.strip()
        ]
    )

    # Request body size boundary (in bytes; default: 64 KB)
    max_body_size_bytes: int = Field(
        default_factory=lambda: int(os.getenv("MAX_BODY_SIZE_BYTES", "65536"))
    )

    # Web concurrency / worker count for CPU-bound model inference
    # NOTE: In-memory sliding-window rate limiting and concurrency semaphores are
    # process-local. For staging deployments using InMemoryRateLimiter, WEB_CONCURRENCY
    # MUST remain 1 to ensure global rate limit and concurrency enforcement.
    # Multi-worker deployments require an external distributed store (e.g., Redis).
    web_concurrency: int = Field(
        default_factory=lambda: int(os.getenv("WEB_CONCURRENCY", "1"))
    )

    # Input validation bounds
    max_input_length: int = Field(default=10000)
    max_history_turns: int = Field(default=20)

    # LLM Provider Selection ('auto', 'gemini', 'groq', 'local')
    # 'auto': Primary Gemini -> Fallback Groq -> Final Fallback Local deterministic
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower().strip()
    )

    # Google AI Studio - Gemini Developer API Configuration
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    enable_gemini: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_GEMINI", "true").lower() == "true"
    )
    gemini_request_timeout: float = Field(
        default_factory=lambda: float(os.getenv("GEMINI_REQUEST_TIMEOUT", "10.0"))
    )
    gemini_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_RETRIES", "1"))
    )

    # Groq Provider Configuration
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    groq_request_timeout: float = Field(
        default_factory=lambda: float(os.getenv("GROQ_REQUEST_TIMEOUT", "10.0"))
    )
    groq_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_RETRIES", "1"))
    )


# Global singleton settings instance
settings = Settings()
