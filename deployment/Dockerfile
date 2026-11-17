# MindBridge 2.0 - Staging & Production Container Definition
# ==========================================================
# Secure, non-root, self-contained deployment image.
# Does not download external model weights at runtime.
# Enforces non-root execution (UID 10001) and embedded-secret prohibition.

FROM python:3.11-slim-bookworm AS runtime

# System runtime dependencies and security configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=staging \
    HOST=0.0.0.0 \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    MAX_CONCURRENT_REQUESTS=10 \
    RATE_LIMIT_PER_MINUTE=60 \
    TRUSTED_PROXIES=127.0.0.1,::1,172.28.0.0/16 \
    DEFAULT_JURISDICTION=GL \
    CRISIS_PROBABILITY_THRESHOLD=0.509765625 \
    SAFE_DEGRADATION_ON_ERROR=true \
    PATH="/app/.venv/bin:$PATH"

# Create unprivileged service user and group (no login shell, no home directory)
RUN groupadd -g 10001 mindbridge && \
    useradd -u 10001 -g mindbridge -s /bin/false -M -d /app mindbridge

WORKDIR /app

# Install Astral uv binary from official multi-arch image
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /bin/uv

# Install Python dependencies using frozen lockfile (CPU-only PyTorch)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application assets, local models, and knowledge base
# (Secrets and raw clinical datasets are excluded via .dockerignore)
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY configs/ ./configs/

# Assign read/execute permissions to unprivileged user
RUN chown -R mindbridge:mindbridge /app && \
    chmod -R 755 /app

# Switch to unprivileged execution
USER 10001:10001

# Expose API port
EXPOSE 8000

# Native Python healthcheck probe (zero external tool dependencies)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).getcode() == 200 else 1)"

# Single-worker staging execution enforcing in-memory rate limiting and concurrency limits
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
