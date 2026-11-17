# MindBridge 2.0 — Staging Deployment Architecture & Operations Guide

## Overview

This directory provides the production-ready staging deployment configuration for MindBridge 2.0. It provides a minimal, reproducible, self-contained deployment topology that runs without paid cloud dependencies or external model downloads.

## Architecture & Topology

The staging environment consists of two isolated containers on a dedicated private Docker bridge network (`staging_net`: `172.28.0.0/16`):

```
                   Internet / Staging Client
                              │
                              ▼ (Port 8080)
            ┌───────────────────────────────────┐
            │      mindbridge-reverse-proxy     │
            │          (Nginx 1.25 Alpine)      │
            │             IP: 172.28.0.2        │
            └─────────────────┬─────────────────┘
                              │
            Internal Bridge   │ X-Forwarded-For: $remote_addr
            (172.28.0.0/16)   │ (Strict Anti-Spoofing)
                              │
                              ▼ (Port 8000)
            ┌───────────────────────────────────┐
            │        mindbridge-backend         │
            │   (Python 3.11 / FastAPI ASGI)    │
            │             IP: 172.28.0.10       │
            │       Non-Root UID/GID 10001      │
            │        WEB_CONCURRENCY=1          │
            └───────────────────────────────────┘
```

## Security & Operational Controls

### 1. Worker Concurrency & Rate Limiting (`WEB_CONCURRENCY=1`)
- **Process-Local Rate Limiter:** The current rate limiter (`InMemoryRateLimiter`) uses an in-memory sliding window deque per client IP.
- **Behavior with Multi-Worker (`WEB_CONCURRENCY > 1`):** When multiple Uvicorn workers are spawned, each process maintains an independent memory space. A client whose requests alternate across workers would have their effective rate limit multiplied by the worker count (e.g. 120 req/min with 2 workers instead of 60 req/min).
- **Staging Resolution:** Staging is explicitly configured with `WEB_CONCURRENCY=1`. This guarantees that in-memory rate limiting (60 requests/minute) and concurrency semaphores (10 max concurrent requests) are strictly and globally enforced on the staging instance without introducing external distributed cache dependencies (such as Redis).
- **Scale-Out Path:** Upgrading to a multi-worker or multi-replica deployment in production requires implementing an external shared Redis/KeyDB rate-limit backend.

### 2. Trusted Proxy Validation & Anti-Spoofing
- **Why `10.0.0.0/8` is Rejected:** The `10.0.0.0/8` CIDR represents over 16.7 million IP addresses. In shared enterprise VPCs or cloud networks, trusting `10.0.0.0/8` allows any machine or compromised workload within the private network to forge `X-Forwarded-For` headers, bypassing IP rate limits and falsifying audit telemetry.
- **Enforced Configuration:** `TRUSTED_PROXIES` is strictly limited to `127.0.0.1,::1,172.28.0.0/16` (the dedicated staging bridge subnet).
- **Untrusted Header Rejection:** When requests arrive from any peer outside `TRUSTED_PROXIES`, all incoming forwarded headers are ignored and discarded. The API resolves the direct peer IP as the authoritative client address.

### 3. Non-Root Execution
- Containers execute under an unprivileged user: `mindbridge` (UID 10001, GID 10001).
- No setuid or sudo binaries are present.
- Filesystem ownership is restricted (`chmod -R 755 /app`).

### 4. Zero Embedded Secrets
- No API keys, passwords, or credentials are baked into the Docker images.
- Configuration is injected strictly via environment variables.
- Models and knowledge base assets are bundled locally into the image so the container is fully self-contained and never downloads weights or resources at runtime.

### 5. Health Check Probes
- **Backend Health Check:** Probes `/health` using Python standard library `urllib.request` every 30 seconds (no external curl binary required):
  ```bash
  python3 -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).getcode() == 200 else 1)"
  ```
- **Reverse Proxy Health Check:** Probes `/` via `wget` every 30 seconds.

## Staging Deployment Instructions

### Prerequisites
- Docker Engine 24+ and Docker Compose v2.

### Step-by-Step Deployment
1. Navigate to the repository root:
   ```bash
   cd /path/to/MindBridge
   ```
2. Prepare staging environment variables:
   ```bash
   cp deployment/.env.staging.example deployment/.env.staging
   ```
3. Build and launch the staging stack:
   ```bash
   docker compose -f deployment/docker-compose.staging.yml up -d --build
   ```
4. Verify container health status:
   ```bash
   docker compose -f deployment/docker-compose.staging.yml ps
   ```
5. Test liveness and readiness endpoints:
   ```bash
   curl -i http://localhost:8080/health
   curl -i http://localhost:8080/ready
   ```
6. Graceful shutdown:
   ```bash
   docker compose -f deployment/docker-compose.staging.yml down
   ```
