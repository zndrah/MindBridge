# MindBridge 2.0 — FastAPI Backend Specification & Integration Guide

## 1. Overview
The MindBridge 2.0 API exposes the completed safety-first, non-clinical mental health first-response chatbot through a high-performance, asynchronous FastAPI backend. The API layer wraps the compiled deterministic orchestration graph (`MindBridgeStateGraph` / `CompiledGraph` in `backend/orchestrator/graph.py`—a custom pure-Python deterministic state machine implementing LangGraph-compatible interface semantics without external library dependencies) without bypassing any safety, retrieval, or verification stages.

---

## 2. Server Architecture & Endpoints

### 2.1 Service Information: `GET /`
Returns service metadata, active version, and documentation endpoints.

**Response (HTTP 200)**:
```json
{
  "name": "MindBridge 2.0 API",
  "version": "0.2.0",
  "description": "Safety-first, non-clinical mental health first-response and knowledge-grounded support system.",
  "documentation": "/docs",
  "health": "/health",
  "ready": "/ready"
}
```

---

### 2.2 Liveness Probe: `GET /health`
Fast process liveness check. Does not evaluate heavy neural models to avoid unnecessary CPU load during frequent orchestrator health checks.

**Response (HTTP 200)**:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "timestamp_utc": "2026-11-02T19:00:00.000000+00:00"
}
```

---

### 2.3 Readiness Probe: `GET /ready`
Deep subsystem readiness check. Confirms that local fine-tuned RoBERTa model weights, BGE-small embeddings, FAISS vector index, and the state graph are compiled and ready for inference.

**Response (HTTP 200 - Ready)**:
```json
{
  "status": "ready",
  "models": {
    "crisis_detector": true,
    "embedding_engine": true,
    "vector_store": true,
    "graph": true
  },
  "details": {
    "crisis_model_path": "models/crisis_detector/roberta",
    "embedding_model_path": "models/embeddings/bge-small-en-v1.5",
    "vector_count": 30,
    "jurisdiction_default": "GL"
  },
  "timestamp_utc": "2026-11-02T19:00:00.000000+00:00"
}
```

**Response (HTTP 503 - Not Ready / Degraded)**:
```json
{
  "status": "not_ready",
  "models": {
    "crisis_detector": false,
    "embedding_engine": true,
    "vector_store": true,
    "graph": false
  },
  "details": { ... },
  "error": "One or more required components are not ready.",
  "timestamp_utc": "2026-11-02T19:00:00.000000+00:00"
}
```

---

### 2.4 Conversational First-Response: `POST /chat`
Executes user input through the complete MindBridge 2.0 state machine:
`input_node` (PII Scrubbing) $\to$ `deterministic_safety_node` $\to$ `classifier_node` (RoBERTa) $\to$ `classifier_decision_node` $\to$ `planning_node` (RAG) $\to$ `response_node` (Generation) $\to$ `verification_node` (Safety Gate).

#### Request Headers:
`Content-Type: application/json`

#### Request Body (`ChatRequest`):
```json
{
  "message": "How do I do box breathing exercises for panic attacks?",
  "session_id": "optional-custom-session-uuid",
  "jurisdiction": "GL",
  "conversation_history": [
    {
      "role": "user",
      "content": "I feel anxious."
    },
    {
      "role": "assistant",
      "content": "I hear what you are experiencing. I am here to support you."
    }
  ]
}
```

#### Request Validation Rules:
- `message`: Required string (1 to 10,000 characters). Must not be empty or whitespace only.
- `session_id`: Optional string (max 64 alphanumeric, underscore, or hyphen characters: `^[a-zA-Z0-9_\-]+$`). Generated automatically via UUID4 if omitted.
- `jurisdiction`: Optional jurisdiction code (`IN`, `US`, `UK`, `CA`, `AU`, `GL`, `UNKNOWN`). Defaults to `GL`.
- `conversation_history`: Optional array of previous turns (max 20 turns, each 1 to 10,000 chars, `role` restricted to `"user"` or `"assistant"`).
- `extra="forbid"`: **Strict boundary security**. Any client attempt to pass unauthorized parameters (`retrieval_authorization`, `is_admin`, `safety_override`, `deterministic_danger_flag`) is rejected with HTTP 422.

#### Routing & Verification Semantics:
- **Emergency Bypass**: Any explicit statement of imminent self-harm, suicidal intent, or crisis trigger immediately invokes `emergency_node` with jurisdiction-filtered hotlines, bypassing RAG retrieval and generation.
- **Universal Verification**: All non-emergency generation passes through `verification_node`. If clinical assertions, ungrounded claims, or fabricated contacts are detected, the response is replaced with a safe neutral fallback.

#### Example Response — Standard Grounded Conversation:
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "response": "I hear what you are experiencing right now. Let's take a pause together. Here is a gentle, supportive grounding exercise—**Box Breathing (4-4-4-4 Parasympathetic Reset)**—that you can explore at your own pace:\n\n1. Inhale gently through the nose for 4 seconds.\n2. Hold the breath calmly for 4 seconds.\n3. Exhale slowly through the mouth for 4 seconds.\n4. Rest in stillness for 4 seconds.\nRepeat for 2 to 4 cycles.\n\nTake this at whatever pace feels comfortable. You are in control of this exercise, and you can pause or stop at any time.",
  "safety_route": "standard_conversation",
  "verification_status": "verified",
  "grounding_status": "grounded",
  "citations": [
    {
      "title": "Box Breathing (4-4-4-4 Parasympathetic Reset)",
      "document_id": "somatic_regulation:01_box_breathing_4_4_4_4",
      "category": "somatic_regulation",
      "jurisdiction": "GL"
    }
  ],
  "error_code": null,
  "timestamp_utc": "2026-11-02T19:00:00.000000+00:00"
}
```

#### Example Response — Universal Emergency Bypass:
```json
{
  "session_id": "7b819f64-1234-4562-b3fc-98765f66afa6",
  "response": "If you are in immediate danger or feeling overwhelmed and considering ending your life, please reach out to support services directly. You do not have to carry this alone, and there is help available right now:\n\n- National Emergency Response Support System (ERSS): Call 112 (Available 24/7, Toll-Free)\n- Tele-MANAS (Tele Mental Health Assistance and Networking Across States): Call 14416 or 1800-891-4416 (Available 24/7, Toll-Free)\n\nPlease contact these services directly by phone. Help is available to support you through this moment.",
  "safety_route": "emergency_bypass",
  "verification_status": "not_run",
  "grounding_status": null,
  "citations": [],
  "error_code": null,
  "timestamp_utc": "2026-11-02T19:00:00.000000+00:00"
}
```

---

## 3. Privacy, Safety & Concurrency Guarantees

1. **Privacy-by-Design & Log Sanitization**:
   - `PrivacyLoggingMiddleware` intercepts all requests. It logs strictly `method`, `path`, `status_code`, `client_ip`, and `latency_ms`.
   - **Zero Raw Disclosure Logging**: User message text and assistant response text are strictly excluded from logging to prevent mental health disclosures from persisting in logs.
2. **Ephemeral Input Isolation**:
   - Raw user input is handled in ephemeral memory within `input_node` and scrubbed of PII (names, phone numbers, emails, addresses). Raw text reference drops upon exit and is never saved in state, checkpoints, or logs. Standard CPython reference counting reclaims memory; no cryptographic RAM page-zeroing is claimed.
3. **No Client Authorization Escalation**:
   - The API layer sets `retrieval_authorization=None` on all client requests.
   - Pydantic enforces `extra="forbid"`, rejecting client attempts to inject authorization objects or safety bypass flags.
4. **Reverse Proxy Security & Rate Limiting**:
   - `RateLimitMiddleware` enforces a sliding-window rate limit (default: 60 requests/min per IP). Exceeding requests receive HTTP 429 (`RATE_LIMIT_EXCEEDED`).
   - Reverse proxy IP spoofing prevention: Forwarded headers (`X-Forwarded-For`, `X-Real-IP`) are only honored when requests originate from explicit `TRUSTED_PROXIES` (e.g. `127.0.0.1`, `::1`). Untrusted requests have forwarded headers ignored, preventing rate limit evasion.
5. **Request Body Size Limit (DoS Prevention)**:
   - `MaxBodySizeMiddleware` enforces an upper bound of 64 KB (`MAX_BODY_SIZE_BYTES=65536`). Oversized requests are rejected with HTTP 413 (`PAYLOAD_TOO_LARGE`) prior to body buffering or JSON parsing.
6. **Concurrency Bounding**:
   - A server-side `asyncio.Semaphore` (default: 10 concurrent requests) prevents CPU thread exhaustion and latency degradation during concurrent model inference.
7. **CORS Restrictions**:
   - Explicit origins configured via `ALLOWED_ORIGINS`. Wildcards (`*`) are prohibited in production.

---

## 4. Local Execution & Model Initialization

### 4.1 Prerequisites
- Python 3.11 with `.venv` and `uv`
- Local model weights in:
  - `models/crisis_detector/roberta/`
  - `models/embeddings/bge-small-en-v1.5/`

### 4.2 Local Start Commands
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run via ASGI entrypoint
python -m backend.main

# Or run via uvicorn directly with hot-reloading for development:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3 Running Tests
```bash
# Run complete test suite (195 tests)
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p "test_*.py"

# Run API test suite specifically
PYTHONPATH=. .venv/bin/python -m unittest tests/test_api.py
```

---

## 5. Known Limitations & Boundaries
1. **CPU Latency Budget**: RoBERTa and BGE inferences run on CPU. While optimized for sub-200ms latency, heavy concurrent loads should be scaled horizontally behind an ASGI reverse proxy (e.g. Nginx).
2. **Deterministic Verification Boundaries**: The verification gate uses deterministic regexes and keyword overlap checks. It prevents explicit diagnostic assertions and fabricated contacts, but does not provide formal mathematical proofs of semantic entailment.
3. **Non-Clinical Boundary**: MindBridge 2.0 is an automated educational first-response tool. It does not provide medical diagnosis, clinical psychotherapy, or direct emergency responder dispatch.
