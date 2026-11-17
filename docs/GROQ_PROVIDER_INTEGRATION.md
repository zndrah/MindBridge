# Groq Provider Integration Guide

**Module:** `backend/generation/llm.py`  
**Provider:** Groq Cloud Chat Completions API (`groq` SDK)  
**Implementation Phase:** Phase 1B  
**Document Version:** 1.0.0  
**Date:** 2026-11-03  

---

## 1. Architectural Overview & Context

In Phase 1B, the MindBridge generation architecture was extended to support **Groq Cloud** as an additional LLM provider alongside Google AI Studio Gemini Developer API, without modifying or breaking existing Gemini functionality.

### 1.1 Provider Matrix & Selection
MindBridge supports three distinct generation modes via `LLM_PROVIDER`:

| `LLM_PROVIDER` | Provider Class | Client / Backend | Default Model | Fallback Target |
| :--- | :--- | :--- | :--- | :--- |
| `groq` | `GroqGenerator` | Groq Cloud Chat Completions | `openai/gpt-oss-120b` | Local deterministic synthesis |
| `gemini` | `GeminiGenerator` | Google AI Studio Developer API | `gemini-2.5-flash` | Local deterministic synthesis |
| `local` | `None` | Built-in offline CPU extraction | Deterministic templates | Self (vetted templates) |

* **Provider Isolation:** The system does not silently switch between cloud providers (e.g., if Groq fails, it does NOT call Gemini). Instead, if the configured provider fails, it immediately activates MindBridge's safe local deterministic grounded synthesis ([`synthesize_grounded_response`](file:///home/adarsh/projects/MindBridge/backend/orchestrator/response_node.py#L150)).
* **Emergency Bypass Independence:** Emergency requests bypass Groq, Gemini, and both RAG collections, routing deterministically to localized emergency hotlines.

---

## 2. Configuration & Secrets

All Groq parameters are configured through environment variables.

### 2.1 Environment Variables

```env
# Provider Selection ('gemini', 'groq', or 'local')
LLM_PROVIDER=groq

# Groq Cloud API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REQUEST_TIMEOUT=10.0
```

### 2.2 Security Invariants
* Zero hardcoded credentials or API keys.
* `GROQ_API_KEY` is read strictly from environment variables.
* Errors and telemetry sanitize credentials and never print keys in logs.
* `.env` remains ignored by Git; `.env.example` contains placeholders only.

---

## 3. Groq Chat Completions API Implementation

### 3.1 Non-Streaming Generation & Verification Gate Preservation
Groq generation uses non-streaming `client.chat.completions.create` to ensure that generated drafts pass completely through MindBridge's deterministic post-generation verification gate ([`verification_node.py`](file:///home/adarsh/projects/MindBridge/backend/orchestrator/verification_node.py)) before exposure to the user:

```python
completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=1.0,
    max_completion_tokens=2048,
    top_p=1.0,
    reasoning_effort="medium"  # dynamically adapted if supported by model
)
```

### 3.2 Parameter Compatibility & Adaptive Fallback
* When using reasoning models such as `openai/gpt-oss-120b`, `reasoning_effort="medium"` is passed.
* If a configured model does not support `reasoning_effort`, `GroqGenerator` catches the parameter error and automatically retries cleanly without `reasoning_effort`, preventing request failure.

---

## 4. Grounded Prompt & Safety Policy Reuse

`GroqGenerator` directly reuses the shared prompt construction pipeline ([`build_grounding_prompt`](file:///home/adarsh/projects/MindBridge/backend/generation/llm.py)):
1. **Clinical Boundaries:** Strict prohibitions against diagnosing mental health disorders, recommending medications, or claiming emergency dispatch.
2. **Community Q&A vs. Approved Knowledge:** CounselChat records are explicitly framed as `Unverified Community Q&A (Peer Perspective — Non-Clinical, Illustrative Only)`.
3. **Citation Provenance:** The model is commanded to use only exact Chunk IDs or titles from reference items, forbidding fabricated citations.

---

## 5. Automated Hermetic Tests

15 focused unit and integration tests are implemented in [`tests/test_groq_generation.py`](file:///home/adarsh/projects/MindBridge/tests/test_groq_generation.py):
* Mocked client initialization
* Missing API key containment
* Missing model containment
* Successful Groq chat completion
* Authentication error (401/403) containment
* Rate limit (429) containment
* Timeout error containment
* Empty response containment
* Local fallback activation in `response_node`
* Prompt construction and community content framing
* Emergency bypass independence
* Post-generation verification auditing
* Provider selection across Gemini, Groq, and local modes
* Parameter adaptation for reasoning effort

All tests run hermetically without network access or live API keys:
```bash
PYTHONPATH=. uv run pytest tests/test_groq_generation.py -v
```

> [!NOTE]
> **No live Groq API calls were performed during implementation or testing.** All functionality was verified using mock objects and synthetic network/error injection.
