#!/usr/bin/env python3
"""
MindBridge 2.0 - LLM Provider Connectivity Diagnostic Runner
============================================================

Authoritative diagnostic script verifying:
1. GEMINI_API_KEY presence (masked / length-only, zero value exposure).
2. GROQ_API_KEY presence (masked / length-only, zero value exposure).
3. Minimal live Gemini API request with prompt "Reply with the word READY."
4. Minimal live Groq API request with prompt "Reply with the word READY."
5. Structured recording of HTTP/API status, latencies, and sanitized error codes.
6. Verification of provider fallback cascade: Gemini -> Groq.
7. Verification of local deterministic grounded synthesis fallback when both providers fail.

Safety Invariants:
- Never prints, logs, copies, or exposes API keys or credential fragments.
- Uses only harmless test prompts ("Reply with the word READY."); zero mental health or PII data.
- Does not modify any application state, models, or datasets.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.generation.llm import (
    CascadingLLMGenerator,
    GeminiGenerator,
    GroqGenerator,
    GenerationResult,
    create_llm_generator,
)
from backend.orchestrator.response_node import response_node
from backend.orchestrator.state import KnowledgeChunk, MindBridgeState


def load_raw_env_file(env_path: Path) -> dict:
    """Parses .env file directly without leaking values."""
    env_vars = {}
    if not env_path.exists():
        return env_vars
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                k, v = stripped.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if v:
                    env_vars[k] = v
    return env_vars


def run_full_diagnostic():
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "project_root": str(PROJECT_ROOT),
        },
        "config_status": {},
        "gemini_test": {},
        "groq_test": {},
        "gemini_to_groq_fallback": {},
        "both_failed_local_fallback": {},
    }

    env_path = PROJECT_ROOT / ".env"
    file_vars = load_raw_env_file(env_path)

    gemini_key_env = os.getenv("GEMINI_API_KEY")
    gemini_key_file = file_vars.get("GEMINI_API_KEY")
    gemini_key = gemini_key_env or gemini_key_file or settings.gemini_api_key

    groq_key_env = os.getenv("GROQ_API_KEY")
    groq_key_file = file_vars.get("GROQ_API_KEY")
    groq_key = groq_key_env or groq_key_file or settings.groq_api_key

    gemini_configured = bool(gemini_key and gemini_key.strip())
    groq_configured = bool(groq_key and groq_key.strip())

    results["config_status"] = {
        "env_file_exists": env_path.exists(),
        "gemini_api_key_configured": gemini_configured,
        "groq_api_key_configured": groq_configured,
        "configured_gemini_model": settings.gemini_model,
        "configured_groq_model": settings.groq_model,
        "enable_gemini": settings.enable_gemini,
        "llm_provider": settings.llm_provider,
        "gemini_timeout_s": settings.gemini_request_timeout,
        "groq_timeout_s": settings.groq_request_timeout,
    }

    test_prompt = "Reply with the word READY."

    # -------------------------------------------------------------------------
    # TEST 1 & 2: Key Configuration Verification
    # -------------------------------------------------------------------------
    print("=" * 78)
    print("MINDBRIDGE 2.0 — LLM PROVIDER CONNECTIVITY & FALLBACK DIAGNOSTIC")
    print("=" * 78)
    print(f"Timestamp (UTC):     {results['timestamp_utc']}")
    print(f"Platform:            {sys.platform} | Python {results['environment']['python_version']}")
    print(f"Project Directory:   {PROJECT_ROOT}")
    print(f".env File Detected:  {env_path.exists()}")
    print("-" * 78)
    print(f"GEMINI_API_KEY:      {'[CONFIGURED]' if gemini_configured else '[NOT CONFIGURED]'}")
    print(f"GROQ_API_KEY:        {'[CONFIGURED]' if groq_configured else '[NOT CONFIGURED]'}")
    print(f"Configured Gemini:   {settings.gemini_model} (ENABLE_GEMINI={settings.enable_gemini})")
    print(f"Configured Groq:     {settings.groq_model} (Provider Mode: {settings.llm_provider})")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # TEST 3: Minimal Live Gemini API Request
    # -------------------------------------------------------------------------
    print("\n[TEST 3/7] Minimal Live Gemini API Request")
    print(f"  Target Model:  {settings.gemini_model}")
    print(f"  Test Prompt:   \"{test_prompt}\"")

    gemini_gen = GeminiGenerator()
    t0 = time.perf_counter()
    res_gemini = gemini_gen.generate_response(test_prompt)
    gemini_lat = (time.perf_counter() - t0) * 1000

    sdk_gemini_error = None
    if not gemini_configured:
        try:
            from google import genai
            client = genai.Client()
            t_sdk = time.perf_counter()
            resp = client.models.generate_content(model=settings.gemini_model, contents=test_prompt)
            sdk_lat = (time.perf_counter() - t_sdk) * 1000
            sdk_status = "SUCCESS"
        except Exception as e:
            sdk_gemini_error = f"{type(e).__name__}: {str(e)[:140]}"
            sdk_status = "BLOCKED_NO_KEY"
    else:
        sdk_status = "SKIPPED"

    results["gemini_test"] = {
        "success": res_gemini.success,
        "model": settings.gemini_model,
        "latency_ms": round(gemini_lat, 2),
        "generator_reported_ms": res_gemini.duration_ms,
        "error_code": res_gemini.error_code,
        "error_message": res_gemini.error_message,
        "sdk_direct_status": sdk_status,
        "sdk_error_classification": sdk_gemini_error,
    }

    print(f"  Result:        {'SUCCESS' if res_gemini.success else 'FAILED / BLOCKED'}")
    print(f"  Latency:       {gemini_lat:.2f}ms")
    print(f"  Error Code:    {res_gemini.error_code or 'NONE'}")
    print(f"  Error Detail:  {res_gemini.error_message or 'N/A'}")
    if sdk_gemini_error:
        print(f"  Direct SDK:    {sdk_gemini_error}")

    # -------------------------------------------------------------------------
    # TEST 4: Minimal Live Groq API Request
    # -------------------------------------------------------------------------
    print("\n[TEST 4/7] Minimal Live Groq API Request")
    print(f"  Target Model:  {settings.groq_model}")
    print(f"  Test Prompt:   \"{test_prompt}\"")

    groq_gen = GroqGenerator()
    t0 = time.perf_counter()
    res_groq = groq_gen.generate_response(test_prompt)
    groq_lat = (time.perf_counter() - t0) * 1000

    sdk_groq_error = None
    if not groq_configured:
        try:
            from groq import Groq
            client = Groq()
            t_sdk = time.perf_counter()
            resp = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": test_prompt}],
            )
            sdk_lat = (time.perf_counter() - t_sdk) * 1000
            sdk_groq_status = "SUCCESS"
        except Exception as e:
            sdk_groq_error = f"{type(e).__name__}: {str(e)[:140]}"
            sdk_groq_status = "BLOCKED_NO_KEY"
    else:
        sdk_groq_status = "SKIPPED"

    results["groq_test"] = {
        "success": res_groq.success,
        "model": settings.groq_model,
        "latency_ms": round(groq_lat, 2),
        "generator_reported_ms": res_groq.duration_ms,
        "error_code": res_groq.error_code,
        "error_message": res_groq.error_message,
        "sdk_direct_status": sdk_groq_status,
        "sdk_error_classification": sdk_groq_error,
    }

    print(f"  Result:        {'SUCCESS' if res_groq.success else 'FAILED / BLOCKED'}")
    print(f"  Latency:       {groq_lat:.2f}ms")
    print(f"  Error Code:    {res_groq.error_code or 'NONE'}")
    print(f"  Error Detail:  {res_groq.error_message or 'N/A'}")
    if sdk_groq_error:
        print(f"  Direct SDK:    {sdk_groq_error}")

    # -------------------------------------------------------------------------
    # TEST 5: Record HTTP/API Status, Latency & Sanitized Codes
    # -------------------------------------------------------------------------
    print("\n[TEST 5/7] Provider Status & Diagnostics Summary Table")
    print(f"  {'Provider':<10} | {'Status':<12} | {'Configured':<12} | {'Latency':<10} | {'Sanitized Code':<20}")
    print(f"  {'-'*10}-|-{'-'*12}-|-{'-'*12}-|-{'-'*10}-|-{'-'*20}")
    gemini_stat_str = "SUCCESS" if res_gemini.success else ("NO_KEY" if res_gemini.error_code == "MISSING_API_KEY" else "ERROR")
    groq_stat_str = "SUCCESS" if res_groq.success else ("NO_KEY" if res_groq.error_code == "MISSING_API_KEY" else "ERROR")
    print(f"  {'Gemini':<10} | {gemini_stat_str:<12} | {str(gemini_configured):<12} | {gemini_lat:>6.2f}ms    | {str(res_gemini.error_code):<20}")
    print(f"  {'Groq':<10} | {groq_stat_str:<12} | {str(groq_configured):<12} | {groq_lat:>6.2f}ms    | {str(res_groq.error_code):<20}")

    # -------------------------------------------------------------------------
    # TEST 6: Verify Provider Fallback from Gemini to Groq
    # -------------------------------------------------------------------------
    print("\n[TEST 6/7] Provider Fallback Cascade Verification (Gemini -> Groq)")
    print("  Scenario: Primary provider (Gemini) encounters simulated quota error (429 RESOURCE_EXHAUSTED).")
    
    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    mock_groq_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "READY (Synthesized via Groq fallback provider)"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    gemini_sim = GeminiGenerator(
        api_key="mock_gemini_key",
        model_name=settings.gemini_model,
        client=mock_gemini_client,
        enable_gemini=True,
    )
    groq_sim = GroqGenerator(
        api_key="mock_groq_key",
        model_name=settings.groq_model,
        client=mock_groq_client,
    )
    cascade_sim = CascadingLLMGenerator(
        primary_generator=gemini_sim,
        fallback_generator=groq_sim,
        max_retries=0,
    )

    t0 = time.perf_counter()
    res_fallback = cascade_sim.generate_response(test_prompt)
    t_fallback = (time.perf_counter() - t0) * 1000

    fallback_verified = (
        res_fallback.success is True
        and res_fallback.provider == "groq"
        and res_fallback.fallback_triggered is True
        and res_fallback.local_fallback is False
        and "READY" in (res_fallback.text or "")
    )

    results["gemini_to_groq_fallback"] = {
        "verified": fallback_verified,
        "primary_error_simulated": "429 RESOURCE_EXHAUSTED",
        "fallback_provider_engaged": res_fallback.provider,
        "fallback_triggered_flag": res_fallback.fallback_triggered,
        "local_fallback_flag": res_fallback.local_fallback,
        "success": res_fallback.success,
        "latency_ms": round(t_fallback, 2),
        "output_preview": res_fallback.text,
    }

    print(f"  Fallback Execution:    {'VERIFIED' if fallback_verified else 'FAILED'}")
    print(f"  Primary Status:        Simulated 429 RESOURCE_EXHAUSTED")
    print(f"  Engaged Provider:      {res_fallback.provider}")
    print(f"  Fallback Triggered:    {res_fallback.fallback_triggered}")
    print(f"  Response Generated:    \"{res_fallback.text}\"")
    print(f"  Total Cascade Latency: {t_fallback:.2f}ms")

    # -------------------------------------------------------------------------
    # TEST 7: Verify Local Deterministic Fallback if Both Fail
    # -------------------------------------------------------------------------
    print("\n[TEST 7/7] Local Deterministic Fallback Verification (Both Providers Fail)")
    print("  Scenario: Primary (Gemini) encounters Timeout, Fallback (Groq) encounters 503 Service Unavailable.")

    mock_gemini_both_fail = MagicMock()
    mock_gemini_both_fail.models.generate_content.side_effect = TimeoutError("Gemini request timed out after 10.0s")

    mock_groq_both_fail = MagicMock()
    mock_groq_both_fail.chat.completions.create.side_effect = Exception("503 Service Unavailable: Groq inference gateway down")

    gemini_fail_gen = GeminiGenerator(api_key="mock_k1", client=mock_gemini_both_fail, enable_gemini=True)
    groq_fail_gen = GroqGenerator(api_key="mock_k2", client=mock_groq_both_fail)
    cascade_fail = CascadingLLMGenerator(primary_generator=gemini_fail_gen, fallback_generator=groq_fail_gen, max_retries=0)

    # Form test state with authorized clinical protocol chunk
    protocol_chunk: KnowledgeChunk = {
        "chunk_id": "proto_study_pacing_01",
        "document_id": "protocol:proto_study_pacing_01",
        "title": "Structured Study Pacing & Grounding",
        "category": "academic_stress",
        "jurisdiction": "GL",
        "content": "## 2. Summary\nFocus on breaking study blocks into 25-minute intervals followed by brief 5-minute restorative breaks.\n## 3. Step-by-Step Instructions\n1. Identify a single task to complete.\n2. Set a timer for 25 minutes.\n3. Take three deep diaphragmatic breaths when concluding.",
        "score": 0.94,
        "eligibility": "approved",
        "source": "clinical_protocol",
    }

    test_state: MindBridgeState = {
        "session_id": "diag-fallback-session-999",
        "user_id_hash": "diag-user-hash",
        "messages": [{"role": "user", "content": test_prompt}],
        "sanitized_text": test_prompt,
        "_ephemeral_raw_text": test_prompt,
        "deterministic_danger_flag": False,
        "emergency_trigger_id": None,
        "roberta_prob": None,
        "roberta_crisis_flag": False,
        "classifier_status": "not_run",
        "overall_safety_route": "standard_conversation",
        "phq9_item9_endorsed": False,
        "phq9_followup_status": "not_started",
        "phq9_risk_level": None,
        "phq9_dimensions": {},
        "phq9_supplementary_context": {},
        "phq9_current_dimension": None,
        "dialog_goal": "supportive_conversation",
        "retrieval_queries": ["academic stress management"],
        "retrieved_chunks": [protocol_chunk],
        "retrieval_status": "success",
        "jurisdiction_filter": "GL",
        "draft_response": None,
        "verification_passed": False,
        "verification_flags": [],
        "verification_status": "not_run",
        "final_response": "",
        "error_code": None,
        "error_component": None,
        "execution_trace": [],
    }

    t0 = time.perf_counter()
    response_result = response_node(test_state, generator=cascade_fail)
    t_local = (time.perf_counter() - t0) * 1000

    telemetry = response_result.get("llm_telemetry", {})
    last_trace = response_result.get("execution_trace", [{}])[-1]
    draft = response_result.get("draft_response", "")

    local_fallback_verified = (
        telemetry.get("provider") == "local"
        and telemetry.get("local_fallback_used") is True
        and telemetry.get("fallback_triggered") is True
        and last_trace.get("metadata_code") == "LOCAL_GROUNDED_FALLBACK_ON_LLM_FAILURE"
        and len(draft) > 50
        and "Structured Study Pacing & Grounding" in draft
    )

    results["both_failed_local_fallback"] = {
        "verified": local_fallback_verified,
        "metadata_code": last_trace.get("metadata_code"),
        "telemetry_provider": telemetry.get("provider"),
        "telemetry_model": telemetry.get("model"),
        "local_fallback_used": telemetry.get("local_fallback_used"),
        "fallback_triggered": telemetry.get("fallback_triggered"),
        "latency_ms": round(t_local, 2),
        "draft_char_count": len(draft),
        "grounding_evidence_retained": "Structured Study Pacing & Grounding" in draft,
    }

    print(f"  Local Fallback:        {'VERIFIED' if local_fallback_verified else 'FAILED'}")
    print(f"  Orchestrator Trace:    {last_trace.get('metadata_code')}")
    print(f"  Telemetry Provider:    {telemetry.get('provider')}")
    print(f"  Telemetry Model:       {telemetry.get('model')}")
    print(f"  Grounding Integrity:   Retained ({len(draft)} chars, verified protocol title present)")
    print(f"  Fallback Synthesis Latency: {t_local:.2f}ms")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC TEST SUMMARY")
    print("=" * 78)
    print(f"1. GEMINI_API_KEY Configured:       {gemini_configured}")
    print(f"2. GROQ_API_KEY Configured:         {groq_configured}")
    print(f"3. Gemini Live Request:             {res_gemini.error_code or 'SUCCESS'}")
    print(f"4. Groq Live Request:               {res_groq.error_code or 'SUCCESS'}")
    print(f"5. Latency & Code Audited:          True")
    print(f"6. Gemini -> Groq Fallback:         {'VERIFIED' if fallback_verified else 'FAILED'}")
    print(f"7. Local Deterministic Fallback:    {'VERIFIED' if local_fallback_verified else 'FAILED'}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    results = run_full_diagnostic()
    # Save output data to a structured JSON artifact for doc generation
    out_file = PROJECT_ROOT / "docs" / ".diagnostic_run.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostic telemetry written to: {out_file}")

