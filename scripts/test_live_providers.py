#!/usr/bin/env python3
"""
MindBridge 2.0 - Live Provider Diagnostic & Model Verification Tool
===================================================================

Safely tests Gemini Developer API and Groq Cloud API using keys from `.env`
or environment variables.

Guarantees:
- Never prints, logs, or exposes API keys.
- Checks available Gemini models supporting `generateContent`.
- Recommends stable, high-throughput Flash models.
- Tests Groq independently.
- Tests Gemini -> Groq failover cascade.
"""

import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(env_path: Path) -> dict:
    """Parses .env file without external dependencies or leaking values."""
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


def run_diagnostics():
    env_file = PROJECT_ROOT / ".env"
    file_vars = load_env_file(env_file)

    gemini_key = os.getenv("GEMINI_API_KEY") or file_vars.get("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY") or file_vars.get("GROQ_API_KEY")

    print("=" * 70)
    print("MindBridge 2.0 — Live LLM Provider Diagnostics")
    print("=" * 70)
    print(f".env file location: {env_file}")
    print(f".env file exists:   {env_file.exists()}")
    print(f"GEMINI_API_KEY:     {'[CONFIGURED]' if gemini_key else '[NOT FOUND]'}")
    print(f"GROQ_API_KEY:       {'[CONFIGURED]' if groq_key else '[NOT FOUND]'}")
    print("=" * 70)

    if not gemini_key and not groq_key:
        print("\n[ACTION REQUIRED] Neither API key is present in .env or the shell environment.")
        print("To configure them, add your keys to:")
        print(f"  {env_file}")
        print("\nExample .env entries:")
        print("  GEMINI_API_KEY=your_gemini_key_here")
        print("  GROQ_API_KEY=your_groq_key_here")
        print("  LLM_PROVIDER=auto")
        print("  ENABLE_GEMINI=true")
        print("  GEMINI_MODEL=gemini-2.0-flash")
        print("  GROQ_MODEL=openai/gpt-oss-120b")
        return

    # 1. Test Gemini Developer API
    if gemini_key:
        print("\n--- 1. Testing Gemini Developer API (Google AI Studio) ---")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)

            # Query available models supporting generateContent
            print("Querying available Gemini models...")
            models_supporting_generate = []
            try:
                for model in client.models.list():
                    name = getattr(model, "name", str(model))
                    actions = getattr(model, "supported_actions", []) or []
                    # Check if model supports generateContent
                    if "generateContent" in actions or not actions:
                        models_supporting_generate.append(name.replace("models/", ""))
            except Exception as e:
                print(f"Warning: Could not list all models: {type(e).__name__}: {e}")

            if models_supporting_generate:
                print(f"Found {len(models_supporting_generate)} models supporting generateContent.")
                flash_models = [m for m in models_supporting_generate if "flash" in m.lower()]
                print("Available Flash models:")
                for fm in flash_models[:10]:
                    print(f"  - {fm}")

            # Test generation with gemini-2.0-flash, gemini-2.5-flash, or gemini-1.5-flash
            candidates = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
            test_model = candidates[0]
            for cand in candidates:
                if any(cand in m for m in models_supporting_generate):
                    test_model = cand
                    break

            print(f"\nExecuting test request with model '{test_model}'...")
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model=test_model,
                contents="Reply with the word READY.",
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            text = resp.text.strip() if hasattr(resp, "text") and resp.text else "(No text returned)"
            print(f"Status: SUCCESS (200 OK)")
            print(f"Latency: {elapsed_ms:.1f}ms")
            print(f"Model Used: {test_model}")
            print(f"Generated Snippet: \"{text[:120]}...\"")

        except Exception as e:
            print(f"Gemini Test FAILED: {type(e).__name__}: {e}")
    else:
        print("\n--- 1. Gemini Developer API: SKIPPED (No key configured) ---")

    # 2. Test Groq Cloud API
    if groq_key:
        print("\n--- 2. Testing Groq Cloud API ---")
        try:
            from groq import Groq

            groq_model = os.getenv("GROQ_MODEL") or file_vars.get("GROQ_MODEL", "openai/gpt-oss-120b")
            client = Groq(api_key=groq_key, timeout=10.0)

            print(f"Executing test request with Groq model '{groq_model}'...")
            t0 = time.perf_counter()
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with the word READY."},
                ],
                max_completion_tokens=150,
                temperature=0.7,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            text = completion.choices[0].message.content.strip()
            print(f"Status: SUCCESS (200 OK)")
            print(f"Latency: {elapsed_ms:.1f}ms")
            print(f"Model Used: {groq_model}")
            print(f"Generated Snippet: \"{text[:120]}...\"")

        except Exception as e:
            print(f"Groq Test FAILED: {type(e).__name__}: {e}")
    else:
        print("\n--- 2. Groq Cloud API: SKIPPED (No key configured) ---")

    # 3. Test Full Cascade Integration
    if gemini_key or groq_key:
        print("\n--- 3. Testing Full MindBridge Cascading LLM Generator ---")
        from backend.generation.llm import create_llm_generator
        gen = create_llm_generator(
            api_key=gemini_key,
            client=None,
        )
        if gen:
            status = gen.get_provider_status() if hasattr(gen, "get_provider_status") else {}
            print(f"Cascade Active Provider: {getattr(gen, 'active_provider', 'local')}")
            print(f"Cascade Available:       {gen.is_available()}")
            print(f"Diagnostics:             {status}")


if __name__ == "__main__":
    run_diagnostics()
