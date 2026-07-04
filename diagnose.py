"""
Phase 0, Step 0.5 — Connectivity Diagnostic
Verifies LM Studio is reachable and both models (LLM + Embedding) work correctly.
"""
import sys
from openai import OpenAI

LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "google/gemma-4-e4b"
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"

def run_diagnostics():
    print("=" * 55)
    print("  Agentic Profile Matching — Connectivity Diagnostic")
    print("=" * 55)
    print(f"\nTarget:          {LM_STUDIO_URL}")
    print(f"LLM Model:       {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print()

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    passed = 0
    failed = 0

    # ── Test 1: LLM basic completion ──────────────────────────
    print("[Test 1] LLM chat completion...")
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: PONG"}],
            max_tokens=10,
            temperature=0.0
        )
        reply = r.choices[0].message.content.strip()
        print(f"  ✅ PASSED — Response: '{reply}'")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED — {e}")
        failed += 1

    # ── Test 2: LLM JSON schema mode ──────────────────────────
    print("[Test 2] LLM JSON schema mode (json_schema format)...")
    try:
        import json
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"]
        }
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user",   "content": 'Return {"status": "ok"}'}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ping", "schema": schema}
            },
            max_tokens=20,
            temperature=0.0
        )
        data = json.loads(r.choices[0].message.content)
        print(f"  ✅ PASSED — Parsed JSON: {data}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED — {e}")
        failed += 1

    # ── Test 3: Embedding generation ──────────────────────────
    print("[Test 3] Embedding model...")
    try:
        e = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input="Software engineer with 5 years of Python experience."
        )
        dim = len(e.data[0].embedding)
        print(f"  ✅ PASSED — Embedding dimension: {dim}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED — {e}")
        failed += 1

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)

    if failed > 0:
        print("\n⚠️  One or more tests failed.")
        print("   Make sure LM Studio is running and the models are loaded.")
        sys.exit(1)
    else:
        print("\n✅  All checks passed. Ready to proceed with Phase 1.")

if __name__ == "__main__":
    run_diagnostics()
