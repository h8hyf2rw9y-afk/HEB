"""Optional local test against a REAL Ollama (llama3.2 + bge-m3). Skipped unless
HEB_LLM_REAL_TESTS=1. Never runs in CI."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("HEB_LLM_REAL_TESTS") != "1",
                                reason="requiere Ollama local con llama3.2 (HEB_LLM_REAL_TESTS=1)")


def test_real_llama_rewrites_a_data_answer_and_passes_validation():
    from app.llm.config import LLMSettings
    from app.llm.service import GroundedWriter

    settings = LLMSettings(enabled=True, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                           model=os.getenv("HEB_LLM_MODEL", "llama3.2"), timeout_s=120)
    writer = GroundedWriter(settings)
    verified = ("Ventas netas por tienda (agosto de 2026), mayor a menor:\n1. Tienda Valle Norte: $2,163,062.36 MXN\n"
                "2. Tienda Cumbres: $1,973,086.37 MXN\n3. Tienda Anáhuac: $949,469.00 MXN\n4. Tienda Lomas del Sur: $833,131.19 MXN")
    out = writer.rewrite(question="¿Qué tienda vendió más en agosto de 2026?", kind="datos", status="answered",
                         verified_answer=verified, evidence=[{"source": "ventas.csv"}, {"source": "tiendas.csv"}])
    print("\nRESULTADO REAL llama3.2:", {
        "generated": out.generated, "fallback_reason": out.fallback_reason, "model": out.model,
        "llm_calls": out.llm_calls, "input_tokens": out.input_tokens, "output_tokens": out.output_tokens,
        "generation_latency_ms": out.generation_latency_ms, "warnings": out.warnings,
    })
    print("RESPUESTA:", out.answer)
    # Ollama must have been reached and reported real counters.
    assert out.fallback_reason not in ("ollama_no_disponible", "timeout"), out.warnings
    assert out.llm_calls == 1 and out.input_tokens > 0 and out.output_tokens > 0
    # Whatever the model wrote, the user-facing answer keeps the verified figures.
    for figure in ("$2,163,062.36 MXN", "$1,973,086.37 MXN", "$949,469.00 MXN", "$833,131.19 MXN"):
        assert figure in out.answer
    if not out.generated:
        pytest.fail(f"llama3.2 respondió pero la redacción no pasó la validación ({out.fallback_reason}): {out.warnings}")
