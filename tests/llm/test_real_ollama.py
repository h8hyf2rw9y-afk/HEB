"""Optional local test against a REAL Ollama (llama3.2 + bge-m3). Skipped unless
HEB_LLM_REAL_TESTS=1. Never runs in CI."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("HEB_LLM_REAL_TESTS") != "1",
                                reason="requiere Ollama local con el modelo configurado (HEB_LLM_REAL_TESTS=1)")


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


def test_real_llama_agent_chooses_a_tool_for_a_data_question():
    """Agent mode against the real model: it must call a verified tool, and the shown
    answer must keep the DuckDB figures whether or not its prose passes validation."""
    from app.api import main
    from app.llm import LLMSettings, ToolAgent

    settings = LLMSettings(enabled=True, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                           model=os.getenv("HEB_LLM_MODEL", "llama3.2"), timeout_s=120, mode="agente")
    out = ToolAgent(settings, main.AGENT_TOOLS).run("¿Qué tienda vendió más en agosto de 2026?")
    print("\nRESULTADO REAL agente:", {
        "handled": out.handled, "generated": out.generated, "fallback_reason": out.fallback_reason,
        "tool_calls": out.tool_calls, "llm_calls": out.llm_calls, "input_tokens": out.input_tokens,
        "output_tokens": out.output_tokens, "generation_latency_ms": out.generation_latency_ms,
        "warnings": out.warnings,
    })
    print("RESPUESTA:", out.answer)
    assert out.fallback_reason not in ("ollama_no_disponible", "timeout"), out.warnings
    assert out.llm_calls >= 1 and out.input_tokens > 0
    if not out.tool_calls:
        pytest.fail(f"{settings.model} no llamó a ninguna herramienta ({out.fallback_reason})")
    assert "$2,163,062.36 MXN" in out.answer


SEVEN_REAL = [
    ("¿Qué tienda vendió más en agosto de 2026?", ["Valle Norte", "$2,163,062.36 MXN"]),
    ("¿Y cómo le fue en julio?", ["$2,153,571.23 MXN"]),
    ("¿Por qué crees que vendió más que las otras?", ["Valle Norte", "no demuestran sus causas"]),
    ("¿Cuántos productos tiene cada tienda?", ["120 SKU"]),
    ("¿Qué es SIGMA?", ["Sistema Integral de Gestión de Mermas y Abasto"]),
    ("¿Cuál es el procedimiento para devolver una licuadora?", ["30 días naturales", "procedimiento_devoluciones.pdf"]),
    ("¿Y si el cliente no tiene ticket?", ["vale de tienda", "identificación oficial"]),
]


def test_real_model_seven_questions_one_conversation():
    """The seven questions from the real chat, with the model configured by the
    environment (HEB_LLM_MODEL, e.g. qwen3:4b-instruct). Prints per-turn calls,
    tokens and stage timings; fails on a missing verified figure or source.
    Simulated tests cannot prove this: run it on the laptop."""
    from uuid import uuid4

    from fastapi.testclient import TestClient

    from app.api import main
    from app.llm import GroundedWriter, LLMSettings, ToolAgent

    base = LLMSettings.from_env()
    settings = LLMSettings(enabled=True, base_url=base.base_url, model=base.model,
                           timeout_s=max(base.timeout_s, 180), mode="agente", keep_alive=base.keep_alive)
    main.AGENT = ToolAgent(settings, main.AGENT_TOOLS)
    main.WRITER = GroundedWriter(settings)
    client, sid, problems = TestClient(main.app), uuid4().hex, []
    for question, must in SEVEN_REAL:
        data = client.post("/chat", json={"question": question, "conversation_id": sid}).json()
        m, t = data["metrics"], data["metrics"].get("timings", {})
        print(f"\n== {question}\n   estado={data['status']} generada={m['generated']} fallback={m['fallback_reason']} "
              f"llamadas={m['llm_calls']} tokens={m['input_tokens']}+{m['output_tokens']} total={m['latency_ms'] / 1000:.1f}s "
              f"prompt={t.get('llm_prompt_eval_ms', 0) / 1000:.1f}s generación={t.get('llm_generation_ms', 0) / 1000:.1f}s "
              f"tokens/s={t.get('tokens_per_s')}\n   resuelta={m.get('resolved_question')}\n   {data['answer'][:400]}")
        if m["warnings"]:
            print("   avisos:", " | ".join(m["warnings"]))
        missing = [text for text in must if text not in data["answer"]]
        if data["status"] != "answered" or missing:
            problems.append((question, data["status"], missing))
    assert not problems, problems
