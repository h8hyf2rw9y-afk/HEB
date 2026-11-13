"""Compare, on the exercise questions and calling /chat in-process:
  Mode A: deterministic answers (no model).
  Mode B: deterministic router + the local LLM rewrites the verified answer.
  Mode C: agent; the LLM chooses the verified tool (DuckDB / RAG) and writes the answer.

    python scripts/eval_llm_modes.py                  # A + B + C (B and C need Ollama + llama3.2)
    python scripts/eval_llm_modes.py --mode A         # Mode A only (no Ollama needed)
    python scripts/eval_llm_modes.py --mode AC --out data/eval_llm_modes.json

Mode B measurements are REAL only when Ollama answers; if it does not, every row
shows a fallback and the report says so. Nothing here simulates a model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# question, kind, expected status, phrases that must be present (verified facts)
CASES = [
    ("¿Qué tienda vendió más en agosto de 2026?", "datos", "answered", ["Tienda Valle Norte", "$2,163,062.36 MXN", "agosto de 2026"]),
    ("¿Qué productos de la tienda T01 están por debajo del punto de reorden?", "datos", "answered", ["SKU-1013", "SKU-1017", "SKU-1036", "1 de septiembre de 2026"]),
    ("¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son?", "datos", "answered", ["0"]),
    ("¿Cuánto vendimos en julio de 2026?", "datos", "answered", ["$5,712,163.10 MXN"]),
    ("¿Cuánto vamos a vender en diciembre?", "datos", "insufficient_data", []),
    ("¿Cuántos días tiene un cliente para devolver una licuadora?", "politicas", "answered", ["30 días naturales", "15 días", "Prevalece procedimiento_devoluciones.pdf", "[procedimiento_devoluciones.pdf, p. 2"]),
    ("¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?", "politicas", "answered", ["vale de tienda", "[procedimiento_devoluciones.pdf, p. 4, sección: 6.1 Cliente sin la tarjeta original]"]),
    ("¿Cuál es la tolerancia de merma en panadería?", "politicas", "answered", ["3.5", "4.5", "[politica_mermas_y_caducidad.pdf, p. 3"]),
    ("¿A qué hora debe llegar el gerente para la apertura de la tienda?", "politicas", "answered", ["5:30", "[faq_gerentes_de_tienda.pdf, p. 1"]),
    ("¿Qué es SIGMA?", "politicas", "answered", ["Sistema Integral de Gestión de Mermas y Abasto"]),
    ("¿Cuál es la tolerancia de merma para electrónica?", "politicas", "insufficient_data", []),
    ("¿Cuál es el salario de un cajero?", "politicas", "insufficient_data", []),
]


def run_mode(client, enabled: bool, mode: str = "redaccion") -> list[dict]:
    from app.api import main
    from app.llm import GroundedWriter, LLMSettings, ToolAgent

    base = LLMSettings.from_env()
    settings = LLMSettings(enabled=enabled, base_url=base.base_url, model=base.model, timeout_s=base.timeout_s, mode=mode)
    main.WRITER = GroundedWriter(settings)
    main.AGENT = ToolAgent(settings, main.AGENT_TOOLS)
    rows = []
    for question, kind, expected, phrases in CASES:
        t0 = time.perf_counter()
        data = client.post("/chat", json={"question": question}).json()
        wall = round((time.perf_counter() - t0) * 1000, 1)
        m = data["metrics"]
        missing = [p for p in phrases if p not in data["answer"]]
        rows.append({
            "question": question, "kind": kind, "expected_status": expected, "status": data["status"],
            "status_ok": data["status"] == expected, "facts_ok": not missing, "missing": missing,
            "generated": m.get("generated", False), "fallback_reason": m.get("fallback_reason"),
            "llm_model": m.get("llm_model"), "llm_calls": m.get("llm_calls", 0),
            "input_tokens": m.get("input_tokens", 0), "output_tokens": m.get("output_tokens", 0),
            "generation_latency_ms": m.get("generation_latency_ms", 0), "latency_ms": m.get("latency_ms", 0),
            "wall_ms": wall, "embedding_calls": m.get("embedding_calls", 0),
            "embedding_input_tokens": m.get("embedding_input_tokens", 0),
            "retrieval_mode": m.get("retrieval_mode"), "warnings": m.get("warnings", []),
            "tool_calls": m.get("tool_calls", []),
            "words": len(data["answer"].split()), "answer": data["answer"],
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["expected_status"] == "answered"]
    gen = [r for r in rows if r["generated"]]
    lat = [r["latency_ms"] for r in rows]
    return {
        "cases": len(rows),
        "status_accuracy": round(sum(r["status_ok"] for r in rows) / len(rows), 3),
        "facts_preserved": round(sum(r["facts_ok"] for r in answered) / len(answered), 3) if answered else None,
        "abstentions_ok": all(r["status_ok"] for r in rows if r["expected_status"] == "insufficient_data"),
        "generated_answers": len(gen),
        "fallbacks": {r["fallback_reason"] or "ninguno": sum(1 for x in rows if x["fallback_reason"] == r["fallback_reason"]) for r in rows},
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "input_tokens_total": sum(r["input_tokens"] for r in rows),
        "output_tokens_total": sum(r["output_tokens"] for r in rows),
        "tokens_per_generated_answer_median": statistics.median([r["input_tokens"] + r["output_tokens"] for r in gen]) if gen else None,
        "generation_latency_ms_median": statistics.median([r["generation_latency_ms"] for r in gen]) if gen else None,
        "latency_ms_median": round(statistics.median(lat), 1), "latency_ms_max": round(max(lat), 1),
        "words_median_answered": statistics.median([r["words"] for r in answered]) if answered else None,
        "embedding_input_tokens_total": sum(r["embedding_input_tokens"] for r in rows),
        "questions_where_model_called_a_tool": sum(1 for r in rows if r.get("tool_calls")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ABC", help="cualquier combinación de A, B, C")
    parser.add_argument("--out", help="archivo JSON para guardar el reporte completo")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    os.chdir(ROOT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from app.api import main as api

    client = TestClient(api.app)
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "ollama_base_url": api.LLM_SETTINGS.base_url, "llm_model": api.LLM_SETTINGS.model, "modes": {}}
    if "A" in args.mode.upper():
        rows = run_mode(client, enabled=False)
        report["modes"]["A_determinista"] = {"summary": summarize(rows), "rows": rows}
    for letter, name, mode in (("B", "B_llm", "redaccion"), ("C", "C_agente", "agente")):
        if letter not in args.mode.upper():
            continue
        rows = run_mode(client, enabled=True, mode=mode)
        s = summarize(rows)
        s["real_model_used"] = s["generated_answers"] > 0 or any(r["llm_calls"] for r in rows)
        report["modes"][name] = {"summary": s, "rows": rows}
    for name, mode in report["modes"].items():
        print(f"\n== Modo {name}")
        for k, v in mode["summary"].items():
            print(f"  {k}: {v}")
        for r in mode["rows"]:
            flag = "OK " if r["status_ok"] and r["facts_ok"] else "REV"
            who = r["llm_model"] if r["generated"] else f"determinista ({r['fallback_reason']})"
            print(f"  [{flag}] {r['status']:17} {r['latency_ms']:8.1f} ms  tok {r['input_tokens']}+{r['output_tokens']:<4} {who:40} {r['question']}")
    for name in ("B_llm", "C_agente"):
        if name in report["modes"] and not report["modes"][name]["summary"]["real_model_used"]:
            print(f"\nAVISO: Ollama no respondió; el modo {name} no contiene mediciones de un modelo real (todo fue fallback).")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # data/ may not exist yet
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReporte guardado en {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
