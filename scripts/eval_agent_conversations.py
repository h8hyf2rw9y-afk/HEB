"""Evaluate actual multi-turn H-E-B conversations against a running local API.

Usage (PowerShell, with API running and HEB_LLM_ENABLED=true):
    python scripts/eval_agent_conversations.py --out data/eval_agent_conversations.json

This is a diagnostic harness, NOT a claim of semantic correctness: manual review
of policy citations and policy precedence remains necessary. No external API.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx


EPISODES = [
    {
        # The seven questions from the real Qwen3 4B chat that failed or were slow.
        # Third element: text that must appear (verified figures / sources).
        "name": "siete_preguntas_reales",
        "turns": [
            ("¿Qué tienda vendió más en agosto de 2026?", "consultar_datos_tienda", ["Valle Norte", "$2,163,062.36 MXN"]),
            ("¿Y cómo le fue en julio?", "consultar_datos_tienda", ["$2,153,571.23 MXN"]),
            ("¿Por qué crees que vendió más que las otras?", "consultar_datos_tienda", ["Valle Norte", "no demuestran sus causas"]),
            ("¿Cuántos productos tiene cada tienda?", "consultar_datos_tienda", ["120 SKU"]),
            ("¿Qué es SIGMA?", "buscar_politicas", ["Sistema Integral de Gestión de Mermas y Abasto"]),
            ("¿Cuál es el procedimiento para devolver una licuadora?", "buscar_politicas", ["30 días naturales", "procedimiento_devoluciones.pdf"]),
            ("¿Y si el cliente no tiene ticket?", "buscar_politicas", ["vale de tienda", "identificación oficial"]),
        ],
    },
    {
        "name": "devolucion_y_ticket",
        "turns": [
            ("¿Qué pasa si un cliente quiere devolver un producto? ¿Cuál es el proceso?", "buscar_politicas"),
            ("¿Y si no tiene ticket?", "buscar_politicas"),
            ("Es una licuadora, la compró hace 20 días. ¿Qué procede?", "buscar_politicas"),
        ],
    },
    {
        "name": "tarjeta_original",
        "turns": [
            ("¿Cómo se reembolsa una compra pagada con tarjeta?", "buscar_politicas"),
            ("¿Y si no trae la tarjeta original?", "buscar_politicas"),
            ("¿El procedimiento indica si se puede dar efectivo?", "buscar_politicas"),
        ],
    },
    {
        "name": "ventas_seguimientos_y_comparacion",
        "turns": [
            ("¿Qué tienda vendió más en agosto de 2026?", "consultar_datos_tienda"),
            ("¿Y cómo le fue en julio?", "consultar_datos_tienda"),
            ("Entonces compara sus productos con los de Cumbres", "consultar_datos_tienda"),
        ],
    },
    {
        "name": "ventas_y_politicas",
        "turns": [
            ("¿Qué tienda vendió más en agosto de 2026?", "consultar_datos_tienda"),
            ("¿Y cuáles fueron las ventas de julio de 2026?", "consultar_datos_tienda"),
            ("¿Cuál es el procedimiento para un cliente que devuelve un producto?", "buscar_politicas"),
        ],
    },
]

BAD_REFUND_SOURCE = "manual_apertura_y_cierre_tienda.pdf"


def percentile(numbers: list[float], p: float) -> float | None:
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    rank = (len(sorted_numbers) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_numbers) - 1)
    return round(sorted_numbers[lower] + (sorted_numbers[upper] - sorted_numbers[lower]) * (rank - lower), 2)


def evaluate(base_url: str, timeout: float, only: str | None = None) -> dict:
    episodes = []
    latency: list[float] = []
    expected_tool_turns = 0
    expected_tool_found = 0
    route_correct = 0
    total_tokens = 0
    fallback_count = 0
    generated_count = 0
    for episode in [e for e in EPISODES if only is None or e["name"] == only]:
        conversation_id = uuid4().hex
        turns = []
        for turn in episode["turns"]:
            question, expected_tool = turn[0], turn[1]
            must_contain = list(turn[2]) if len(turn) > 2 else []
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        base_url.rstrip("/") + "/chat",
                        json={"question": question, "conversation_id": conversation_id},
                    )
                    response.raise_for_status()
                    data = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                turns.append({"question": question, "request_error": str(exc)})
                continue
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            metrics = data.get("metrics") or {}
            tool_calls = metrics.get("tool_calls") or []
            # Tools the MODEL chose; a verified route (backend retrieved first, model
            # only wrote) is recorded separately and is not a model tool choice.
            tools_used = [c.get("name") for c in tool_calls if c.get("planner") != "verified_route"]
            verified_route = any(c.get("planner") == "verified_route" for c in tool_calls)
            expected_tool_turns += 1
            expected_tool_found += int(expected_tool in tools_used)
            # A deterministic route (fast SQL, return-policy route, ...) that answered from
            # the right kind of source is a correct path, not a tool-selection failure.
            sources_now = [e.get("source", "") for e in (data.get("evidence") or [])]
            expected_ext = ".csv" if expected_tool == "consultar_datos_tienda" else ".pdf"
            planner_action = ("model_tool_call" if tools_used else "verified_route" if verified_route else
                              "deterministic_route" if metrics.get("answer_mode", "").startswith("grounded") else "none")
            route_ok = expected_tool in tools_used or (
                planner_action in ("verified_route", "deterministic_route")
                and any(src.endswith(expected_ext) for src in sources_now))
            route_correct += int(route_ok)
            fallback_count += int(bool(metrics.get("fallback_reason")))
            generated_count += int(bool(metrics.get("generated")))
            total_tokens += int(metrics.get("total_tokens") or 0)
            latency.append(float(metrics.get("latency_ms") or elapsed))
            answer = data.get("answer", "")
            evidence = data.get("evidence") or []
            source_names = [e.get("source", "") for e in evidence]
            turns.append({
                "question": question,
                "expected_tool": expected_tool,
                "tools_used": tools_used,
                "tool_selection_matches": expected_tool in tools_used,
                "planner_action": planner_action,
                "route_correct": route_ok,
                "status": data.get("status"),
                "answer": answer,
                "evidence_sources": source_names,
                "conflicts": data.get("conflicts", []),
                "generated": metrics.get("generated", False),
                "fallback_reason": metrics.get("fallback_reason"),
                "resolved_question": metrics.get("resolved_question"),
                "answer_mode": metrics.get("answer_mode"),
                "grounded_source_matches": (
                    any(source.endswith(".csv") for source in source_names)
                    if expected_tool == "consultar_datos_tienda"
                    else any(source.endswith(".pdf") for source in source_names)
                ),
                "expected_text_found": all(text in answer for text in must_contain),
                "missing_expected_text": [text for text in must_contain if text not in answer],
                "llm_calls": metrics.get("llm_calls", 0),
                "timings": {k: v for k, v in (metrics.get("timings") or {}).items() if k != "llm_calls"},
                "tokens": metrics.get("total_tokens", 0),
                "input_tokens": metrics.get("input_tokens", 0),
                "output_tokens": metrics.get("output_tokens", 0),
                "llm_model": metrics.get("llm_model"),
                "latency_ms": metrics.get("latency_ms", elapsed),
                "requires_manual_review": True,
                "suspicious_it_ticket_in_return": (
                    "devolucion" in episode["name"] and
                    (BAD_REFUND_SOURCE in answer or BAD_REFUND_SOURCE in source_names)
                ),
            })
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    base_url.rstrip("/") + "/chat/reset",
                    json={"conversation_id": conversation_id},
                )
        except httpx.HTTPError:
            pass
        episodes.append({"name": episode["name"], "turns": turns})
    return {
        "scope": "real local API; results need human factual review",
        "summary": {
            "turns_with_response": expected_tool_turns,
            "model_selected_expected_tool": expected_tool_found,
            "tool_selection_rate": round(expected_tool_found / expected_tool_turns, 3)
            if expected_tool_turns else None,
            # Counts equivalent deterministic routes as correct (tool_selection_rate does not).
            "route_correct_rate": round(route_correct / expected_tool_turns, 3)
            if expected_tool_turns else None,
            "generated_answers": generated_count,
            "fallback_answers": fallback_count,
            "total_tokens": total_tokens,
            "latency_p50_ms": percentile(latency, 0.5),
            "latency_p95_ms": percentile(latency, 0.95),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="data/eval_agent_conversations.json")
    parser.add_argument("--timeout", type=float, default=540.0)
    parser.add_argument("--episode", choices=[e["name"] for e in EPISODES],
                        help="ejecutar solo un episodio, por ejemplo siete_preguntas_reales")
    args = parser.parse_args()
    result = evaluate(args.base_url, args.timeout, args.episode)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Evaluation saved: {target} (review answers and citations manually)")


if __name__ == "__main__":
    main()
