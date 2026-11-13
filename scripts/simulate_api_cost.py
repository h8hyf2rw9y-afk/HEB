"""Simulate what each interaction would cost on the Anthropic (Claude) API, using the
token counts that Ollama actually reported while running the project locally.

    python scripts/simulate_api_cost.py data/eval_llm_modes.json
    python scripts/simulate_api_cost.py data/eval_agent_conversations.json --usd-mxn 18.5
    python scripts/simulate_api_cost.py --manual 1000 200          # one interaction: 1000 in, 200 out
    python scripts/simulate_api_cost.py data/eval_llm_modes.json --out data/simulacion_costos.json

The real cost of this project is $0 per token: the model runs locally in Ollama.
This is a SIMULATION. Read the caveats it prints:
- Token counts come from the local model's tokenizer (Llama/Qwen), not Claude's;
  the same text tokenizes differently. For exact figures, use the Anthropic
  token counting endpoint on the real prompts.
- Claude models also bill thinking tokens as output, and Claude's tool definitions
  and system prompt formatting add input tokens. Real cost can be higher.
- Prices are list prices for the first-party API, without prompt caching, batch
  discounts or taxes. Check current prices before quoting them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PRICES_AS_OF = "2026-11-03"
# USD per 1M tokens (input, output). First-party Anthropic API list prices.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-2": (8.00, 40.00),
}

CAVEATS = [
    f"Simulación con precios de lista de la API de Anthropic al {PRICES_AS_OF}, sin caché, lotes ni impuestos.",
    "El costo real de este proyecto es $0 por token: el modelo corre localmente en Ollama.",
    "Los tokens los contó el tokenizador del modelo local (Llama/Qwen), no el de Claude: el mismo texto da otra cantidad.",
    "Claude cobra el razonamiento interno (thinking) como salida y sus definiciones de herramientas suman entrada: el costo real puede ser mayor.",
]


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    price_in, price_out = PRICES_USD_PER_MTOK[model]
    return input_tokens * price_in / 1_000_000 + output_tokens * price_out / 1_000_000


def load_interactions(path: Path) -> list[dict]:
    """Rows with question, input/output tokens and LLM calls, from either eval report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    if "modes" in data:  # scripts/eval_llm_modes.py
        for mode, block in data["modes"].items():
            for r in block.get("rows", []):
                rows.append({"source": f"{path.name}:{mode}", "question": r.get("question", ""),
                             "input_tokens": int(r.get("input_tokens") or 0),
                             "output_tokens": int(r.get("output_tokens") or 0),
                             "llm_calls": int(r.get("llm_calls") or 0)})
    elif "episodes" in data:  # scripts/eval_agent_conversations.py
        for ep in data["episodes"]:
            for t in ep.get("turns", []):
                if "request_error" in t:
                    continue
                if "input_tokens" not in t and t.get("tokens"):
                    raise SystemExit(
                        f"{path}: este reporte solo trae el total de tokens por turno. "
                        "Vuelve a correr scripts/eval_agent_conversations.py para obtener entrada y salida por separado.")
                rows.append({"source": f"{path.name}:{ep.get('name')}", "question": t.get("question", ""),
                             "input_tokens": int(t.get("input_tokens") or 0),
                             "output_tokens": int(t.get("output_tokens") or 0),
                             "llm_calls": int(t.get("llm_calls") or 0)})
    else:
        raise SystemExit(f"{path}: formato no reconocido (se esperaba un reporte de eval_llm_modes o eval_agent_conversations)")
    return rows


def simulate(rows: list[dict], models: list[str]) -> dict:
    priced = []
    for r in rows:
        priced.append({**r, "cost_usd": {m: round(cost_usd(r["input_tokens"], r["output_tokens"], m), 6) for m in models}})
    with_llm = [r for r in priced if r["input_tokens"] or r["output_tokens"]]
    summary: dict = {"interactions": len(priced), "interactions_with_llm_tokens": len(with_llm)}
    if with_llm:
        ins = [r["input_tokens"] for r in with_llm]
        outs = [r["output_tokens"] for r in with_llm]
        summary.update({
            "input_tokens_median": statistics.median(ins), "output_tokens_median": statistics.median(outs),
            "input_tokens_total": sum(ins), "output_tokens_total": sum(outs),
            "llm_calls_total": sum(r["llm_calls"] for r in with_llm),
        })
        summary["per_model"] = {}
        for m in models:
            costs = [r["cost_usd"][m] for r in with_llm]
            summary["per_model"][m] = {
                "median_usd_per_interaction": round(statistics.median(costs), 6),
                "mean_usd_per_interaction": round(statistics.mean(costs), 6),
                "max_usd_per_interaction": round(max(costs), 6),
                "usd_per_1000_interactions_at_mean": round(statistics.mean(costs) * 1000, 2),
            }
    return {"prices_as_of": PRICES_AS_OF, "prices_usd_per_mtok": {m: PRICES_USD_PER_MTOK[m] for m in models},
            "caveats": CAVEATS, "summary": summary, "interactions": priced}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="*", type=Path, help="JSON de eval_llm_modes.py o eval_agent_conversations.py")
    parser.add_argument("--manual", nargs=2, type=int, metavar=("ENTRADA", "SALIDA"), help="tokens de una interacción")
    parser.add_argument("--models", default=",".join(PRICES_USD_PER_MTOK), help="modelos a simular, separados por coma")
    parser.add_argument("--usd-mxn", type=float, help="tipo de cambio para mostrar también pesos (tú lo proporcionas)")
    parser.add_argument("--out", type=Path, help="guardar el resultado completo en JSON")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in PRICES_USD_PER_MTOK]
    if unknown:
        parser.error(f"modelos sin precio: {unknown}")
    rows: list[dict] = []
    if args.manual:
        rows.append({"source": "manual", "question": "(manual)", "input_tokens": args.manual[0],
                     "output_tokens": args.manual[1], "llm_calls": 1})
    for path in args.reports:
        rows.extend(load_interactions(path))
    if not rows:
        parser.error("indica al menos un reporte JSON o --manual ENTRADA SALIDA")
    result = simulate(rows, models)

    print("SIMULACIÓN de costo con la API de Anthropic (no es un costo real del proyecto)")
    for line in CAVEATS:
        print("  -", line)
    s = result["summary"]
    print(f"\nInteracciones: {s['interactions']} · con tokens del LLM: {s['interactions_with_llm_tokens']}")
    if "per_model" not in s:
        print("Ninguna interacción usó el LLM (todas deterministas): costo simulado $0.")
        return 0
    print(f"Tokens por interacción (mediana): {s['input_tokens_median']} de entrada + {s['output_tokens_median']} de salida")
    print(f"\n{'Modelo':20} {'USD/interacción (mediana)':>26} {'USD por 1,000 interacciones':>28}"
          + (f" {'MXN por 1,000':>14}" if args.usd_mxn else ""))
    for m, v in s["per_model"].items():
        line = f"{m:20} {v['median_usd_per_interaction']:>26.6f} {v['usd_per_1000_interactions_at_mean']:>28.2f}"
        if args.usd_mxn:
            line += f" {v['usd_per_1000_interactions_at_mean'] * args.usd_mxn:>14.2f}"
        print(line)
    print("\nDetalle por pregunta (USD):")
    for r in result["interactions"]:
        if r["input_tokens"] or r["output_tokens"]:
            costs = "  ".join(f"{m.replace('claude-', '')}={r['cost_usd'][m]:.5f}" for m in models)
            print(f"  {r['input_tokens']:>6}+{r['output_tokens']:<5} {costs}  {r['question'][:70]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.usd_mxn:
            result["usd_mxn_rate_provided_by_user"] = args.usd_mxn
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultado guardado en {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
