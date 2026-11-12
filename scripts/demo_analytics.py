"""Print the three live, grounded analytical responses used for the H-E-B demo.

Run from the repository root: python scripts/demo_analytics.py
This reads only the synthetic CSVs bundled with the exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.analytics import answer_data_question

QUESTIONS = (
    "¿Qué tienda vendió más en agosto de 2026?",
    "¿Qué productos de la tienda T01 están por debajo del punto de reorden?",
    "¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son?",
)


def main() -> int:
    for number, question in enumerate(QUESTIONS, 1):
        result = answer_data_question(question)
        print(f"\nA{number} · {question}\nEstado: {result.status}\n{result.answer}")
        print("Fuentes:", ", ".join(f"{e['source']} ({e['reference']})" for e in result.evidence) or "Sin evidencia")
        if result.status != "answered":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
