"""Verificación de entrega del asistente H-E-B (independiente del desarrollo de DuckDB).

Uso (Windows PowerShell, macOS o Linux), desde la raíz del repositorio y con el
entorno virtual del proyecto:

    python scripts/verify_delivery.py            # verificación completa
    python scripts/verify_delivery.py --json     # resumen en JSON

Qué hace (solo lectura sobre el código; escribe únicamente el índice del RAG en
``data/rag_index``, carpeta ignorada por Git):
1. Comprueba versión de Python, dependencias, PDF y CSV del ejercicio.
2. Construye el índice del RAG con los 4 PDF (BM25, sin Ollama).
3. Llama a la API en proceso (FastAPI TestClient): /health, /health/rag y /chat.
4. Preguntas de políticas: deben responder con cita verificable (documento y página).
5. Preguntas analíticas: solo se REPORTA el estado que devuelve hoy la API; no se
   evalúan ni se marcan como cumplidas (las implementa Codex).

Código de salida: 0 si todo lo implementado funciona; 1 si algo implementado falla.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "candidatos" / "politicas"
CSV_DIR = ROOT / "candidatos" / "datos"
INDEX_DIR = ROOT / "data" / "rag_index"

REQUIRED_MODULES = ["fastapi", "uvicorn", "duckdb", "pydantic", "pymupdf", "streamlit", "httpx"]
EXPECTED_PDFS = [
    "faq_gerentes_de_tienda.pdf",
    "manual_apertura_y_cierre_tienda.pdf",
    "politica_mermas_y_caducidad.pdf",
    "procedimiento_devoluciones.pdf",
]
EXPECTED_CSVS = ["ventas.csv", "inventario.csv", "catalogo_productos.csv", "tiendas.csv", "tickets_mesa_servicio.csv"]

# Policy questions used in docs/DEMO_SCRIPT.md: (question, expected status, document that must be cited)
POLICY_CHECKS = [
    ("¿Cuántos días tiene un cliente para devolver una licuadora?", "answered", "procedimiento_devoluciones.pdf"),
    ("¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?", "answered", "procedimiento_devoluciones.pdf"),
    ("¿Cuál es la tolerancia de merma en panadería?", "answered", "politica_mermas_y_caducidad.pdf"),
    ("¿A qué hora debe llegar el gerente para la apertura de la tienda?", "answered", "faq_gerentes_de_tienda.pdf"),
    ("¿Cuál es la tolerancia de merma para electrónica?", "insufficient_data", None),
]
# Analytic questions: reported only (pending Codex / PR #11).
ANALYTIC_PROBES = [
    "¿Qué tienda vendió más en agosto?",
    "¿Qué productos de la tienda T01 están por debajo del punto de reorden?",
    "¿Cuántos tickets críticos siguen abiertos?",
]


def _ok(flag: bool) -> str:
    return "OK " if flag else "FALLA"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="imprime el resumen en JSON")
    args = parser.parse_args(argv)
    try:  # Windows consoles default to cp1252; answers contain accents and symbols.
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    report: dict = {"platform": platform.platform(), "python": platform.python_version(), "checks": []}
    failures = 0

    def check(name: str, passed: bool, detail: str = "", *, counts: bool = True) -> None:
        nonlocal failures
        report["checks"].append({"name": name, "ok": passed, "detail": detail, "counts": counts})
        if counts and not passed:
            failures += 1
        if not args.json:
            label = _ok(passed) if counts else "INFO "
            print(f"[{label}] {name}" + (f" — {detail}" if detail else ""))

    def finish(code: int) -> int:
        report["total_seconds"] = round(time.perf_counter() - t_start, 1)
        report["failures"] = failures
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return code

    t_start = time.perf_counter()
    check("Python >= 3.10", sys.version_info >= (3, 10), platform.python_version())

    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    check("Dependencias instaladas", not missing, "faltan: " + ", ".join(missing) if missing else "")
    if missing:
        if not args.json:
            print("Instala las dependencias (ver README.md, sección Instalación).")
        return finish(1)

    pdfs = [p for p in EXPECTED_PDFS if not (PDF_DIR / p).is_file()]
    check("4 PDF del ejercicio", not pdfs, "faltan: " + ", ".join(pdfs) if pdfs else str(PDF_DIR.relative_to(ROOT)))
    csvs = [c for c in EXPECTED_CSVS if not (CSV_DIR / c).is_file()]
    check("CSV del ejercicio", not csvs, "faltan: " + ", ".join(csvs) if csvs else str(CSV_DIR.relative_to(ROOT)))
    if pdfs:
        return finish(1)

    # Build the RAG index (same command as the README).
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("HEB_RAG_INDEX_DIR", str(INDEX_DIR))
    os.environ.setdefault("HEB_RAG_EMBEDDINGS", "off")
    from app.rag.cli import main as rag_cli

    t0 = time.perf_counter()
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        code = rag_cli(["build", "--pdf-dir", str(PDF_DIR), "--index-dir", os.environ["HEB_RAG_INDEX_DIR"], "--embeddings", "off"])
    build_s = time.perf_counter() - t0
    check("Índice RAG construido", code == 0, f"{build_s:.1f} s")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # starlette.testclient deprecation notice
        from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    health = client.get("/health")
    check("GET /health", health.status_code == 200 and health.json().get("status") == "ok")
    rag = client.get("/health/rag").json()
    check("GET /health/rag", bool(rag.get("ready")) and rag.get("documents") == 4,
          f"documentos={rag.get('documents')} fragmentos={rag.get('chunks')} embeddings={rag.get('embeddings_enabled')}")

    report["policy"] = []
    for question, expected, doc in POLICY_CHECKS:
        data = client.post("/chat", json={"question": question}).json()
        cited = any(e.get("source") == doc for e in data.get("evidence", [])) if doc else True
        passed = data.get("status") == expected and cited and (expected != "answered" or f"[{doc}, p." in data.get("answer", ""))
        report["policy"].append({"question": question, "status": data.get("status"), "ok": passed,
                                 "latency_ms": data.get("metrics", {}).get("latency_ms"),
                                 "llm_calls": data.get("metrics", {}).get("llm_calls"),
                                 "total_tokens": data.get("metrics", {}).get("total_tokens")})
        check(f"Política: {question}", passed, f"estado={data.get('status')}")

    report["analytics"] = []
    for question in ANALYTIC_PROBES:
        data = client.post("/chat", json={"question": question}).json()
        report["analytics"].append({"question": question, "status": data.get("status"), "answer": data.get("answer", "")[:200]})
        check(f"Analítica (solo informativo): {question}", True, f"estado actual={data.get('status')}", counts=False)

    report["index_build_seconds"] = round(build_s, 1)
    if not args.json:
        print(f"\nTiempo total de verificación: {round(time.perf_counter() - t_start, 1)} s · fallos: {failures}")
        print("Las consultas analíticas se reportan sin evaluarlas: dependen del trabajo de Codex (PR #11).")
    return finish(1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
