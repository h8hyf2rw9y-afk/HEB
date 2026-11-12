"""Documentation checks for the delivery docs: links, evidence, and that quoted
outputs in the demo script still match what the system returns."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
PDF_DIR = ROOT / "candidatos" / "politicas"
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _rows(markdown: str) -> list[list[str]]:
    rows = []
    for line in markdown.splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*-", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    broken = []
    for target in LINK_RE.findall(doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (doc.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert broken == []


def test_compliance_done_rows_cite_existing_evidence():
    text = (ROOT / "docs" / "COMPLIANCE.md").read_text(encoding="utf-8")
    problems = []
    for row in _rows(text):
        status = row[-1]
        if not status.startswith("✅"):
            continue
        cited = re.findall(r"`([\w./-]+\.(?:py|md|json|csv|pdf))(?:::[\w\[\]-]+)?`", " ".join(row))
        cited += re.findall(r"`(test_\w+)`", " ".join(row))  # bare test names
        if not cited and "h8hyf2rw9y-afk/HEB#" not in " ".join(row):
            problems.append(("sin evidencia", row[1][:60]))
        for ref in cited:
            if ref.startswith("test_"):
                found = any(ref in p.read_text(encoding="utf-8") for p in (ROOT / "tests").rglob("test_*.py"))
            else:
                found = (ROOT / ref).exists()  # repo-relative paths only
            if not found:
                problems.append(("evidencia inexistente", ref))
    assert problems == []


def test_analytics_are_not_marked_done_while_catalog_is_empty():
    """Guard against marking DuckDB features as done before Codex delivers them."""
    from app.data.catalog import build_catalog

    catalog_empty = not build_catalog(ROOT / "data" / "raw") and not build_catalog(ROOT / "candidatos" / "datos")
    if not catalog_empty:
        pytest.skip("El catálogo ya tiene datasets: revisar la matriz a mano")
    text = (ROOT / "docs" / "COMPLIANCE.md").read_text(encoding="utf-8")
    analytic = re.compile(r"ventas\.csv|inventario\.csv|tickets_mesa|cifras salen de los datos", re.IGNORECASE)
    wrongly_done = [r[1][:60] for r in _rows(text) if analytic.search(" ".join(r)) and r[-1].startswith("✅")]
    assert wrongly_done == []


def test_readme_commands_reference_existing_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for ref in ("app/rag/requirements.txt", "frontend/requirements.txt", "candidatos/politicas",
                "frontend/app.py", "scripts/verify_delivery.py"):
        assert ref in readme.replace("\\", "/"), ref
        assert (ROOT / ref).exists(), ref
    assert importlib.util.find_spec("app.api.main") is not None
    assert importlib.util.find_spec("app.rag.cli") is not None


def test_verify_script_questions_are_the_demo_questions():
    spec = importlib.util.spec_from_file_location("verify_delivery", ROOT / "scripts" / "verify_delivery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = (ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    for question, _status, _doc in module.POLICY_CHECKS:
        if "apertura" in question:  # in README/TOKEN_COSTS, not in the video
            continue
        assert question in demo, question


# --------------------------------------------------- demo outputs still true
DEMO_EXPECTATIONS = [
    ("¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?", "answered",
     ["el reembolso se realiza en vale de tienda",
      "ni reembolsos a una tarjeta distinta de la original. [procedimiento_devoluciones.pdf, p. 4, sección: 6.1 Cliente sin la tarjeta original]"]),
    ("¿Cuál es la tolerancia de merma en panadería?", "answered",
     ["Categoría: Panadería | Tolerancia mensual (% de venta): 3.5 | Umbral de alerta (%): 4.5 | Frecuencia de conteo: Diaria. "
      "[politica_mermas_y_caducidad.pdf, p. 3, sección: 4. Tabla de tolerancias de merma por categoría]"]),
    ("¿Cuál es la tolerancia de merma para electrónica?", "insufficient_data",
     ["no incluye «electrónica»; incluye: Lácteos, Panadería, Carnes, Abarrotes, Bebidas, Limpieza, Frutas y verduras, Cuidado personal."]),
    ("¿Cuántos días tiene un cliente para devolver una licuadora?", "answered",
     ["Plazo para devolución o cambio: 30 días naturales", "el cliente tiene 15 días con ticket",
      "Prevalece procedimiento_devoluciones.pdf"]),
    ("¿Cuál es el plazo para devolver abarrotes?", "answered",
     ["Tipo de producto: Abarrotes y no perecederos | Plazo para devolución o cambio: 30 días"]),
]


@pytest.fixture(scope="module")
def retriever():
    from app.rag.config import RagSettings
    from app.rag.extraction import discover_pdfs
    from app.rag.index import PolicyIndex
    from app.rag.service import PolicyRetriever

    pdfs = discover_pdfs(PDF_DIR)
    if len(pdfs) != 4:
        pytest.skip("candidatos/politicas no disponible")
    return PolicyRetriever(PolicyIndex.build(pdfs, RagSettings(embeddings="off")), RagSettings(embeddings="off"))


@pytest.mark.parametrize("question,status,phrases", DEMO_EXPECTATIONS, ids=lambda v: v[:30] if isinstance(v, str) else None)
def test_demo_script_outputs_match_the_system(retriever, question, status, phrases):
    demo = (ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    assert question in demo
    ans = retriever.answer(question)
    assert ans.status == status
    for phrase in phrases:
        assert phrase in ans.answer, phrase


def test_demo_script_does_not_contain_invented_analytics():
    demo = (ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    section = demo.split("### A1–A3", 1)[1].split("## Si algo falla", 1)[0]
    rows = [r for r in _rows(section) if r and r[0].startswith("A")]
    assert rows, "tabla analítica no encontrada"
    from app.data.catalog import build_catalog

    if not build_catalog(ROOT / "data" / "raw"):
        assert all("[PENDIENTE" in r[3] for r in rows), "resultados analíticos sin implementación en main"
