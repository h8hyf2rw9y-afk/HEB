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


CSV_DIR = ROOT / "candidatos" / "datos"


def test_analytics_are_not_marked_done_while_catalog_is_empty():
    """Guard: data features can only be marked done if the API's catalog is populated
    (the API registers the CSVs from candidatos/datos)."""
    from app.data.catalog import build_catalog

    if build_catalog(CSV_DIR):
        assert set(build_catalog(CSV_DIR)) == {"sales", "inventory", "tickets"}
        return
    text = (ROOT / "docs" / "COMPLIANCE.md").read_text(encoding="utf-8")
    analytic = re.compile(r"ventas\.csv|inventario\.csv|tickets_mesa|cifras salen de los datos", re.IGNORECASE)
    wrongly_done = [r[1][:60] for r in _rows(text) if analytic.search(" ".join(r)) and r[-1].startswith("✅")]
    assert wrongly_done == []


def test_api_uses_the_exercise_csvs():
    pytest.importorskip("fastapi")
    from app.api import main

    assert {spec.csv_path.parent.resolve() for spec in main.CATALOG.values()} == {CSV_DIR.resolve()}


def test_readme_commands_reference_existing_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in readme
    assert (ROOT / "requirements.txt").read_text(encoding="utf-8").strip(), "requirements.txt vacío"
    for ref in ("requirements.txt", "candidatos/politicas", "frontend/app.py",
                "scripts/verify_delivery.py", "scripts/demo_analytics.py"):
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
    for question in (module.A1, module.A2, module.A3, module.A4):
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


def _verify_module():
    spec = importlib.util.spec_from_file_location("verify_delivery", ROOT / "scripts" / "verify_delivery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_block_after(demo: str, heading: str) -> list[str]:
    part = demo.split(heading, 1)[1]
    block = part.split("```", 2)[1]
    return [line for line in block.splitlines() if line.strip()]


ANALYTIC_HEADINGS = {"A1": "### A1", "A2": "### A2", "A3": "### A3", "A4": "### A4"}


def test_demo_script_analytics_match_system_and_independent_csv_math():
    pytest.importorskip("duckdb")
    from app.tools.analytics import answer_data_question

    module = _verify_module()
    demo = (ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    independent = module.independent_expectations()
    for key, question in (("A1", module.A1), ("A2", module.A2), ("A3", module.A3), ("A4", module.A4)):
        result = answer_data_question(question)
        documented = _code_block_after(demo, ANALYTIC_HEADINGS[key])
        live = [line for line in result.answer.splitlines() if line.strip()]
        assert documented == live, f"{key}: el guion no coincide con la salida actual"
        for fragment in independent.get(question, []):
            assert fragment in result.answer, f"{key}: {fragment}"
    assert "[PENDIENTE" not in demo


def test_demo_script_reflects_issue_12_state(retriever):
    """If #12 is fixed, the script must drop the P17 bullet and the warning."""
    demo = (ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    answer = retriever.answer("¿Cuántos días tiene un cliente para devolver una licuadora?").answer
    p17_live = "sección: P17." in answer
    assert p17_live == ("sección: P17." in demo)
    assert p17_live == ("h8hyf2rw9y-afk/HEB#12, abierto" in demo)


def test_limitations_list_exactly_the_currently_misrouted_policy_questions():
    import json

    from app.tools.analytics import looks_like_data

    cases = []
    for name in ("candidatos_eval.json", "holdout_eval.json"):
        cases += json.loads((ROOT / "tests" / "rag" / "fixtures" / name).read_text(encoding="utf-8"))
    policy = [c["question"] for c in cases if c["expect"] == "answered"]
    misrouted = {q for q in policy if looks_like_data(q)}
    section = (ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8").split("## 2.", 1)[1].split("## 3.", 1)[0]
    documented = {r[0] for r in _rows(section)[1:]}
    assert documented == misrouted
    assert f"**{len(misrouted)} se desvían" in section
