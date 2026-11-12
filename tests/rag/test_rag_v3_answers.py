"""Phase 3: answer precision, abstention rules and determinism on the exercise PDFs."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from app.rag.answering import analyse_question, prune_table_siblings, row_cells, unknown_terms
from app.rag.config import RagSettings
from app.rag.evaluation import evaluate, load_cases
from app.rag.extraction import discover_pdfs
from app.rag.index import PolicyIndex
from app.rag.service import _DATA_QUESTION_RE, PolicyRetriever
from app.rag.text import normalize

from conftest import FIXTURES, ROOT

PDF_DIR = ROOT / "candidatos" / "politicas"
needs_pdfs = pytest.mark.skipif(len(discover_pdfs(PDF_DIR)) != 4, reason="candidatos/politicas no disponible")


@pytest.fixture(scope="module")
def real() -> PolicyRetriever:
    index = PolicyIndex.build(discover_pdfs(PDF_DIR), RagSettings(embeddings="off"))
    return PolicyRetriever(index, RagSettings(embeddings="off"))


def _bullets(answer: str) -> list[str]:
    return [line for line in answer.splitlines() if line.startswith("- ")]


# ------------------------------------------------------------ pure helpers
def test_focus_extraction():
    assert analyse_question("¿Cuál es el precio de la leche?").focus == {"precio"}
    assert analyse_question("¿Cuánto cuesta el estacionamiento?").focus == {"cuesta"}
    assert analyse_question("¿Puedo usar el fondo fijo?").focus == set()
    assert analyse_question("¿Qué descuento hay a 3 días?").quantities == {"días": {"3"}}


@pytest.mark.parametrize("q,is_data", [
    ("¿Qué tienda vendió más en agosto?", True),
    ("¿Cuánto vendimos en julio?", True),
    ("¿Cuáles son las ventas totales de la tienda 2?", True),
    ("¿Qué producto se vende más?", True),
    ("¿Qué porcentaje de devoluciones sobre venta se reporta?", False),
    ("¿Cuál es el último día de exhibición y venta de un perecedero?", False),
    ("¿Cuántas devoluciones sobre venta neta se reportan?", False),
    ("¿Se puede vender un producto con la fecha vencida?", False),
])
def test_sales_figure_questions_are_recognised(q, is_data):
    assert bool(_DATA_QUESTION_RE.search(normalize(q))) is is_data


def test_row_cells():
    assert row_cells("Tipo: Ropa | Plazo: 30 días.") == [("Tipo", "Ropa"), ("Plazo", "30 días")]


def test_unknown_terms_ignore_inflections_and_clitics():
    q = analyse_question("¿Puedo darle el bitcoin si supero el umbral?")
    known = {"umbral", "supera"}
    prefixes = {t[:4] for t in known} | {"dar"[:4]}
    out = unknown_terms(q, lambda t: t in known, lambda p: p in prefixes)
    assert out == ["bitcoi"]


# ----------------------------------------------------- answers (real PDFs)
@needs_pdfs
def test_card_refund_without_card_is_direct_and_complete(real):
    ans = real.answer("¿Cómo se reembolsa si el cliente pagó con tarjeta y no presenta la tarjeta original?")
    bullets = _bullets(ans.answer)
    assert len(bullets) == 1
    assert "el reembolso se realiza en vale de tienda" in bullets[0]
    assert "No se realizan reembolsos en efectivo" in bullets[0]
    assert "ni reembolsos a una tarjeta distinta de la original" in bullets[0]
    assert "sección: 6.1 Cliente sin la tarjeta original" in bullets[0]
    assert "Tarjeta de débito" not in ans.answer and "Tarjeta de crédito" not in ans.answer


@needs_pdfs
def test_unrelated_sentences_are_left_out(real):
    ans = real.answer("¿Cuál es el horario estándar de atención al público?")
    assert "7:00 a 22:00" in ans.answer
    assert "recepción de proveedores" not in ans.answer


@needs_pdfs
def test_faq_entry_matching_the_question_is_quoted_with_its_restrictions(real):
    ans = real.answer("¿Puedo aceptar una devolución sin ticket?")
    for phrase in ("Sí, con restricciones", "identificación oficial", "exclusivamente en vale de tienda",
                   "como máximo tres devoluciones sin ticket"):
        assert phrase in ans.answer
    assert "Consulta el Procedimiento" not in ans.answer  # cross-reference dropped


@needs_pdfs
def test_numeric_entity_selects_the_exact_table_row(real):
    ans = real.answer("¿Qué descuento se aplica a un producto con 3 días antes de caducar?")
    assert _bullets(ans.answer) == [
        "- Días restantes antes de la fecha de caducidad: 3 días | Descuento sobre precio regular: 35 %. "
        "[politica_mermas_y_caducidad.pdf, p. 4, sección: 6. Rebajas escalonadas para producto próximo a vencer]"
    ]


@needs_pdfs
def test_entity_in_another_column(real):
    ans = real.answer("¿Qué monto de devolución puede autorizar el Gerente de Tienda?")
    assert _bullets(ans.answer)[0].startswith("- Rango de monto (pesos): De 10,000.01 a 30,000.00 | Quién autoriza: Gerente de Tienda")


@needs_pdfs
def test_policy_leads_over_the_faq_that_defers_to_it(real):
    ans = real.answer("¿Con qué frecuencia se cuenta el inventario de abarrotes?")
    first = _bullets(ans.answer)[0]
    assert "Categoría: Abarrotes" in first and "Frecuencia de conteo: Mensual" in first
    assert "politica_mermas_y_caducidad.pdf, p. 3" in first


@needs_pdfs
@pytest.mark.parametrize("q,reason", [
    ("¿Cuál es el precio de la leche?", "no indican «precio»"),
    ("¿Cuánto cuesta el estacionamiento para clientes?", "no indican «cuesta»"),
    ("¿Cuál es la tolerancia de merma para electrónica?", "no incluye «electrónica»"),
    ("¿Qué tienda vendió más en agosto?", "datos de ventas"),
])
def test_unsupported_questions_abstain_and_say_why(real, q, reason):
    ans = real.answer(q)
    assert ans.status == "insufficient_data"
    assert ans.evidence == []
    assert reason in ans.answer


@needs_pdfs
def test_table_gap_message_lists_what_the_table_contains(real):
    ans = real.answer("¿Cuál es la tolerancia de merma para electrónica?")
    assert "politica_mermas_y_caducidad.pdf, p. 3" in ans.answer
    for category in ("Lácteos", "Panadería", "Abarrotes", "Cuidado personal"):
        assert category in ans.answer


@needs_pdfs
def test_conflict_answer_keeps_both_sides_and_marks_the_non_prevailing_one(real):
    ans = real.answer("¿Cuántos días tiene un cliente para devolver una licuadora?")
    bullets = _bullets(ans.answer)
    assert "30 días naturales" in bullets[0] and "procedimiento_devoluciones.pdf" in bullets[0]
    assert any("15 días" in b and "no prevalece según la fuente" in b for b in bullets)
    assert "⚠️ Contradicción entre documentos" in ans.answer


def test_sibling_rows_are_pruned_only_when_the_entity_row_exists():
    from types import SimpleNamespace as NS

    def row(label, entity, section="T"):
        return NS(hit=NS(chunk=NS(kind="table_row", document="d", section=section, label=label)), entity_match=entity)

    rows = [row("3 días", True), row("5 días", False), row("Otra", False, section="U")]
    kept = prune_table_siblings(rows)
    assert [r.hit.chunk.label for r in kept] == ["3 días", "Otra"]
    assert prune_table_siblings([row("5 días", False)])[0].hit.chunk.label == "5 días"


# ----------------------------------------------------------- determinism
@needs_pdfs
def test_answers_do_not_depend_on_hash_seed(tmp_path):
    script = (
        "import json,sys; sys.path.insert(0, %r)\n"
        "from app.rag.extraction import discover_pdfs\n"
        "from app.rag.index import PolicyIndex\n"
        "from app.rag.service import PolicyRetriever\n"
        "from app.rag.config import RagSettings\n"
        "r = PolicyRetriever(PolicyIndex.build(discover_pdfs(%r), RagSettings(embeddings='off')))\n"
        "qs = [c['question'] for c in json.load(open(%r))]\n"
        "print(json.dumps([r.answer(q).answer for q in qs], ensure_ascii=False))\n"
    ) % (str(ROOT), str(PDF_DIR), str(FIXTURES / "candidatos_eval.json"))
    outs = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", script], env=env, check=True,
                                   capture_output=True, text=True).stdout)
    assert outs[0] == outs[1]


# ------------------------------------------------------- held-out battery
@needs_pdfs
def test_holdout_battery_quality_floors(real):
    """Independent questions written before phase-3 changes; floors, not tuning targets."""
    report = evaluate(real, load_cases(FIXTURES / "holdout_eval.json"), top_k=5, pdf_dir=PDF_DIR)
    print(report.model_dump_json(indent=1))
    assert report.cases == 49
    assert report.citation_verified == 1.0
    assert report.false_abstention_rate == 0.0
    assert report.abstention_accuracy >= 0.9
    assert report.hit_at_k >= 0.95
