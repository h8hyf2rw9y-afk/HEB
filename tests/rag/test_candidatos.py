"""Evaluation on the four exercise PDFs in candidatos/politicas (fictional documents
published with the exercise). Skipped if the folder is not present."""

from __future__ import annotations


import pytest

from app.rag.config import RagSettings
from app.rag.evaluation import CitationVerifier, evaluate, load_cases
from app.rag.extraction import discover_pdfs, extract_pdf
from app.rag.index import PolicyIndex
from app.rag.interface import PolicySearchRequest, search_policies
from app.rag.service import PolicyRetriever

from conftest import FIXTURES, ROOT

PDF_DIR = ROOT / "candidatos" / "politicas"
FAQ, MAN, POL, PRO = (
    "faq_gerentes_de_tienda.pdf",
    "manual_apertura_y_cierre_tienda.pdf",
    "politica_mermas_y_caducidad.pdf",
    "procedimiento_devoluciones.pdf",
)

pytestmark = pytest.mark.skipif(
    len(discover_pdfs(PDF_DIR)) != 4, reason="candidatos/politicas no disponible"
)


@pytest.fixture(scope="module")
def real_index() -> PolicyIndex:
    return PolicyIndex.build(discover_pdfs(PDF_DIR), RagSettings(embeddings="off"))


@pytest.fixture()
def real(real_index) -> PolicyRetriever:
    return PolicyRetriever(real_index, RagSettings(embeddings="off"))


# ------------------------------------------------------------- indexing
def test_all_four_pdfs_are_indexed_with_document_facts(real_index):
    docs = real_index.documents
    assert set(docs) == {FAQ, MAN, POL, PRO}
    assert real_index.warnings == []
    expected = {
        FAQ: ("FAQ-OPS-001", "1.4", "2026-01-12"),
        MAN: ("MAN-OPS-007", "2.5", "2025-10-01"),
        POL: ("POL-OPS-014", "3.2", "2025-03-01"),
        PRO: ("PRO-SAC-021", "4.0", "2025-06-15"),
    }
    for name, (code, version, date) in expected.items():
        info = docs[name]
        assert (info.code, info.version, info.effective_date) == (code, version, date)
        assert info.empty_pages == []
        assert info.supersedes == []  # change-log rows are not supersession statements
    assert "PRO-SAC-021" in docs[FAQ].references
    assert any("prevalece siempre la política o el procedimiento" in s for s in docs[FAQ].precedence_statements)


def test_running_headers_are_removed_and_sections_detected(real_index):
    texts = [c.text for c in real_index.chunks]
    assert not any("Documento interno ficticio para ejercicio" in t for t in texts)
    assert not any(t.startswith("Página ") for t in texts)
    faq_sections = {c.section for c in real_index.chunks if c.document == FAQ and c.section}
    import re

    numbered = {m.group(0) for s in faq_sections if (m := re.match(r"P\d+", s))}
    assert numbered == {f"P{i}" for i in range(1, 26)}
    # multi-line bold question headings are merged
    assert any(s.startswith("P18.") and s.endswith("devuelven en tienda?") for s in faq_sections)


def test_tables_become_atomic_rows_with_headers(real_index):
    rows = {c.label: c for c in real_index.chunks if c.document == PRO and c.kind == "table_row"}
    electro = rows["Electrodomésticos y electrónicos menores"]
    assert electro.page == 2 and electro.section.startswith("3. Plazos de devolución")
    assert "Plazo para devolución o cambio: 30 días naturales" in electro.text
    tol = next(c for c in real_index.chunks if c.document == POL and c.label == "Panadería" and "Tolerancia" in c.text)
    assert "Tolerancia mensual (% de venta): 3.5 | Umbral de alerta (%): 4.5" in tol.text
    # narrow-column word breaks repaired ("Priori dad" -> "Prioridad", "Versi ón" -> "Versión")
    assert any(c.text.startswith("Prioridad: Crítica") for c in real_index.chunks)
    assert any(c.kind == "changelog_row" and c.text.startswith("Versión: 4.0") for c in real_index.chunks)


def test_every_extracted_passage_is_verifiable_in_the_pdf(real_index):
    from app.rag.models import Evidence

    verifier = CitationVerifier(PDF_DIR)
    bad = [
        (c.document, c.page, c.text[:60])
        for c in real_index.chunks
        if not verifier.verify(Evidence(document=c.document, page=c.page, passage=c.text, section=c.section))
    ]
    assert bad == []


def test_original_pdfs_are_not_modified():
    import hashlib

    digests = {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in discover_pdfs(PDF_DIR)}
    extract_pdf(PDF_DIR / FAQ)
    assert digests == {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in discover_pdfs(PDF_DIR)}


# -------------------------------------------------------------- answers
def test_appliance_conflict_shows_both_sides_and_follows_the_faq_rule(real):
    ans = real.answer("¿Cuántos días tiene un cliente para devolver una licuadora?")
    assert ans.status == "answered"
    assert len(ans.conflicts) == 1
    c = ans.conflicts[0]
    assert {(cl.document, cl.page, tuple(cl.values)) for cl in c.claims} == {(FAQ, 4, ("15",)), (PRO, 2, ("30",))}
    assert c.resolution == "resolved_by_source" and c.preferred_document == PRO
    assert "prevalece siempre la política o el procedimiento" in c.basis and "PRO-SAC-021" in c.basis
    assert "30 días naturales" in ans.answer and "15 días" in ans.answer
    assert "no prevalece según la fuente" in ans.answer


def test_dairy_rule_is_not_a_conflict_with_the_non_dairy_discount_row(real):
    ans = real.answer("¿Cuántos días antes de la caducidad se retiran los lácteos?")
    assert ans.conflicts == []
    assert "dos días antes" in ans.answer


@pytest.mark.parametrize("q", [
    "¿Cuál es el salario de un cajero?",
    "¿Cuántos días de vacaciones tiene un empleado?",
    "¿Se permiten mascotas dentro de la tienda?",
    "¿Qué cambia en la política de devoluciones de 2027?",
])
def test_unsupported_questions_abstain(real, q):
    ans = real.answer(q)
    assert ans.status == "insufficient_data" and ans.evidence == []


def test_ambiguous_question_asks_for_clarification_with_options(real):
    ans = real.answer("¿Cuál es el plazo?")
    assert ans.status == "needs_clarification"
    assert "¿A cuál te refieres?" in ans.answer
    assert ans.evidence  # options are backed by passages


def test_change_log_rows_only_answer_document_history_questions(real):
    hist = real.answer("¿Cuál es la fecha de vigencia de la política de mermas y caducidad?")
    assert "1 de marzo de 2025" in hist.answer
    other = real.answer("¿Cuál es la vigencia de un vale de tienda?")
    assert all(e.section != "Política de mermas y caducidad" for e in other.evidence)
    assert not any("Fecha de vigencia" in e.passage for e in other.evidence)


def test_contract_response_on_real_documents(real):
    resp = search_policies(PolicySearchRequest(query="¿Qué es SIGMA?"), real)
    assert resp.contract_version == "rag-v1" and resp.status == "answered"
    assert "Sistema Integral de Gestión de Mermas y Abasto" in resp.answer
    assert all(e.score > 0 and e.page >= 1 and e.passage for e in resp.evidence)


# ----------------------------------------------------------- battery
def test_evaluation_battery(real):
    cases = load_cases(FIXTURES / "candidatos_eval.json")
    report = evaluate(real, cases, top_k=5, pdf_dir=PDF_DIR)
    print(report.model_dump_json(indent=1))
    assert report.cases == len(cases) >= 60
    # Every case not documented as a known limitation must pass.
    assert report.failures == []
    assert report.citation_verified == 1.0
    assert report.false_abstention_rate == 0.0
    assert report.conflict_accuracy == 1.0
    assert report.retrieval_mode == "bm25"
    assert report.llm_tokens == 0
