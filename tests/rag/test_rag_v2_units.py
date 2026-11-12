"""Unit tests for phase-2 retrieval improvements (synthetic inputs only)."""

from __future__ import annotations

import json

import pytest

from app.rag import interface
from app.rag.bm25 import BM25Index
from app.rag.chunking import chunk_document
from app.rag.conflicts import _label_tokens, detect_conflicts, resolve_precedence
from app.rag.extraction import ExtractedDocument, PageBlock, _clean_cell, extract_pdf, pymupdf
from app.rag.index import INDEX_VERSION, PolicyIndex, SearchHit, definition_expansions
from app.rag.models import Chunk, DocumentInfo
from app.rag.service import PolicyRetriever


# ------------------------------------------------------------ extraction
def _table_pdf(path, rows, kv=False, header_lines=False, pages=1):
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=612, height=792)
        if header_lines:
            page.insert_text((72, 30), "Empresa X — Documento de prueba", fontsize=8)
            page.insert_text((72, 760), f"Página {n + 1}", fontsize=8)
        page.insert_text((72, 80), "3. Plazos de prueba", fontname="hebo", fontsize=13)
        page.insert_text((72, 100), "Texto de introducción de la sección de prueba.", fontsize=10)
        top, cw, rh = 120, 170, 24
        ncol, nrow = len(rows[0]), len(rows)
        for r in range(nrow + 1):  # ruling lines so find_tables sees a grid
            page.draw_line((72, top + r * rh), (72 + ncol * cw, top + r * rh))
        for c in range(ncol + 1):
            page.draw_line((72 + c * cw, top), (72 + c * cw, top + nrow * rh))
        for r, row in enumerate(rows):
            for col, cell in enumerate(row):
                bold = (r == 0 and not kv) or (kv and col == 0)
                page.insert_text((76 + col * cw, top + r * rh + 16), cell,
                                 fontname="hebo" if bold else "helv", fontsize=9)
        y = top + nrow * rh
        page.insert_text((72, y + 30), "Texto posterior a la tabla en la misma sección.", fontsize=10)
    doc.save(path)
    doc.close()
    return path


def test_table_rows_are_extracted_with_headers(tmp_path):
    path = _table_pdf(tmp_path / "t.pdf", [["Tipo", "Plazo"], ["Ropa", "30 días"], ["Perecederos", "48 horas"]])
    doc = extract_pdf(path)
    rows = [b for b in doc.blocks if b.kind == "table_row"]
    assert [(b.label, b.text) for b in rows] == [
        ("Ropa", "Tipo: Ropa | Plazo: 30 días."),
        ("Perecederos", "Tipo: Perecederos | Plazo: 48 horas."),
    ]
    assert all(b.section == "3. Plazos de prueba" for b in rows)
    # prose before and after the table keeps its order and is not duplicated
    prose = [b.text for b in doc.blocks if b.kind == "text"]
    assert prose[0].startswith("Texto de introducción") and prose[-1].startswith("Texto posterior")
    assert not any("Ropa" in t for t in prose)


def test_key_value_table_feeds_document_facts(tmp_path):
    path = _table_pdf(tmp_path / "kv.pdf", [["Código de documento", "ABC-DEF-001"], ["Versión", "2.0"],
                                             ["Fecha de vigencia", "5 de mayo de 2025"]], kv=True)
    doc = extract_pdf(path)
    assert {b.kind for b in doc.blocks if "ABC" in b.text} == {"metadata_row"}
    assert (doc.info.code, doc.info.version, doc.info.effective_date) == ("ABC-DEF-001", "2.0", "2025-05-05")


def test_repeated_headers_and_page_markers_are_dropped(tmp_path):
    path = _table_pdf(tmp_path / "h.pdf", [["A", "B"], ["x", "1 día"]], header_lines=True, pages=3)
    text = extract_pdf(path).full_text
    assert "Documento de prueba" not in text and "Página" not in text


def test_changelog_table_is_tagged_and_not_a_supersession(tmp_path):
    path = _table_pdf(tmp_path / "c.pdf", [["Versión", "Fecha", "Descripción del cambio"],
                                            ["2.0", "1 de abril de 2023", "Se sustituye la bitácora en papel."]])
    doc = extract_pdf(path)
    assert [b.kind for b in doc.blocks if "bitácora" in b.text] == ["changelog_row"]
    assert doc.info.supersedes == []


@pytest.mark.parametrize("raw,clean", [
    ("Versi\nón", "Versión"), ("Priori\ndad", "Prioridad"),
    ("Tiempo de primera\nrespuesta", "Tiempo de primera respuesta"), ("Plazo para\ndevolución o\ncambio", "Plazo para devolución o cambio"),
])
def test_clean_cell_repairs_narrow_column_breaks(raw, clean):
    assert _clean_cell(raw) == clean


def test_multiline_bold_heading_is_merged(tmp_path):
    path = tmp_path / "q.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 60, 300, 120),
                        "P18. ¿Cómo se atienden los pedidos en línea que se recogen o devuelven en tienda?",
                        fontname="hebo", fontsize=11)
    page.insert_text((72, 140), "Los pedidos se preparan en el área de recoge en tienda.", fontsize=10)
    doc.save(path)
    doc.close()
    blocks = extract_pdf(path).blocks
    assert blocks[0].section == "P18. ¿Cómo se atienden los pedidos en línea que se recogen o devuelven en tienda?"


def test_rows_are_atomic_chunks():
    doc = ExtractedDocument(document="d.pdf", path=None, blocks=[
        PageBlock(1, "S", "intro"),
        PageBlock(1, "S", "Tipo: A | Plazo: 1 día.", kind="table_row", label="A"),
        PageBlock(1, "S", "Tipo: B | Plazo: 2 días.", kind="table_row", label="B"),
        PageBlock(1, "S", "cierre"),
    ])
    chunks = chunk_document(doc)
    assert [(c.kind, c.label, c.text) for c in chunks] == [
        ("text", None, "intro"), ("table_row", "A", "Tipo: A | Plazo: 1 día."),
        ("table_row", "B", "Tipo: B | Plazo: 2 días."), ("text", None, "cierre"),
    ]


# -------------------------------------------------------------- retrieval
def _chunk(i, text, section="S", kind="text", label=None, doc="d.pdf", page=1):
    return Chunk(chunk_id=f"c{i}", document=doc, page=page, section=section, text=text, kind=kind, label=label)


def test_definition_expansions_come_only_from_definition_sections():
    chunks = [
        _chunk(0, "Electrodomésticos menores: aparatos tales como licuadoras, tostadores y planchas.", section="2. Definiciones"),
        _chunk(1, "Ejemplo fuera de definiciones: tales como bicicletas.", section="5. Otra"),
    ] + [_chunk(i, f"relleno {i}") for i in range(2, 30)]
    exp = definition_expansions(chunks, BM25Index([c.text.lower().split() for c in chunks]))
    assert exp["licuad"] == {"electr", "menor"}
    assert "biciclet" not in exp and "bicicl" not in exp


def test_soft_matches_count_half_in_coverage():
    bm = BM25Index([["devolu", "plazo"], ["otro"]])
    full = bm.coverage(["devolu", "plazo"], 0)
    soft = bm.coverage(["devolv", "plazo"], 0, {"devolv": {"devolu"}}, alternative_weight=0.5)
    none = bm.coverage(["devolv", "plazo"], 0)
    assert full == 1.0 and none < soft < full


def test_history_rows_need_a_history_question_about_a_document():
    chunks = [
        _chunk(0, "Versión: 1.0 | Fecha: 1 de agosto de 2023 | Descripción del cambio: Emisión inicial.", kind="changelog_row", label="1.0"),
        _chunk(1, "Fecha de vigencia: 1 de agosto de 2025.", kind="metadata_row", label="Fecha de vigencia"),
        _chunk(2, "El vale de tienda tiene vigencia de doce meses."),
        _chunk(3, "Las ventas se reportan cada mes."),
    ]
    index = PolicyIndex(chunks=chunks, documents={})
    ids = lambda q: {h.chunk.chunk_id for h in index.search(q, top_k=5)[0]}  # noqa: E731
    assert not ids("¿Qué tienda vendió más en agosto?") & {"c0", "c1"}
    assert "c1" not in ids("¿Cuál es la vigencia de un vale de tienda?")
    assert "c1" in ids("¿Cuál es la fecha de vigencia del manual?")


def test_one_term_question_with_many_subjects_needs_clarification():
    chunks = [_chunk(i, f"Tipo: {t} | Plazo: {d}.", kind="table_row", label=t)
              for i, (t, d) in enumerate([("Ropa", "30 días"), ("Perecederos", "48 horas"), ("Abarrotes", "30 días")])]
    chunks += [_chunk(9, "Texto sin relación alguna con nada.")]
    r = PolicyRetriever(PolicyIndex(chunks=chunks, documents={}))
    ans = r.answer("¿Cuál es el plazo?")
    assert ans.status == "needs_clarification" and len(ans.evidence) == 3
    assert r.answer("¿Cuál es el plazo de ropa?").status == "answered"


def test_definitional_question_is_not_ambiguous():
    chunks = [_chunk(0, "SIGMA: Sistema Integral de Gestión.", section="2. Definiciones")] + [
        _chunk(i, f"Se registra en SIGMA el movimiento {i}.", section=f"Sección {i}") for i in range(1, 5)
    ]
    ans = PolicyRetriever(PolicyIndex(chunks=chunks, documents={})).answer("¿Qué es SIGMA?")
    assert ans.status == "answered"
    assert ans.evidence[0].section == "2. Definiciones"
    assert "SIGMA: Sistema Integral de Gestión." in ans.answer


# -------------------------------------------------------------- conflicts
def _hit(chunk):
    return SearchHit(chunk, 1.0, 1.0)


FAQ = DocumentInfo(document="faq.pdf", code="FAQ-OPS-001", references=["PRO-SAC-021"],
                   precedence_statements=["En caso de duda o de diferencia con un documento normativo, "
                                          "prevalece siempre la política o el procedimiento vigente."])
PRO = DocumentInfo(document="pro.pdf", code="PRO-SAC-021", title="Procedimiento de devoluciones")
MAN = DocumentInfo(document="man.pdf", code="MAN-OPS-007", title="Manual de apertura")


def test_subordination_rule_resolves_only_for_referenced_policies_or_procedures():
    docs = {d.document: d for d in (FAQ, PRO, MAN)}
    assert resolve_precedence("faq.pdf", "pro.pdf", docs)[0] == "pro.pdf"
    # A manual is neither a policy nor a procedure: stays unresolved.
    assert resolve_precedence("faq.pdf", "man.pdf", docs)[0] is None
    # A procedure the FAQ never cites: stays unresolved.
    other = DocumentInfo(document="pro2.pdf", code="PRO-XYZ-999")
    assert resolve_precedence("faq.pdf", "pro2.pdf", {**docs, "pro2.pdf": other})[0] is None
    # Without the subordination sentence: unresolved.
    plain = FAQ.model_copy(update={"precedence_statements": []})
    assert resolve_precedence("faq.pdf", "pro.pdf", {"faq.pdf": plain, "pro.pdf": PRO})[0] is None


def test_unrelated_policy_sentence_with_prevalece_is_not_a_precedence_rule():
    pol = DocumentInfo(document="pol.pdf", code="POL-OPS-014",
                       precedence_statements=["En esos casos prevalece el descuento mayor, nunca se acumulan."],
                       references=["PRO-SAC-021"])
    assert resolve_precedence("pol.pdf", "pro.pdf", {"pol.pdf": pol, "pro.pdf": PRO})[0] is None


def test_row_label_must_match_the_other_statement():
    docs = {"faq.pdf": FAQ, "pro.pdf": PRO}
    faq = _chunk(0, "Para electrodomésticos y electrónicos menores como licuadoras, el cliente tiene 15 días con ticket para la devolución.", doc="faq.pdf")
    electro = _chunk(1, "Tipo de producto: Electrodomésticos y electrónicos menores | Plazo para devolución o cambio: 30 días naturales | Requisitos: Ticket.",
                     kind="table_row", label="Electrodomésticos y electrónicos menores", doc="pro.pdf", page=2)
    abarrotes = _chunk(2, "Tipo de producto: Abarrotes y no perecederos | Plazo para devolución o cambio: 30 días | Requisitos: Ticket.",
                       kind="table_row", label="Abarrotes y no perecederos", doc="pro.pdf", page=2)
    q = "¿Cuántos días hay para devolver una licuadora?"
    found = detect_conflicts(q, [_hit(faq), _hit(abarrotes)], docs)
    assert found == []  # different subject (abarrotes)
    found = detect_conflicts(q, [_hit(faq), _hit(electro)], docs, extra_terms={"electr", "menor"})
    assert len(found) == 1 and found[0].preferred_document == "pro.pdf"


def test_negated_words_are_removed_from_row_labels():
    assert "lacteo" not in _label_tokens("1 día (solo perecederos no lácteos)")
    assert "perece" in _label_tokens("1 día (solo perecederos no lácteos)")


def test_changelog_and_metadata_rows_never_conflict():
    docs = {"a.pdf": DocumentInfo(document="a.pdf"), "b.pdf": DocumentInfo(document="b.pdf")}
    a = _chunk(0, "Versión: 1.0 | Fecha: 1 de agosto de 2023 | Plazo de 30 días.", kind="changelog_row", label="1.0", doc="a.pdf")
    b = _chunk(1, "Versión: 2.0 | Fecha: 1 de agosto de 2024 | Plazo de 15 días.", kind="changelog_row", label="2.0", doc="b.pdf")
    assert detect_conflicts("plazo versión", [_hit(a), _hit(b)], docs) == []


# ------------------------------------------------------------ compatibility
def test_v1_index_still_loads_with_rebuild_warning(tmp_path, bm25_index):
    target = bm25_index.save(tmp_path)
    payload = json.loads(target.read_text())
    assert payload["version"] == INDEX_VERSION == 2
    payload["version"] = 1
    for c in payload["chunks"]:
        c.pop("kind"), c.pop("label")
    for d in payload["documents"].values():
        d.pop("code"), d.pop("version"), d.pop("references")
    target.write_text(json.dumps(payload))
    loaded = PolicyIndex.load(tmp_path)
    assert len(loaded.chunks) == len(bm25_index.chunks)
    assert any("reconstrúyelo" in w for w in loaded.warnings)


def test_rag_v1_public_surface_is_unchanged():
    assert set(interface.PolicySearchRequest.model_fields) == {"query", "top_k"}
    assert set(interface.PolicySearchResponse.model_fields) == {
        "contract_version", "status", "answer", "evidence", "conflicts", "coverage", "metrics", "error"}
    assert set(interface.PolicyEvidence.model_fields) == {"document", "page", "passage", "section", "score", "chunk_id"}
    assert interface.CONTRACT_VERSION == "rag-v1"
    from app.rag.models import PolicyConflict, RetrievalMetrics
    assert set(PolicyConflict.model_fields) == {"kind", "topic", "claims", "resolution", "preferred_document", "basis"}
    assert set(RetrievalMetrics.model_fields) == {
        "retrieval_mode", "latency_ms", "candidates", "embedding_calls", "embedding_input_tokens",
        "embedding_latency_ms", "llm_calls", "input_tokens", "output_tokens", "warnings"}
    from app.rag import service
    assert callable(interface.search_policies) and callable(interface.rag_health) and callable(service.reset_retriever)
