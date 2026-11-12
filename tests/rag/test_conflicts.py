from app.rag.conflicts import detect_conflicts, extract_quantities, polarity, resolve_precedence
from app.rag.index import SearchHit
from app.rag.models import Chunk, DocumentInfo

V1 = "SINTETICO_politica_devoluciones_v1.pdf"
V2 = "SINTETICO_politica_devoluciones_v2.pdf"


def test_quantity_extraction():
    q = extract_quantities("Plazo de 30 días naturales; ajustes mayores al 5% y 5 días hábiles, quince horas, $1,500")
    assert q["días"] == {"30"}
    assert q["%"] == {"5"}
    assert q["días hábiles"] == {"5"}
    assert q["horas"] == {"15"}
    assert q["moneda"] == {"1500"}


def test_polarity():
    assert polarity("Está prohibido operar sin chaleco") == -1
    assert polarity("Se permite operar con chaleco") == 1
    assert polarity("El conteo es semanal") == 0


def test_superseded_policy_conflict_shows_both_sides_and_cites_source(retriever):
    ans = retriever.answer("¿Cuántos días tengo para devolver un producto?")
    assert ans.status == "answered"
    assert len(ans.conflicts) == 1
    c = ans.conflicts[0]
    assert c.kind == "numeric" and c.topic == "días"
    assert {cl.document for cl in c.claims} == {V1, V2}
    assert {tuple(cl.values) for cl in c.claims} == {("15",), ("30",)}
    assert c.resolution == "resolved_by_source"
    assert c.preferred_document == V2
    assert "reemplaza" in c.basis  # basis quotes the document text
    # both passages are returned as evidence
    assert {e.document for e in ans.evidence} >= {V1, V2}
    assert "Prevalece SINTETICO_politica_devoluciones_v2.pdf" in ans.answer


def test_unresolved_conflict_does_not_invent_a_winner(retriever):
    ans = retriever.answer("¿Quién aprueba los ajustes de inventario?")
    assert len(ans.conflicts) == 1
    c = ans.conflicts[0]
    assert c.topic == "%"
    assert c.resolution == "unresolved"
    assert c.preferred_document is None
    assert {cl.document for cl in c.claims} == {
        "SINTETICO_lineamientos_inventario.pdf",
        "SINTETICO_manual_operativo.pdf",
    }
    assert "Ningún documento establece" in c.basis
    assert "sin resolver" in ans.answer
    assert "Prevalece" not in ans.answer


def test_no_false_conflict_for_unrelated_quantities(retriever):
    for q in ("¿Cuánto tarda un reembolso?", "¿Quién puede operar montacargas?", "¿Qué hago si hay un derrame?"):
        assert retriever.answer(q).conflicts == [], q


def _hit(doc, text, page=1, section="S"):
    return SearchHit(Chunk(chunk_id=f"{doc}:{page}:{section}", document=doc, page=page, section=section, text=text), 1.0, 1.0)


def test_effective_dates_alone_do_not_resolve():
    docs = {
        "a.pdf": DocumentInfo(document="a.pdf", effective_date="2025-01-01"),
        "b.pdf": DocumentInfo(document="b.pdf", effective_date="2020-01-01"),
    }
    preferred, basis = resolve_precedence("a.pdf", "b.pdf", docs)
    assert preferred is None
    assert "2025-01-01" in basis and "2020-01-01" in basis


def test_explicit_precedence_statement_resolves():
    docs = {
        "a.pdf": DocumentInfo(document="a.pdf", title="Manual de Seguridad",
                              precedence_statements=["En caso de conflicto prevalece el Manual de Seguridad"]),
        "b.pdf": DocumentInfo(document="b.pdf", title="Guía Rápida"),
    }
    assert resolve_precedence("a.pdf", "b.pdf", docs)[0] == "a.pdf"


def test_modality_conflict_detected():
    hits = [
        _hit("a.pdf", "Se permite el uso de celular en piso de venta durante el turno."),
        _hit("b.pdf", "Está prohibido el uso de celular en piso de venta durante el turno."),
    ]
    docs = {d: DocumentInfo(document=d) for d in ("a.pdf", "b.pdf")}
    conflicts = detect_conflicts("¿Se puede usar el celular en piso de venta?", hits, docs)
    assert [c.kind for c in conflicts] == ["modality"]
    assert conflicts[0].resolution == "unresolved"


def test_same_chunk_never_conflicts_with_itself():
    hits = [_hit("a.pdf", "Plazo de 30 días para devoluciones. Plazo de 15 días para devoluciones.")]
    assert detect_conflicts("plazo devoluciones", hits, {"a.pdf": DocumentInfo(document="a.pdf")}) == []
