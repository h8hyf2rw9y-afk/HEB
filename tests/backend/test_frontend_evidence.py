"""Regression checks for evidence labels and policy warning formatting (issue #13)."""
from frontend.app import cited_evidence, display_answer


def test_warning_starts_new_markdown_block():
    answer = "Respuesta:\n- Dato [p.pdf, p. 2]\n⚠️ Contradicción entre documentos"
    result = display_answer(answer)
    assert "\n\n⚠️" in result
    assert "Dato [p.pdf, p. 2]" in result


def test_only_evidence_cited_in_answer_is_shown():
    evidence = [
        {"source_type": "pdf", "source": "procedure.pdf", "reference": "p. 4, sección: 6.1"},
        {"source_type": "pdf", "source": "faq.pdf", "reference": "p. 1, sección: P17"},
        {"source_type": "pdf", "source": "procedure.pdf", "reference": "p. 4, sección: 6.1"},
    ]
    answer = "Reembolso en vale [procedure.pdf, p. 4, sección: 6.1]"
    result = cited_evidence(evidence, answer)
    assert len(result) == 1
    assert result[0]["source"] == "procedure.pdf"


def test_csv_sources_still_show():
    evidence = [
        {"source_type": "csv", "source": "ventas.csv", "reference": "agosto de 2026"},
        {"source_type": "csv", "source": "tiendas.csv", "reference": "tienda_id"},
    ]
    assert len(cited_evidence(evidence, "Ventas totales de agosto: 12 MXN")) == 2
