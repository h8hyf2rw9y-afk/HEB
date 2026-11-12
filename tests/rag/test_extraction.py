from pathlib import Path

import pytest

from app.rag.extraction import PdfExtractionError, discover_pdfs, extract_pdf, pymupdf
from app.rag.index import PolicyIndex
from app.rag.service import PolicyRetriever


def test_extracts_pages_sections_and_document_facts(corpus_dir):
    doc = extract_pdf(corpus_dir / "SINTETICO_politica_devoluciones_v1.pdf")
    assert doc.info.pages == 2
    assert doc.info.title == "Política de Devoluciones v1"
    assert doc.info.effective_date == "2023-01-01"
    by_section = {b.section: b for b in doc.blocks}
    assert by_section["2. Plazo de devolución"].page == 1
    assert "30 días naturales" in by_section["2. Plazo de devolución"].text
    assert by_section["4. Reembolsos"].page == 2


def test_supersession_phrase_is_taken_from_text(corpus_dir):
    info = extract_pdf(corpus_dir / "SINTETICO_politica_devoluciones_v2.pdf").info
    assert info.effective_date == "2024-07-01"
    assert any("reemplaza" in s for s in info.supersedes)


def _blank_pdf(path: Path, pages: int = 1) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def test_empty_pdf_reports_empty_pages_and_yields_no_chunks(tmp_path):
    path = _blank_pdf(tmp_path / "vacio.pdf", pages=2)
    doc = extract_pdf(path)
    assert doc.blocks == []
    assert doc.info.empty_pages == [1, 2]
    index = PolicyIndex.build([path])
    assert index.chunks == []
    assert any("sin texto extraíble" in w for w in index.warnings)
    ans = PolicyRetriever(index).answer("¿Cuántos días para devolver?")
    assert ans.status == "insufficient_data"
    assert ans.evidence == []


def test_corrupted_pdf_raises_extraction_error(tmp_path):
    bad = tmp_path / "roto.pdf"
    bad.write_bytes(b"esto no es un PDF")
    with pytest.raises(PdfExtractionError):
        extract_pdf(bad)


def test_missing_file_raises_extraction_error(tmp_path):
    with pytest.raises(PdfExtractionError):
        extract_pdf(tmp_path / "no_existe.pdf")


def test_encrypted_pdf_raises_extraction_error(tmp_path):
    path = tmp_path / "cifrado.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secreto")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    doc.close()
    with pytest.raises(PdfExtractionError, match="contraseña"):
        extract_pdf(path)


def test_build_skips_broken_files_and_keeps_valid_ones(tmp_path, corpus_dir):
    bad = tmp_path / "roto.pdf"
    bad.write_bytes(b"%PDF-1.7 basura")
    good = corpus_dir / "SINTETICO_manual_seguridad_tienda.pdf"
    index = PolicyIndex.build([bad, good])
    assert set(index.documents) == {good.name}
    assert any("roto.pdf" in w for w in index.warnings)


def test_discover_pdfs(tmp_path, corpus_dir):
    assert discover_pdfs(tmp_path / "nada") == []
    assert len(discover_pdfs(corpus_dir)) == 5
