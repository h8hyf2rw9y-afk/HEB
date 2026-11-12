import pytest

from app.rag.chunking import chunk_document
from app.rag.extraction import ExtractedDocument, PageBlock
from app.rag.text import normalize, split_sentences, stem, tokenize


def test_normalize_strips_accents_and_case():
    assert normalize("Devolución ÁRBOL Ñandú") == "devolucion arbol nandu"


def test_tokenize_removes_stopwords_and_unifies_plurals():
    assert tokenize("Las devoluciones") == tokenize("la devolución")
    assert "30" in tokenize("plazo de 30 días")


def test_stem_keeps_numbers():
    assert stem("10.5") == "10.5"


def test_split_sentences():
    assert split_sentences("Uno dos. Tres cuatro.") == ["Uno dos.", "Tres cuatro."]


def _doc(blocks):
    return ExtractedDocument(document="x.pdf", path=None, blocks=blocks)  # type: ignore[arg-type]


def test_chunks_never_cross_pages_or_sections():
    doc = _doc([
        PageBlock(1, "A", "uno dos tres"),
        PageBlock(1, "B", "cuatro cinco"),
        PageBlock(2, "B", "seis siete"),
    ])
    chunks = chunk_document(doc, max_words=50, overlap_words=5)
    assert [(c.page, c.section, c.text) for c in chunks] == [
        (1, "A", "uno dos tres"), (1, "B", "cuatro cinco"), (2, "B", "seis siete")
    ]
    assert len({c.chunk_id for c in chunks}) == 3


def test_sliding_window_overlap():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_document(_doc([PageBlock(3, None, words)]), max_words=10, overlap_words=4)
    assert chunks[0].text.split()[-4:] == chunks[1].text.split()[:4]
    assert chunks[-1].text.split()[-1] == "w24"
    assert all(c.page == 3 for c in chunks)


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_params(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_document(_doc([]), max_words=max_words, overlap_words=overlap)
