import json

from app.rag.config import RagSettings
from app.rag.index import PolicyIndex
from app.rag.service import PolicyRetriever

from rag_fakes import DownEmbedder, FakeEmbedder

Q = "¿Cuántos días tengo para devolver un producto?"


def test_bm25_only_when_no_embedder(bm25_index):
    assert bm25_index.embeddings is None
    hits, metrics = bm25_index.search(Q, top_k=3)
    assert metrics.retrieval_mode == "bm25"
    assert metrics.embedding_calls == 0
    assert hits[0].chunk.section == "2. Plazo de devolución"


def test_build_falls_back_to_bm25_when_ollama_down(pdf_paths):
    index = PolicyIndex.build(pdf_paths, RagSettings(embeddings="auto"), embedder=DownEmbedder())
    assert index.embeddings is None
    assert any("solo BM25" in w for w in index.warnings)
    ans = PolicyRetriever(index, embedder=DownEmbedder()).answer(Q)
    assert ans.status == "answered"
    assert ans.metrics.retrieval_mode == "bm25"


def test_required_embeddings_fail_loudly(pdf_paths):
    import pytest

    from app.rag.embeddings import EmbeddingError

    with pytest.raises(EmbeddingError):
        PolicyIndex.build(pdf_paths, RagSettings(embeddings="required"), embedder=DownEmbedder())


def test_query_time_outage_falls_back_to_bm25(hybrid_index):
    ans = PolicyRetriever(hybrid_index, embedder=DownEmbedder()).answer(Q)
    assert ans.status == "answered"
    assert ans.metrics.retrieval_mode == "bm25"
    assert any("solo BM25" in w for w in ans.metrics.warnings)


def test_hybrid_rrf_active_with_embeddings(hybrid_index):
    assert hybrid_index.embed_model == "fake-embed"
    assert len(hybrid_index.embeddings) == len(hybrid_index.chunks)
    hits, metrics = hybrid_index.search(Q, top_k=3, embedder=FakeEmbedder())
    assert metrics.retrieval_mode == "hybrid"
    assert metrics.embedding_calls == 1
    assert all(h.embedding_similarity is not None for h in hits)
    # RRF scores are bounded by 2 / (k + 1)
    assert all(0 < h.score <= 2 / 61 + 1e-6 for h in hits)
    assert hits[0].chunk.section == "2. Plazo de devolución"


def test_embedding_model_mismatch_uses_bm25(hybrid_index):
    _, metrics = hybrid_index.search(Q, embedder=FakeEmbedder(model="otro"))
    assert metrics.retrieval_mode == "bm25"


def test_every_result_has_document_page_passage_score(retriever):
    ev = retriever.retrieve("¿Quién puede operar montacargas?", top_k=5)
    assert ev
    for e in ev:
        assert e.document.endswith(".pdf")
        assert e.page >= 1
        assert e.passage
        assert e.score is not None and e.score > 0
    http = ev[0].to_http_evidence()
    assert {k: http[k] for k in ("source_type", "source", "reference", "page", "chunk_id")} == {
        "source_type": "pdf",
        "source": "SINTETICO_manual_seguridad_tienda.pdf",
        "reference": "p. 1, sección: 1. Montacargas",
        "page": 1,
        "chunk_id": ev[0].chunk_id,
    }
    # The passage shown to the user is the retrieved text itself (whitespace-normalised).
    assert http["passage"] and " ".join(ev[0].passage.split()).startswith(http["passage"].removesuffix(" …"))


def test_http_evidence_passage_is_bounded():
    from app.rag.interface import PolicyEvidence

    ev = PolicyEvidence(document="d.pdf", page=2, passage="palabra " * 400, section="S", score=1.0, chunk_id="d.pdf#3")
    http = ev.to_http_evidence()
    assert len(http["passage"]) <= 602 and http["passage"].endswith(" …") and http["chunk_id"] == "d.pdf#3"


def test_save_load_roundtrip_and_gitignore(tmp_path, hybrid_index):
    target = hybrid_index.save(tmp_path / "idx")
    assert (tmp_path / "idx" / ".gitignore").read_text() == "*\n"
    from app.rag.index import INDEX_VERSION

    assert json.loads(target.read_text())["version"] == INDEX_VERSION
    loaded = PolicyIndex.load(tmp_path / "idx")
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in hybrid_index.chunks]
    a, _ = loaded.search(Q, embedder=FakeEmbedder())
    b, _ = hybrid_index.search(Q, embedder=FakeEmbedder())
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]


def test_load_missing_index_raises(tmp_path):
    import pytest

    from app.rag.index import IndexNotFoundError

    with pytest.raises(IndexNotFoundError):
        PolicyIndex.load(tmp_path)
