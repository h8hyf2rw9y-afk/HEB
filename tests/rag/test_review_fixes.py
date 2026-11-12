"""Regression tests for Codex review on PR #5 (one section per review point)."""

import json
import math
import shutil
import subprocess

import pytest

from app.rag import interface, service
from app.rag.config import RagSettings
from app.rag.conflicts import resolve_precedence
from app.rag.embeddings import EmbeddingError, EmbeddingUsage, cosine_similarities, validate_vectors
from app.rag.extraction import extract_pdf
from app.rag.index import PolicyIndex
from app.rag.models import Chunk, DocumentInfo
from app.rag.service import PolicyRetriever
from app.rag.synthetic import write_synthetic_corpus
from rag_fakes import FakeEmbedder


# --------------------------------------------------------------------------
# 1. Explicit settings must not replace the shared retriever singleton.
# --------------------------------------------------------------------------
@pytest.fixture()
def clean_singleton():
    service.reset_retriever()
    yield
    service.reset_retriever()


def _index_with(tmp_path, name, pdf):
    index = PolicyIndex.build([pdf])
    index.save(tmp_path / name)
    return RagSettings(index_dir=tmp_path / name, embeddings="off")


def test_health_with_explicit_settings_does_not_mutate_singleton(tmp_path, pdf_paths, clean_singleton, monkeypatch):
    main = _index_with(tmp_path, "main", pdf_paths[0])
    other = _index_with(tmp_path, "other", pdf_paths[1])
    monkeypatch.setattr(RagSettings, "from_env", classmethod(lambda cls: main))

    shared = service.get_retriever()
    assert set(shared.index.documents) == {pdf_paths[0].name}

    health = interface.rag_health(other)
    assert health["ready"] and health["documents"] == 1
    temp = service.get_retriever(other)
    assert temp is not shared
    assert set(temp.index.documents) == {pdf_paths[1].name}

    # Subsequent default requests still use the original index.
    assert service.get_retriever() is shared
    assert set(service.get_retriever().index.documents) == {pdf_paths[0].name}


def test_reset_retriever_reloads(tmp_path, pdf_paths, clean_singleton, monkeypatch):
    main = _index_with(tmp_path, "main", pdf_paths[0])
    monkeypatch.setattr(RagSettings, "from_env", classmethod(lambda cls: main))
    first = service.get_retriever()
    service.reset_retriever()
    assert service.get_retriever() is not first


# --------------------------------------------------------------------------
# 2. Ambiguous supersession phrases never pick a winner by date.
# --------------------------------------------------------------------------
def _docs(*infos):
    return {d.document: d for d in infos}


def test_ambiguous_title_with_dates_stays_unresolved():
    # Both versions share the title; the phrase matches both. Dates must NOT decide.
    a = DocumentInfo(document="pol_a.pdf", title="Política de Devoluciones", effective_date="2024-07-01",
                     supersedes=["Este documento reemplaza a la Política de Devoluciones anterior."])
    b = DocumentInfo(document="pol_b.pdf", title="Política de Devoluciones", effective_date="2023-01-01")
    preferred, basis = resolve_precedence("pol_a.pdf", "pol_b.pdf", _docs(a, b))
    assert preferred is None
    assert "ambiguas" in basis and "no deciden por sí solas" in basis


def test_similar_file_names_v1_vs_v10():
    # v2 replaces v10, not v1. Conflict v2 vs v1 must stay unresolved.
    v1 = DocumentInfo(document="politica_v1.pdf")
    v10 = DocumentInfo(document="politica_v10.pdf")
    v2 = DocumentInfo(document="politica_v2.pdf", supersedes=["Este documento reemplaza a politica_v10.pdf."])
    docs = _docs(v1, v10, v2)
    assert resolve_precedence("politica_v2.pdf", "politica_v1.pdf", docs)[0] is None
    assert resolve_precedence("politica_v2.pdf", "politica_v10.pdf", docs)[0] == "politica_v2.pdf"


def test_similar_titles_with_dotted_versions():
    v1 = DocumentInfo(document="a.pdf", title="Política de Devoluciones v1")
    v12 = DocumentInfo(document="b.pdf", title="Política de Devoluciones v1.2")
    v2 = DocumentInfo(document="c.pdf", title="Política de Devoluciones v2",
                      supersedes=["Este documento reemplaza a la Política de Devoluciones v1.2 vigente."])
    docs = _docs(v1, v12, v2)
    assert resolve_precedence("c.pdf", "a.pdf", docs)[0] is None
    assert resolve_precedence("c.pdf", "b.pdf", docs)[0] == "c.pdf"


def test_short_file_stem_does_not_match_common_words():
    # "a.pdf" must not be "mentioned" by the preposition in "reemplaza a la ...".
    a = DocumentInfo(document="a.pdf")
    new = DocumentInfo(document="nueva.pdf", supersedes=["Este documento reemplaza a la guía anterior."])
    assert resolve_precedence("nueva.pdf", "a.pdf", _docs(a, new))[0] is None


def test_phrase_matching_several_documents_is_ambiguous():
    a = DocumentInfo(document="nuevo.pdf", supersedes=["Reemplaza a manual.pdf y a manual_anexo.pdf."])
    b = DocumentInfo(document="manual.pdf")
    c = DocumentInfo(document="manual_anexo.pdf")
    assert resolve_precedence("nuevo.pdf", "manual.pdf", _docs(a, b, c))[0] is None


def test_supersession_sentence_not_truncated_at_version_dot(tmp_path):
    from app.rag.extraction import pymupdf

    path = tmp_path / "pol.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Este documento reemplaza a la Política v1.2 vigente desde 2023.")
    doc.save(path)
    doc.close()
    info = extract_pdf(path).info
    assert info.supersedes == ["Este documento reemplaza a la Política v1.2 vigente desde 2023."]


def test_synthetic_supersession_still_resolves(retriever):
    ans = retriever.answer("¿Cuántos días tengo para devolver un producto?")
    assert ans.conflicts[0].preferred_document == "SINTETICO_politica_devoluciones_v2.pdf"


# --------------------------------------------------------------------------
# 3. Generated synthetic PDFs protect themselves from Git.
# --------------------------------------------------------------------------
def test_synthetic_dir_has_gitignore(tmp_path):
    write_synthetic_corpus(tmp_path / "pdfs")
    assert (tmp_path / "pdfs" / ".gitignore").read_text() == "*\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="git no disponible")
def test_generated_pdfs_and_index_are_ignored_by_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    pdfs = write_synthetic_corpus(repo / "data" / "synthetic_policies")
    PolicyIndex.build(pdfs).save(repo / "data" / "rag_index")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert status == ""


# --------------------------------------------------------------------------
# 4. Embedding vectors are validated; incompatible vectors degrade to BM25.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("vectors,count,msg", [
    ([[1.0, 2.0], [1.0]], 2, "dimensión"),
    ([[1.0, math.nan]], 1, "no finito"),
    ([[1.0, math.inf]], 1, "no finito"),
    ([["a", 1.0]], 1, "no numérico"),
    ([[]], 1, "vacío"),
    ([[1.0]], 2, "Se esperaban 2"),
    ("no-list", 1, "Se esperaban"),
])
def test_validate_vectors_rejects_bad_input(vectors, count, msg):
    with pytest.raises(EmbeddingError, match=msg):
        validate_vectors(vectors, expected_count=count)


def test_validate_vectors_expected_dim():
    assert validate_vectors([[1, 2, 3]], expected_count=1) == 3
    with pytest.raises(EmbeddingError):
        validate_vectors([[1, 2]], expected_count=1, expected_dim=3)


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="incompatible"):
        cosine_similarities([1.0, 0.0], [[1.0, 0.0, 0.0]])


class RaggedEmbedder(FakeEmbedder):
    """Returns vectors of inconsistent dimension at build time."""

    def embed(self, texts):
        vecs, usage = super().embed(texts)
        vecs[-1] = vecs[-1][:-1]
        return vecs, usage


class WrongDimQueryEmbedder(FakeEmbedder):
    """Correct model name, but query vectors with a different dimension."""

    def embed(self, texts):
        return [[0.1] * (self.dim + 5) for _ in texts], EmbeddingUsage(calls=1)


class NanQueryEmbedder(FakeEmbedder):
    def embed(self, texts):
        return [[math.nan] * self.dim for _ in texts], EmbeddingUsage(calls=1)


Q = "¿Cuántos días tengo para devolver un producto?"


def test_build_auto_degrades_on_inconsistent_vectors(pdf_paths):
    index = PolicyIndex.build(pdf_paths, RagSettings(embeddings="auto"), embedder=RaggedEmbedder())
    assert index.embeddings is None and index.embed_model is None
    assert any("solo BM25" in w and "dimensión" in w for w in index.warnings)
    assert PolicyRetriever(index).answer(Q).status == "answered"


def test_build_required_raises_on_inconsistent_vectors(pdf_paths):
    with pytest.raises(EmbeddingError):
        PolicyIndex.build(pdf_paths, RagSettings(embeddings="required"), embedder=RaggedEmbedder())


@pytest.mark.parametrize("embedder_cls", [WrongDimQueryEmbedder, NanQueryEmbedder])
def test_query_with_incompatible_vector_falls_back_to_bm25(hybrid_index, embedder_cls):
    ans = PolicyRetriever(hybrid_index, embedder=embedder_cls()).answer(Q)
    assert ans.status == "answered"
    assert ans.metrics.retrieval_mode == "bm25"
    assert any("incompatibles" in w for w in ans.metrics.warnings)
    assert all(e.score is not None for e in ans.evidence)


@pytest.mark.parametrize("corrupt", [
    lambda e: e[:-1],                                   # fewer vectors than chunks
    lambda e: [e[0][:-1]] + e[1:],                       # ragged dimensions
    lambda e: [[float("nan")] * len(e[0])] + e[1:],      # non-finite values
])
def test_load_with_corrupted_embeddings_degrades_to_bm25(tmp_path, hybrid_index, corrupt):
    target = hybrid_index.save(tmp_path / "idx")
    payload = json.loads(target.read_text())
    payload["embeddings"] = corrupt(payload["embeddings"])
    target.write_text(json.dumps(payload))
    loaded = PolicyIndex.load(tmp_path / "idx")
    assert loaded.embeddings is None and loaded.embed_model is None
    assert any("inválidos" in w for w in loaded.warnings)
    ans = PolicyRetriever(loaded, embedder=FakeEmbedder()).answer(Q)
    assert ans.status == "answered" and ans.metrics.retrieval_mode == "bm25"


def test_saved_index_records_dimension_and_reloads_hybrid(tmp_path, hybrid_index):
    target = hybrid_index.save(tmp_path / "idx")
    assert json.loads(target.read_text())["embed_dim"] == 128
    loaded = PolicyIndex.load(tmp_path / "idx")
    assert loaded.embed_dim == 128
    _, metrics = loaded.search(Q, embedder=FakeEmbedder())
    assert metrics.retrieval_mode == "hybrid"


def test_index_constructed_directly_validates_embeddings():
    chunks = [Chunk(chunk_id=f"c{i}", document="d.pdf", page=1, text=f"texto {i}") for i in range(2)]
    index = PolicyIndex(chunks=chunks, documents={}, embeddings=[[1.0, 0.0], [1.0]], embed_model="m")
    assert index.embeddings is None and index.embed_model is None
