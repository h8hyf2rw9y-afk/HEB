"""Shared fixtures for RAG tests. All PDFs are synthetic and generated at test time."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.config import RagSettings  # noqa: E402
from app.rag.index import PolicyIndex  # noqa: E402
from app.rag.service import PolicyRetriever  # noqa: E402
from app.rag.synthetic import write_synthetic_corpus  # noqa: E402
from rag_fakes import FakeEmbedder  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthetic_pdfs")
    write_synthetic_corpus(d)
    return d


@pytest.fixture(scope="session")
def pdf_paths(corpus_dir) -> list[Path]:
    return sorted(corpus_dir.glob("*.pdf"))


@pytest.fixture(scope="session")
def settings() -> RagSettings:
    return RagSettings(embeddings="auto")


@pytest.fixture(scope="session")
def bm25_index(pdf_paths, settings) -> PolicyIndex:
    return PolicyIndex.build(pdf_paths, settings, embedder=None)


@pytest.fixture(scope="session")
def hybrid_index(pdf_paths, settings) -> PolicyIndex:
    return PolicyIndex.build(pdf_paths, settings, embedder=FakeEmbedder())


@pytest.fixture()
def retriever(bm25_index, settings) -> PolicyRetriever:
    return PolicyRetriever(bm25_index, settings, embedder=None)
