"""Stable integration surface for the backend (FastAPI / tool layer owned by Codex).

Contract version ``rag-v1``. Backward-compatible additions only; breaking changes
require an agreed GitHub Issue and an update to docs/CONTRACTS.md.

Usage from the backend::

    from app.rag.interface import PolicySearchRequest, search_policies, rag_health

    resp = search_policies(PolicySearchRequest(query="¿Cuántos días para devolver?", top_k=5))
    resp.status        # answered | insufficient_data | needs_clarification | error
    resp.evidence      # list[PolicyEvidence]: document, page, passage, section, score
    resp.conflicts     # contradictory passages, both sides included
    resp.metrics       # latency, retrieval mode, embedding calls/tokens

``search_policies`` never raises: failures are returned as ``status="error"``.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from .config import RagSettings
from .index import IndexNotFoundError
from .models import AnswerStatus, PolicyConflict, RetrievalMetrics, http_evidence
from .service import PolicyRetriever, get_retriever

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "rag-v1"


class PolicySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query vacía")
        return v


class PolicyEvidence(BaseModel):
    """One retrieved passage. ``score`` is the relevance score used for ranking
    (BM25 score in bm25 mode, RRF score in hybrid mode; only comparable within a response)."""

    document: str
    page: int = Field(ge=1)
    passage: str
    section: str | None = None
    score: float
    chunk_id: str

    def to_http_evidence(self, max_passage_chars: int = 600) -> dict:
        """HTTP evidence for /chat. Besides the citation it carries the retrieved passage
        and its chunk id, so a user can audit the exact text behind each claim."""
        return http_evidence(self.document, self.page, self.section, self.passage, self.chunk_id, max_passage_chars)


class PolicySearchResponse(BaseModel):
    contract_version: str = CONTRACT_VERSION
    status: AnswerStatus
    answer: str = Field(description="Extractive, cited answer (no LLM). Safe to show to users.")
    evidence: list[PolicyEvidence] = Field(default_factory=list)
    conflicts: list[PolicyConflict] = Field(default_factory=list)
    coverage: float = 0.0
    metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    error: str | None = None


def search_policies(
    request: PolicySearchRequest, retriever: PolicyRetriever | None = None
) -> PolicySearchResponse:
    try:
        retriever = retriever or get_retriever()
        ans = retriever.answer(request.query, request.top_k)
    except IndexNotFoundError as exc:
        return PolicySearchResponse(status="error", answer="Índice de políticas no disponible.", error=str(exc))
    except Exception as exc:  # never leak a stack trace through the tool boundary
        logger.exception("RAG search failed")
        return PolicySearchResponse(
            status="error", answer="Error interno al consultar políticas.", error=type(exc).__name__
        )
    return PolicySearchResponse(
        status=ans.status,
        answer=ans.answer,
        evidence=[
            PolicyEvidence(
                document=e.document,
                page=e.page,
                passage=e.passage,
                section=e.section,
                score=e.score if e.score is not None else 0.0,
                chunk_id=e.chunk_id or "",
            )
            for e in ans.evidence
        ],
        conflicts=ans.conflicts,
        coverage=ans.coverage,
        metrics=ans.metrics,
    )


def rag_health(settings: RagSettings | None = None) -> dict:
    """Lightweight status for a backend /health endpoint."""
    try:
        r = get_retriever(settings)
    except IndexNotFoundError as exc:
        return {"contract_version": CONTRACT_VERSION, "ready": False, "error": str(exc)}
    return {
        "contract_version": CONTRACT_VERSION,
        "ready": True,
        "documents": len(r.index.documents),
        "chunks": len(r.index.chunks),
        "embed_model": r.index.embed_model,
        "embeddings_enabled": r.embedder is not None,
        "built_at": r.index.built_at,
        "index_warnings": r.index.warnings,
        "document_list": [
            {"document": d.document, "title": d.title, "pages": d.pages, "code": d.code, "version": d.version}
            for d in sorted(r.index.documents.values(), key=lambda d: d.document)
        ],
    }
