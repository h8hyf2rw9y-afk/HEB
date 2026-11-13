"""Pydantic models exposed by the RAG module.

``Evidence`` follows ``docs/CONTRACTS.md`` (document name, page number, passage,
optional section). Extra fields are optional so the contract stays backward
compatible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Status values allowed by docs/CONTRACTS.md for POST /chat.
AnswerStatus = Literal["answered", "needs_clarification", "insufficient_data", "error"]
RetrievalMode = Literal["bm25", "hybrid", "embedding"]


class Chunk(BaseModel):
    """Indexed unit of text. Never spans more than one PDF page."""

    chunk_id: str
    document: str = Field(description="PDF file name, e.g. politica_devoluciones.pdf")
    page: int = Field(ge=1, description="1-based page number")
    section: str | None = None
    text: str
    # Internal metadata (not part of the public Evidence contract).
    kind: str = "text"  # text | table_row | metadata_row | changelog_row
    label: str | None = None  # first cell of a table row


def http_evidence(document: str, page: int, section: str | None, passage: str,
                  chunk_id: str | None, max_passage_chars: int = 600) -> dict:
    """Shared by every evidence type so /chat always exposes the same auditable shape."""
    reference = f"p. {page}" + (f", sección: {section}" if section else "")
    text = " ".join(passage.split())
    if len(text) > max_passage_chars:
        text = text[:max_passage_chars].rsplit(" ", 1)[0] + " …"
    return {"source_type": "pdf", "source": document, "reference": reference,
            "page": page, "chunk_id": chunk_id, "passage": text}


class Evidence(BaseModel):
    """Retrieved passage with provenance (contract: retrieve_policy -> list[Evidence])."""

    document: str
    page: int = Field(ge=1)
    passage: str
    section: str | None = None
    # Optional extensions (backward compatible with contract v0).
    chunk_id: str | None = None
    score: float | None = None

    def to_http_evidence(self, max_passage_chars: int = 600) -> dict:
        """``POST /chat`` evidence item: citation plus the retrieved passage and chunk id."""
        return http_evidence(self.document, self.page, self.section, self.passage, self.chunk_id, max_passage_chars)


class DocumentInfo(BaseModel):
    """Per-document facts extracted *from the document text* (never invented)."""

    document: str
    title: str | None = None
    pages: int = 0
    code: str | None = Field(default=None, description="Document code found in the text, e.g. PRO-SAC-021")
    version: str | None = None
    references: list[str] = Field(
        default_factory=list, description="Other document codes cited in the text"
    )
    effective_date: str | None = Field(
        default=None, description="ISO date found in the text (e.g. 'vigente a partir del ...')"
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description="Raw phrases where the document says it replaces another one",
    )
    precedence_statements: list[str] = Field(
        default_factory=list,
        description="Raw phrases stating which document prevails in case of conflict",
    )
    empty_pages: list[int] = Field(
        default_factory=list, description="Pages without extractable text (possibly scanned)"
    )


class ConflictClaim(BaseModel):
    document: str
    page: int
    section: str | None = None
    sentence: str
    values: list[str] = Field(default_factory=list)


class PolicyConflict(BaseModel):
    """Two passages that appear to state incompatible rules on the same topic."""

    kind: Literal["numeric", "modality"]
    topic: str = Field(description="Unit or modality that differs, e.g. 'días'")
    claims: list[ConflictClaim]
    resolution: Literal["resolved_by_source", "unresolved"] = "unresolved"
    preferred_document: str | None = None
    basis: str = Field(
        default="",
        description="Why the conflict was (or was not) resolved; quotes source text when resolved",
    )


class RetrievalMetrics(BaseModel):
    retrieval_mode: RetrievalMode = "bm25"
    latency_ms: float = 0.0
    candidates: int = 0
    embedding_calls: int = 0
    embedding_input_tokens: int = 0
    embedding_latency_ms: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: list[str] = Field(default_factory=list)


class PolicyAnswer(BaseModel):
    """Full RAG answer: status, grounded answer text, evidence, conflicts, metrics."""

    question: str
    status: AnswerStatus
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[PolicyConflict] = Field(default_factory=list)
    coverage: float = Field(default=0.0, description="IDF-weighted share of query terms found")
    metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
