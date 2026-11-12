"""Policy index: build from PDFs, persist to disk, hybrid search (BM25 + optional embeddings)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bm25 import BM25Index
from .chunking import chunk_document
from .config import RagSettings
from .embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingUsage,
    cosine_similarities,
    validate_vectors,
)
from .extraction import PdfExtractionError, extract_pdf
from .models import Chunk, DocumentInfo, RetrievalMetrics
from .text import tokenize

logger = logging.getLogger(__name__)

INDEX_FILE = "index.json"
INDEX_VERSION = 1
RRF_K = 60
PREFIX_LEN = 5


class IndexNotFoundError(FileNotFoundError):
    pass


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    coverage: float
    bm25_score: float = 0.0
    embedding_similarity: float | None = None


def _index_text(chunk: Chunk) -> str:
    # Section titles are indexed together with the passage to boost topical matches.
    return f"{chunk.section or ''} {chunk.text}"


@dataclass
class PolicyIndex:
    chunks: list[Chunk]
    documents: dict[str, DocumentInfo]
    embeddings: list[list[float]] | None = None
    embed_model: str | None = None
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = field(default_factory=list)
    build_usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)

    def __post_init__(self) -> None:
        self.embed_dim: int | None = None
        if self.embeddings is not None:
            try:
                self.embed_dim = validate_vectors(self.embeddings, expected_count=len(self.chunks))
            except EmbeddingError as exc:
                self.warnings.append(f"Embeddings del índice inválidos, se usa solo BM25: {exc}")
                self.embeddings = None
                self.embed_model = None
        self._tokens = [tokenize(_index_text(c)) for c in self.chunks]
        self.bm25 = BM25Index(self._tokens)
        # prefix -> most frequent vocabulary term, to map unseen inflections
        # (e.g. "devolvi" -> "devolv") onto indexed terms.
        self._prefix: dict[str, str] = {}
        counts: dict[str, int] = {}
        for toks in self._tokens:
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        for term, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(term) >= PREFIX_LEN:
                self._prefix.setdefault(term[:PREFIX_LEN], term)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        pdf_paths: list[Path],
        settings: RagSettings | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> "PolicyIndex":
        settings = settings or RagSettings()
        chunks: list[Chunk] = []
        documents: dict[str, DocumentInfo] = {}
        warnings: list[str] = []
        seen: set[str] = set()
        for path in pdf_paths:
            if path.name in seen:
                warnings.append(f"Nombre de archivo duplicado ignorado: {path}")
                continue
            seen.add(path.name)
            try:
                extracted = extract_pdf(path)
            except PdfExtractionError as exc:
                warnings.append(str(exc))
                continue
            assert extracted.info is not None
            documents[extracted.document] = extracted.info
            if extracted.info.empty_pages:
                warnings.append(
                    f"{extracted.document}: páginas sin texto extraíble {extracted.info.empty_pages} "
                    "(¿escaneadas? requieren OCR, no indexadas)"
                )
            chunks.extend(
                chunk_document(
                    extracted,
                    max_words=settings.chunk_max_words,
                    overlap_words=settings.chunk_overlap_words,
                )
            )

        embeddings = None
        embed_model = None
        usage = EmbeddingUsage()
        if embedder is not None and chunks and settings.embeddings != "off":
            try:
                embeddings, usage = embedder.embed([_index_text(c) for c in chunks])
                validate_vectors(embeddings, expected_count=len(chunks))
                embed_model = embedder.model
            except EmbeddingError as exc:
                embeddings = None
                if settings.embeddings == "required":
                    raise
                warnings.append(f"Embeddings deshabilitados, se usa solo BM25: {exc}")
        for w in warnings:
            logger.warning(w)
        return cls(
            chunks=chunks,
            documents=documents,
            embeddings=embeddings,
            embed_model=embed_model,
            warnings=warnings,
            build_usage=usage,
        )

    # ---------------------------------------------------------- persistence
    def save(self, index_dir: str | Path) -> Path:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        # The index contains passages of internal PDFs: never let Git pick it up.
        (index_dir / ".gitignore").write_text("*\n", encoding="utf-8")
        payload = {
            "version": INDEX_VERSION,
            "built_at": self.built_at,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "warnings": self.warnings,
            "documents": {k: v.model_dump() for k, v in self.documents.items()},
            "chunks": [c.model_dump() for c in self.chunks],
            "embeddings": self.embeddings,
        }
        target = index_dir / INDEX_FILE
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, index_dir: str | Path) -> "PolicyIndex":
        target = Path(index_dir) / INDEX_FILE
        if not target.is_file():
            raise IndexNotFoundError(
                f"No existe el índice {target}. Ejecuta: python -m app.rag build --pdf-dir <carpeta>"
            )
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(f"Versión de índice incompatible: {payload.get('version')}")
        return cls(
            chunks=[Chunk(**c) for c in payload["chunks"]],
            documents={k: DocumentInfo(**v) for k, v in payload["documents"].items()},
            embeddings=payload.get("embeddings"),
            embed_model=payload.get("embed_model"),
            built_at=payload.get("built_at", ""),
            warnings=payload.get("warnings", []),
        )

    # --------------------------------------------------------------- search
    def query_tokens(self, query: str) -> list[str]:
        out = []
        for t in tokenize(query):
            if t not in self.bm25.idf and len(t) >= PREFIX_LEN:
                t = self._prefix.get(t[:PREFIX_LEN], t)
            out.append(t)
        return out

    def search(
        self,
        query: str,
        top_k: int = 5,
        embedder: EmbeddingProvider | None = None,
        candidate_pool: int = 50,
    ) -> tuple[list[SearchHit], RetrievalMetrics]:
        t0 = time.perf_counter()
        metrics = RetrievalMetrics()
        q_tokens = self.query_tokens(query)
        bm25_hits = self.bm25.search(q_tokens, candidate_pool)
        bm25_scores = dict(bm25_hits)

        sims: list[float] | None = None
        if self.embeddings is not None and embedder is not None:
            if embedder.model != self.embed_model:
                metrics.warnings.append(
                    f"Modelo de embeddings distinto al del índice ({embedder.model} != {self.embed_model}); solo BM25"
                )
            else:
                try:
                    vectors, usage = embedder.embed([query])
                    metrics.embedding_calls = usage.calls
                    metrics.embedding_input_tokens = usage.input_tokens
                    metrics.embedding_latency_ms = round(usage.latency_ms, 2)
                    validate_vectors(vectors, expected_count=1, expected_dim=self.embed_dim)
                    sims = cosine_similarities(vectors[0], self.embeddings)
                except (EmbeddingError, ValueError) as exc:
                    sims = None
                    metrics.warnings.append(
                        f"Embeddings no disponibles o incompatibles en consulta, solo BM25: {exc}"
                    )
        elif self.embeddings is not None:
            metrics.warnings.append("Índice con embeddings pero sin proveedor en consulta; solo BM25")

        if sims is None:
            metrics.retrieval_mode = "bm25"
            fused = {i: s for i, s in bm25_hits}
        else:
            metrics.retrieval_mode = "hybrid"
            emb_rank = sorted(range(len(sims)), key=lambda i: (-sims[i], i))[:candidate_pool]
            fused = {}
            for rank, (i, _) in enumerate(bm25_hits):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, i in enumerate(emb_rank):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        hits = [
            SearchHit(
                chunk=self.chunks[i],
                score=round(score, 6),
                coverage=round(self.bm25.coverage(q_tokens, i), 4),
                bm25_score=round(bm25_scores.get(i, 0.0), 4),
                embedding_similarity=None if sims is None else round(sims[i], 4),
            )
            for i, score in ordered
        ]
        metrics.candidates = len(fused)
        metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return hits, metrics
