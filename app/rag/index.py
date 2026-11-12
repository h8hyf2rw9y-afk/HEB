"""Policy index: build from PDFs, persist to disk, hybrid search (BM25 + optional embeddings)."""

from __future__ import annotations

import json
import re
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
from .text import normalize, split_sentences, tokenize

logger = logging.getLogger(__name__)

INDEX_FILE = "index.json"
INDEX_VERSION = 2  # v2: table rows, chunk kind/label, document code/version/references
COMPATIBLE_VERSIONS = {1, 2}
RRF_K = 60
PREFIX_LEN = 5
HISTORY_KINDS = {"metadata_row", "changelog_row"}
HISTORY_TERMS = set(tokenize(
    "version versiones vigencia vigente actualizacion actualizo actualizada modificacion "
    "modifico historial emision codigo responsable clasificacion autorizo"
))
# A history question must also name a document ("vigencia de la política ..."),
# so "vigencia de un vale de tienda" does not open the cover-metadata rows.
DOCUMENT_WORDS = ("politica", "manual", "procedimiento", "documento", "faq", "preguntas frecuentes", "lineamiento")
DEFINITIONAL_RE = re.compile(r"\bqu[eé]\s+(?:es|son|significa|significan|quiere decir)\b", re.IGNORECASE)
DEFINITION_BOOST = 2.0
EXPANSION_WEIGHT = 0.5  # definition-derived terms count half as much as literal ones


class IndexNotFoundError(FileNotFoundError):
    pass


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    coverage: float
    bm25_score: float = 0.0
    embedding_similarity: float | None = None


_DEFINITION_ITEM = re.compile(r"^\s*([^:•]{3,80}):\s*(.+)$")
_EXAMPLES = re.compile(r"(?:tales como|por ejemplo|como|:)\s+([^.;]+)", re.IGNORECASE)
_LIST_SPLIT = re.compile(r",|\s+y\s+|\s+o\s+")


def definition_expansions(
    chunks: list[Chunk], bm25: BM25Index | None = None, max_df_ratio: float = 0.05
) -> dict[str, set[str]]:
    """Map example words to the term that the documents define with them.

    Only sections titled "Definiciones" are used and only explicit example lists
    ("tales como A, B y C", "por ejemplo ...", "...: A, B, C"). Example items of
    more than four words are ignored, and only example words that are rare in the
    corpus (document frequency <= ``max_df_ratio``) are expanded, so generic words
    such as "producto" never trigger an expansion. Nothing comes from outside the
    corpus.
    """
    out: dict[str, set[str]] = {}
    max_df = max(1, int(max_df_ratio * len(chunks)))

    def rare(token: str) -> bool:
        if bm25 is None:
            return True
        return sum(1 for f in bm25.doc_freqs if token in f) <= max_df

    for chunk in chunks:
        if chunk.kind != "text" or "definicion" not in normalize(chunk.section or ""):
            continue
        for item in split_sentences(chunk.text.replace("•", " ")):
            m = _DEFINITION_ITEM.match(item)
            if not m:
                continue
            term_tokens = set(tokenize(m.group(1)))
            if not term_tokens:
                continue
            for ex in _EXAMPLES.finditer(m.group(2)):
                for example in _LIST_SPLIT.split(ex.group(1)):
                    words = tokenize(example)
                    if 0 < len(example.split()) <= 4:
                        for w in words:
                            if w not in term_tokens and rare(w):
                                out.setdefault(w, set()).update(term_tokens)
    return out


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
        self.expansions = definition_expansions(self.chunks, self.bm25)
        self._by_prefix: dict[str, set[str]] = {}
        for term in counts:
            if len(term) >= PREFIX_LEN and not term[:1].isdigit():
                self._by_prefix.setdefault(term[:PREFIX_LEN], set()).add(term)
        regular = {t for toks, c in zip(self._tokens, self.chunks) if c.kind not in HISTORY_KINDS for t in toks}
        self._history_only_terms = {t for toks in self._tokens for t in toks} - regular
        self._definition_chunks = {
            i for i, c in enumerate(self.chunks) if "definicion" in normalize(c.section or "")
        }

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
        version = payload.get("version")
        if version not in COMPATIBLE_VERSIONS:
            raise ValueError(f"Versión de índice incompatible: {version}")
        warnings = list(payload.get("warnings", []))
        if version != INDEX_VERSION:
            warnings.append(
                f"Índice v{version} construido con un extractor anterior (sin filas de tabla ni "
                "metadatos de documento); reconstrúyelo con: python -m app.rag build"
            )
        return cls(
            chunks=[Chunk(**c) for c in payload["chunks"]],
            documents={k: DocumentInfo(**v) for k, v in payload["documents"].items()},
            embeddings=payload.get("embeddings"),
            embed_model=payload.get("embed_model"),
            built_at=payload.get("built_at", ""),
            warnings=warnings,
        )

    # --------------------------------------------------------------- search
    def query_tokens(self, query: str) -> list[str]:
        out = []
        for t in tokenize(query):
            if t not in self.bm25.idf and len(t) >= PREFIX_LEN:
                t = self._prefix.get(t[:PREFIX_LEN], t)
            out.append(t)
        return out

    def query_alternatives(self, q_tokens: list[str]) -> dict[str, set[str]]:
        """Soft matches per query term: corpus-defined expansions plus indexed
        terms sharing the first 5 characters ("devolv" ~ "devolu")."""
        out: dict[str, set[str]] = {}
        for t in q_tokens:
            alts = set(self.expansions.get(t, ()))
            if len(t) >= PREFIX_LEN and not t[:1].isdigit():
                alts |= self._by_prefix.get(t[:PREFIX_LEN], set()) - {t}
            if alts:
                out[t] = alts
        return out

    def history_question(self, query: str, q_tokens: list[str]) -> bool:
        norm_query = normalize(query)
        return bool(set(q_tokens) & HISTORY_TERMS) and any(w in norm_query for w in DOCUMENT_WORDS)

    def unseen_terms(self, query: str, q_tokens: list[str]) -> set[str]:
        """Terms that only occur in chunks excluded from this query's search."""
        return set() if self.history_question(query, q_tokens) else self._history_only_terms


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
        # Expand with terms the documents themselves define ("Electrodomésticos ...:
        # tales como licuadoras ..."), so a question about "licuadora" also reaches
        # the row "Electrodomésticos y electrónicos menores | Plazo: 30 días".
        alternatives = self.query_alternatives(q_tokens)
        expanded = q_tokens + sorted({x for alts in alternatives.values() for x in alts} - set(q_tokens))
        extra = set(expanded) - set(q_tokens)
        bm25_hits = self.bm25.search(
            expanded, candidate_pool, {t: EXPANSION_WEIGHT for t in extra} if extra else None
        )
        # Cover metadata and change-log rows only answer questions about document
        # history (version, vigencia, actualización...). Otherwise their dates
        # ("1 de agosto de 2023") would match unrelated questions ("ventas de agosto").
        history_ok = self.history_question(query, q_tokens)

        def allowed(i: int) -> bool:
            return history_ok or self.chunks[i].kind not in HISTORY_KINDS

        bm25_hits = [(i, sc) for i, sc in bm25_hits if allowed(i)]
        unseen = self.unseen_terms(query, q_tokens)
        # "¿Qué es SIGMA?": prefer the documents' own "Definiciones" sections.
        if DEFINITIONAL_RE.search(query) and self._definition_chunks:
            bm25_hits = sorted(
                ((i, sc * DEFINITION_BOOST if i in self._definition_chunks else sc) for i, sc in bm25_hits),
                key=lambda kv: (-kv[1], kv[0]),
            )
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
            emb_rank = [
                i for i in sorted(range(len(sims)), key=lambda i: (-sims[i], i)) if allowed(i)
            ][:candidate_pool]
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
                coverage=round(self.bm25.coverage(q_tokens, i, alternatives, unseen, EXPANSION_WEIGHT), 4),
                bm25_score=round(bm25_scores.get(i, 0.0), 4),
                embedding_similarity=None if sims is None else round(sims[i], 4),
            )
            for i, score in ordered
        ]
        metrics.candidates = len(fused)
        metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return hits, metrics
