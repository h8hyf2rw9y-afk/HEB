"""Public entry points of the RAG module.

Contract (docs/CONTRACTS.md): ``retrieve_policy(query: str, top_k: int) -> list[Evidence]``.
``answer_policy_question`` adds status, conflicts, abstention and metrics, without
calling an LLM (extractive answer). LLM generation belongs to the shared
orchestration layer; see ``app.rag.prompts.build_grounded_prompt``.
"""

from __future__ import annotations

import re
import threading
import time

from .config import RagSettings
from .answering import (
    PREFIX,
    analyse_question,
    assess_hit,
    compose_answer,
    prune_table_siblings,
    rerank,
    surface_words,
    table_entity_gaps,
)
from .conflicts import _SUBORDINATE_RE, _is_normative, detect_conflicts, topic_tokens
from .embeddings import EmbeddingProvider, OllamaEmbedder
from .index import DEFINITIONAL_RE, PolicyIndex, SearchHit
from .models import Evidence, PolicyAnswer, RetrievalMetrics
from .text import normalize

# Sales figures: a sales verb or plural "ventas" together with a quantity or
# ranking word. Singular "venta" is excluded on purpose: policies use it in
# "piso de venta", "venta neta", "último día de venta".
_SALES = r"(?:vend(?:i|io|ieron|imos|e|en|emos)\w*|ventas)"
_QUANT = r"(?:cuant[oa]s?|mas|menos|total\w*|mayor|menor)"
_DATA_QUESTION_RE = re.compile(rf"\b{_SALES}\b[^?]*\b{_QUANT}\b|\b{_QUANT}\b[^?]*\b{_SALES}\b")
INSUFFICIENT_MSG = (
    "No encontré información suficiente en los documentos de políticas disponibles para "
    "responder esta pregunta. No se infiere ninguna política que no esté en los documentos."
)
CLARIFY_OPTIONS_MSG = (
    "Tu pregunta es muy general y los documentos tratan varios temas distintos. "
    "¿A cuál te refieres?"
)
CLARIFY_MSG = "La pregunta está vacía o no contiene términos suficientes. ¿Puedes reformularla?"


def _to_evidence(hit: SearchHit) -> Evidence:
    c = hit.chunk
    return Evidence(
        document=c.document,
        page=c.page,
        passage=c.text,
        section=c.section,
        chunk_id=c.chunk_id,
        score=hit.score,
    )


class PolicyRetriever:
    def __init__(
        self,
        index: PolicyIndex,
        settings: RagSettings | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.index = index
        self.settings = settings or RagSettings()
        self.embedder = embedder
        self._prefixes4: set[str] | None = None

    @classmethod
    def from_settings(cls, settings: RagSettings | None = None) -> "PolicyRetriever":
        settings = settings or RagSettings.from_env()
        index = PolicyIndex.load(settings.index_dir)
        embedder = None
        if settings.embeddings != "off" and index.embed_model:
            embedder = OllamaEmbedder(
                model=settings.embed_model,
                base_url=settings.ollama_base_url,
                timeout_s=settings.ollama_timeout_s,
            )
        return cls(index, settings, embedder)

    # -------------------------------------------------------------------
    def _known_prefix(self, prefix: str) -> bool:
        if self._prefixes4 is None:
            self._prefixes4 = {t[:4] for t in self.index.bm25.idf}
        return prefix in self._prefixes4

    def _is_relevant(self, hit: SearchHit) -> bool:
        if hit.coverage >= self.settings.min_coverage:
            return True
        return (
            hit.embedding_similarity is not None
            and hit.embedding_similarity >= self.settings.min_embedding_similarity
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        """Contract function. Returns an empty list when evidence is insufficient."""
        return self.answer(query, top_k).evidence

    def answer(self, question: str, top_k: int | None = None) -> PolicyAnswer:
        t0 = time.perf_counter()
        top_k = self.settings.top_k if top_k is None else top_k
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k debe estar entre 1 y 50")
        question = (question or "").strip()
        if not question or not self.index.query_tokens(question):
            return PolicyAnswer(question=question, status="needs_clarification", answer=CLARIFY_MSG)

        # Operational-data questions ("¿Qué tienda vendió más en agosto?") ask for
        # figures that policy documents do not contain; they belong to the
        # structured-data tools, so the RAG abstains instead of matching words.
        if _DATA_QUESTION_RE.search(normalize(question)):
            return PolicyAnswer(
                question=question,
                status="insufficient_data",
                answer=INSUFFICIENT_MSG + "\nℹ️ Parece una consulta de datos de ventas; "
                "los documentos de políticas no contienen cifras de ventas.",
                metrics=RetrievalMetrics(warnings=["Consulta de datos de ventas; fuera del alcance del RAG"]),
            )

        # Retrieve a wider pool, then keep and order passages by how directly they answer.
        pool = max(top_k * 3, 15)
        hits, metrics = self.index.search(question, top_k=pool, embedder=self.embedder)
        coverage = max((h.coverage for h in hits), default=0.0)
        q = analyse_question(question)
        q_tokens = self.index.query_tokens(question)
        alternatives = self.index.query_alternatives(q_tokens)
        extra = {x for alts in alternatives.values() for x in alts}
        q_pfx = {t[:PREFIX] for t in q.terms | extra}
        weight = self.index.bm25.term_idf
        top_score = max((h.score for h in hits), default=0.0)
        assessed = [assess_hit(q, h, q_pfx, top_score) for h in hits]
        relevant_a = [
            a for a in assessed
            if self._is_relevant(a.hit) or (a.entity_match and a.attribute_match)
        ]

        def insufficient(reason: str | None = None) -> PolicyAnswer:
            metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if reason:
                metrics.warnings.append(reason)
            return PolicyAnswer(
                question=question,
                status="insufficient_data",
                answer=INSUFFICIENT_MSG + (f"\nℹ️ {reason}" if reason else ""),
                coverage=coverage,
                metrics=metrics,
            )

        if not relevant_a:
            return insufficient()

        words = surface_words(question)
        # A table that matches the asked attribute but not the asked entity.
        for gap in table_entity_gaps(q, relevant_a, weight):
            gap_rows = {
                a.hit.chunk.chunk_id for a in relevant_a
                if a.hit.chunk.kind == "table_row"
                and (a.hit.chunk.document, a.hit.chunk.section) == (gap.document, gap.section)
            }
            relevant_a = [a for a in relevant_a if a.hit.chunk.chunk_id not in gap_rows]
            others_cover = any(
                {t[:PREFIX] for t in topic_tokens(a.hit.chunk.text)} & {t[:PREFIX] for t in gap.missing}
                for a in relevant_a
            )
            if not others_cover:
                asked = ", ".join(sorted(words.get(t, t) for t in gap.missing))
                listed = ", ".join(lbl for lbl in gap.labels if lbl)
                return insufficient(
                    f"La tabla «{gap.section}» ({gap.document}, p. {gap.page}) no incluye «{asked}»; "
                    f"incluye: {listed}."
                )
        if not relevant_a:
            return insufficient()

        # The asked attribute ("¿Cuál es el *precio* de ...?") must be in the evidence.
        if q.focus:
            focus_pfx = {t[:PREFIX] for t in q.focus}
            if not any({t[:PREFIX] for t in topic_tokens(a.hit.chunk.text)} & focus_pfx for a in relevant_a):
                asked = ", ".join(sorted(words.get(t, t) for t in q.focus))
                return insufficient(f"Los documentos no indican «{asked}» para lo que se pregunta.")

        ranked = rerank(prune_table_siblings(relevant_a))[:top_k]
        relevant = [a.hit for a in ranked]
        conflicts = detect_conflicts(question, relevant, self.index.documents, extra_terms=extra)
        evidence = [_to_evidence(h) for h in relevant]
        options = _ambiguity_options(question, q_tokens, relevant)
        if options:
            metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return PolicyAnswer(
                question=question,
                status="needs_clarification",
                answer=CLARIFY_OPTIONS_MSG + "\n" + "\n".join(f"- {o}" for o in options),
                evidence=evidence,
                conflicts=conflicts,
                coverage=coverage,
                metrics=metrics,
            )
        answer_text = compose_answer(
            q,
            ranked,
            {e.chunk_id: e for e in evidence},
            conflicts,
            weight,
            known=lambda t: t in self.index.bm25.idf,
            known_prefix=self._known_prefix,
            normative_docs={d for d, info in self.index.documents.items() if _is_normative(info)},
            subordinate_docs={
                d for d, info in self.index.documents.items()
                if any(_SUBORDINATE_RE.search(normalize(s)) for s in info.precedence_statements)
            },
            original_words=words,
        )
        metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return PolicyAnswer(
            question=question,
            status="answered",
            answer=answer_text,
            evidence=evidence,
            conflicts=conflicts,
            coverage=coverage,
            metrics=metrics,
        )


def _ambiguity_options(question: str, q_tokens: list[str], relevant: list[SearchHit]) -> list[str]:
    """A one-term question ("¿Cuál es el plazo?") whose evidence covers three or
    more different subjects is ambiguous: ask which one is meant instead of
    picking one. Definitional questions ("¿Qué es SIGMA?") are not ambiguous."""
    if len(set(q_tokens)) != 1 or DEFINITIONAL_RE.search(question):
        return []
    subjects: list[str] = []
    for h in relevant:
        c = h.chunk
        subject = f"{c.label} — {c.section}" if c.label else (c.section or c.document)
        subject = f"{subject} ({c.document}, p. {c.page})"
        if subject not in subjects:
            subjects.append(subject)
    return subjects if len(subjects) >= 3 else []


# ---------------------------------------------------------------------------
# Module-level convenience API (lazy, thread-safe singleton built from env vars).
# ---------------------------------------------------------------------------
_default: PolicyRetriever | None = None
_lock = threading.Lock()


def get_retriever(settings: RagSettings | None = None) -> PolicyRetriever:
    """Return the shared retriever (built once from env vars).

    Passing explicit ``settings`` returns a *separate*, uncached retriever and never
    replaces the shared one, so temporary configurations (health checks, tests,
    CLI) cannot change the index used by later requests.
    """
    global _default
    if settings is not None:
        return PolicyRetriever.from_settings(settings)
    with _lock:
        if _default is None:
            _default = PolicyRetriever.from_settings(None)
        return _default


def reset_retriever() -> None:
    """Drop the shared retriever (e.g. after rebuilding the index on disk)."""
    global _default
    with _lock:
        _default = None


def retrieve_policy(query: str, top_k: int = 5) -> list[Evidence]:
    return get_retriever().retrieve(query, top_k)


def answer_policy_question(question: str, top_k: int = 5) -> PolicyAnswer:
    return get_retriever().answer(question, top_k)
