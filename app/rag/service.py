"""Public entry points of the RAG module.

Contract (docs/CONTRACTS.md): ``retrieve_policy(query: str, top_k: int) -> list[Evidence]``.
``answer_policy_question`` adds status, conflicts, abstention and metrics, without
calling an LLM (extractive answer). LLM generation belongs to the shared
orchestration layer; see ``app.rag.prompts.build_grounded_prompt``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .config import RagSettings
from .conflicts import topic_tokens, detect_conflicts
from .embeddings import EmbeddingProvider, OllamaEmbedder
from .index import DEFINITIONAL_RE, PolicyIndex, SearchHit
from .models import Evidence, PolicyAnswer, PolicyConflict
from .text import split_sentences

INSUFFICIENT_MSG = (
    "No encontré información suficiente en los documentos de políticas disponibles para "
    "responder esta pregunta. No se infiere ninguna política que no esté en los documentos."
)
CLARIFY_OPTIONS_MSG = (
    "Tu pregunta es muy general y los documentos tratan varios temas distintos. "
    "¿A cuál te refieres?"
)
CLARIFY_MSG = "La pregunta está vacía o no contiene términos suficientes. ¿Puedes reformularla?"


def _cite(e: Evidence) -> str:
    ref = f"{e.document}, p. {e.page}"
    return f"[{ref}, sección: {e.section}]" if e.section else f"[{ref}]"


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

        hits, metrics = self.index.search(question, top_k=top_k, embedder=self.embedder)
        relevant = [h for h in hits if self._is_relevant(h)]
        coverage = max((h.coverage for h in hits), default=0.0)
        if not relevant:
            metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return PolicyAnswer(
                question=question,
                status="insufficient_data",
                answer=INSUFFICIENT_MSG,
                coverage=coverage,
                metrics=metrics,
            )

        q_tokens = self.index.query_tokens(question)
        alternatives = self.index.query_alternatives(q_tokens)
        extra = {x for alts in alternatives.values() for x in alts}
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
        answer_text = compose_extractive_answer(
            question,
            evidence,
            conflicts,
            atomic_ids={h.chunk.chunk_id for h in relevant if h.chunk.kind != "text"},
            weight=self.index.bm25.term_idf,
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


def _best_sentence(
    question: str, passage: str, weight: Callable[[str], float] | None = None
) -> str:
    """Sentence of ``passage`` sharing the most (IDF-weighted) topic terms with the question."""
    q = topic_tokens(question)
    w = weight or (lambda _t: 1.0)
    sentences = split_sentences(passage) or [passage]
    return max(
        enumerate(sentences),
        key=lambda it: (sum(w(t) for t in topic_tokens(it[1]) & q), -it[0]),
    )[1]


def _sentences(passage: str) -> list[str]:
    """Sentences, with very short ones ("No.", "Sí.") merged into the next one."""
    out: list[str] = []
    carry = ""
    for s in split_sentences(passage) or [passage]:
        if len(s.split()) < 4:
            carry = f"{carry} {s}".strip()
            continue
        out.append(f"{carry} {s}".strip() if carry else s)
        carry = ""
    if carry:
        out.append(carry)
    return out


def _primary_excerpt(question: str, e: Evidence, weight: Callable[[str], float] | None, limit: int = 3) -> str:
    """Up to ``limit`` sentences of the top passage, in document order: its lead
    sentence when the section heading matches the question (FAQ entries answer
    their heading first) plus the best-scoring sentences."""
    q = topic_tokens(question)
    w = weight or (lambda _t: 1.0)
    sents = _sentences(e.passage)
    scored = [(sum(w(t) for t in topic_tokens(s) & q), i) for i, s in enumerate(sents)]
    chosen = {i for sc, i in sorted(scored, key=lambda x: (-x[0], x[1]))[: limit - 1] if sc > 0}
    heading = topic_tokens(e.section or "")
    if q and len(heading & q) / len(q) >= 0.5:
        chosen.add(0)
    if not chosen:
        chosen = {max(scored)[1]} if scored else {0}
    return " ".join(sents[i] for i in sorted(chosen)[:limit])


def compose_extractive_answer(
    question: str,
    evidence: list[Evidence],
    conflicts: list[PolicyConflict],
    max_points: int = 3,
    *,
    atomic_ids: set[str] | None = None,
    weight: Callable[[str], float] | None = None,
) -> str:
    """Grounded answer built only from retrieved sentences, each with its citation.

    Passages whose chunk id is in ``atomic_ids`` (table rows) are quoted whole.
    """
    atomic_ids = atomic_ids or set()
    preferred = {c.preferred_document for c in conflicts if c.preferred_document}
    superseded = {
        claim.document
        for c in conflicts
        if c.preferred_document
        for claim in c.claims
        if claim.document != c.preferred_document
    }
    ordered = sorted(
        evidence,
        key=lambda e: (e.document in superseded, e.document not in preferred),
    )
    lines = ["Según los documentos de políticas:"]
    seen: set[str] = set()
    for e in ordered:
        if len(seen) >= max_points:
            break
        if e.chunk_id in atomic_ids:
            sentence = e.passage
        elif not seen:  # primary passage: lead sentence + best supporting sentences
            sentence = _primary_excerpt(question, e, weight)
        else:
            sentence = _best_sentence(question, e.passage, weight)
        if sentence in seen:
            continue
        seen.add(sentence)
        note = " (no prevalece según la fuente; ver conflicto)" if e.document in superseded else ""
        lines.append(f"- {sentence} {_cite(e)}{note}")

    for c in conflicts:
        claims = "; ".join(
            f"{cl.document} p. {cl.page}: {', '.join(cl.values) or cl.sentence}" for cl in c.claims
        )
        if c.resolution == "resolved_by_source":
            lines.append(
                f"⚠️ Política contradictoria ({c.topic}): {claims}. Prevalece {c.preferred_document}. {c.basis}"
            )
        else:
            lines.append(
                f"⚠️ Política contradictoria sin resolver ({c.topic}): {claims}. {c.basis} "
                "Se recomienda confirmar con el área responsable."
            )
    return "\n".join(lines)


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
