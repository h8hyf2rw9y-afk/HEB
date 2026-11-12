"""Public entry points of the RAG module.

Contract (docs/CONTRACTS.md): ``retrieve_policy(query: str, top_k: int) -> list[Evidence]``.
``answer_policy_question`` adds status, conflicts, abstention and metrics, without
calling an LLM (extractive answer). LLM generation belongs to the shared
orchestration layer; see ``app.rag.prompts.build_grounded_prompt``.
"""

from __future__ import annotations

import threading
import time

from .config import RagSettings
from .conflicts import topic_tokens, detect_conflicts
from .embeddings import EmbeddingProvider, OllamaEmbedder
from .index import PolicyIndex, SearchHit
from .models import Evidence, PolicyAnswer, PolicyConflict
from .text import split_sentences

INSUFFICIENT_MSG = (
    "No encontré información suficiente en los documentos de políticas disponibles para "
    "responder esta pregunta. No se infiere ninguna política que no esté en los documentos."
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

        conflicts = detect_conflicts(question, relevant, self.index.documents)
        evidence = [_to_evidence(h) for h in relevant]
        answer_text = compose_extractive_answer(question, evidence, conflicts)
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


def _best_sentence(question: str, passage: str) -> str:
    q = topic_tokens(question)
    sentences = split_sentences(passage) or [passage]
    return max(sentences, key=lambda s: (len(topic_tokens(s) & q), -sentences.index(s)))


def compose_extractive_answer(
    question: str, evidence: list[Evidence], conflicts: list[PolicyConflict], max_points: int = 3
) -> str:
    """Grounded answer built only from retrieved sentences, each with its citation."""
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
        sentence = _best_sentence(question, e.passage)
        if sentence in seen:
            continue
        seen.add(sentence)
        note = " (documento reemplazado según la fuente)" if e.document in superseded else ""
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
    global _default
    with _lock:
        if _default is None or settings is not None:
            _default = PolicyRetriever.from_settings(settings)
        return _default


def retrieve_policy(query: str, top_k: int = 5) -> list[Evidence]:
    return get_retriever().retrieve(query, top_k)


def answer_policy_question(question: str, top_k: int = 5) -> PolicyAnswer:
    return get_retriever().answer(question, top_k)
