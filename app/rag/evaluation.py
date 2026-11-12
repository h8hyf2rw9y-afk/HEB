"""Retrieval evaluation: hit@k, MRR, citations, abstention, conflicts, latency, tokens.

Cases are JSON objects (see ``tests/rag/fixtures/*.json``). Fields beyond
``id/question/expect`` are optional, so older case files keep working.

Cases marked ``known_limitation`` are evaluated and reported, but their failures
are listed separately (``known_limitation_failures``) instead of in ``failures``.
They document behaviour that is known to be wrong, not hidden.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Evidence
from .service import PolicyRetriever
from .text import normalize

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]


class EvalCase(BaseModel):
    id: str
    question: str
    expect: str = Field(description="answered | insufficient_data | needs_clarification")
    category: str = "general"
    # Any (document, page) pair listed here counts as a correct hit.
    relevant: list[tuple[str, int]] = Field(default_factory=list)
    # Substring (accent/case-insensitive) of the section of a correct hit.
    expected_section: str | None = None
    # Phrases (verified by hand in the PDFs) that must appear in the answer text.
    expected_phrases: list[str] = Field(default_factory=list)
    expect_conflict: bool | None = None
    expect_preferred: str | None = None
    known_limitation: bool = False
    note: str | None = None


class CategoryReport(BaseModel):
    cases: int
    passed: int


class EvalReport(BaseModel):
    cases: int
    retrieval_mode: str = "bm25"
    hit_at_k: float
    mrr: float
    citation_accuracy: float | None = None
    citation_verified: float | None = None
    phrase_accuracy: float | None = None
    abstention_accuracy: float
    false_abstention_rate: float | None = None
    status_accuracy: float
    conflict_accuracy: float | None
    latency_ms_p50: float
    latency_ms_p95: float | None = None
    latency_ms_max: float
    embedding_input_tokens: int
    llm_tokens: int
    passed_cases: int | None = None
    by_category: dict[str, CategoryReport] = Field(default_factory=dict)
    failures: list[str]
    known_limitation_failures: list[str] = Field(default_factory=list)


def load_cases(path: str | Path) -> list[EvalCase]:
    return [EvalCase(**c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


class CitationVerifier:
    """Checks that a cited passage really is on the cited page of the PDF.

    Every word of the passage must occur on that page (table rows are rebuilt as
    "Header: value | ...", so word presence rather than exact substring is used),
    and the section heading words must occur in the document.
    """

    def __init__(self, pdf_dir: str | Path) -> None:
        self.pdf_dir = Path(pdf_dir)
        self._pages: dict[str, list[set[str]]] = {}

    def _doc(self, name: str) -> list[set[str]] | None:
        if name not in self._pages:
            path = self.pdf_dir / name
            if not path.is_file():
                return None
            with pymupdf.open(path) as doc:
                self._pages[name] = [set(_words(p.get_text())) for p in doc]
        return self._pages[name]

    def verify(self, e: Evidence) -> bool:
        pages = self._doc(e.document)
        if pages is None or not 1 <= e.page <= len(pages):
            return False
        page_words = pages[e.page - 1]
        # Words split across lines in narrow table cells ("Priori-dad") may be
        # glued in the passage; accept them if both halves are on the page.
        missing = [w for w in _words(e.passage) if w not in page_words]
        if any(not any(w.startswith(p) and w[len(p):] in page_words for p in page_words) for w in missing):
            return False
        if e.section:
            doc_words = set().union(*pages)
            return all(w in doc_words or len(w) <= 2 for w in _words(e.section))
        return True


def evaluate(
    retriever: PolicyRetriever,
    cases: list[EvalCase],
    top_k: int = 5,
    pdf_dir: str | Path | None = None,
) -> EvalReport:
    verifier = CitationVerifier(pdf_dir) if pdf_dir else None
    hits, rr, latencies, failures, known_failures = [], [], [], [], []
    citation_ok, verified, phrases_ok = [], [], []
    status_ok = abst_ok = abst_n = false_abst = pos_n = 0
    conflict_ok = conflict_n = 0
    emb_tokens = llm_tokens = 0
    modes: set[str] = set()
    by_cat: dict[str, list[bool]] = defaultdict(list)

    for case in cases:
        ans = retriever.answer(case.question, top_k)
        modes.add(ans.metrics.retrieval_mode)
        latencies.append(ans.metrics.latency_ms)
        emb_tokens += ans.metrics.embedding_input_tokens
        llm_tokens += ans.metrics.input_tokens + ans.metrics.output_tokens
        problems: list[str] = []

        if ans.status == case.expect:
            status_ok += 1
        else:
            problems.append(f"status {ans.status} != {case.expect}")
        if case.expect == "insufficient_data":
            abst_n += 1
            abst_ok += ans.status == "insufficient_data"
        elif case.expect == "answered":
            pos_n += 1
            false_abst += ans.status == "insufficient_data"

        if case.relevant:
            wanted = {(d, p) for d, p in case.relevant}
            rank = next((i for i, e in enumerate(ans.evidence, 1) if (e.document, e.page) in wanted), None)
            hits.append(1.0 if rank else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
            if not rank:
                problems.append(f"no relevant passage in top-{top_k}")
            if case.expected_section is not None:
                sec = normalize(case.expected_section)
                ok = any(
                    (e.document, e.page) in wanted and sec in normalize(e.section or "")
                    for e in ans.evidence
                )
                citation_ok.append(ok)
                if not ok:
                    problems.append(f"no citation with section ~ '{case.expected_section}'")
        if case.expected_phrases:
            text = normalize(ans.answer)
            missing = [p for p in case.expected_phrases if normalize(p) not in text]
            phrases_ok.append(not missing)
            if missing:
                problems.append(f"answer lacks {missing}")
        if verifier is not None:
            for e in ans.evidence:
                ok = verifier.verify(e)
                verified.append(ok)
                if not ok:
                    problems.append(f"unverifiable citation {e.document} p.{e.page}")
        if case.expect_conflict is not None:
            conflict_n += 1
            ok = bool(ans.conflicts) == case.expect_conflict
            if ok and case.expect_preferred is not None:
                ok = any(c.preferred_document == case.expect_preferred for c in ans.conflicts)
            conflict_ok += ok
            if not ok:
                got = [(c.topic, c.resolution, c.preferred_document) for c in ans.conflicts]
                problems.append(f"conflict expectation not met: {got}")

        by_cat[case.category].append(not problems)
        for p in problems:
            (known_failures if case.known_limitation else failures).append(f"{case.id}: {p}")

    def avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    def rate(xs: list[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    p95 = None
    if len(latencies) >= 2:
        p95 = round(statistics.quantiles(latencies, n=20, method="inclusive")[18], 2)
    return EvalReport(
        cases=len(cases),
        retrieval_mode="+".join(sorted(modes)) or "bm25",
        hit_at_k=avg(hits),
        mrr=avg(rr),
        citation_accuracy=rate(citation_ok),
        citation_verified=rate(verified),
        phrase_accuracy=rate(phrases_ok),
        abstention_accuracy=round(abst_ok / abst_n, 4) if abst_n else 1.0,
        false_abstention_rate=round(false_abst / pos_n, 4) if pos_n else None,
        status_accuracy=round(status_ok / len(cases), 4) if cases else 0.0,
        conflict_accuracy=round(conflict_ok / conflict_n, 4) if conflict_n else None,
        latency_ms_p50=round(statistics.median(latencies), 2) if latencies else 0.0,
        latency_ms_p95=p95,
        latency_ms_max=round(max(latencies), 2) if latencies else 0.0,
        embedding_input_tokens=emb_tokens,
        llm_tokens=llm_tokens,
        passed_cases=sum(ok for oks in by_cat.values() for ok in oks),
        by_category={k: CategoryReport(cases=len(v), passed=sum(v)) for k, v in sorted(by_cat.items())},
        failures=failures,
        known_limitation_failures=known_failures,
    )
