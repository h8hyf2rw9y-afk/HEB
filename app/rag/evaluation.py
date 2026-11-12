"""Retrieval evaluation: hit@k, MRR, abstention accuracy, conflict detection, latency."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from .service import PolicyRetriever


class EvalCase(BaseModel):
    id: str
    question: str
    expect: str = Field(description="'answered' or 'insufficient_data'")
    # Any (document, page) pair listed here counts as a correct hit.
    relevant: list[tuple[str, int]] = Field(default_factory=list)
    expect_conflict: bool | None = None
    expect_preferred: str | None = None


class EvalReport(BaseModel):
    cases: int
    hit_at_k: float
    mrr: float
    abstention_accuracy: float
    status_accuracy: float
    conflict_accuracy: float | None
    latency_ms_p50: float
    latency_ms_max: float
    embedding_input_tokens: int
    llm_tokens: int
    failures: list[str]


def load_cases(path: str | Path) -> list[EvalCase]:
    return [EvalCase(**c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]


def evaluate(retriever: PolicyRetriever, cases: list[EvalCase], top_k: int = 5) -> EvalReport:
    hits, rr, latencies, failures = [], [], [], []
    status_ok = abst_ok = abst_n = 0
    conflict_ok = conflict_n = 0
    emb_tokens = llm_tokens = 0
    for case in cases:
        ans = retriever.answer(case.question, top_k)
        latencies.append(ans.metrics.latency_ms)
        emb_tokens += ans.metrics.embedding_input_tokens
        llm_tokens += ans.metrics.input_tokens + ans.metrics.output_tokens
        if ans.status == case.expect:
            status_ok += 1
        else:
            failures.append(f"{case.id}: status {ans.status} != {case.expect}")
        if case.expect == "insufficient_data":
            abst_n += 1
            abst_ok += ans.status == "insufficient_data"
        if case.relevant:
            wanted = {(d, p) for d, p in case.relevant}
            rank = next(
                (i for i, e in enumerate(ans.evidence, 1) if (e.document, e.page) in wanted), None
            )
            hits.append(1.0 if rank else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
            if not rank:
                failures.append(f"{case.id}: no relevant passage in top-{top_k}")
        if case.expect_conflict is not None:
            conflict_n += 1
            detected = bool(ans.conflicts)
            ok = detected == case.expect_conflict
            if ok and case.expect_preferred is not None:
                ok = any(c.preferred_document == case.expect_preferred for c in ans.conflicts)
            conflict_ok += ok
            if not ok:
                failures.append(f"{case.id}: conflict expectation not met ({ans.conflicts})")

    def avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return EvalReport(
        cases=len(cases),
        hit_at_k=avg(hits),
        mrr=avg(rr),
        abstention_accuracy=round(abst_ok / abst_n, 4) if abst_n else 1.0,
        status_accuracy=round(status_ok / len(cases), 4) if cases else 0.0,
        conflict_accuracy=round(conflict_ok / conflict_n, 4) if conflict_n else None,
        latency_ms_p50=round(statistics.median(latencies), 2) if latencies else 0.0,
        latency_ms_max=round(max(latencies), 2) if latencies else 0.0,
        embedding_input_tokens=emb_tokens,
        llm_tokens=llm_tokens,
        failures=failures,
    )
