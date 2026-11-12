"""FastAPI entry points for structured queries and grounded policy answers."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data.catalog import build_catalog
from app.rag.interface import PolicySearchRequest, rag_health, search_policies
from app.tools.structured import StructuredRequest, query_dataset

app = FastAPI(title="HEB AI Assistant", version="0.2.0")
CATALOG = build_catalog(Path("data/raw"))

# Explicit, conservative routing: structured sales questions are not sent to policy RAG.
_POLICY_TERMS = re.compile(
    r"\b(?:pol[ií]tica(?:s)?|procedimiento(?:s)?|devoluci[oó]n(?:es)?|"
    r"devolver|devuelv[oaes]|reembolso(?:s)?|merma(?:s)?|caducidad|"
    r"caduc[oa]s?|apertura|abrir\s+(?:la\s+)?tienda|"
    r"cierre\s+(?:de\s+)?tienda|cerrar\s+(?:la\s+)?tienda|"
    r"manual(?:es)?|faq|reglamento(?:s)?)\b",
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class Metrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    latency_ms: float = 0
    retrieval_mode: str | None = None
    embedding_calls: int = 0
    embedding_input_tokens: int = 0
    embedding_latency_ms: float = 0
    warnings: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    evidence: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    status: Literal["answered", "needs_clarification", "insufficient_data", "error"]


class AggregateRequest(BaseModel):
    dataset: Literal["sales", "inventory", "tickets"]
    metric: str = Field(min_length=1)
    operation: Literal["sum", "avg", "min", "max", "count"] = "sum"
    filters: dict[str, str] = Field(default_factory=dict)
    group_by: str | None = None
    limit: int = Field(default=25, ge=1, le=100)


@app.get("/health")
def health() -> dict[str, str]:
    # Preserve the existing health contract for current clients and tests.
    return {"status": "ok"}


@app.get("/health/rag")
def health_rag() -> dict:
    # No explicit settings: inspect the same shared retriever used by /chat.
    return rag_health()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    question = request.question.strip()
    if not question:
        return ChatResponse(
            answer="Escribe una pregunta para continuar.",
            status="needs_clarification",
        )
    if not _POLICY_TERMS.search(question):
        return ChatResponse(
            answer="Todavía no está disponible la consulta conversacional de datos estructurados. "
                   "Puedes consultar políticas de devoluciones, mermas y procedimientos.",
            status="insufficient_data",
            metrics=Metrics(latency_ms=round((time.perf_counter() - start) * 1000, 2)),
        )

    # The RAG boundary handles index absence and search failures as status='error'.
    # Do not pass the whole question to PolicySearchRequest if it exceeds rag-v1's limit.
    if len(question) > 1000:
        return ChatResponse(
            answer="La consulta de políticas es demasiado larga (máximo 1000 caracteres).",
            status="needs_clarification",
        )
    result = search_policies(PolicySearchRequest(query=question, top_k=5))
    rag_metrics = result.metrics
    return ChatResponse(
        answer=result.answer,
        status=result.status,
        evidence=[item.to_http_evidence() for item in result.evidence],
        conflicts=[item.model_dump() for item in result.conflicts],
        metrics=Metrics(
            input_tokens=rag_metrics.input_tokens,
            output_tokens=rag_metrics.output_tokens,
            total_tokens=rag_metrics.input_tokens + rag_metrics.output_tokens,
            llm_calls=rag_metrics.llm_calls,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            retrieval_mode=rag_metrics.retrieval_mode,
            embedding_calls=rag_metrics.embedding_calls,
            embedding_input_tokens=rag_metrics.embedding_input_tokens,
            embedding_latency_ms=rag_metrics.embedding_latency_ms,
            warnings=rag_metrics.warnings,
        ),
    )


@app.post("/query/aggregate")
def query_aggregate(request: AggregateRequest) -> dict:
    try:
        result = query_dataset(StructuredRequest(**request.model_dump()), CATALOG)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "rows": result.rows,
        "evidence": [{
            "source_type": "csv",
            "source": result.source,
            "reference": result.as_of,
        }],
        "metric": result.metric,
        "operation": result.operation,
        "filters": result.filters,
    }
