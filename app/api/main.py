"""FastAPI entry points for structured queries and grounded policy answers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data.catalog import build_catalog
from app.llm import GroundedWriter, LLMSettings
from app.rag.interface import PolicySearchRequest, rag_health, search_policies
from app.tools.structured import StructuredRequest, query_dataset
from app.tools.analytics import answer_data_question, looks_like_data

app = FastAPI(title="HEB AI Assistant", version="0.3.0")
CATALOG = build_catalog(Path(__file__).resolve().parents[2] / "candidatos" / "datos")
LLM_SETTINGS = LLMSettings.from_env()
WRITER = GroundedWriter(LLM_SETTINGS)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class Metrics(BaseModel):
    # Generative model (Ollama /api/chat): real counters prompt_eval_count / eval_count.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    llm_model: str | None = None
    generation_latency_ms: float = 0
    generated: bool = False  # True when the shown answer was written by the model
    fallback_reason: str | None = None  # why the deterministic answer was kept
    # Whole request, including retrieval/SQL and generation.
    latency_ms: float = 0
    retrieval_mode: str | None = None
    # Embeddings (Ollama /api/embed), counted separately from the generative model.
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


@app.get("/health/llm")
def health_llm() -> dict:
    """Generative layer configuration (never calls Ollama)."""
    return {
        "enabled": LLM_SETTINGS.enabled,
        "model": LLM_SETTINGS.model,
        "base_url": LLM_SETTINGS.base_url,
        "timeout_s": LLM_SETTINGS.timeout_s,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    question = request.question.strip()
    if not question:
        return ChatResponse(
            answer="Escribe una pregunta para continuar.",
            status="needs_clarification",
        )
    # Data queries use fixed, parameterized SQL. Everything else goes to the RAG.
    # This also fixes previously unreachable policy questions such as SIGMA.
    if looks_like_data(question):
        try:
            result = answer_data_question(question)
        except Exception:
            return ChatResponse(
                answer="No pude consultar los CSV en este momento. Verifica que DuckDB esté instalado y los archivos estén disponibles.",
                status="error",
                metrics=Metrics(latency_ms=round((time.perf_counter() - start) * 1000, 2)),
            )
        outcome = WRITER.rewrite(
            question=question, kind="datos", status=result.status,
            verified_answer=result.answer, evidence=result.evidence,
        )
        return ChatResponse(
            answer=outcome.answer,
            status=result.status,
            evidence=result.evidence,
            metrics=Metrics(
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                total_tokens=outcome.total_tokens,
                llm_calls=outcome.llm_calls,
                llm_model=outcome.model if outcome.llm_calls else None,
                generation_latency_ms=outcome.generation_latency_ms,
                generated=outcome.generated,
                fallback_reason=outcome.fallback_reason,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                warnings=outcome.warnings,
            ),
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
    evidence = [item.to_http_evidence() for item in result.evidence]
    conflicts = [item.model_dump() for item in result.conflicts]
    outcome = WRITER.rewrite(
        question=question, kind="politicas", status=result.status,
        verified_answer=result.answer, evidence=evidence, conflicts=conflicts,
    )
    return ChatResponse(
        answer=outcome.answer,
        status=result.status,
        evidence=evidence,
        conflicts=conflicts,
        metrics=Metrics(
            input_tokens=rag_metrics.input_tokens + outcome.input_tokens,
            output_tokens=rag_metrics.output_tokens + outcome.output_tokens,
            total_tokens=rag_metrics.input_tokens + rag_metrics.output_tokens + outcome.total_tokens,
            llm_calls=rag_metrics.llm_calls + outcome.llm_calls,
            llm_model=outcome.model if outcome.llm_calls else None,
            generation_latency_ms=outcome.generation_latency_ms,
            generated=outcome.generated,
            fallback_reason=outcome.fallback_reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            retrieval_mode=rag_metrics.retrieval_mode,
            embedding_calls=rag_metrics.embedding_calls,
            embedding_input_tokens=rag_metrics.embedding_input_tokens,
            embedding_latency_ms=rag_metrics.embedding_latency_ms,
            warnings=rag_metrics.warnings + outcome.warnings,
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
