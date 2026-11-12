"""FastAPI foundation. /chat is deliberately not an LLM implementation yet."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data.catalog import build_catalog
from app.tools.structured import StructuredRequest, query_dataset

app = FastAPI(title="HEB AI Assistant", version="0.1.0")
CATALOG = build_catalog(Path("data/raw"))


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class Metrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    latency_ms: int = 0


class ChatResponse(BaseModel):
    answer: str
    evidence: list[dict] = Field(default_factory=list)
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
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """No fabricated answer while LLM/RAG orchestration is unimplemented."""
    _ = request
    return ChatResponse(
        answer="La integración conversacional todavía no está disponible.",
        status="insufficient_data",
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
