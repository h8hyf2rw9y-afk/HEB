"""Typed entrypoints for structured data tools.

Dataset schemas are registered explicitly, never inferred from an LLM prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.data.catalog import DatasetSpec
from app.data.engine import QueryResult, aggregate


@dataclass(frozen=True)
class StructuredRequest:
    dataset: str
    metric: str
    operation: str = "sum"
    filters: dict[str, str] | None = None
    group_by: str | None = None
    limit: int = 25


def query_dataset(
    request: StructuredRequest, catalog: dict[str, DatasetSpec]
) -> QueryResult:
    spec = catalog.get(request.dataset)
    if spec is None:
        raise ValueError("Unknown or unconfigured dataset")
    return aggregate(
        spec,
        metric=request.metric,
        operation=request.operation,
        filters=request.filters,
        group_by=request.group_by,
        limit=request.limit,
    )


def query_sales(request: StructuredRequest, catalog: dict[str, DatasetSpec]) -> QueryResult:
    if request.dataset != "sales":
        raise ValueError("Expected sales dataset")
    return query_dataset(request, catalog)


def query_inventory(request: StructuredRequest, catalog: dict[str, DatasetSpec]) -> QueryResult:
    if request.dataset != "inventory":
        raise ValueError("Expected inventory dataset")
    return query_dataset(request, catalog)


def query_tickets(request: StructuredRequest, catalog: dict[str, DatasetSpec]) -> QueryResult:
    if request.dataset != "tickets":
        raise ValueError("Expected tickets dataset")
    return query_dataset(request, catalog)
