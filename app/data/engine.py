"""Deterministic, read-only DuckDB aggregates over an approved CSV catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog import DatasetSpec, identifier


@dataclass(frozen=True)
class QueryResult:
    rows: list[dict[str, Any]]
    source: str
    filters: dict[str, Any]
    metric: str
    operation: str
    as_of: str | None = None


def aggregate(
    spec: DatasetSpec,
    *,
    metric: str,
    operation: str,
    filters: dict[str, str] | None = None,
    group_by: str | None = None,
    limit: int = 25,
) -> QueryResult:
    """Run a safe allowlisted aggregation; no arbitrary SQL accepted."""
    if operation not in {"sum", "avg", "min", "max", "count"}:
        raise ValueError("Unsupported aggregation")
    if metric not in spec.measures:
        raise ValueError("Metric is not allowlisted")
    if group_by is not None and group_by not in spec.dimensions:
        raise ValueError("Grouping is not allowlisted")
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100")
    filters = filters or {}
    if any(key not in spec.dimensions for key in filters):
        raise ValueError("Filter is not allowlisted")
    if not spec.csv_path.is_file():
        raise FileNotFoundError("Approved CSV file is unavailable")

    # DuckDB is imported only when querying so documentation-only checkout works.
    import duckdb

    source = str(spec.csv_path.resolve())
    group_sql = identifier(group_by) if group_by else None
    expr = f'{operation.upper()}({identifier(metric)})'
    select = f"{group_sql} AS dimension, {expr} AS value" if group_sql else f"{expr} AS value"
    clauses = [f"{identifier(col)} = ?" for col in filters]
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    group = f" GROUP BY {group_sql}" if group_sql else ""
    order = " ORDER BY value DESC NULLS LAST" if group_sql else ""
    sql = f"SELECT {select} FROM read_csv_auto(?)" + where + group + order + " LIMIT ?"
    params = [source, *filters.values(), limit]
    with duckdb.connect(database=":memory:", read_only=False) as connection:
        cursor = connection.execute(sql, params)
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return QueryResult(
        rows=rows,
        source=spec.csv_path.name,
        filters=dict(filters),
        metric=metric,
        operation=operation,
    )
