"""Explicit, validated DuckDB dataset catalog; no model-generated SQL."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid SQL identifier")
    return '"' + value + '"'


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    dimensions: tuple[str, ...]
    measures: tuple[str, ...]
    date_column: str | None = None

    def __post_init__(self) -> None:
        identifier(self.name)
        for col in (*self.dimensions, *self.measures):
            identifier(col)
        if self.date_column:
            identifier(self.date_column)
        if not self.csv_path.suffix.lower() == ".csv":
            raise ValueError("Only CSV inputs are allowed")


def build_catalog(data_dir: Path) -> dict[str, DatasetSpec]:
    """No schema is assumed until the HEB data dictionary is inspected.

    Replace this empty catalog with approved DatasetSpec declarations.
    Do not infer a table layout from model output or from arbitrary uploads.
    """
    _ = data_dir
    return {}
