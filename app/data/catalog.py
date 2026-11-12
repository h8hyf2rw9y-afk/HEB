"""Allowlisted schemas for the synthetic H-E-B exercise CSVs."""
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
        if self.csv_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV inputs are allowed")


_SCHEMAS = {
    "sales": ("ventas.csv", ("tienda_id", "sku", "canal"), ("venta_neta_mxn", "unidades", "descuento_mxn"), "fecha"),
    "inventory": ("inventario.csv", ("tienda_id", "sku"), ("existencia", "punto_reorden"), None),
    "tickets": ("tickets_mesa_servicio.csv", ("tienda_id", "categoria", "prioridad", "estado"), ("ticket_id",), "fecha_creacion"),
}


def build_catalog(data_dir: Path) -> dict[str, DatasetSpec]:
    """Register only the three approved analytical datasets, if present."""
    return {
        name: DatasetSpec(name, data_dir / filename, dimensions, measures, date_column)
        for name, (filename, dimensions, measures, date_column) in _SCHEMAS.items()
        if (data_dir / filename).is_file()
    }
