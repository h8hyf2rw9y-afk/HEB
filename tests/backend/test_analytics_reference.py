"""Independent CSV reference checks for the three analytical demo questions.

These expected results are computed separately with Python's csv/Decimal,
not by the DuckDB implementation under test.
"""
import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
from app.tools.analytics import answer_data_question

DATA = Path(__file__).resolve().parents[2] / "candidatos" / "datos"


def records(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_demo_a1_top_august_store_from_independent_csv_sum():
    stores = {r["tienda_id"]: r["nombre"] for r in records("tiendas.csv")}
    amounts = defaultdict(lambda: Decimal("0"))
    for row in records("ventas.csv"):
        if row["fecha"].startswith("2026-08"):
            amounts[row["tienda_id"]] += Decimal(row["venta_neta_mxn"])
    top = sorted(amounts.items(), key=lambda pair: (-pair[1], stores[pair[0]]))[0]
    response = answer_data_question("¿Qué tienda vendió más en agosto de 2026?", DATA)
    assert response.status == "answered"
    assert f"1. {stores[top[0]]}: ${top[1]:,.2f} MXN" in response.answer
    assert response.evidence[0]["source"] == "ventas.csv"
    assert "agosto de 2026" in response.answer


def test_demo_a2_t01_replenishment_matches_independent_csv_rows():
    products = {r["sku"]: r["nombre"] for r in records("catalogo_productos.csv")}
    expected = [
        r for r in records("inventario.csv")
        if r["tienda_id"] == "T01" and int(r["existencia"]) <= int(r["punto_reorden"])
    ]
    expected.sort(key=lambda row: row["sku"])
    assert expected, "Dataset should contain at least one replenishment example"
    response = answer_data_question(
        "¿Qué productos de la tienda T01 están por debajo del punto de reorden?", DATA
    )
    assert response.status == "answered"
    assert "1 de septiembre de 2026" in response.answer
    assert response.evidence[0]["source"] == "inventario.csv"
    for row in expected[:30]:
        fragment = (
            f"{row['sku']} · {products[row['sku']]}: existencia {row['existencia']}, "
            f"punto de reorden {row['punto_reorden']}"
        )
        assert fragment in response.answer
    assert "Tienda Cumbres" not in response.answer


def test_demo_a3_critical_open_tickets_match_independent_csv_count():
    stores = {r["tienda_id"]: r["nombre"] for r in records("tiendas.csv")}
    expected = defaultdict(int)
    for row in records("tickets_mesa_servicio.csv"):
        if row["prioridad"].casefold() == "crítica" and row["estado"].casefold() in {"abierto", "en proceso"}:
            expected[row["tienda_id"]] += 1
    response = answer_data_question(
        "¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son?", DATA
    )
    if not expected:
        assert response.status == "insufficient_data"
        return
    assert response.status == "answered"
    assert f"Total: {sum(expected.values())} tickets." in response.answer
    for store_id, count in expected.items():
        assert f"{stores[store_id]}: {count} tickets críticos pendientes." in response.answer
    assert response.evidence[0]["source"] == "tickets_mesa_servicio.csv"
