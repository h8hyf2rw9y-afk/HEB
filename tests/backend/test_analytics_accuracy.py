"""Regression cases where a plausible answer would otherwise be misleading."""
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from app.tools.analytics import answer_data_question


@pytest.fixture
def data(tmp_path: Path) -> Path:
    (tmp_path / "tiendas.csv").write_text("tienda_id,nombre\nT01,Norte\nT02,Sur\n", encoding="utf-8")
    (tmp_path / "catalogo_productos.csv").write_text(
        "sku,nombre,categoria\nSKU-1002,Leche,Lácteos\nSKU-1003,Pan,Panadería\n",
        encoding="utf-8",
    )
    (tmp_path / "ventas.csv").write_text(
        "fecha,tienda_id,sku,unidades,venta_neta_mxn,descuento_mxn,canal\n"
        "2026-06-01,T01,SKU-1002,3,120,0,Piso\n"
        "2026-06-02,T01,SKU-1002,1,40,0,Piso\n"
        "2026-06-03,T01,SKU-1003,2,200,0,Piso\n"
        "2026-06-04,T02,SKU-1002,-1,-40,0,Piso\n",
        encoding="utf-8",
    )
    return tmp_path


def test_requested_sku_filters_other_products(data: Path):
    result = answer_data_question("¿Cuánto vendimos del SKU-1002 en junio?", data)
    assert result.status == "answered"
    assert "SKU-1002" in result.answer
    assert "$120.00 MXN" in result.answer
    assert "$320.00 MXN" not in result.answer


def test_requested_sku_can_combine_store_filter(data: Path):
    result = answer_data_question("¿Cuánto vendimos del SKU-1002 en T01 en junio?", data)
    assert result.status == "answered"
    assert "T01" not in result.answer or "Norte" in result.answer
    assert "$160.00 MXN" in result.answer


def test_unknown_sku_does_not_fall_back_to_all_products(data: Path):
    result = answer_data_question("¿Cuánto vendimos del SKU-9999 en junio?", data)
    assert result.status == "insufficient_data"
    assert result.evidence == []


def test_year_not_in_dataset_abstains(data: Path):
    result = answer_data_question("¿Cuánto vendimos en junio de 2024?", data)
    assert result.status == "insufficient_data"
    assert result.evidence == []


def test_mean_is_not_a_sum(data: Path):
    result = answer_data_question("¿Cuál fue el promedio de ventas en junio?", data)
    assert result.status == "answered"
    assert "$80.00 MXN" in result.answer
    assert "$320.00 MXN" not in result.answer


def test_count_of_sales_rows_is_not_revenue(data: Path):
    result = answer_data_question("¿Cuántas ventas tuvimos en junio?", data)
    assert result.status == "answered"
    assert "4 registros" in result.answer
    assert "MXN" not in result.answer


def test_unsupported_margin_abstains(data: Path):
    result = answer_data_question("¿Cuál fue la ganancia de ventas en junio?", data)
    assert result.status == "insufficient_data"
    assert result.evidence == []
