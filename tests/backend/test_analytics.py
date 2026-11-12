"""End-to-end, deterministic DuckDB checks using a tiny synthetic CSV fixture."""
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
from fastapi.testclient import TestClient

from app.api.main import app
from app.tools.analytics import answer_data_question, looks_like_data


@pytest.fixture
def data(tmp_path: Path) -> Path:
    (tmp_path / "tiendas.csv").write_text(
        "tienda_id,nombre\nT01,Norte\nT02,Sur\n", encoding="utf-8"
    )
    (tmp_path / "catalogo_productos.csv").write_text(
        "sku,nombre,categoria\nSKU-1,Leche,Lácteos\nSKU-2,Pan,Panadería\n",
        encoding="utf-8",
    )
    (tmp_path / "ventas.csv").write_text(
        "fecha,tienda_id,sku,unidades,venta_neta_mxn,descuento_mxn,canal\n"
        "2026-06-02,T01,SKU-1,10,250,20,Piso\n"
        "2026-06-03,T02,SKU-2,5,100,0,Ecommerce\n"
        "2026-07-01,T01,SKU-1,-2,-50,0,Piso\n"
        "2026-08-04,T02,SKU-2,3,60,0,Piso\n",
        encoding="utf-8",
    )
    (tmp_path / "inventario.csv").write_text(
        "tienda_id,sku,existencia,punto_reorden\n"
        "T01,SKU-1,0,10\nT01,SKU-2,15,10\nT02,SKU-1,5,10\n",
        encoding="utf-8",
    )
    (tmp_path / "tickets_mesa_servicio.csv").write_text(
        "tienda_id,estado,prioridad\n"
        "T01,Abierto,Crítica\nT01,Resuelto,Media\nT02,En proceso,Alta\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("q", [
    "¿Qué tienda vendió más en agosto?",
    "¿Cuánto vendimos en julio?",
    "¿Cuántos tickets siguen pendientes?",
    "¿Qué existencias tenemos?",
])
def test_data_router(q: str):
    assert looks_like_data(q)


@pytest.mark.parametrize("q", ["¿Qué es SIGMA?", "¿Puedo usar el fondo fijo?", "¿Cuál es la política de devoluciones?"])
def test_policy_router(q: str):
    assert not looks_like_data(q)


def test_net_sales_includes_returns(data: Path):
    a = answer_data_question("¿Cuánto vendimos en julio?", data)
    assert a.status == "answered"
    assert "-50.00 MXN" in a.answer
    assert a.evidence[0]["source"] == "ventas.csv"


def test_store_ranking(data: Path):
    a = answer_data_question("¿Qué tienda vendió más en junio?", data)
    assert a.status == "answered"
    assert "Norte: $250.00 MXN" in a.answer
    assert a.answer.index("Norte") < a.answer.index("Sur")


def test_product_ranking(data: Path):
    a = answer_data_question("¿Qué producto vendió más en junio?", data)
    assert a.status == "answered"
    assert "Leche: $250.00 MXN" in a.answer
    assert a.evidence[-1]["source"] == "catalogo_productos.csv"


def test_inventory_snapshot(data: Path):
    a = answer_data_question("¿Qué productos están bajo el punto de reorden en inventario?", data)
    assert a.status == "answered"
    assert "Norte · SKU-1 · Leche: existencia 0, punto de reorden 10." in a.answer
    assert "Sur · SKU-1 · Leche: existencia 5, punto de reorden 10." in a.answer
    assert "2026-09-01" in a.evidence[0]["reference"]


def test_tickets(data: Path):
    a = answer_data_question("¿Cuántos tickets están pendientes?", data)
    assert a.status == "answered"
    assert "Norte: 1 tickets pendientes." in a.answer
    assert "Sur: 1 tickets pendientes." in a.answer
    assert "Total: 2 tickets." in a.answer


def test_unsupported_period_abstains(data: Path):
    a = answer_data_question("¿Cuánto vendimos en noviembre de 2026?", data)
    assert a.status == "insufficient_data"
    assert not a.evidence


def test_no_files_abstains(tmp_path: Path):
    a = answer_data_question("¿Cuánto vendimos en junio?", tmp_path)
    assert a.status == "insufficient_data"




def test_specific_store_does_not_mix_totals(data: Path):
    a = answer_data_question("¿Cuánto vendimos en la tienda Norte en junio?", data)
    assert a.status == "answered"
    assert "Norte" in a.answer
    assert "$250.00 MXN" in a.answer
    assert "$350.00 MXN" not in a.answer


def test_specific_store_by_code(data: Path):
    a = answer_data_question("¿Cuánto vendimos en la T02 en junio?", data)
    assert a.status == "answered"
    assert "$100.00 MXN" in a.answer


def test_unknown_store_abstains(data: Path):
    a = answer_data_question("¿Cuánto vendimos en T99 en junio?", data)
    assert a.status == "insufficient_data"
    assert a.evidence == []




def test_critical_open_tickets_only(data: Path):
    a = answer_data_question("¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son?", data)
    assert a.status == "answered"
    assert "Norte: 1 tickets críticos pendientes." in a.answer
    assert "Sur:" not in a.answer
    assert "Total: 1 tickets." in a.answer


def test_inventory_specific_store_lists_skus(data: Path):
    a = answer_data_question("¿Qué productos de la tienda T01 están por debajo del punto de reorden?", data)
    assert a.status == "answered"
    assert "SKU-1 · Leche: existencia 0, punto de reorden 10" in a.answer
    assert "Sur" not in a.answer
    assert "SKU-2" not in a.answer


def test_future_prediction_abstains(data: Path):
    a = answer_data_question("¿Cuánto vamos a vender en diciembre?", data)
    assert a.status == "insufficient_data"
    assert not a.evidence

def test_chat_sales_real_data():
    result = TestClient(app).post("/chat", json={"question": "¿Cuánto vendimos en agosto?"})
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "answered"
    assert body["evidence"][0]["source"] == "ventas.csv"
    assert body["metrics"]["llm_calls"] == 0
