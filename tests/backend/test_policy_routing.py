"""Protect policy questions from being misrouted to analytical SQL."""
import pytest
from app.tools.analytics import looks_like_data


@pytest.mark.parametrize("question", [
    "¿Cuántas devoluciones sin ticket puede hacer un mismo cliente en doce meses?",
    "¿Con qué frecuencia se cuenta el inventario de abarrotes?",
    "¿Cuánto tiempo puede permanecer en tienda un producto mermado después de su registro?",
])
def test_policy_questions_never_route_to_duckdb(question: str):
    assert not looks_like_data(question)


@pytest.mark.parametrize("question", [
    "¿Cuántos tickets críticos siguen abiertos?",
    "¿Cuánto vendimos por tienda en agosto?",
    "¿Qué productos están por debajo del punto de reorden del inventario?",
    "¿Cuántos tickets hay en T01?",
])
def test_analytics_questions_still_route_to_duckdb(question: str):
    assert looks_like_data(question)
