"""Return and receipt routing: never replace store-return context with IT tickets."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.api.return_intents import route_returns, check_return_evidence
from app.rag.interface import PolicyEvidence, PolicySearchResponse

client = TestClient(main.app)

RETURN_QUESTION = "Que pasa si un cliente quiere devolver un producto cual es el proceso ?"


def test_generic_return_question_selects_general_process_not_blender_faq():
    route = route_returns(RETURN_QUESTION)
    assert route is not None and route.kind == "general_process"
    assert "procedimiento_devoluciones.pdf" in route.query


@pytest.mark.parametrize("question", [
    "Si no tiene el ticket",
    "¿Y si no trae ticket?",
    "Pero no presenta el ticket",
])
def test_followup_preserves_return_receipt_semantics(question):
    route = route_returns(question, RETURN_QUESTION)
    assert route is not None and route.kind == "no_receipt"
    assert "sin ticket de compra" in route.query


def test_standalone_short_ticket_question_requests_context():
    route = route_returns("Si no tiene el ticket")
    assert route is not None and route.kind == "needs_context"


@pytest.mark.parametrize("question", [
    "Cómo levantar un ticket de Mesa de Servicio?",
    "Si no tiene el ticket de soporte",
])
def test_it_service_questions_are_not_return_questions(question):
    assert route_returns(question, RETURN_QUESTION) is None


def test_no_receipt_answer_requires_pertinent_source_and_conditions():
    answer = (
        "Sí, con restricciones: presentar identificación oficial, "
        "se devuelve exclusivamente en vale de tienda. Se permiten "
        "como máximo tres devoluciones sin ticket. "
        "[faq_gerentes_de_tienda.pdf, p. 3, sección: P?]"
    )
    assert check_return_evidence(
        "no_receipt", status="answered", answer=answer,
        evidence=[{"source": "faq_gerentes_de_tienda.pdf"}]
    )
    assert not check_return_evidence(
        "no_receipt", status="answered", answer="Se levanta un ticket en Mesa de Servicio.",
        evidence=[{"source": "manual_apertura_y_cierre_tienda.pdf"}]
    )


def test_first_return_question_rejects_appliance_only_faq(monkeypatch):
    def wrong_result(*args, **kwargs):
        return PolicySearchResponse(
            status="answered", answer=(
                "Para licuadoras hay 15 días con ticket. "
                "[faq_gerentes_de_tienda.pdf, p. 4, sección: P15]"
            ),
            evidence=[PolicyEvidence(
                document="faq_gerentes_de_tienda.pdf", page=4, passage="15 días",
                section="P15", score=1.0, chunk_id="fake",
            )],
        )
    monkeypatch.setattr(main, "search_policies", wrong_result)
    data = client.post("/chat", json={"question": RETURN_QUESTION}).json()
    assert data["status"] == "needs_clarification"
    assert data["evidence"] == []
    assert "15 días" not in data["answer"]


def test_followup_does_not_hit_technical_tickets(monkeypatch):
    def wrong_result(*args, **kwargs):
        return PolicySearchResponse(
            status="answered", answer=(
                "Se levantan tickets de soporte en el portal de Mesa de Servicio."
                " [manual_apertura_y_cierre_tienda.pdf, p. 7, sección: 9.3]"
            ),
            evidence=[PolicyEvidence(
                document="manual_apertura_y_cierre_tienda.pdf", page=7,
                passage="Mesa de Servicio", section="9.3", score=1, chunk_id="fake",
            )],
        )
    monkeypatch.setattr(main, "search_policies", wrong_result)
    data = client.post("/chat", json={
        "question": "Si no tiene el ticket", "previous_question": RETURN_QUESTION
    }).json()
    assert data["status"] == "needs_clarification"
    assert data["evidence"] == []
    assert "Mesa de Servicio" not in data["answer"]


def test_standalone_receipt_followup_does_not_call_rag(monkeypatch):
    monkeypatch.setattr(main, "search_policies", lambda *a, **kw: pytest.fail("Ambiguous ticket should not query RAG"))
    data = client.post("/chat", json={"question": "Si no tiene el ticket"}).json()
    assert data["status"] == "needs_clarification"
    assert "ticket de compra" in data["answer"]
    assert "Mesa de Servicio" in data["answer"]
    assert data["evidence"] == []


def test_no_receipt_full_question_uses_verified_faq(monkeypatch):
    def verified(*args, **kwargs):
        return PolicySearchResponse(
            status="answered", answer=(
                "Sí, con restricciones: identificación oficial, devolución sin ticket "
                "exclusivamente en vale de tienda. "
                "[faq_gerentes_de_tienda.pdf, p. 3, sección: P?]"
            ),
            evidence=[PolicyEvidence(
                document="faq_gerentes_de_tienda.pdf", page=3,
                section="P?", passage="Sí, con restricciones: identificación oficial.",
                score=1, chunk_id="fake",
            )],
        )
    monkeypatch.setattr(main, "search_policies", verified)
    data = client.post("/chat", json={
        "question": "Si no tiene el ticket",
        "previous_question": RETURN_QUESTION,
    }).json()
    assert data["status"] == "answered"
    assert "vale de tienda" in data["answer"]
    assert data["evidence"][0]["source"] == "faq_gerentes_de_tienda.pdf"
    assert data["metrics"]["generated"] is False
