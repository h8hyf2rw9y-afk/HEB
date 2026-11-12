"""Regression tests for store catalogue, source discovery and HR absences."""
from fastapi.testclient import TestClient
import pytest

from app.api import main
from app.api.business_intents import route_business_question

client = TestClient(main.app)


@pytest.mark.parametrize("question", [
    "¿Cuántas tiendas tengo?",
    "Cuantas sucursales tenemos?",
])
def test_store_count_is_grounded_in_real_catalogue(question, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Store-count questions must not search policies or sales")
    monkeypatch.setattr(main, "search_policies", forbidden)
    monkeypatch.setattr(main, "answer_data_question", forbidden)
    response = client.post("/chat", json={"question": question})
    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "answered"
    assert "4 tiendas" in data["answer"]
    for code in ("T01", "T02", "T03", "T04"):
        assert code in data["answer"]
    assert data["evidence"] == [
        {"source_type": "csv", "source": "tiendas.csv", "reference": "tienda_id, nombre"}
    ]
    assert data["metrics"]["llm_calls"] == 0


@pytest.mark.parametrize("question", [
    "¿Qué documentos me puedes dar o qué información?",
    "Qué archivos tienes disponibles?",
    "Qué fuentes puedes consultar?",
])
def test_document_discovery_not_random_policy(question, monkeypatch):
    monkeypatch.setattr(main, "search_policies", lambda *a, **kw: pytest.fail("Must not call RAG"))
    data = client.post("/chat", json={"question": question}).json()
    assert data["status"] == "answered"
    assert "ventas.csv" in data["answer"]
    assert "tiendas.csv" in data["answer"]
    assert "faq_gerentes_de_tienda.pdf" in data["answer"]
    assert "P25." not in data["answer"]


@pytest.mark.parametrize("question", [
    "Tengo un colaborador que lleva semanas sin poder ir al trabajo y quiero saber qué hacer",
    "Un empleado falta desde hace varias semanas, ¿qué procede?",
    "Un trabajador lleva meses de ausencia, ¿lo puedo despedir?",
])
def test_prolonged_employee_absence_does_not_hallucinate_hr_procedure(question, monkeypatch):
    monkeypatch.setattr(main, "search_policies", lambda *a, **kw: pytest.fail("Must not call unrelated RAG"))
    data = client.post("/chat", json={"question": question}).json()
    assert data["status"] == "needs_clarification"
    assert "Recursos Humanos" in data["answer"]
    assert "incapacidad" in data["answer"]
    assert "reembolso" not in data["answer"].lower()
    assert data["evidence"] == []


def test_one_shift_staffing_problem_not_forced_into_long_term_hr_case():
    assert route_business_question("¿Qué hago si un empleado no se presenta a su turno?") is None


def test_policy_questions_not_misclassified_as_documents_list():
    for q in [
        "¿Qué dice el documento sobre devoluciones sin ticket?",
        "¿Qué información establece el procedimiento de devoluciones sobre el plazo?",
    ]:
        assert route_business_question(q) is None
