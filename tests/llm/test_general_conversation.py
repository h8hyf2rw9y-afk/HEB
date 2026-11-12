"""Non-business Ollama conversation must not be presented as sourced company facts."""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import main, assistant_chat
from app.llm.config import LLMSettings
from app.llm.general import CasualResult, generate_general_answer
from app.llm.ollama_client import OllamaChatClient

client = TestClient(main.app)


@pytest.mark.parametrize("question", [
    "¿Cómo estás?", "Cuéntame un chiste", "¿Qué es Python?",
    "¿Por qué es azul el cielo?", "Háblame de inteligencia artificial",
])
def test_free_conversation_is_recognized(question):
    assert assistant_chat.is_general_conversation(question)


@pytest.mark.parametrize("question", [
    "¿Qué es SIGMA?", "¿Qué es la tolerancia de merma?",
    "¿Cuántas tiendas tengo?", "¿Qué documentos tienes?",
    "¿Qué hago si un empleado falta?", "¿Qué pasa con la política de devoluciones?",
])
def test_company_questions_are_never_general_chat(question):
    assert not assistant_chat.is_general_conversation(question)


def test_general_question_uses_llm_and_never_queries_corporate_sources(monkeypatch):
    monkeypatch.setattr(main, "LLM_SETTINGS", LLMSettings(enabled=True))
    monkeypatch.setattr(main, "search_policies", lambda *args, **kw: pytest.fail("No PDF RAG"))
    monkeypatch.setattr(main, "answer_data_question", lambda *args, **kw: pytest.fail("No DuckDB"))
    monkeypatch.setattr(main, "generate_general_answer", lambda *args, **kwargs: CasualResult(
        answer="¡Claro! Aquí tienes un chiste breve.", generated=True,
        model="llama3.2", llm_calls=1, input_tokens=25, output_tokens=11, latency_ms=15,
    ))
    response = client.post("/chat", json={"question": "Cuéntame un chiste"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert "chiste" in data["answer"]
    assert data["evidence"] == []
    assert data["metrics"]["generated"]
    assert data["metrics"]["llm_calls"] == 1
    assert data["metrics"]["total_tokens"] == 36


def test_general_model_falls_back_if_ollama_is_unavailable():
    def broken(request):
        raise httpx.ConnectError("Offline")
    settings = LLMSettings(enabled=True)
    fake = OllamaChatClient(settings.base_url, settings.model, 2, transport=httpx.MockTransport(broken))
    out = generate_general_answer(
        "Cuéntame un chiste", settings, default_answer="Sin modelo disponible", client=fake
    )
    assert not out.generated and out.answer == "Sin modelo disponible"
    assert out.fallback_reason == "ollama_no_disponible"


def test_casual_system_prompt_does_not_claim_corporate_references():
    from app.llm.general import CASUAL_SYSTEM_PROMPT
    assert "no tiene documentos ni datos corporativos" in CASUAL_SYSTEM_PROMPT
    assert "no des consejo laboral individual" in CASUAL_SYSTEM_PROMPT

@pytest.mark.parametrize("question", [
    "¿Qué opinas de despedir a alguien por faltar dos semanas?",
    "¿Qué opinas de sancionar a alguien por faltar?",
    "¿Qué es una incapacidad laboral?",
    "Explícame los permisos del personal",
    "¿Qué puedes hacer con un empleado que no vino al turno?",
    "¿Cómo manejamos una baja de trabajadores?",
])
def test_labor_and_operational_topics_never_enter_ungrounded_chat(question):
    assert not assistant_chat.is_general_conversation(question)


def test_unverified_answer_is_marked_as_such_in_api(monkeypatch):
    monkeypatch.setattr(main, "LLM_SETTINGS", LLMSettings(enabled=True))
    monkeypatch.setattr(main, "generate_general_answer", lambda *args, **kwargs: CasualResult(
        answer="Un chiste breve.", generated=True, model="llama3.2",
        llm_calls=1, input_tokens=11, output_tokens=6,
    ))
    data = client.post("/chat", json={"question": "Cuéntame un chiste"}).json()
    assert data["metrics"]["answer_mode"] == "general_unverified"
    assert data["evidence"] == []


def test_verifiable_company_answer_is_not_marked_general(monkeypatch):
    monkeypatch.setattr(main, "LLM_SETTINGS", LLMSettings(enabled=False))
    data = client.post("/chat", json={"question": "¿Cuántas tiendas tengo?"}).json()
    assert data["status"] == "answered"
    assert data["metrics"]["answer_mode"] == "grounded_data"
    assert data["evidence"]
