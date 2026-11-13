"""Assistant greetings and Ollama diagnostics must not be passed to PDF RAG."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import assistant_chat, main
from app.llm.config import LLMSettings

client = TestClient(main.app)


@pytest.mark.parametrize("question,intent", [
    ("Hola", "greeting"),
    ("¡Buenos días!", "greeting"),
    ("Gracias", "thanks"),
    ("¿Qué puedes hacer?", "capabilities"),
    ("¿Quién eres?", "identity"),
    ("¿Qué modelo usas?", "model"),
    ("Hola, estás conectado con Ollma", "ollama_status"),
    ("¿Estás conectado con Ollama?", "ollama_status"),
    ("¿Ollama está funcionando?", "ollama_status"),
    ("¿Tienes llama3.2 instalado?", "ollama_status"),
])
def test_meta_intent_classification(question, intent):
    assert assistant_chat.classify_assistant_question(question) == intent


@pytest.mark.parametrize("question", [
    "¿Cuántos días tiene un cliente para devolver una licuadora?",
    "¿Qué tienda vendió más en agosto de 2026?",
    "¿Cuánto inventario hay?",
    "¿Con qué frecuencia se cuenta el inventario de abarrotes?",
])
def test_business_questions_do_not_match_assistant_meta(question):
    assert assistant_chat.classify_assistant_question(question) is None


@pytest.mark.parametrize("question", ["Hola", "¿Qué puedes hacer?", "¿Quién eres?", "Gracias"])
def test_meta_chat_does_not_touch_analytics_or_rag(monkeypatch, question):
    def unexpected(*args, **kwargs):
        raise AssertionError("No se debe consultar DuckDB, RAG ni Ollama para esta pregunta")
    monkeypatch.setattr(main, "answer_data_question", unexpected)
    monkeypatch.setattr(main, "search_policies", unexpected)
    monkeypatch.setattr(assistant_chat, "check_ollama", unexpected)
    response = client.post("/chat", json={"question": question})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered" and data["answer"]
    assert data["evidence"] == [] and data["conflicts"] == []
    assert data["metrics"]["llm_calls"] == 0
    assert data["metrics"]["generated"] is False


@pytest.mark.parametrize("enabled,health,expected", [
    (True, {"reachable": True, "model_available": True}, "redacción generativa está activada"),
    (False, {"reachable": True, "model_available": True}, "redacción generativa está desactivada"),
    (True, {"reachable": True, "model_available": False}, "no aparece entre los instalados"),
    (True, {"reachable": False, "model_available": False}, "No pude verificar conexión"),
])
def test_ollama_status_never_invents_connectivity(monkeypatch, enabled, health, expected):
    monkeypatch.setattr(main, "LLM_SETTINGS", LLMSettings(enabled=enabled, model="llama3.2"))
    monkeypatch.setattr(assistant_chat, "check_ollama", lambda settings: health)
    def unexpected(*args, **kwargs):
        raise AssertionError("Una consulta sobre Ollama no debe llegar al RAG ni a DuckDB")
    monkeypatch.setattr(main, "search_policies", unexpected)
    monkeypatch.setattr(main, "answer_data_question", unexpected)
    data = client.post("/chat", json={"question": "Hola, estas conectado con Ollma"}).json()
    assert data["status"] == "answered"
    assert expected in data["answer"]
    assert data["evidence"] == []
    assert data["metrics"]["llm_calls"] == 0


def test_live_status_probe_checks_model_inventory(monkeypatch):
    original_client = httpx.Client

    def make_client(**kwargs):
        return original_client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}, {"name": "bge-m3:latest"}]})
        ), **kwargs)

    monkeypatch.setattr(assistant_chat.httpx, "Client", make_client)
    health = assistant_chat.check_ollama(LLMSettings(model="llama3.2"))
    assert health == {"reachable": True, "model_available": True}


def test_status_probe_does_not_claim_connectivity_on_error(monkeypatch):
    original_client = httpx.Client
    def make_client(**kwargs):
        return original_client(transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text="offline")
        ), **kwargs)
    monkeypatch.setattr(assistant_chat.httpx, "Client", make_client)
    health = assistant_chat.check_ollama(LLMSettings())
    assert health == {"reachable": False, "model_available": False}

@pytest.mark.parametrize("question", [
    "¿Qué puedes hacer con un empleado que no vino al turno?",
    "¿Qué puedes hacer cuando el personal no asiste?",
    "¿Quién eres para decidir sobre un despido?",
    "¿Qué modelo usas para calcular las ventas de mi tienda?",
])
def test_business_phrasing_never_gets_assistant_meta_intent(question):
    assert assistant_chat.classify_assistant_question(question) is None


@pytest.mark.parametrize("question", [
    "Para que fuiste creado ?",
    "¿Cuál es tu propósito?",
    "¿Por qué fuiste creado?",
])
def test_origin_questions_remain_assistant_meta(question):
    assert assistant_chat.classify_assistant_question(question) == "identity"


def test_origin_question_never_calls_csv_when_agent_enabled(monkeypatch):
    from app.llm.agent import AgentOutcome

    class Agent:
        settings = LLMSettings(enabled=True, mode="agente")

        def run(self, *args, **kwargs):
            raise AssertionError("Assistant origin questions must not call the agent")

    monkeypatch.setattr(main, "AGENT", Agent())
    monkeypatch.setattr(main, "LLM_SETTINGS", LLMSettings(enabled=False))
    data = client.post("/chat", json={"question": "Para que fuiste creado ?"}).json()
    assert data["status"] == "answered"
    assert "asistente" in data["answer"].lower()
    assert data["metrics"]["tool_calls"] == []


def test_generic_off_topic_capability_claim_is_discarded(monkeypatch):
    from app.llm.general import CasualResult

    class Agent:
        settings = LLMSettings(enabled=True, mode="agente")

    monkeypatch.setattr(main, "AGENT", Agent())
    monkeypatch.setattr(main, "generate_general_answer", lambda *a, **k:
                        CasualResult(answer="Puedo darte recetas, consejos de cuidado personal y horóscopos.",
                                     generated=True, model="llama3.2", llm_calls=1,
                                     input_tokens=33, output_tokens=20))
    response = client.post("/chat", json={"question": "Que puedes hacer por mi ?"}).json()
    assert response["status"] == "answered"
    assert response["metrics"]["fallback_reason"] == "respuesta_meta_fuera_de_alcance"
    assert response["metrics"]["generated"] is False
    assert "ventas" in response["answer"].lower()
    assert response["metrics"]["llm_calls"] == 1


def test_catalogue_queries_bypass_tool_agent_in_agent_mode(monkeypatch):
    class Agent:
        settings = LLMSettings(enabled=True, mode="agente")
        def run(self, *args, **kwargs):
            raise AssertionError("Exact catalogue queries must use verified routing")

    monkeypatch.setattr(main, "AGENT", Agent())
    count = client.post("/chat", json={"question": "Dime cuantas tiendas tengo ?"}).json()
    assert count["status"] == "answered"
    assert "4 tiendas" in count["answer"]
    assert count["metrics"]["llm_calls"] == 0
    sources = client.post("/chat", json={
        "question": "¿Qué información tienes disponible para responder mis preguntas?"
    }).json()
    assert sources["status"] == "answered"
    assert "ventas.csv" in sources["answer"]
    assert sources["metrics"]["llm_calls"] == 0
