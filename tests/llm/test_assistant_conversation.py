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
