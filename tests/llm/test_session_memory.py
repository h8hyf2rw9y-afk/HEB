"""Bounded in-memory conversation context and reset API."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.api.session_memory import MAX_TURNS, SESSIONS, VolatileConversationStore
from app.llm.agent import AgentOutcome
from app.llm.config import LLMSettings

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def clean_sessions():
    SESSIONS.clear_all()
    yield
    SESSIONS.clear_all()


def test_session_is_isolated_bounded_and_resettable():
    store = VolatileConversationStore()
    for i in range(MAX_TURNS + 3):
        store.add("session-a", f"pregunta {i}")
    assert len(store.history("session-a")) == MAX_TURNS
    assert store.history("session-a")[0] == "pregunta 3"
    assert store.history("session-b") == []
    store.add("session-b", "otra consulta")
    store.clear("session-a")
    assert store.history("session-a") == []
    assert store.history("session-b") == ["otra consulta"]


def test_expired_session_does_not_return_previous_history(monkeypatch):
    from app.api import session_memory
    store = VolatileConversationStore()
    monkeypatch.setattr(session_memory.time, "monotonic", lambda: 100.0)
    store.add("session", "no tengo ticket")
    monkeypatch.setattr(session_memory.time, "monotonic", lambda: 100.0 + session_memory.TTL_SECONDS + 1)
    assert store.history("session") == []


def test_api_remembers_three_turns_but_not_other_session(monkeypatch):
    captured = []

    class CapturingAgent:
        settings = LLMSettings(enabled=True, mode="agente")

        def run(self, question, previous_question=None, history=None):
            captured.append((question, list(history or [])))
            return AgentOutcome(handled=True, answer="Respuesta verificada de demostración.",
                                status="answered", kind="politicas")

    monkeypatch.setattr(main, "AGENT", CapturingAgent())
    sid = uuid4().hex
    for question in [
        "Quiero devolver una licuadora",
        "No tengo el ticket",
        "¿Y si la compró hace 20 días?",
    ]:
        result = client.post("/chat", json={"question": question, "conversation_id": sid})
        assert result.status_code == 200
    assert captured[0][1] == []
    assert captured[1][1] == ["Quiero devolver una licuadora"]
    assert captured[2][1] == ["Quiero devolver una licuadora", "No tengo el ticket"]

    other = client.post("/chat", json={
        "question": "¿Y qué pasa después?", "conversation_id": uuid4().hex,
    })
    assert other.status_code == 200
    assert captured[-1][1] == []

    reset = client.post("/chat/reset", json={"conversation_id": sid})
    assert reset.json() == {"cleared": True}
    assert SESSIONS.history(sid) == []


def test_no_id_is_stateless_and_history_can_be_explicit(monkeypatch):
    captured = []

    class CapturingAgent:
        settings = LLMSettings(enabled=True, mode="agente")

        def run(self, question, previous_question=None, history=None):
            captured.append(list(history or []))
            return AgentOutcome(handled=True, answer="Respuesta.", status="answered",
                                kind="politicas")

    monkeypatch.setattr(main, "AGENT", CapturingAgent())
    client.post("/chat", json={"question": "¿Qué política aplica?"})
    client.post("/chat", json={
        "question": "¿Y si no tengo ticket?",
        "history": ["Quiero devolver una licuadora"],
    })
    assert captured == [[], ["Quiero devolver una licuadora"]]
