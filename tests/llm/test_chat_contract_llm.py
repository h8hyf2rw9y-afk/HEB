"""/chat keeps its contract with the generative layer on and off (simulated Ollama)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx
import pytest

from app.api import main
from app.llm.config import LLMSettings
from app.llm.ollama_client import OllamaChatClient
from app.llm.service import GroundedWriter
from conftest import FakeOllama, ollama_reply

pytest.importorskip("duckdb")
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)
CONTRACT_KEYS = {"answer", "evidence", "conflicts", "metrics", "status"}
LEGACY_METRIC_KEYS = {"input_tokens", "output_tokens", "total_tokens", "llm_calls", "latency_ms", "retrieval_mode",
                      "embedding_calls", "embedding_input_tokens", "embedding_latency_ms", "warnings"}
NEW_METRIC_KEYS = {"llm_model", "generation_latency_ms", "generated", "fallback_reason"}


def _install(monkeypatch, behaviour, enabled=True):
    fake = FakeOllama(behaviour)
    settings = LLMSettings(enabled=enabled, model="llama3.2", timeout_s=2)
    writer = GroundedWriter(settings, OllamaChatClient(settings.base_url, settings.model, 2, transport=fake.transport))
    monkeypatch.setattr(main, "WRITER", writer)
    return fake


def test_contract_fields_with_llm_disabled(monkeypatch):
    _install(monkeypatch, ollama_reply("x"), enabled=False)
    data = client.post("/chat", json={"question": "¿Qué tienda vendió más en agosto de 2026?"}).json()
    assert set(data) == CONTRACT_KEYS
    assert LEGACY_METRIC_KEYS | NEW_METRIC_KEYS <= set(data["metrics"])
    assert data["status"] == "answered" and data["metrics"]["llm_calls"] == 0
    assert data["metrics"]["generated"] is False and data["metrics"]["fallback_reason"] == "llm_desactivado"
    assert data["answer"].startswith("Ventas netas por tienda")


def test_data_answer_rewritten_keeps_evidence_and_counts_tokens(monkeypatch):
    def behaviour(req):
        verified = req["messages"][1]["content"].split("<respuesta_verificada>\n", 1)[1].split("\n</respuesta_verificada>")[0]
        lines = [line.split(". ", 1)[1] for line in verified.splitlines()[1:]]
        return ollama_reply("En agosto de 2026 la tienda con mayores ventas netas fue " + lines[0] + ". Le siguen " + "; ".join(lines[1:]) + ".",
                            prompt_tokens=610, completion_tokens=72)

    fake = _install(monkeypatch, behaviour)
    data = client.post("/chat", json={"question": "¿Qué tienda vendió más en agosto de 2026?"}).json()
    m = data["metrics"]
    assert data["status"] == "answered" and m["generated"] is True and m["llm_model"] == "llama3.2"
    assert (m["llm_calls"], m["input_tokens"], m["output_tokens"], m["total_tokens"]) == (1, 610, 72, 682)
    assert m["generation_latency_ms"] >= 0 and m["latency_ms"] >= m["generation_latency_ms"]
    assert "$2,163,062.36 MXN" in data["answer"] and "Tienda Valle Norte" in data["answer"]
    assert data["evidence"][0]["source"] == "ventas.csv"  # evidence untouched
    assert len(fake.requests) == 1


def test_policy_answer_rewritten_keeps_citations_and_conflicts(monkeypatch):
    def behaviour(req):
        body = req["messages"][1]["content"]
        verified = body.split("<respuesta_verificada>\n", 1)[1].split("\n</respuesta_verificada>")[0]
        # Echo the verified answer minus the header: a faithful "rewrite".
        return ollama_reply(verified.replace("Respuesta según los documentos:\n", "Según los documentos: "))

    _install(monkeypatch, behaviour)
    data = client.post("/chat", json={"question": "¿Cuántos días tiene un cliente para devolver una licuadora?"}).json()
    assert data["status"] == "answered" and data["metrics"]["generated"] is True
    assert data["conflicts"] and data["conflicts"][0]["resolution"] == "resolved_by_source"
    assert "[procedimiento_devoluciones.pdf, p. 2" in data["answer"] and "Prevalece procedimiento_devoluciones.pdf" in data["answer"]
    assert data["metrics"]["retrieval_mode"] == "bm25"


def test_abstention_is_never_rewritten(monkeypatch):
    fake = _install(monkeypatch, ollama_reply("En diciembre venderemos $1,000,000.00 MXN."))
    data = client.post("/chat", json={"question": "¿Cuánto vamos a vender en diciembre?"}).json()
    assert data["status"] == "insufficient_data" and "$1,000,000" not in data["answer"]
    assert fake.requests == [] and data["metrics"]["fallback_reason"] == "estado_insufficient_data_no_se_redacta"
    data = client.post("/chat", json={"question": "¿Cuál es la tolerancia de merma para electrónica?"}).json()
    assert data["status"] == "insufficient_data" and fake.requests == []


def test_ollama_down_keeps_chat_working(monkeypatch):
    _install(monkeypatch, httpx.ConnectError("refused"))
    data = client.post("/chat", json={"question": "¿Qué tienda vendió más en agosto de 2026?"}).json()
    assert data["status"] == "answered" and data["answer"].startswith("Ventas netas por tienda")
    assert data["metrics"]["fallback_reason"] == "ollama_no_disponible" and data["metrics"]["llm_calls"] == 0


def test_tampered_figure_from_model_is_discarded(monkeypatch):
    _install(monkeypatch, ollama_reply("La tienda que más vendió fue Tienda Valle Norte con $9,163,062.36 MXN."))
    data = client.post("/chat", json={"question": "¿Qué tienda vendió más en agosto de 2026?"}).json()
    assert "$2,163,062.36 MXN" in data["answer"] and "$9,163,062.36" not in data["answer"]
    assert data["metrics"]["fallback_reason"] == "validacion_fallida" and data["metrics"]["llm_calls"] == 1


def test_health_llm_reports_configuration_without_calling_ollama():
    data = client.get("/health/llm").json()
    assert set(data) == {"enabled", "model", "base_url", "timeout_s"}


def test_settings_from_env_defaults_and_overrides(monkeypatch):
    for key in ("HEB_LLM_ENABLED", "HEB_LLM_MODEL", "OLLAMA_BASE_URL", "HEB_LLM_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    s = LLMSettings.from_env()
    assert (s.enabled, s.model, s.base_url, s.timeout_s) == (False, "llama3.2", "http://localhost:11434", 60.0)
    monkeypatch.setenv("HEB_LLM_ENABLED", "true")
    monkeypatch.setenv("HEB_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    monkeypatch.setenv("HEB_LLM_TIMEOUT_S", "15")
    s = LLMSettings.from_env()
    assert (s.enabled, s.model, s.base_url, s.timeout_s) == (True, "llama3.1", "http://127.0.0.1:11434", 15.0)


def test_app_starts_with_llm_enabled_and_no_ollama():
    """A fresh process with HEB_LLM_ENABLED=true but no Ollama must still answer deterministically."""
    code = (
        "import json,sys; sys.path.insert(0, %r)\n"
        "from fastapi.testclient import TestClient\n"
        "from app.api import main\n"
        "d = TestClient(main.app).post('/chat', json={'question': '¿Cuál es la tolerancia de merma en panadería?'}).json()\n"
        "print(json.dumps({'status': d['status'], 'reason': d['metrics']['fallback_reason'], 'ok': 'Panadería' in d['answer']}))\n"
    ) % str(main.__file__.rsplit("/app/", 1)[0] if "/app/" in main.__file__ else ".")
    env = {**os.environ, "HEB_LLM_ENABLED": "true", "OLLAMA_BASE_URL": "http://127.0.0.1:9", "HEB_LLM_TIMEOUT_S": "2"}
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True).stdout
    result = json.loads(out.strip().splitlines()[-1])
    assert result == {"status": "answered", "reason": "ollama_no_disponible", "ok": True}


@pytest.mark.parametrize("field,value", [("prompt_eval_count", "abc"), ("eval_count", []), ("eval_count", -3)])
def test_malformed_ollama_counters_do_not_cause_http_500(monkeypatch, field, value):
    reply = ollama_reply("La tienda que más vendió fue Tienda Valle Norte con $2,163,062.36 MXN.")
    reply[field] = value
    _install(monkeypatch, reply)
    response = client.post("/chat", json={"question": "¿Qué tienda vendió más en agosto de 2026?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert data["answer"].startswith("Ventas netas por tienda") and "$2,163,062.36 MXN" in data["answer"]
    assert data["evidence"][0]["source"] == "ventas.csv"
    m = data["metrics"]
    assert m["generated"] is False and m["fallback_reason"] == "respuesta_invalida"
    assert (m["llm_calls"], m["total_tokens"]) == (0, 0) and m["llm_model"] is None
    assert any("Contador de tokens" in w for w in m["warnings"])
