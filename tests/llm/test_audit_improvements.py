"""Audit follow-ups: stage timings, live Ollama probe, sources listing, auditable
policy passages and the agent's retry with the user's own question."""

from __future__ import annotations

import pytest
from llm_fakes import FakeOllama, ollama_reply, policy_index, policy_index_dir  # noqa: F401 (fixtures); first

from app.api import main
from app.llm.agent import ToolAgent
from app.llm.config import LLMSettings
from app.llm.ollama_client import OllamaChatClient

pytest.importorskip("duckdb")
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)


def tool_reply(name, arguments):
    reply = ollama_reply("", prompt_tokens=300, completion_tokens=20)
    reply["message"]["tool_calls"] = [{"function": {"name": name, "arguments": arguments}}]
    return reply


class Script:
    def __init__(self, *steps):
        self.steps = list(steps)

    def __call__(self, req):
        step = self.steps.pop(0)
        return step(req) if callable(step) else step


def _install_agent(monkeypatch, behaviour) -> FakeOllama:
    fake = FakeOllama(behaviour)
    settings = LLMSettings(enabled=True, model="llama3.2", timeout_s=2, mode="agente", verified_first=False)
    monkeypatch.setattr(main, "AGENT", ToolAgent(settings, main.AGENT_TOOLS,
                        OllamaChatClient(settings.base_url, settings.model, 2, transport=fake.transport)))
    monkeypatch.delenv("HEB_FAST_ANALYTICS", raising=False)
    return fake


def _echo_tool_output(req):
    tool_msgs = [m for m in req["messages"] if m["role"] == "tool"]
    return ollama_reply(tool_msgs[-1]["content"].split("\n", 1)[1], prompt_tokens=500, completion_tokens=60)


def test_every_chat_reports_stage_timings_even_without_model():
    m = client.post("/chat", json={"question": "¿Qué tienda vendió más en agosto de 2026?"}).json()["metrics"]
    t = m["timings"]
    assert t["llm_ms"] == 0 and t["llm_calls"] == [] and t["total_ms"] == m["latency_ms"]
    assert t["other_ms"] >= 0


def test_agent_timings_split_model_stages_and_tool_time(monkeypatch, policy_index):
    _install_agent(monkeypatch, Script(tool_reply("buscar_politicas", {"consulta": "tolerancia de merma en panadería"}),
                                       _echo_tool_output))
    m = client.post("/chat", json={"question": "¿Cuál es la tolerancia de merma en panadería?"}).json()["metrics"]
    t = m["timings"]
    assert len(t["llm_calls"]) == 2 and t["llm_calls"][0]["used_tools"] is True
    # Fake Ollama reports 120 ms load, 900 ms prompt eval and 1500 ms generation per call.
    assert (t["llm_load_ms"], t["llm_prompt_eval_ms"], t["llm_generation_ms"]) == (240.0, 1800.0, 3000.0)
    assert t["tokens_per_s"] > 0
    assert m["tool_calls"][0]["duration_ms"] >= 0 and t["tools_ms"] == m["tool_calls"][0]["duration_ms"]


def test_agent_retries_a_failed_reformulation_with_the_users_question(monkeypatch, policy_index):
    _install_agent(monkeypatch, Script(tool_reply("buscar_politicas", {"consulta": "zzz qqq sin coincidencias"}),
                                       _echo_tool_output))
    data = client.post("/chat", json={"question": "¿Qué es SIGMA?"}).json()
    call = data["metrics"]["tool_calls"][0]
    assert call["retried_with_original_question"] is True and call["first_query_status"] != "answered"
    assert call["status"] == "answered" and call["query"] == "¿Qué es SIGMA?"
    assert data["status"] == "answered" and "Sistema Integral de Gestión de Mermas y Abasto" in data["answer"]


def test_policy_evidence_carries_the_cited_passage(policy_index):
    data = client.post("/chat", json={"question": "¿Cuál es la tolerancia de merma en panadería?"}).json()
    ev = next(e for e in data["evidence"] if e["source"] == "politica_mermas_y_caducidad.pdf")
    assert ev["chunk_id"] and ev["page"] == 3 and "3.5" in ev["passage"]


def test_health_ollama_is_a_live_probe(monkeypatch):
    monkeypatch.setattr(main, "check_ollama", lambda settings: {"reachable": True, "model_available": False})
    data = client.get("/health/ollama").json()
    assert data["reachable"] is True and data["model_available"] is False and "model" in data


def test_sources_lists_real_csv_files_and_indexed_pdfs(policy_index):
    data = client.get("/sources").json()
    files = {d["file"]: d for d in data["datasets"]}
    assert {"ventas.csv", "inventario.csv", "tickets_mesa_servicio.csv", "tiendas.csv"} <= set(files)
    assert files["ventas.csv"]["date_min"].startswith("2026-06") and files["ventas.csv"]["date_max"].startswith("2026-08")
    assert files["ventas.csv"]["rows"] > 0 and "venta_neta_mxn" in files["ventas.csv"]["columns"]
    assert data["policies"]["ready"] is True and len(data["policies"]["documents"]) == 4
