import json

import pytest
from pydantic import ValidationError

from app.rag import interface
from app.rag.config import RagSettings
from app.rag.evaluation import evaluate, load_cases
from app.rag.index import IndexNotFoundError
from app.rag.interface import PolicySearchRequest, PolicySearchResponse, search_policies
from app.rag.prompts import build_grounded_prompt

from conftest import FIXTURES

CONTRACT_STATUSES = {"answered", "needs_clarification", "insufficient_data", "error"}


@pytest.mark.parametrize("q", [
    "¿Cuál es la política de bonos para gerentes regionales?",
    "¿Cuál es el horario de la farmacia?",
    "¿Cuántos días de vacaciones tiene un empleado?",
])
def test_unanswerable_questions_abstain(retriever, q):
    ans = retriever.answer(q)
    assert ans.status == "insufficient_data"
    assert ans.evidence == []
    assert "No encontré información suficiente" in ans.answer
    assert retriever.retrieve(q) == []


@pytest.mark.parametrize("q", ["", "   ", "¿Qué?"])
def test_empty_questions_need_clarification(retriever, q):
    assert retriever.answer(q).status == "needs_clarification"


def test_top_k_validation(retriever):
    with pytest.raises(ValueError):
        retriever.answer("montacargas", top_k=0)


def test_answer_is_extractive_and_cited(retriever):
    ans = retriever.answer("¿Quién puede operar montacargas?")
    assert ans.status == "answered"
    assert "Solo el personal certificado puede operar montacargas." in ans.answer
    assert "[SINTETICO_manual_seguridad_tienda.pdf, p. 1, sección: 1. Montacargas]" in ans.answer
    assert ans.metrics.llm_calls == 0 and ans.metrics.latency_ms >= 0


def test_grounded_prompt_contains_numbered_passages_and_conflicts(retriever):
    prompt = build_grounded_prompt(retriever.answer("¿Cuántos días tengo para devolver un producto?"))
    assert "[1] (SINTETICO_politica_devoluciones_v2.pdf, p. 1" in prompt
    assert "Conflictos detectados" in prompt


def test_interface_request_validation():
    with pytest.raises(ValidationError):
        PolicySearchRequest(query="   ")
    with pytest.raises(ValidationError):
        PolicySearchRequest(query="x", top_k=21)


def test_interface_response_shape(retriever):
    resp = search_policies(PolicySearchRequest(query="¿Cuántos días tengo para devolver un producto?"), retriever)
    assert resp.contract_version == "rag-v1"
    assert resp.status in CONTRACT_STATUSES
    first = resp.evidence[0]
    assert {"document", "page", "passage", "section", "score", "chunk_id"} <= set(first.model_dump())
    assert first.to_http_evidence()["source_type"] == "pdf"
    # JSON-serializable for FastAPI
    json.loads(resp.model_dump_json())
    assert PolicySearchResponse.model_validate_json(resp.model_dump_json()) == resp


def test_interface_never_raises_when_index_missing(monkeypatch):
    def boom(*a, **k):
        raise IndexNotFoundError("sin índice")

    monkeypatch.setattr(interface, "get_retriever", boom)
    resp = search_policies(PolicySearchRequest(query="montacargas"))
    assert resp.status == "error" and resp.error == "sin índice"
    assert interface.rag_health()["ready"] is False


def test_interface_hides_internal_errors(monkeypatch, retriever):
    monkeypatch.setattr(retriever, "answer", lambda *a, **k: 1 / 0)
    resp = search_policies(PolicySearchRequest(query="montacargas"), retriever)
    assert resp.status == "error" and resp.error == "ZeroDivisionError"


def test_health_with_index(tmp_path, bm25_index):
    bm25_index.save(tmp_path)
    h = interface.rag_health(RagSettings(index_dir=tmp_path, embeddings="off"))
    assert h["ready"] is True and h["documents"] == 5 and h["embeddings_enabled"] is False


@pytest.mark.parametrize("index_name", ["bm25_index", "hybrid_index"])
def test_evaluation_on_synthetic_set(request, index_name):
    from app.rag.service import PolicyRetriever
    from rag_fakes import FakeEmbedder

    index = request.getfixturevalue(index_name)
    embedder = FakeEmbedder() if index.embeddings else None
    report = evaluate(PolicyRetriever(index, embedder=embedder), load_cases(FIXTURES / "synthetic_eval.json"))
    print(index_name, report.model_dump_json())
    assert report.failures == []
    assert report.hit_at_k == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.conflict_accuracy == 1.0
