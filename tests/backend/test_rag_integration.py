"""Contract-level tests for the FastAPI ↔ rag-v1 adapter."""
from fastapi.testclient import TestClient

from app.api import main
from app.rag.interface import PolicyEvidence, PolicySearchResponse
from app.rag.models import ConflictClaim, PolicyConflict, RetrievalMetrics

client = TestClient(main.app)


def test_policy_question_maps_evidence_and_metrics(monkeypatch):
    def fake_search(request):
        assert request.query == "¿Cuál es la política de devoluciones?"
        return PolicySearchResponse(
            status="answered",
            answer="Se aceptan devoluciones. [devoluciones.pdf, p. 2]",
            evidence=[
                PolicyEvidence(
                    document="devoluciones.pdf", page=2, section="Reglas",
                    passage="Se aceptan devoluciones.", score=1.0, chunk_id="c1",
                )
            ],
            metrics=RetrievalMetrics(
                retrieval_mode="bm25", embedding_calls=1,
                embedding_input_tokens=12, embedding_latency_ms=4.5,
            ),
        )

    monkeypatch.setattr(main, "search_policies", fake_search)
    data = client.post("/chat", json={"question": "¿Cuál es la política de devoluciones?"}).json()
    assert data["status"] == "answered"
    assert data["evidence"] == [
        {"source_type": "pdf", "source": "devoluciones.pdf", "reference": "p. 2, sección: Reglas"}
    ]
    assert data["metrics"]["embedding_calls"] == 1
    assert data["metrics"]["embedding_input_tokens"] == 12
    assert data["metrics"]["llm_calls"] == 0
    assert data["metrics"]["latency_ms"] >= 0


def test_unresolved_conflict_is_preserved(monkeypatch):
    conflict = PolicyConflict(
        kind="numeric", topic="días",
        claims=[
            ConflictClaim(document="a.pdf", page=1, sentence="10 días", values=["10 días"]),
            ConflictClaim(document="b.pdf", page=1, sentence="15 días", values=["15 días"]),
        ],
        resolution="unresolved",
        basis="No hay precedencia explícita",
    )
    monkeypatch.setattr(main, "search_policies", lambda req: PolicySearchResponse(
        status="answered", answer="Hay políticas contradictorias sin resolver.",
        conflicts=[conflict],
    ))
    data = client.post("/chat", json={"question": "¿Cuántos días para devolver?"}).json()
    assert data["conflicts"][0]["resolution"] == "unresolved"
    assert len(data["conflicts"][0]["claims"]) == 2


def test_policy_index_unavailable_returns_error(monkeypatch):
    monkeypatch.setattr(main, "search_policies", lambda req: PolicySearchResponse(
        status="error", answer="Índice de políticas no disponible.", error="missing index",
    ))
    data = client.post("/chat", json={"question": "¿Cuál es el procedimiento de cierre?"}).json()
    assert data["status"] == "error"
    assert data["evidence"] == []


def test_structured_question_does_not_call_rag(monkeypatch):
    def unexpected(_):
        raise AssertionError("Sales questions must not invoke policy RAG")

    monkeypatch.setattr(main, "search_policies", unexpected)
    data = client.post("/chat", json={"question": "¿Cuánto vendimos en agosto?"}).json()
    assert data["status"] == "answered"
    assert data["evidence"][0]["source"] == "ventas.csv"


def test_rag_health_uses_shared_configuration(monkeypatch):
    monkeypatch.setattr(main, "rag_health", lambda: {"ready": True, "documents": 4})
    assert client.get("/health/rag").json() == {"ready": True, "documents": 4}


def test_long_policy_question_is_rejected_before_rag(monkeypatch):
    def unexpected(_):
        raise AssertionError("Should reject >1000 characters")

    monkeypatch.setattr(main, "search_policies", unexpected)
    data = client.post("/chat", json={"question": "política " + "x" * 1000}).json()
    assert data["status"] == "needs_clarification"
