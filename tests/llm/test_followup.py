"""Grounded multi-turn analytics: derive new queries, never reuse stale figures."""
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api import main
from app.api.followup import resolve_data_followup
from app.api.session_memory import SESSIONS, VerifiedTurn

client = TestClient(main.app)

def setup_function():
    SESSIONS.clear_all()

def teardown_function():
    SESSIONS.clear_all()

EVIDENCE = [{"source_type": "csv", "source": "ventas.csv", "reference": "agosto de 2026"}]
PREVIOUS = VerifiedTurn(
    question="¿Qué tienda vendió más en agosto de 2026?",
    answer="Ventas netas por tienda (agosto de 2026):\n1. Tienda Valle Norte: $2,163,062.36 MXN\n2. Tienda Cumbres: $1,000,000.00 MXN",
    evidence=EVIDENCE,
    kind="grounded_data",
)


def test_followup_resolves_store_and_recomputes_new_period():
    assert resolve_data_followup("¿Y cómo le fue en julio?", PREVIOUS) == (
        "¿Cuánto vendió Tienda Valle Norte en julio de 2026?"
    )
    assert resolve_data_followup("¿Y cuáles fueron sus productos más vendidos?", PREVIOUS) == (
        "¿Qué productos tuvieron más ventas en Tienda Valle Norte en agosto de 2026?"
    )


def test_new_month_reranks_all_stores_instead_of_filtering_to_old_winner():
    assert resolve_data_followup("¿Y en julio?", PREVIOUS) == (
        "¿Qué tienda vendió más en julio de 2026?"
    )


def test_no_unverified_or_cross_domain_context_injection():
    bad = VerifiedTurn(question=PREVIOUS.question, answer=PREVIOUS.answer,
                       evidence=[], kind="grounded_data")
    assert resolve_data_followup("¿Y en julio?", bad) is None
    assert resolve_data_followup("¿Y cómo le fue en julio?", None) is None
    assert resolve_data_followup("¿Y qué dice la política de devoluciones?", PREVIOUS) is None
    assert resolve_data_followup("¿Y en diciembre de 2026?", PREVIOUS) is None


def test_api_stores_verified_answer_and_requeries_sales(monkeypatch):
    # A deterministic run is sufficient for the grounded continuity contract.
    sid = uuid4().hex
    first = client.post("/chat", json={"question": PREVIOUS.question, "conversation_id": sid}).json()
    assert first["status"] == "answered" and first["metrics"]["answer_mode"] == "grounded_data"
    saved = SESSIONS.latest_verified(sid)
    assert saved is not None and saved.is_grounded
    follow = client.post("/chat", json={
        "question": "¿Y cómo le fue en julio?", "conversation_id": sid,
    }).json()
    assert follow["status"] == "answered"
    assert follow["metrics"]["resolved_question"] == "¿Cuánto vendió Tienda Valle Norte en julio de 2026?"
    assert "julio de 2026" in follow["answer"] and "Tienda Valle Norte" in follow["answer"]
    assert "ventas.csv" in {e["source"] for e in follow["evidence"]}
    # A different conversation cannot borrow the first conversation's result.
    other = client.post("/chat", json={
        "question": "¿Y cómo le fue en julio?", "conversation_id": uuid4().hex,
    }).json()
    assert other["metrics"]["resolved_question"] is None
    assert SESSIONS.latest_verified(sid).question == follow["metrics"]["resolved_question"]
    client.post("/chat/reset", json={"conversation_id": sid})
    assert SESSIONS.latest_verified(sid) is None


def test_memory_never_persists_generated_answers_without_sources():
    sid = uuid4().hex
    SESSIONS.add(sid, "¿Qué tienda vendió más?", answer="Tienda inventada: 999 millones",
                 evidence=[], kind="general_unverified", status="answered")
    assert SESSIONS.latest_verified(sid) is None
