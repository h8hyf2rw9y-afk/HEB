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


def test_comparison_reuses_verified_winner_and_month():
    resolved = resolve_data_followup("Entonces compara sus productos con los de Cumbres", PREVIOUS)
    assert resolved == "Compara productos de Tienda Valle Norte y Tienda Cumbres en agosto de 2026."


def test_compare_two_explicit_stores_has_real_csv_sources():
    from app.tools.analytics import answer_data_question
    answer = answer_data_question(
        "Compara productos de Tienda Valle Norte y Tienda Cumbres en agosto de 2026."
    )
    assert answer.status == "answered"
    assert "Tienda Valle Norte" in answer.answer and "Tienda Cumbres" in answer.answer
    assert "no demuestra sus causas" in answer.answer
    assert {"ventas.csv", "tiendas.csv", "catalogo_productos.csv"} <= {
        item["source"] for item in answer.evidence
    }


def test_api_can_drill_into_products_without_invented_causes():
    sid = uuid4().hex
    client.post("/chat", json={"question": PREVIOUS.question, "conversation_id": sid})
    result = client.post("/chat", json={
        "question": "Entonces compara sus productos con los de Cumbres",
        "conversation_id": sid,
    }).json()
    assert result["status"] == "answered"
    assert "Tienda Valle Norte" in result["answer"] and "Tienda Cumbres" in result["answer"]
    assert "no demuestra sus causas" in result["answer"]
    assert result["metrics"]["resolved_question"] is not None



def test_policy_followup_requeries_instead_of_replaying_old_answer():
    from app.api.followup import resolve_policy_followup

    previous = VerifiedTurn(
        question="¿Cómo se hace una devolución de licuadora?",
        answer="Respuesta documentada [procedimiento_devoluciones.pdf, p. 2].",
        evidence=[{"source_type": "pdf", "source": "procedimiento_devoluciones.pdf",
                   "reference": "p. 2"}],
        kind="grounded_policy",
    )
    resolved = resolve_policy_followup("¿Y si no tiene el ticket?", previous)
    assert resolved is not None and "devolución de licuadora" in resolved
    assert "no tiene el ticket" in resolved
    assert resolve_policy_followup("¿Qué tienda vendió más en agosto?", previous) is None
    assert resolve_policy_followup("¿Y si no tiene el ticket?", PREVIOUS) is None



def test_model_rewording_cannot_change_store_identity():
    tampered = VerifiedTurn(
        question=PREVIOUS.question,
        answer="1. Tienda Cumbres: $999,999,999.99 MXN",
        evidence=EVIDENCE, kind="grounded_data",
    )
    resolved = resolve_data_followup("¿Y cómo le fue en julio?", tampered)
    assert resolved == "¿Cuánto vendió Tienda Valle Norte en julio de 2026?"



def test_why_followup_uses_product_breakdown_not_unverified_causes():
    resolved = resolve_data_followup("¿Y por qué crees que vendió más?", PREVIOUS)
    assert resolved == "¿Por qué Tienda Valle Norte vendió más que las otras tiendas en agosto de 2026?"



def test_causal_sales_question_is_not_free_chat():
    from app.api.assistant_chat import is_general_conversation
    assert not is_general_conversation("¿Por qué crees que vendió más?")
    assert not is_general_conversation("¿Por qué crees que vendió más que otra?")
    assert is_general_conversation("¿Por qué el cielo es azul?")


def test_real_chat_retains_sales_context_after_unsupported_month():
    sid = uuid4().hex
    first = client.post("/chat", json={
        "question": "¿Qué tienda vendió más en agosto de 2026?",
        "conversation_id": sid,
    }).json()
    assert first["status"] == "answered"
    january = client.post("/chat", json={
        "question": "¿Y cómo le fue en enero?",
        "conversation_id": sid,
    }).json()
    assert january["status"] in {"insufficient_data", "needs_clarification"}
    last = SESSIONS.latest_verified(sid)
    assert last is not None and "agosto" in last.question.lower()
    why = client.post("/chat", json={
        "question": "¿Por qué crees que vendió más que otra?",
        "conversation_id": sid,
    }).json()
    assert why["status"] == "answered"
    assert why["metrics"]["resolved_question"] is not None
    assert "no demuestra sus causas" in why["answer"]


def test_opt_in_fast_exact_sales_ranking(monkeypatch):
    monkeypatch.setenv("HEB_FAST_ANALYTICS", "true")
    class ForbiddenAgent:
        def __init__(self):
            from app.llm.config import LLMSettings
            self.settings = LLMSettings(enabled=True, mode="agente")
        def run(self, *args, **kwargs):
            raise AssertionError("Simple ranking should not call Ollama")
    monkeypatch.setattr(main, "AGENT", ForbiddenAgent())
    result = client.post("/chat", json={
        "question": "¿Qué tienda vendió más en agosto de 2026?",
        "conversation_id": uuid4().hex,
    }).json()
    assert result["status"] == "answered"
    assert "Tienda Valle Norte" in result["answer"]
    assert result["metrics"]["llm_calls"] == 0
    assert result["metrics"]["answer_mode"] == "grounded_data"
    assert {e["source"] for e in result["evidence"]} == {"ventas.csv", "tiendas.csv"}



def test_elliptical_agent_answer_remembers_verified_ranking_tool_query(monkeypatch):
    from app.llm.agent import AgentOutcome
    from app.llm.config import LLMSettings

    class ScriptedAgent:
        settings = LLMSettings(enabled=True, mode="agente")

        def run(self, question, previous_question=None, history=None):
            assert question == "¿Y cómo le fue en agosto?"
            return AgentOutcome(
                handled=True, status="answered", kind="datos",
                answer=("Ventas netas por tienda en agosto de 2026: "
                        "Tienda Valle Norte, Tienda Cumbres, Tienda Anáhuac y Tienda Lomas del Sur."),
                evidence=EVIDENCE,
                tool_calls=[
                    {"name": "consultar_datos_tienda",
                     "query": "¿Cuántas tiendas tienen ventas en agosto de 2026?",
                     "status": "answered"},
                    {"name": "consultar_datos_tienda",
                     "query": "¿Qué tienda vendió más en agosto de 2026?",
                     "status": "answered"},
                ],
            )

    monkeypatch.setattr(main, "AGENT", ScriptedAgent())
    monkeypatch.delenv("HEB_FAST_ANALYTICS", raising=False)
    sid = uuid4().hex
    result = client.post("/chat", json={
        "question": "¿Y cómo le fue en agosto?", "conversation_id": sid,
    }).json()
    assert result["status"] == "answered"
    cached = SESSIONS.latest_verified(sid)
    assert cached is not None
    assert cached.question == "¿Qué tienda vendió más en agosto de 2026?"
    assert resolve_data_followup("¿Por qué crees que vendió más?", cached) is not None



def test_fast_analytics_avoids_ollama_for_followup_comparison(monkeypatch):
    monkeypatch.setenv("HEB_FAST_ANALYTICS", "true")
    sid = uuid4().hex
    first = client.post("/chat", json={
        "question": "¿Qué tienda vendió más en agosto de 2026?",
        "conversation_id": sid,
    }).json()
    assert first["metrics"]["llm_calls"] == 0
    why = client.post("/chat", json={
        "question": "¿Por qué crees que vendió más que otra?",
        "conversation_id": sid,
    }).json()
    assert why["status"] == "answered"
    assert why["metrics"]["resolved_question"] == (
        "Compara productos de Tienda Valle Norte y Tienda Cumbres en agosto de 2026."
    )
    assert why["metrics"]["llm_calls"] == 0
    assert "no demuestra sus causas" in why["answer"]
    assert "Cumbres" in why["answer"] and "Valle Norte" in why["answer"]
