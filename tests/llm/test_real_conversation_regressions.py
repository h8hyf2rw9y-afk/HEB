"""The seven questions that failed or were slow with Qwen3 4B in the real chat,
replayed as ONE conversation in agent mode with a simulated Ollama.

Simulated replies cannot prove how the real model behaves; they prove the
routing, the evidence, the number of model calls and the validation around it.
The same seven questions run against real Ollama in test_real_ollama.py and
scripts/eval_agent_conversations.py."""

from __future__ import annotations

import re
from uuid import uuid4

import pytest
from llm_fakes import FakeOllama, ollama_reply, policy_index, policy_index_dir  # noqa: F401 (fixtures); first

from app.api import main
from app.llm.agent import ToolAgent
from app.llm.config import LLMSettings
from app.llm.ollama_client import OllamaChatClient
from app.llm.service import GroundedWriter
from app.llm.validation import validate_data_summary

pytest.importorskip("duckdb")
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)

SEVEN = [
    "¿Qué tienda vendió más en agosto de 2026?",
    "¿Y cómo le fue en julio?",
    "¿Por qué crees que vendió más que las otras?",
    "¿Cuántos productos tiene cada tienda?",
    "¿Qué es SIGMA?",
    "¿Cuál es el procedimiento para devolver una licuadora?",
    "¿Y si el cliente no tiene ticket?",
]


def qwen_like(req: dict) -> dict:
    """A well-behaved model: a short summary for data, a faithful rewrite for policies."""
    assert "tools" not in req, "a clear question must not spend a tool-selection call"
    body = req["messages"][1]["content"]
    if "<resultado_verificado>" in body:
        verified = body.split("<resultado_verificado>\n", 1)[1].split("\n</resultado_verificado>")[0]
        line = next(line for line in verified.splitlines() if "$" in line or "SKU" in line)
        text = re.sub(r"^\s*(?:\d+\.|-)\s*", "", line).rstrip(".")
        return ollama_reply(f"Según los datos verificados: {text}.", prompt_tokens=380, completion_tokens=40)
    if "<texto_verificado>" in body:  # long policy answer: short summary, cited text shown below
        return ollama_reply('Para una licuadora el plazo es de 30 días naturales con ticket de compra y empaque original completo; hay una contradicción con la FAQ, que menciona 15 días, y prevalece procedimiento_devoluciones.pdf.', prompt_tokens=900, completion_tokens=60)
    verified = body.split("<respuesta_verificada>\n", 1)[1].split("\n</respuesta_verificada>")[0]
    return ollama_reply(verified.replace("Respuesta según los documentos:\n", ""), prompt_tokens=700, completion_tokens=150)


def _install(monkeypatch, behaviour, verified_first=True) -> FakeOllama:
    fake = FakeOllama(behaviour)
    settings = LLMSettings(enabled=True, model="qwen3:4b-instruct", timeout_s=2, mode="agente",
                           verified_first=verified_first)
    llm = OllamaChatClient(settings.base_url, settings.model, 2, transport=fake.transport)
    monkeypatch.setattr(main, "AGENT", ToolAgent(settings, main.AGENT_TOOLS, llm))
    # In production AGENT and WRITER come from the same settings (catalogue routes use WRITER).
    monkeypatch.setattr(main, "WRITER", GroundedWriter(settings, llm))
    monkeypatch.delenv("HEB_FAST_ANALYTICS", raising=False)
    return fake


def test_seven_real_questions_one_conversation(monkeypatch, policy_index):
    fake = _install(monkeypatch, qwen_like)
    sid = uuid4().hex
    answers = []
    for question in SEVEN:
        data = client.post("/chat", json={"question": question, "conversation_id": sid}).json()
        answers.append(data)
        m = data["metrics"]
        # Evidence first, then exactly one model call that only writes.
        assert data["status"] == "answered", (question, data["answer"])
        assert m["llm_calls"] == 1 and m["generated"] is True, (question, m["warnings"])
        assert m["answer_mode"] in ("grounded_data", "grounded_policy") and data["evidence"]
        assert all(call.get("planner") == "verified_route" for call in m["tool_calls"])
        assert len(m["timings"]["llm_calls"]) == 1 and m["timings"]["tools_ms"] == 0
    assert len(fake.requests) == 7

    ranking, july, why, products, sigma, blender, no_receipt = answers
    assert "Tienda Valle Norte" in ranking["answer"] and "$2,163,062.36 MXN" in ranking["answer"]
    assert {e["source"] for e in ranking["evidence"]} >= {"ventas.csv", "tiendas.csv"}

    assert july["metrics"]["resolved_question"] == "¿Cuánto vendió Tienda Valle Norte en julio de 2026?"
    assert "$2,153,571.23 MXN" in july["answer"]

    # "Why?" keeps the store and the month, compares against the other stores and
    # separates observed data from what the files cannot show.
    assert why["metrics"]["resolved_question"] == "¿Por qué Tienda Valle Norte vendió más que las otras tiendas en julio de 2026?"
    for text in ("Lo que muestran los datos:", "Lo que los datos no permiten afirmar:", "no demuestran sus causas",
                 "Volumen:", "Venta promedio por unidad:", "Abarrotes"):
        assert text in why["answer"]
    assert "catalogo_productos.csv" in {e["source"] for e in why["evidence"]}

    assert "120 SKU registrados" in products["answer"] and "inventario.csv" in {e["source"] for e in products["evidence"]}

    assert "Sistema Integral de Gestión de Mermas y Abasto" in sigma["answer"]

    # The product the user already named is never asked for again.
    assert "¿Qué producto" not in blender["answer"] and "licuadora" in blender["answer"].lower()
    assert "30 días naturales" in blender["answer"] and "Prevalece procedimiento_devoluciones.pdf" in blender["answer"]
    assert "7. Flujo de atención en tienda" in blender["answer"]
    assert blender["conflicts"] and blender["conflicts"][0]["resolution"] == "resolved_by_source"

    assert "vale de tienda" in no_receipt["answer"] and "identificación oficial" in no_receipt["answer"]


def test_causal_claim_from_the_model_is_rejected_and_the_table_shown(monkeypatch, policy_index):
    def causal(req):
        return ollama_reply("Tienda Valle Norte vendió más debido a sus promociones y a su ubicación.")

    _install(monkeypatch, causal)
    sid = uuid4().hex
    client.post("/chat", json={"question": SEVEN[0], "conversation_id": sid})
    data = client.post("/chat", json={"question": SEVEN[2], "conversation_id": sid}).json()
    m = data["metrics"]
    assert m["generated"] is False and m["fallback_reason"] == "validacion_fallida"
    assert "afirma causas" in " ".join(m["warnings"])
    assert "promociones y a su ubicación" not in data["answer"] and "Lo que muestran los datos:" in data["answer"]


def test_unclear_question_still_reaches_the_tool_calling_agent(monkeypatch, policy_index):
    def agent(req):
        assert "tools" in req  # nothing verified answered it, so the model plans the query
        return ollama_reply("No tengo información verificada para eso.")

    fake = _install(monkeypatch, agent)
    data = client.post("/chat", json={"question": "Dame un panorama de Cumbres en agosto"}).json()
    assert len(fake.requests) == 1 and data["metrics"]["fallback_reason"] == "modelo_no_consulto_herramientas"


VERIFIED_RANKING = ("Ventas netas por tienda (agosto de 2026), mayor a menor:\n"
                    "1. Tienda Valle Norte: $2,163,062.36 MXN\n2. Tienda Cumbres: $1,973,086.37 MXN\n"
                    "3. Tienda Anáhuac: $949,469.00 MXN\n4. Tienda Lomas del Sur: $833,131.19 MXN")


@pytest.mark.parametrize("summary", [
    "En agosto de 2026 la tienda con más ventas fue Valle Norte, con $2,163,062.36 MXN; le siguió Cumbres con $1,973,086.37 MXN.",
    "Valle Norte lideró agosto de 2026 con unos 2.16 millones, por delante de Cumbres (1.97 millones).",
    "Las cuatro tiendas vendieron en agosto de 2026; la primera fue Tienda Valle Norte.",
])
def test_natural_summaries_are_accepted(summary):
    check = validate_data_summary(question="¿Qué tienda vendió más en agosto de 2026?", verified=VERIFIED_RANKING,
                                  summary=summary, allowed_sources={"ventas.csv", "tiendas.csv"})
    assert check.ok, check.failures


@pytest.mark.parametrize("summary,reason", [
    ("Cumbres fue la tienda con más ventas en agosto de 2026.", "declara como primera"),
    ("Valle Norte vendió $1,973,086.37 MXN en agosto de 2026.", "atribuida a"),
    ("Valle Norte vendió $2,500,000.00 MXN en agosto de 2026.", "cifras o fechas nuevas"),
    ("Valle Norte vendió 9.6% más que Cumbres en agosto de 2026.", "cifras o fechas nuevas"),
    ("Valle Norte lideró con 2.5 millones en agosto de 2026.", "aproximada sin correspondencia"),
    ("Valle Norte vendió más en septiembre.", "cifras o fechas nuevas"),
    ("Valle Norte vendió más porque tiene mejor ubicación.", "afirma causas"),
])
def test_wrong_summaries_are_rejected(summary, reason):
    check = validate_data_summary(question="¿Qué tienda vendió más en agosto de 2026?", verified=VERIFIED_RANKING,
                                  summary=summary, allowed_sources={"ventas.csv", "tiendas.csv"})
    assert not check.ok and any(reason in f for f in check.failures), check.failures


def _ask(sid: str, question: str) -> dict:
    return client.post("/chat", json={"question": question, "conversation_id": sid}).json()


def test_reference_to_the_store_survives_a_policy_question_in_between(policy_index):
    sid = uuid4().hex
    _ask(sid, "¿Qué tienda vendió más en agosto de 2026?")
    _ask(sid, "¿Cuáles fueron sus productos más vendidos?")
    _ask(sid, "¿Qué es SIGMA?")
    data = _ask(sid, "¿Y cuánto vendió esa tienda en junio?")
    assert data["metrics"]["resolved_question"] == "¿Cuánto vendió Tienda Valle Norte en junio de 2026?"
    assert "$2,085,066.63 MXN" in data["answer"]  # re-queried in DuckDB, not replayed


def test_a_new_self_contained_question_does_not_inherit_the_old_store(policy_index):
    sid = uuid4().hex
    _ask(sid, "¿Cuánto vendió Tienda Cumbres en agosto de 2026?")
    data = _ask(sid, "¿Qué tienda vendió más en junio de 2026?")
    assert data["metrics"].get("resolved_question") is None
    assert "Ventas netas por tienda (junio de 2026)" in data["answer"]


def test_comparala_compares_the_two_stores_in_the_same_month(policy_index):
    sid = uuid4().hex
    _ask(sid, "¿Qué tienda vendió más en julio de 2026?")
    data = _ask(sid, "Compárala con Cumbres")
    assert data["metrics"]["resolved_question"] == "Compara productos de Tienda Valle Norte y Tienda Cumbres en julio de 2026."
    assert data["status"] == "answered" and "Tienda Cumbres:" in data["answer"]


def test_why_question_for_a_store_that_did_not_lead_says_so(policy_index):
    sid = uuid4().hex
    _ask(sid, "¿Cuánto vendió Tienda Cumbres en agosto de 2026?")
    data = _ask(sid, "¿Por qué vendió más que las otras?")
    assert "no fue la tienda con más ventas" in data["answer"] and "Tienda Valle Norte" in data["answer"]


def test_return_followup_with_elapsed_days_keeps_the_product_deadline(policy_index):
    sid = uuid4().hex
    _ask(sid, "¿Cuál es el procedimiento para devolver una licuadora?")
    data = _ask(sid, "¿Y si ya pasaron 20 días?")
    assert data["status"] == "answered" and "30 días naturales" in data["answer"]
    assert "Prevalece procedimiento_devoluciones.pdf" in data["answer"]


def test_conversations_stay_isolated(policy_index):
    a, b = uuid4().hex, uuid4().hex
    _ask(a, "¿Cuánto vendió Tienda Cumbres en agosto de 2026?")
    data = _ask(b, "¿Y cuánto vendió esa tienda en junio?")
    assert data["metrics"].get("resolved_question") is None


def test_generic_return_question_asks_only_for_the_missing_product(policy_index):
    data = client.post("/chat", json={"question": "¿Cómo hago una devolución?"}).json()
    assert data["status"] == "answered" and "7. Flujo de atención en tienda" in data["answer"]
    assert data["answer"].rstrip().endswith("¿Qué producto quiere devolver el cliente y tiene ticket de compra?")


def test_sales_drivers_figures_reconcile_with_pure_python():
    """The «why» analysis is new SQL: recompute its figures from the CSVs without DuckDB."""
    import csv
    from pathlib import Path

    from app.tools.analytics import answer_data_question

    data = Path(__file__).resolve().parents[2] / "candidatos" / "datos"
    with (data / "catalogo_productos.csv").open(encoding="utf-8-sig") as fh:
        category = {r["sku"]: r["categoria"] for r in csv.DictReader(fh)}
    sales, units, cat = {}, {}, {}
    with (data / "ventas.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["fecha"].startswith("2026-07"):
                sales[r["tienda_id"]] = sales.get(r["tienda_id"], 0) + float(r["venta_neta_mxn"])
                units[r["tienda_id"]] = units.get(r["tienda_id"], 0) + int(r["unidades"])
                key = (category[r["sku"]], r["tienda_id"])
                cat[key] = cat.get(key, 0) + float(r["venta_neta_mxn"])
    others = [t for t in sales if t != "T01"]
    avg_sales = sum(sales[t] for t in others) / len(others)
    avg_units = sum(units[t] for t in others) / len(others)
    avg_abarrotes = sum(cat.get(("Abarrotes", t), 0) for t in others) / len(others)
    answer = answer_data_question("¿Por qué Tienda Valle Norte vendió más que las otras tiendas en julio de 2026?").answer
    assert f"${sales['T01']:,.2f} MXN" in answer and f"${avg_sales:,.2f} MXN" in answer
    assert f"{units['T01']:,.0f} unidades" in answer and f"{avg_units:,.0f} en las otras" in answer
    assert f"Abarrotes: ${cat[('Abarrotes', 'T01')]:,.2f} MXN frente a ${avg_abarrotes:,.2f} MXN" in answer


# --- The blender: long policy answer summarized, cited text shown below -------------------

from app.llm.validation import validate_policy_summary  # noqa: E402

LICUADORA_CONFLICT = [{
    "topic": "días", "resolution": "resolved_by_source", "preferred_document": "procedimiento_devoluciones.pdf",
    "claims": [{"document": "faq_gerentes_de_tienda.pdf", "page": 4, "values": ["15 días"]},
               {"document": "procedimiento_devoluciones.pdf", "page": 2, "values": ["30 días"]}],
}]
LICUADORA_TEXT = ("- Plazo para devolución o cambio: 30 días naturales | Requisitos: Ticket de compra, empaque original. "
                  "[procedimiento_devoluciones.pdf, p. 2, sección: 3. Plazos]\n"
                  "- El cliente tiene 15 días con ticket. [faq_gerentes_de_tienda.pdf, p. 4, sección: P15] "
                  "Prevalece procedimiento_devoluciones.pdf.")


def _policy_check(summary):
    return validate_policy_summary(question="¿Cuál es el procedimiento para devolver una licuadora?",
                                   verified=LICUADORA_TEXT, summary=summary, conflicts=LICUADORA_CONFLICT,
                                   allowed_sources={"procedimiento_devoluciones.pdf", "faq_gerentes_de_tienda.pdf"})


@pytest.mark.parametrize("summary", [
    "El cliente puede devolver la licuadora dentro de 30 días naturales con ticket de compra y empaque original.",
    "El plazo es de 30 días naturales; la FAQ menciona 15 días, pero prevalece procedimiento_devoluciones.pdf.",
])
def test_policy_summary_accepted(summary):
    assert _policy_check(summary).ok, _policy_check(summary).failures


@pytest.mark.parametrize("summary,reason", [
    ("El cliente tiene 15 días para devolver la licuadora.", "no prevalece"),
    ("El plazo es de 15 días; prevalece faq_gerentes_de_tienda.pdf.", "contra la fuente"),
    ("El plazo es de 45 días naturales.", "cifras o fechas nuevas"),
    ("Ver [procedimiento_devoluciones.pdf, p. 9, sección: 12. Garantías].", "citas inexistentes"),
    ("Según manual_garantias.pdf el plazo es de 30 días naturales.", "fuentes inexistentes"),
])
def test_policy_summary_rejected(summary, reason):
    check = _policy_check(summary)
    assert not check.ok and any(reason in f for f in check.failures), check.failures


def test_blender_answer_is_a_short_model_summary_plus_the_cited_text(monkeypatch, policy_index):
    fake = _install(monkeypatch, qwen_like)
    data = client.post("/chat", json={"question": SEVEN[5]}).json()
    m, sent = data["metrics"], fake.requests[-1]
    assert m["generated"] is True and m["llm_calls"] == 1 and m["fallback_reason"] is None
    assert "<texto_verificado>" in sent["messages"][1]["content"] and sent["options"]["num_predict"] <= 260
    summary, _, cited = data["answer"].partition("**Texto verificado de las políticas:**")
    assert "30 días naturales" in summary and cited  # every citation and condition stays below
    assert "[procedimiento_devoluciones.pdf, p. 2" in cited and "7. Flujo de atención en tienda" in cited
