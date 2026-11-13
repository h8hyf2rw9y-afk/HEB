"""FastAPI entry points for structured queries and grounded policy answers."""
from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data.catalog import build_catalog
from app.api.assistant_chat import classify_assistant_question, assistant_answer, check_ollama, is_general_conversation
from app.api.business_intents import route_business_question
from app.api.return_intents import route_returns, check_return_evidence
from app.api.session_memory import SESSIONS
from app.api.followup import EXPLICIT_DATA_REF, resolve_data_followup, resolve_policy_followup
from app.llm import AgentOutcome, GroundedWriter, LLMSettings, ToolAgent, ToolResult
from app.llm.general import generate_general_answer
from app.llm.ollama_client import record_llm_calls
from app.rag.interface import PolicySearchRequest, PolicySearchResponse, policy_section, rag_health, search_policies
from app.tools.structured import StructuredRequest, query_dataset
from app.tools.analytics import answer_data_question, looks_like_data

app = FastAPI(title="HEB AI Assistant", version="0.3.0")
CATALOG = build_catalog(Path(__file__).resolve().parents[2] / "candidatos" / "datos")
LLM_SETTINGS = LLMSettings.from_env()
WRITER = GroundedWriter(LLM_SETTINGS)
# Used on the deterministic path after the agent already spent its model calls.
_NO_REWRITE = GroundedWriter(LLMSettings(enabled=False))


def _tool_store_data(query: str) -> ToolResult:
    """Agent tool: fixed, parameterized DuckDB analytics (the model never writes SQL)."""
    r = answer_data_question(query)
    return ToolResult(kind="datos", status=r.status, answer=r.answer, evidence=list(r.evidence))


def _tool_policies(query: str) -> ToolResult:
    """Agent tool: verify policy relevance, especially purchase vs IT tickets."""
    returns = route_returns(query)
    if returns and returns.kind == "needs_context":
        return ToolResult(kind="politicas", status="needs_clarification",
                          answer="¿Hablas del ticket de compra para una devolución o de soporte técnico?")
    effective_query = returns.query if returns else query
    r = search_policies(PolicySearchRequest(query=effective_query, top_k=5))
    evidence = [e.to_http_evidence() for e in r.evidence]
    if returns and not check_return_evidence(returns.kind, status=r.status,
                                             answer=r.answer, evidence=evidence):
        message = (
            "La evidencia recuperada no permite explicar de manera verificada "
            "ambos conceptos (cambio de producto y reembolso). "
            "No debo equipararlos basándome únicamente en la definición de cambio."
            if returns.kind == "exchange_vs_refund" else
            "No se encontró evidencia pertinente para esta devolución. "
            "Indica tipo de producto y comprobante de compra."
        )
        return ToolResult(kind="politicas", status="needs_clarification", answer=message)
    m = r.metrics
    return ToolResult(
        kind="politicas", status=r.status, answer=r.answer,
        evidence=[e.to_http_evidence() for e in r.evidence],
        conflicts=[c.model_dump() for c in r.conflicts],
        metrics={"retrieval_mode": m.retrieval_mode, "embedding_calls": m.embedding_calls,
                 "embedding_input_tokens": m.embedding_input_tokens, "embedding_latency_ms": m.embedding_latency_ms},
    )


AGENT_TOOLS = {"consultar_datos_tienda": _tool_store_data, "buscar_politicas": _tool_policies}
AGENT = ToolAgent(LLM_SETTINGS, AGENT_TOOLS)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=128)
    previous_question: str | None = Field(default=None, max_length=4000)
    history: list[Annotated[str, Field(max_length=400)]] = Field(default_factory=list, max_length=6)


class Metrics(BaseModel):
    # Generative model (Ollama /api/chat): real counters prompt_eval_count / eval_count.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    llm_model: str | None = None
    generation_latency_ms: float = 0
    generated: bool = False  # True when the shown answer was written by the model
    answer_mode: str = "unknown"  # general_unverified | assistant_info | grounded_data | grounded_policy | needs_context | source_listing
    fallback_reason: str | None = None  # why the deterministic answer was kept
    # Whole request, including retrieval/SQL and generation.
    latency_ms: float = 0
    retrieval_mode: str | None = None
    # Embeddings (Ollama /api/embed), counted separately from the generative model.
    embedding_calls: int = 0
    embedding_input_tokens: int = 0
    embedding_latency_ms: float = 0
    warnings: list[str] = Field(default_factory=list)
    # Agent mode: verified tools the model chose to run, with the query actually executed.
    tool_calls: list[dict] = Field(default_factory=list)
    # Where the time went: every Ollama call (load / prompt reading / generation, tokens/s),
    # tool execution in agent mode, and the rest (routing, SQL, retrieval, validation).
    timings: dict = Field(default_factory=dict)
    resolved_question: str | None = None  # grounded continuation query, never a cached answer


class ChatResponse(BaseModel):
    answer: str
    evidence: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    status: Literal["answered", "needs_clarification", "insufficient_data", "error"]


class ResetConversationRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)


@app.post("/chat/reset")
def reset_conversation(request: ResetConversationRequest) -> dict[str, bool]:
    SESSIONS.clear(request.conversation_id)
    return {"cleared": True}


class AggregateRequest(BaseModel):
    dataset: Literal["sales", "inventory", "tickets"]
    metric: str = Field(min_length=1)
    operation: Literal["sum", "avg", "min", "max", "count"] = "sum"
    filters: dict[str, str] = Field(default_factory=dict)
    group_by: str | None = None
    limit: int = Field(default=25, ge=1, le=100)


@app.get("/health")
def health() -> dict[str, str]:
    # Preserve the existing health contract for current clients and tests.
    return {"status": "ok"}


@app.get("/health/rag")
def health_rag() -> dict:
    # No explicit settings: inspect the same shared retriever used by /chat.
    return rag_health()


@app.get("/health/ollama")
def health_ollama() -> dict:
    """Live probe: asks Ollama which models are installed (GET /api/tags, no generation).
    Unlike /health/llm, this proves whether the server answers right now."""
    status = check_ollama(LLM_SETTINGS)
    return {"enabled": LLM_SETTINGS.enabled, "base_url": LLM_SETTINGS.base_url,
            "model": LLM_SETTINGS.model, **status}


_DATA_DIR = Path(__file__).resolve().parents[2] / "candidatos" / "datos"
_DATE_COLUMNS = ("fecha", "fecha_creacion", "fecha_ultimo_conteo", "fecha_corte")


def _csv_summary(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        date_col = next((c for c in _DATE_COLUMNS if c in columns), None)
        rows, lo, hi = 0, None, None
        for row in reader:
            rows += 1
            value = (row.get(date_col) or "").strip() if date_col else ""
            if value:
                lo = value if lo is None or value < lo else lo
                hi = value if hi is None or value > hi else hi
    return {"file": path.name, "rows": rows, "columns": columns,
            "date_column": date_col, "date_min": lo, "date_max": hi}


@app.get("/sources")
def sources() -> dict:
    """What the assistant can actually answer from: the CSV files queried with fixed SQL
    and the PDF policies in the RAG index. Read-only; there is no upload endpoint."""
    datasets = []
    for path in sorted(_DATA_DIR.glob("*.csv")):
        try:
            datasets.append(_csv_summary(path))
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            datasets.append({"file": path.name, "error": type(exc).__name__})
    rag = rag_health()
    return {"datasets": datasets, "policies": {
        "ready": rag.get("ready", False), "documents": rag.get("document_list", []),
        "chunks": rag.get("chunks"), "embeddings_enabled": rag.get("embeddings_enabled"),
        "built_at": rag.get("built_at"), "error": rag.get("error"),
    }}


@app.get("/health/llm")
def health_llm() -> dict:
    """Generative layer configuration (never calls Ollama)."""
    return {
        "enabled": LLM_SETTINGS.enabled,
        "model": LLM_SETTINGS.model,
        "base_url": LLM_SETTINGS.base_url,
        "timeout_s": LLM_SETTINGS.timeout_s,
        "mode": LLM_SETTINGS.mode,
    }


def _agent_applies(question: str) -> bool:
    """LLM selects tools for real analysis. Exact catalogue/meta questions are
    answered by their existing trusted routes instead of speculative tool calls."""
    return (AGENT.settings.enabled and AGENT.settings.mode == "agente" and bool(question)
            and len(question) <= 1000
            and not classify_assistant_question(question)
            and not is_general_conversation(question)
            and route_business_question(question) is None)


def _fast_analytics(question: str, start: float) -> ChatResponse | None:
    """Optional reliable low-latency path for already-supported exact CSV facts.

    Natural-language explorations still go to the agent; no SQL is generated.
    This is opt-in to preserve tool-calling benchmarks and existing behavior.
    """
    if os.getenv("HEB_FAST_ANALYTICS", "false").lower() not in {"1", "true", "yes"}:
        return None
    from app.tools.analytics import plain as _plain_data
    q = _plain_data(question).strip(" ¡!¿?.,;:")
    rank = bool(re.search(
        r"\b(?:que|cual)\s+tienda\s+vendio\s+mas\b"
        r"|\btienda\s+(?:con\s+)?(?:mas|mayores)\s+ventas\b",
        q,
    ))
    comparison = bool(re.search(
        r"^compara\s+productos\s+de\s+tienda\s+.+\s+y\s+tienda\s+.+\s+en\s+"
        r"(?:junio|julio|agosto)\s+de\s+2026$", q,
    ))
    # Follow-up questions resolved to these exact, supported SQL templates
    # should not pay for a second expensive Ollama drafting round.
    if not (rank or comparison):
        return None
    result = answer_data_question(question)
    # Fast means "skip slow model tool-selection", not "skip model writing".
    # The model receives only the result already computed by safe SQL.
    prose = WRITER.rewrite(question=question, kind="datos", status=result.status,
                           verified_answer=result.answer, evidence=result.evidence)
    return ChatResponse(
        answer=prose.answer, status=result.status, evidence=result.evidence,
        metrics=Metrics(
            answer_mode="grounded_data",
            input_tokens=prose.input_tokens, output_tokens=prose.output_tokens,
            total_tokens=prose.total_tokens, llm_calls=prose.llm_calls,
            llm_model=prose.model if prose.llm_calls else None,
            generation_latency_ms=prose.generation_latency_ms,
            generated=prose.generated, fallback_reason=prose.fallback_reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            warnings=["Consulta verificada en DuckDB; el modelo redacta el resultado cuando está disponible."]
                     + prose.warnings,
        ),
    )


def _agent_metrics(out: AgentOutcome, start: float) -> Metrics:
    emb = out.tool_metrics
    modes = [m.get("retrieval_mode") for m in emb if m.get("retrieval_mode")]
    return Metrics(
        answer_mode="grounded_policy" if out.kind == "politicas" else "grounded_data",
        input_tokens=out.input_tokens, output_tokens=out.output_tokens,
        total_tokens=out.input_tokens + out.output_tokens, llm_calls=out.llm_calls,
        llm_model=out.model if out.llm_calls else None,
        generation_latency_ms=out.generation_latency_ms, generated=out.generated,
        fallback_reason=out.fallback_reason,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        retrieval_mode=modes[0] if modes else None,
        embedding_calls=sum(m.get("embedding_calls", 0) for m in emb),
        embedding_input_tokens=sum(m.get("embedding_input_tokens", 0) for m in emb),
        embedding_latency_ms=round(sum(m.get("embedding_latency_ms", 0) for m in emb), 1),
        warnings=out.warnings,
        tool_calls=out.tool_calls,
    )


def _stage_timings(llm_log: list[dict], metrics: "Metrics") -> dict:
    llm_total = round(sum(c["latency_ms"] for c in llm_log), 1)
    tools_ms = round(sum(float(c.get("duration_ms") or 0) for c in metrics.tool_calls), 1)
    timings = {
        "total_ms": metrics.latency_ms,
        "llm_ms": llm_total,
        "llm_load_ms": round(sum(c["load_ms"] for c in llm_log), 1),
        "llm_prompt_eval_ms": round(sum(c["prompt_eval_ms"] for c in llm_log), 1),
        "llm_generation_ms": round(sum(c["eval_ms"] for c in llm_log), 1),
        "tools_ms": tools_ms,
        "other_ms": round(max(metrics.latency_ms - llm_total - tools_ms, 0.0), 1),
        "llm_calls": llm_log,
    }
    rates = [c["tokens_per_s"] for c in llm_log if c.get("tokens_per_s")]
    if rates:
        timings["tokens_per_s"] = round(sum(rates) / len(rates), 1)
    return timings


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    with record_llm_calls() as llm_log:
        response = _chat(request)
    response.metrics.timings = _stage_timings(llm_log, response.metrics)
    return response


def _chat(request: ChatRequest) -> ChatResponse:
    # Client-provided history has priority. Otherwise a bounded, volatile
    # conversation_id session lets a direct API client continue a dialogue.
    saved = SESSIONS.history(request.conversation_id)
    if not request.history and saved:
        request = request.model_copy(update={
            "history": saved,
            "previous_question": request.previous_question or saved[-1],
        })
    # Resolve only narrowly-defined elliptical analytics against the latest
    # verified CSV result. Never treat old answers as fresh facts.
    latest = SESSIONS.latest_verified(request.conversation_id)
    # Missing historical months have a known, exact boundary: do not spend
    # two Ollama calls deciding whether data outside the CSV exists.
    from app.tools.analytics import plain as _plain_data
    q_normalized = _plain_data(request.question)
    if (latest is not None and latest.is_grounded and latest.kind == "grounded_data"
            and any(e.get("source") == "ventas.csv" for e in latest.evidence)
            and re.search(r"\b(?:enero|febrero|marzo|abril|mayo|septiembre|octubre|noviembre|diciembre)\b", q_normalized)
            and re.search(r"\b(?:le fue|vend\w*|ventas|factur\w*)\b", q_normalized)):
        SESSIONS.add(request.conversation_id, request.question.strip())
        return ChatResponse(
            answer="No tengo ventas registradas para ese mes: el archivo de ventas "
                   "disponible abarca únicamente junio, julio y agosto de 2026. "
                   "Puedo comparar cualquiera de esos tres meses.",
            status="insufficient_data",
            metrics=Metrics(
                answer_mode="grounded_data",
                warnings=["El periodo solicitado queda fuera de la cobertura del CSV."],
            ),
        )
    resolved = (resolve_data_followup(request.question, latest)
                or resolve_policy_followup(request.question, latest))
    if not resolved and latest is not None and latest.kind != "grounded_data" \
            and EXPLICIT_DATA_REF.search(_plain_data(request.question)):
        # "¿Y esa tienda en junio?" after a policy question: the explicit reference
        # points to the last verified DATA turn, which is re-queried, not replayed.
        resolved = resolve_data_followup(
            request.question, SESSIONS.latest_verified_of(request.conversation_id, "grounded_data"))
    effective_request = request.model_copy(update={"question": resolved}) if resolved else request
    response = _chat_once(effective_request, verified_context=latest)
    if resolved:
        response.metrics.resolved_question = resolved
    # If an elliptical utterance was answered via multiple model-selected
    # CSV queries, keep the precise *executed ranking query* as the memory
    # anchor. It is checked against DuckDB again, not trusted from model prose.
    memory_question = effective_request.question.strip()
    if (response.status == "answered"
            and response.metrics.answer_mode == "grounded_data"
            and any(e.get("source") == "ventas.csv" for e in response.evidence)
            and not looks_like_data(memory_question)):
        from app.tools.analytics import plain as _plain_data
        import re as _re
        requested_months = set(_re.findall(
            r"\b(?:junio|julio|agosto)\b", _plain_data(memory_question)
        ))
        for record in reversed(response.metrics.tool_calls):
            candidate = record.get("query", "")
            if (record.get("name") != "consultar_datos_tienda"
                    or record.get("status") != "answered"
                    or not isinstance(candidate, str)):
                continue
            q = _plain_data(candidate)
            if not _re.search(r"\b(?:tienda|sucursal)\b", q):
                continue
            if not _re.search(r"\b(?:vendio|mas|mayor)\b", q):
                continue
            if requested_months and not requested_months <= set(_re.findall(
                    r"\b(?:junio|julio|agosto)\b", q)):
                continue
            # A verified CSV result, not the model's answer, authorizes
            # using this query to ground later pronoun references.
            verified = answer_data_question(candidate)
            if verified.status == "answered" and verified.intent == "sales":
                memory_question = candidate
                break
    SESSIONS.add(
        request.conversation_id,
        memory_question,
        answer=response.answer,
        evidence=response.evidence,
        kind=response.metrics.answer_mode,
        status=response.status,
    )
    return response


def _chat_once(request: ChatRequest, verified_context=None) -> ChatResponse:
    start = time.perf_counter()
    question = request.question.strip()
    fast = _fast_analytics(question, start)
    if fast is not None:
        return fast
    if _agent_applies(question) and getattr(AGENT.settings, "verified_first", False):
        # Verified-first: the backend can already classify and answer most
        # business questions from DuckDB/RAG in milliseconds. Asking the model to
        # pick a tool first costs a full call (tens of seconds on a laptop) and it
        # sometimes answers without any tool. Retrieve first; the model only writes.
        probe = _deterministic_chat(request, start, _NO_REWRITE)
        if probe.status == "needs_clarification" and probe.metrics.answer_mode == "needs_context":
            return probe  # a precise clarifying question beats a speculative tool call
        if (probe.status == "answered" and probe.evidence
                and probe.metrics.answer_mode in ("grounded_data", "grounded_policy")):
            return _write_verified(probe, question, start)
    if _agent_applies(question):
        # A source-backed prior answer is context for the real ToolAgent only.
        # Test doubles and older implementations keep the three-argument API.
        if isinstance(AGENT, ToolAgent):
            out = AGENT.run(question, request.previous_question, request.history,
                            verified_context=verified_context)
        else:
            out = AGENT.run(question, request.previous_question, request.history)
        if out.handled:
            return ChatResponse(answer=out.answer, status=out.status, evidence=out.evidence,
                                conflicts=out.conflicts, metrics=_agent_metrics(out, start))
        # The model did not consult a tool (or Ollama failed before any tool ran):
        # answer deterministically, report the attempt, and do not call the model again.
        response = _deterministic_chat(request, start, _NO_REWRITE)
        m = response.metrics
        m.llm_calls += out.llm_calls
        m.input_tokens += out.input_tokens
        m.output_tokens += out.output_tokens
        m.total_tokens += out.input_tokens + out.output_tokens
        m.llm_model = out.model if out.llm_calls else m.llm_model
        m.generation_latency_ms = out.generation_latency_ms
        m.generated = False
        m.fallback_reason = out.fallback_reason
        m.warnings = out.warnings + list(m.warnings)
        m.tool_calls = out.tool_calls
        m.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return response
    return _deterministic_chat(request, start, None)


_RETURN_PROCEDURE = "procedimiento_devoluciones.pdf"
_RETURN_FLOW_SECTION = "7. Flujo de atención en tienda"


def _return_policy(route) -> PolicySearchResponse:
    """Evidence for a return question. For "how do I return X" the steps come from
    the procedure's in-store flow section, and the deadline/requirements from a
    search with the user's own words (which names the product). Only the missing
    piece of information is asked for, never what the user already said."""
    if route.kind != "general_process":
        return search_policies(PolicySearchRequest(query=route.query, top_k=5))
    flow = policy_section(_RETURN_PROCEDURE, _RETURN_FLOW_SECTION)
    if flow.status != "answered":
        return flow
    steps = "Pasos de atención en tienda:\n" + flow.answer
    if not route.product:
        return flow.model_copy(update={"answer": steps + (
            "\n\nEl plazo y los requisitos dependen del tipo de producto (tabla de la sección 3 del "
            "procedimiento). ¿Qué producto quiere devolver el cliente y tiene ticket de compra?")})
    specific = None
    # The user's words first; a follow-up merged with the previous question can be
    # too diffuse for retrieval, so retry with a query focused on the named product.
    for query in (route.query, f"Plazo y requisitos para devolver {route.product}"):
        specific = search_policies(PolicySearchRequest(query=query, top_k=5))
        if specific.status == "answered" and any(e.document == _RETURN_PROCEDURE for e in specific.evidence):
            break
    if specific.status != "answered" or not any(e.document == _RETURN_PROCEDURE for e in specific.evidence):
        return flow.model_copy(update={"answer": steps + (
            f"\n\nNo encontré en el procedimiento un plazo específico para «{route.product}». "
            "¿A qué tipo de producto corresponde y tiene ticket de compra?")})
    body = specific.answer.replace("Respuesta según los documentos:\n", "", 1).strip()
    return specific.model_copy(update={
        "answer": f"Plazo y requisitos para {route.product}:\n{body}\n\n{steps}",
        "evidence": list(specific.evidence) + list(flow.evidence),
    })


def _write_verified(probe: ChatResponse, question: str, start: float) -> ChatResponse:
    """One model call: the agent's model writes the verified answer; validation and
    the deterministic fallback stay exactly as in the rewrite path."""
    kind = "datos" if probe.metrics.answer_mode == "grounded_data" else "politicas"
    writer = GroundedWriter(AGENT.settings, AGENT.client) if isinstance(AGENT, ToolAgent) else WRITER
    prose = writer.rewrite(question=question, kind=kind, status=probe.status,
                           verified_answer=probe.answer, evidence=probe.evidence, conflicts=probe.conflicts)
    m = probe.metrics
    m.input_tokens, m.output_tokens = prose.input_tokens, prose.output_tokens
    m.total_tokens = prose.total_tokens
    m.llm_calls = prose.llm_calls
    m.llm_model = prose.model if prose.llm_calls else None
    m.generation_latency_ms = prose.generation_latency_ms
    m.generated, m.fallback_reason = prose.generated, prose.fallback_reason
    m.warnings = list(m.warnings) + prose.warnings
    m.tool_calls = [{"name": "consultar_datos_tienda" if kind == "datos" else "buscar_politicas",
                     "query": question, "status": probe.status, "planner": "verified_route"}]
    m.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return probe.model_copy(update={"answer": prose.answer, "metrics": m})


def _deterministic_chat(request: ChatRequest, start: float, writer: GroundedWriter | None) -> ChatResponse:
    writer = writer or WRITER  # resolved at call time so tests can swap WRITER
    question = request.question.strip()
    if not question:
        return ChatResponse(
            answer="Escribe una pregunta para continuar.",
            status="needs_clarification",
        )
    # Returns and receipts require policy-specific routing. The previous question
    # is explicit client-supplied context (not a server-side remembered session).
    # UI-provided history is bounded, explicit context, not server memory.
    context_question = request.previous_question
    if request.history:
        # Pick a past return-related question for short receipt follow-ups,
        # rather than treating an intervening "sin ticket" as a new topic.
        from app.api.return_intents import plain as _plain_return
        prior = next((item for item in reversed(request.history)
                      if "devol" in _plain_return(item) or "reembols" in _plain_return(item)), None)
        context_question = prior or request.history[-1]
    return_route = route_returns(question, context_question)
    if return_route is not None:
        if return_route.kind == "needs_context":
            return ChatResponse(
                answer=("¿Te refieres al ticket de compra para devolver un producto "
                        "o a un ticket de soporte de la Mesa de Servicio?"),
                status="needs_clarification",
                metrics=Metrics(answer_mode="needs_context",
                                latency_ms=round((time.perf_counter() - start) * 1000, 2)),
            )
        result = _return_policy(return_route)
        evidence = [item.to_http_evidence() for item in result.evidence]
        if not check_return_evidence(return_route.kind, status=result.status,
                                     answer=result.answer, evidence=evidence):
            message = (
                "No pude verificar en los documentos disponibles tanto la definición "
                "de cambio de producto como las condiciones de reembolso; no son "
                "automáticamente equivalentes. ¿Quieres consultar los requisitos de "
                "cambio o las condiciones de reembolso?"
                if return_route.kind == "exchange_vs_refund" else
                "No encontré una sección suficientemente pertinente del procedimiento "
                "general de devoluciones. ¿Qué tipo de producto desea devolver el cliente "
                "y tiene ticket de compra?"
                if return_route.kind == "general_process" else
                "No pude comprobar con evidencia pertinente las condiciones para devolver "
                "sin ticket de compra. Consulta la política vigente de devoluciones "
                "antes de autorizarla."
            )
            return ChatResponse(
                answer=message, status="needs_clarification",
                metrics=Metrics(answer_mode="needs_context",
                                latency_ms=round((time.perf_counter() - start) * 1000, 2)),
            )
        # A model may explain verified policy evidence, but policy rewriting
        # remains strictly validated: never relax citations or precedence.
        conflicts = [item.model_dump() for item in result.conflicts]
        prose = writer.rewrite(
            question=question, kind="politicas", status="answered",
            verified_answer=result.answer, evidence=evidence, conflicts=conflicts,
        )
        return ChatResponse(
            answer=prose.answer, status="answered", evidence=evidence,
            conflicts=conflicts,
            metrics=Metrics(
                answer_mode="grounded_policy", retrieval_mode=result.metrics.retrieval_mode,
                input_tokens=prose.input_tokens, output_tokens=prose.output_tokens,
                total_tokens=prose.total_tokens, llm_calls=prose.llm_calls,
                llm_model=prose.model if prose.llm_calls else None,
                generation_latency_ms=prose.generation_latency_ms,
                generated=prose.generated, fallback_reason=prose.fallback_reason,
                embedding_calls=result.metrics.embedding_calls,
                embedding_input_tokens=result.metrics.embedding_input_tokens,
                embedding_latency_ms=result.metrics.embedding_latency_ms,
                warnings=result.metrics.warnings + prose.warnings,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            ),
        )
    # Conversational/meta questions are not business data or policy evidence.
    # A connection question probes Ollama live; it must never claim connectivity
    # solely because a model name is configured.
    assistant_intent = classify_assistant_question(question)
    if assistant_intent:
        default_answer = assistant_answer(assistant_intent, LLM_SETTINGS)
        # Live availability diagnostics must be exact, not generated by a model.
        if assistant_intent in {"ollama_status", "model"}:
            return ChatResponse(
                answer=default_answer, status="answered",
                metrics=Metrics(answer_mode="assistant_info", latency_ms=round((time.perf_counter() - start) * 1000, 2)),
            )
        response = generate_general_answer(
            question, LLM_SETTINGS, default_answer=default_answer,
        )
        # Meta answers must describe the actual H-E-B assistant, not a
        # generic model that offers recipes and unrelated services.
        answer = response.answer
        generated = response.generated
        fallback = response.fallback_reason
        if generated and assistant_intent in {"capabilities", "identity"}:
            from app.api.assistant_chat import _plain as _normalize_meta
            lowered = _normalize_meta(answer)
            if not any(term in lowered for term in ("h-e-b", "heb", "duckdb", "rag", "ventas", "inventario", "politic", "tiendas", "asistente local")):
                answer = default_answer
                generated = False
                fallback = "respuesta_meta_fuera_de_alcance"
        return ChatResponse(
            answer=answer, status="answered",
            metrics=Metrics(
                answer_mode="general_unverified" if generated else "assistant_info",
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                total_tokens=response.input_tokens + response.output_tokens,
                llm_calls=response.llm_calls,
                llm_model=response.model if response.llm_calls else None,
                generation_latency_ms=response.latency_ms,
                generated=generated, fallback_reason=fallback,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            ),
        )
    # Catalog questions and unsupported employment cases need exact, relevant
    # evidence (or an explicit clarification), not a random policy passage.
    business_result = route_business_question(question)
    if business_result is not None:
        # The factual catalogue is deterministic, but the presentation belongs
        # to the model. Abstentions remain exact and are never hallucinated.
        source_kind = ("datos" if business_result.evidence and
                       all(e.get("source_type") == "csv" for e in business_result.evidence)
                       else "politicas")
        prose = writer.rewrite(
            question=question, kind=source_kind, status=business_result.status,
            verified_answer=business_result.answer, evidence=business_result.evidence,
        ) if business_result.evidence else None
        return ChatResponse(
            answer=prose.answer if prose else business_result.answer,
            status=business_result.status,
            evidence=business_result.evidence,
            metrics=Metrics(
                answer_mode=("grounded_data" if source_kind == "datos" else
                             "source_listing" if business_result.evidence else "needs_context"),
                input_tokens=prose.input_tokens if prose else 0,
                output_tokens=prose.output_tokens if prose else 0,
                total_tokens=prose.total_tokens if prose else 0,
                llm_calls=prose.llm_calls if prose else 0,
                llm_model=prose.model if prose and prose.llm_calls else None,
                generation_latency_ms=prose.generation_latency_ms if prose else 0,
                generated=prose.generated if prose else False,
                fallback_reason=prose.fallback_reason if prose else None,
                warnings=prose.warnings if prose else [],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            ),
        )
    # Non-business conversation uses the local LLM directly. It is not
    # represented as CSV/PDF-grounded evidence and cannot invent policy citations.
    if is_general_conversation(question):
        response = generate_general_answer(
            question, LLM_SETTINGS,
            default_answer=(
                "Puedo conversar sobre temas generales cuando Ollama está activado y disponible. "
                "Por ahora puedes preguntarme sobre las fuentes H-E-B cargadas."
            ),
        )
        return ChatResponse(
            answer=response.answer, status="answered",
            metrics=Metrics(
                answer_mode="general_unverified",
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                total_tokens=response.input_tokens + response.output_tokens,
                llm_calls=response.llm_calls,
                llm_model=response.model if response.llm_calls else None,
                generation_latency_ms=response.latency_ms, generated=response.generated,
                fallback_reason=response.fallback_reason,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            ),
        )
    # Data queries use fixed, parameterized SQL. Everything else goes to the RAG.
    # This also fixes previously unreachable policy questions such as SIGMA.
    if looks_like_data(question):
        try:
            result = answer_data_question(question)
        except Exception:
            return ChatResponse(
                answer="No pude consultar los CSV en este momento. Verifica que DuckDB esté instalado y los archivos estén disponibles.",
                status="error",
                metrics=Metrics(latency_ms=round((time.perf_counter() - start) * 1000, 2)),
            )
        outcome = writer.rewrite(
            question=question, kind="datos", status=result.status,
            verified_answer=result.answer, evidence=result.evidence,
        )
        return ChatResponse(
            answer=outcome.answer,
            status=result.status,
            evidence=result.evidence,
            metrics=Metrics(
                answer_mode="grounded_data",
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                total_tokens=outcome.total_tokens,
                llm_calls=outcome.llm_calls,
                llm_model=outcome.model if outcome.llm_calls else None,
                generation_latency_ms=outcome.generation_latency_ms,
                generated=outcome.generated,
                fallback_reason=outcome.fallback_reason,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                warnings=outcome.warnings,
            ),
        )

    # The RAG boundary handles index absence and search failures as status='error'.
    # Do not pass the whole question to PolicySearchRequest if it exceeds rag-v1's limit.
    if len(question) > 1000:
        return ChatResponse(
            answer="La consulta de políticas es demasiado larga (máximo 1000 caracteres).",
            status="needs_clarification",
        )
    result = search_policies(PolicySearchRequest(query=question, top_k=5))
    rag_metrics = result.metrics
    evidence = [item.to_http_evidence() for item in result.evidence]
    conflicts = [item.model_dump() for item in result.conflicts]
    outcome = writer.rewrite(
        question=question, kind="politicas", status=result.status,
        verified_answer=result.answer, evidence=evidence, conflicts=conflicts,
    )
    return ChatResponse(
        answer=outcome.answer,
        status=result.status,
        evidence=evidence,
        conflicts=conflicts,
        metrics=Metrics(
            answer_mode="grounded_policy",
            input_tokens=rag_metrics.input_tokens + outcome.input_tokens,
            output_tokens=rag_metrics.output_tokens + outcome.output_tokens,
            total_tokens=rag_metrics.input_tokens + rag_metrics.output_tokens + outcome.total_tokens,
            llm_calls=rag_metrics.llm_calls + outcome.llm_calls,
            llm_model=outcome.model if outcome.llm_calls else None,
            generation_latency_ms=outcome.generation_latency_ms,
            generated=outcome.generated,
            fallback_reason=outcome.fallback_reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            retrieval_mode=rag_metrics.retrieval_mode,
            embedding_calls=rag_metrics.embedding_calls,
            embedding_input_tokens=rag_metrics.embedding_input_tokens,
            embedding_latency_ms=rag_metrics.embedding_latency_ms,
            warnings=rag_metrics.warnings + outcome.warnings,
        ),
    )


@app.post("/query/aggregate")
def query_aggregate(request: AggregateRequest) -> dict:
    try:
        result = query_dataset(StructuredRequest(**request.model_dump()), CATALOG)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "rows": result.rows,
        "evidence": [{
            "source_type": "csv",
            "source": result.source,
            "reference": result.as_of,
        }],
        "metric": result.metric,
        "operation": result.operation,
        "filters": result.filters,
    }
