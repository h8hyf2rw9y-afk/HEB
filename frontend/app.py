"""Local-first Streamlit interface for the H-E-B retail assistant.

Run: streamlit run frontend/app.py
The API is started separately: uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import os
import re
from uuid import uuid4

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
# A tool-calling agent may make several sequential Ollama calls; the UI
# must not time out before the backend's 120 s/model-call budget.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("HEB_CHAT_TIMEOUT_S", "540"))


def api_url() -> str:
    return os.getenv("HEB_API_URL", DEFAULT_API_URL).rstrip("/")


def health_check(base_url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{base_url}/health", timeout=3.0)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("status") != "ok":
            return False, "El servidor respondió, pero no reporta estado correcto."
        return True, "API conectada"
    except (httpx.HTTPError, ValueError) as exc:
        return False, f"Sin conexión: {type(exc).__name__}"


def send_question(base_url: str, question: str, previous_question: str | None = None,
                  history: list[str] | None = None, conversation_id: str | None = None) -> dict:
    response = httpx.post(
        f"{base_url}/chat",
        json={"question": question, "conversation_id": conversation_id,
              "previous_question": previous_question,
              "history": (history or [])[-6:]},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise ValueError("Respuesta inesperada del backend.")
    return data


def recent_user_history(messages: list[dict]) -> list[str]:
    """Send up to six preceding user messages, never assistant claims."""
    return [
        str(item.get("content", ""))[:400]
        for item in messages[:-1]
        if item.get("role") == "user"
    ][-6:]


def recent_question_context(messages: list[dict]) -> str | None:
    """Bounded context for return/receipt follow-ups, not backend memory."""
    prior_users = [
        str(message.get("content", ""))
        for message in messages[:-1]
        if message.get("role") == "user"
    ]
    if not prior_users:
        return None
    latest = prior_users[-1]
    # If the user repeats "sin ticket", retain the original return question
    # from at most three recent turns.
    if re.search(r"(?i)\\b(?:sin (?:el )?ticket|no (?:tiene|trae) (?:el )?ticket)\\b", latest):
        for prior in reversed(prior_users[-3:]):
            if re.search(r"(?i)\\b(?:devoluci[oó]n|devolver|reembolso)\\b", prior):
                return prior
    return latest


def display_answer(answer: str) -> str:
    """Separate policy warnings from Markdown list items in Streamlit."""
    return re.sub(r"(?<!\n)\n(?=[⚠ℹ])", "\n\n", answer)


def cited_evidence(evidence: object, answer: str) -> list[dict]:
    """Display only cited policy passages; CSV tables are always query sources."""
    if not isinstance(evidence, list):
        return []
    result = []
    seen = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("document") or "")
        reference = str(item.get("reference") or "")
        if item.get("source_type") == "pdf":
            if f"[{source}, {reference}]" not in answer:
                continue
        key = (source, reference)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def render_evidence(evidence: object) -> None:
    if not isinstance(evidence, list) or not evidence:
        return
    with st.expander(f"Fuentes ({len(evidence)})"):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or item.get("document") or "Fuente"
            reference = item.get("reference") or item.get("page") or ""
            st.markdown(f"**{source}** · {reference}")
            if item.get("passage"):
                st.caption(str(item["passage"]))


def fetch_json(url: str) -> dict | None:
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except (httpx.HTTPError, ValueError):
        return None


def render_sources(base_url: str) -> None:
    """Read-only view of what the assistant answers from (GET /sources, /health/ollama)."""
    st.subheader("Fuentes de datos")
    st.caption("Solo lectura: el asistente responde únicamente con estos archivos. No hay carga de archivos desde la interfaz.")
    sources = fetch_json(f"{base_url}/sources")
    if sources is None:
        st.warning("No se pudo consultar /sources. Revisa que FastAPI esté activo.")
        return
    st.markdown("**Datos (CSV, consultados con SQL fijo en DuckDB)**")
    st.dataframe([
        {"Archivo": d.get("file"), "Filas": d.get("rows"),
         "Rango de fechas": (f'{d["date_min"]} a {d["date_max"]}' if d.get("date_min") else "—"),
         "Columnas": ", ".join(d.get("columns") or [])}
        for d in sources.get("datasets", [])
    ], hide_index=True)
    policies = sources.get("policies") or {}
    st.markdown("**Políticas (PDF, índice RAG)**")
    if not policies.get("ready"):
        st.warning(f'Índice RAG no disponible: {policies.get("error") or "sin construir"}')
    else:
        st.dataframe([
            {"Documento": d.get("document"), "Título": d.get("title"), "Páginas": d.get("pages"),
             "Código": d.get("code"), "Versión": d.get("version")}
            for d in policies.get("documents", [])
        ], hide_index=True)
        st.caption(f'{policies.get("chunks")} fragmentos indexados · embeddings '
                   f'{"activos" if policies.get("embeddings_enabled") else "desactivados (BM25)"} · construido {policies.get("built_at")}')
    ollama = fetch_json(f"{base_url}/health/ollama")
    if ollama:
        st.markdown("**Modelo local (Ollama)**")
        state = ("responde" if ollama.get("reachable") else "no responde")
        model = ("instalado" if ollama.get("model_available") else "no encontrado")
        st.caption(f'Generación {"activada" if ollama.get("enabled") else "desactivada"} · '
                   f'{ollama.get("model")} {model} · Ollama {state} en {ollama.get("base_url")}')


def timing_line(metrics: dict) -> str | None:
    """Where the time went, from metrics.timings (real Ollama counters)."""
    t = metrics.get("timings") or {}
    if not t or not t.get("llm_ms"):
        return None
    line = (f'Tiempo del modelo: {t["llm_ms"] / 1000:.1f} s (carga {t.get("llm_load_ms", 0) / 1000:.1f} s · '
            f'lectura del prompt {t.get("llm_prompt_eval_ms", 0) / 1000:.1f} s · '
            f'generación {t.get("llm_generation_ms", 0) / 1000:.1f} s)')
    if t.get("tokens_per_s"):
        line += f' · {t["tokens_per_s"]} tokens/s'
    if t.get("tools_ms"):
        line += f' · herramientas {t["tools_ms"] / 1000:.2f} s'
    return line + f' · resto {t.get("other_ms", 0) / 1000:.2f} s'


def render_metrics(metrics: object) -> None:
    if not isinstance(metrics, dict):
        return
    with st.expander("Métricas de la consulta"):
        a, b, c = st.columns(3)
        a.metric("Latencia total", f'{metrics.get("latency_ms", 0)} ms')
        b.metric("Llamadas LLM", str(metrics.get("llm_calls", 0)))
        c.metric("Tokens LLM", f'{metrics.get("input_tokens", 0)} + {metrics.get("output_tokens", 0)}')
        st.caption(llm_summary(metrics))
        timing = timing_line(metrics)
        if timing:
            st.caption(timing)
        for line in tool_call_lines(metrics):
            st.caption(line)
        if metrics.get("embedding_calls"):
            st.caption(f'Embeddings: {metrics["embedding_calls"]} llamadas · {metrics.get("embedding_input_tokens", 0)} tokens')
        for warning in metrics.get("warnings") or []:
            st.caption(f"⚠ {warning}")


def tool_call_lines(metrics: dict) -> list[str]:
    """Agent mode: which verified query the model decided to run, and with what text."""
    lines = []
    for call in metrics.get("tool_calls") or []:
        label = {"consultar_datos_tienda": "Datos de tienda (DuckDB)", "buscar_politicas": "Políticas (PDF)"}.get(call.get("name"), call.get("name"))
        query = call.get("query") or call.get("arguments")
        lines.append(f'🔧 El modelo consultó {label}: «{query}» → {call.get("status")}')
        if call.get("warning"):
            lines.append(f'⚠ {call["warning"]}')
    return lines


def llm_summary(metrics: dict) -> str:
    """One line that says who wrote the answer, with real counters."""
    if metrics.get("generated"):
        grounded = metrics.get("answer_mode") in {"grounded_data", "grounded_policy"}
        suffix = (
            "Redacción validada contra los datos o documentos disponibles."
            if grounded else
            "Conversación general: no se consultaron CSV ni políticas; respuesta no verificada."
        )
        return (f'Redactada por {metrics.get("llm_model")} en {metrics.get("generation_latency_ms", 0)} ms '
                f'({metrics.get("input_tokens", 0)} tokens de entrada, {metrics.get("output_tokens", 0)} de salida). '
                + suffix)
    reason = metrics.get("fallback_reason") or "sin modelo"
    return f"Respuesta determinista (sin redacción del modelo): {reason}."


def answer_origin_notice(metrics: object) -> str | None:
    """Make the lack of enterprise sources visible outside the metrics expander."""
    if not isinstance(metrics, dict):
        return None
    mode = metrics.get("answer_mode")
    if mode == "general_unverified":
        return "Conversación general: respuesta sin consultar ni verificar fuentes de H-E-B."
    if mode == "needs_context":
        return "Orientación para precisar tu consulta; no se encontró una política laboral aplicable."
    if mode == "assistant_info":
        return "Información sobre el asistente; no se consultaron datos o políticas."
    return None


def main() -> None:
    st.set_page_config(page_title="HEB | Asistente de tienda", page_icon="🛒", layout="wide")
    st.title("Asistente de tienda")
    st.caption("Consultas sobre operación, ventas, inventario y políticas · entorno local")

    with st.sidebar:
        st.header("Entorno")
        base_url = st.text_input("API local", value=api_url())
        connected, status = health_check(base_url.rstrip("/"))
        if connected:
            st.success(status)
        else:
            st.warning(status)
        st.divider()
        page = st.radio("Navegación", ["Consultas", "Fuentes de datos"], label_visibility="collapsed")
        st.divider()
        st.caption("Los archivos de la tienda permanecen en tu equipo. No se envían a GitHub desde esta interfaz.")

    if page == "Fuentes de datos":
        render_sources(base_url.rstrip("/"))
        return

    st.subheader("Consultas")
    st.caption("Pregunta por las ventas de junio a agosto de 2026, el inventario, los tickets o las políticas internas.")
    st.caption("La conversación conserva hasta seis preguntas previas durante 30 minutos de inactividad (solo en memoria local). El historial no es evidencia de políticas; «Nueva conversación» borra la sesión.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = uuid4().hex
    if st.button("Nueva conversación"):
        try:
            httpx.post(
                f"{base_url.rstrip('/')}/chat/reset",
                json={"conversation_id": st.session_state.conversation_id},
                timeout=2.0,
            )
        except httpx.HTTPError:
            pass  # Session will expire on its own if the API is unavailable.
        st.session_state.messages = []
        st.session_state.conversation_id = uuid4().hex
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(display_answer(message["content"]))
            if message["role"] == "assistant":
                if message.get("status") not in (None, "answered"):
                    st.caption(f'Estado: {message["status"]}')
                notice = answer_origin_notice(message.get("metrics"))
                if notice:
                    st.caption(notice)
                render_evidence(cited_evidence(message.get("evidence"), message["content"]))
                render_metrics(message.get("metrics"))

    question = st.chat_input("Ej. ¿Qué tienda vendió más en agosto de 2026?", disabled=not connected)
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                result = send_question(
                    base_url.rstrip("/"), question,
                    recent_question_context(st.session_state.messages),
                    recent_user_history(st.session_state.messages),
                    st.session_state.conversation_id,
                )
                answer = result["answer"]
                st.markdown(display_answer(answer))
                if result.get("status") != "answered":
                    st.caption(f'Estado: {result.get("status", "desconocido")}')
                notice = answer_origin_notice(result.get("metrics"))
                if notice:
                    st.caption(notice)
                render_evidence(cited_evidence(result.get("evidence"), answer))
                render_metrics(result.get("metrics"))
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "status": result.get("status"),
                    "evidence": result.get("evidence", []),
                    "metrics": result.get("metrics", {}),
                })
            except httpx.ReadTimeout:
                message = ("La consulta tardó más que el tiempo de espera configurado. "
                           "FastAPI podría seguir procesándola. Revisa la terminal del backend "
                           "y aumenta HEB_CHAT_TIMEOUT_S si tu equipo necesita más tiempo.")
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message, "status": "error"})
            except (httpx.HTTPError, ValueError) as exc:
                message = f"No se pudo completar la consulta: {type(exc).__name__}. Revisa que FastAPI esté activo."
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message, "status": "error"})


if __name__ == "__main__":
    main()
