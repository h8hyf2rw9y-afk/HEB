"""Local-first Streamlit interface for the H-E-B retail assistant.

Run: streamlit run frontend/app.py
The API is started separately: uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 30.0


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


def send_question(base_url: str, question: str) -> dict:
    response = httpx.post(
        f"{base_url}/chat",
        json={"question": question, "conversation_id": None},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise ValueError("Respuesta inesperada del backend.")
    return data


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


def render_metrics(metrics: object) -> None:
    if not isinstance(metrics, dict):
        return
    with st.expander("Métricas de la consulta"):
        a, b, c = st.columns(3)
        a.metric("Latencia", f'{metrics.get("latency_ms", 0)} ms')
        b.metric("Llamadas LLM", str(metrics.get("llm_calls", 0)))
        c.metric("Tokens", str(metrics.get("total_tokens", 0)))


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
        st.subheader("Fuentes de datos")
        st.info("La carga de Excel, CSV, ZIP y PDF estará disponible en una fase posterior. Esta versión no almacena ni procesa archivos.")
        st.markdown("**Flujo previsto:** seleccionar periodo → revisar archivos → validar esquema y vista previa → confirmar importación → consultar versión activa.")
        st.markdown("**Formatos previstos:** Excel (.xlsx), CSV, PDF y ZIP. El soporte se implementará cuando existan endpoints seguros de ingestión.")
        st.caption("Por ahora, la carga e indexación se preparan localmente mediante los procesos del backend.")
        return

    st.subheader("Consultas")
    st.caption("El backend actual puede responder que la integración conversacional todavía no está disponible. La interfaz mostrará ese estado sin inventar resultados.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Nueva conversación"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("status") not in (None, "answered"):
                    st.caption(f'Estado: {message["status"]}')
                render_evidence(message.get("evidence"))
                render_metrics(message.get("metrics"))

    question = st.chat_input("Ej. ¿Qué tienda vendió más este mes?", disabled=not connected)
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                result = send_question(base_url.rstrip("/"), question)
                answer = result["answer"]
                st.markdown(answer)
                if result.get("status") != "answered":
                    st.caption(f'Estado: {result.get("status", "desconocido")}')
                render_evidence(result.get("evidence"))
                render_metrics(result.get("metrics"))
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "status": result.get("status"),
                    "evidence": result.get("evidence", []),
                    "metrics": result.get("metrics", {}),
                })
            except (httpx.HTTPError, ValueError) as exc:
                message = f"No se pudo completar la consulta: {type(exc).__name__}. Revisa que FastAPI esté activo."
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message, "status": "error"})


if __name__ == "__main__":
    main()
