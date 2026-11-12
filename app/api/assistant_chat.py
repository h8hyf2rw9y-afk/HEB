"""Deterministic assistant meta-conversation, isolated from business evidence.

Meta questions are not about the CSVs or policy PDFs. In particular, being
configured to use Ollama does not prove the Ollama server is reachable.
"""
from __future__ import annotations

import re
import unicodedata

import httpx

from app.llm.config import LLMSettings


def _plain(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


_OLLAMA = re.compile(r"\boll(?:a)?ma\b|\bllama\s*3(?:[.]\d)?\b")
_CONNECTION = re.compile(
    r"\b(conectad[oa]?|conexion|conexiones|disponible|funcionando|"
    r"funciona|activo|activa|corriendo|encendid[oa]|online|respondiendo|"
    r"acceso|comunicad[oa]?)\b"
)
_GREETING = re.compile(r"^(?:hola|buenos dias|buenas tardes|buenas noches|hey|que tal|saludos|holi)[!?,.\s]*$")
_THANKS = re.compile(r"^(?:gracias|muchas gracias|te agradezco)[!,.\s]*$")
_WHO = re.compile(r"\b(quien eres|que eres|como te llamas)\b")
_CAPABILITIES = re.compile(r"\b(que puedes hacer|en que me puedes ayudar|como me ayudas|para que sirves|ayuda|que sabes hacer)\b")
_MODEL = re.compile(r"\b(que modelo|cual modelo|modelo usas|modelo utilizas|que ia usas)\b")


def classify_assistant_question(question: str) -> str | None:
    """Route *only* specific assistant-meta intents, not arbitrary questions.

    A greeting combined with a health question takes precedence over greeting.
    """
    q = _plain(question).strip().strip(" ¡!¿?.,;:")
    if _OLLAMA.search(q) and (_CONNECTION.search(q) or "estas" in q or "esta" in q):
        return "ollama_status"
    if _OLLAMA.search(q) and (_MODEL.search(q) or re.search(r"\b(tienes|usas|utilizas|habilitado|instalado)\b", q)):
        return "ollama_status"
    if _MODEL.search(q):
        return "model"
    if _CAPABILITIES.search(q):
        return "capabilities"
    if _WHO.search(q):
        return "identity"
    if _GREETING.fullmatch(q):
        return "greeting"
    if _THANKS.fullmatch(q):
        return "thanks"
    return None


def check_ollama(settings: LLMSettings, *, timeout_s: float = 2.5) -> dict[str, object]:
    """Check live Ollama availability and model installation via /api/tags.

    Never infer server connectivity from environment variables. Never generate
    text or access user documents during the status check.
    """
    try:
        with httpx.Client(timeout=timeout_s) as client:
            res = client.get(f"{settings.base_url}/api/tags")
            res.raise_for_status()
            data = res.json()
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise ValueError("Ollama returned invalid model inventory")
        names = {
            m.get("name") for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)
        }
        configured = settings.model
        model_found = configured in names or (
            ":" not in configured and f"{configured}:latest" in names
        )
        return {"reachable": True, "model_available": model_found}
    except (httpx.HTTPError, ValueError, TypeError):
        return {"reachable": False, "model_available": False}


def assistant_answer(intent: str, settings: LLMSettings) -> str:
    if intent == "greeting":
        return ("¡Hola! Soy el asistente de H-E-B. Puedo consultar ventas, inventario, "
                "tickets de mesa de servicio y políticas internas con sus fuentes. ¿Qué necesitas?")
    if intent == "thanks":
        return "¡Con gusto! Puedes preguntarme sobre ventas, inventario, tickets o políticas."
    if intent == "identity":
        return ("Soy el asistente de consulta H-E-B. DuckDB calcula los resultados de los CSV, "
                "el RAG consulta políticas y Ollama puede redactar respuestas si está activado.")
    if intent == "capabilities":
        return ("Puedo responder consultas de ventas por tienda y periodo, inventario y punto de reorden, "
                "tickets de soporte, y procedimientos de los PDF con citas. También puedo informarte "
                "si Ollama está disponible. No invento datos que no existen en las fuentes.")
    if intent == "model":
        mode = "activada" if settings.enabled else "desactivada"
        return (f"El modelo generativo configurado es {settings.model} mediante Ollama; "
                f"la redacción generativa está {mode}. "
                "Si quieres comprobar la conexión real, pregúntame «¿Estás conectado con Ollama?».")
    if intent == "ollama_status":
        health = check_ollama(settings)
        if not health["reachable"]:
            return (f"No pude verificar conexión con Ollama en {settings.base_url}. "
                    "El asistente puede seguir respondiendo con DuckDB y RAG sin generación. "
                    "Comprueba que Ollama esté ejecutándose.")
        if not health["model_available"]:
            return (f"Ollama responde en {settings.base_url}, pero el modelo configurado "
                    f"({settings.model}) no aparece entre los instalados. "
                    "El asistente sigue disponible en modo determinista.")
        if settings.enabled:
            return (f"Sí, comprobé que Ollama responde y que {settings.model} está instalado. "
                    "La redacción generativa está activada; su funcionamiento por respuesta "
                    "se confirma en las métricas de cada consulta.")
        return (f"Sí, comprobé que Ollama responde y que {settings.model} está instalado, "
                "pero la redacción generativa está desactivada (HEB_LLM_ENABLED=false). "
                "Por ahora las respuestas se devuelven en modo determinista.")
    raise ValueError(f"Unknown assistant intent: {intent}")
