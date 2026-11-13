"""Optional general conversation. Never use this path to answer policy or CSV questions.

This is deliberately separate from grounded rewriting: answers from this path
must not be represented as verified or cited business evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import LLMSettings
from .ollama_client import LLMError, OllamaChatClient


@dataclass(frozen=True)
class CasualResult:
    answer: str
    generated: bool = False
    model: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    fallback_reason: str | None = None


CASUAL_SYSTEM_PROMPT = """Eres el asistente conversacional de una prueba local para H-E-B.
Responde en español claro y natural, de manera breve y cálida.
Esta conversación es GENERAL y no tiene documentos ni datos corporativos como contexto.
No inventes acceso a sistemas, documentos, Internet, cuentas ni historial.
No afirmes haber verificado datos de H-E-B, sus políticas, personal ni inventarios.
Tu función real es ayudar a gerentes H-E-B a consultar ventas por tienda y periodo,
inventario, catálogo de productos, tickets de soporte y políticas de los PDF.
Las consultas de negocio se resuelven con herramientas verificadas de DuckDB y RAG;
no tienes acceso general a Internet, a cuentas privadas ni a todos los datos de H-E-B.
Si preguntan "¿qué puedes hacer?", describe principalmente estas capacidades
con ejemplos reales, no un catálogo genérico de recetas, horóscopos o cuidados personales.
Si te preguntan "¿para qué fuiste creado?", explica el propósito del asistente de tienda.
Si te piden un hecho interno de H-E-B, pide que lo consulten mediante los datos o políticas disponibles.
Si preguntan por tu identidad, di que eres un asistente local y que Ollama puede generar tu redacción.
Nunca afirmes que Ollama está conectado si no se ha realizado una comprobación real.
No incluyas citas ni cifras corporativas inventadas, y no des consejo laboral individual como política interna."""


def generate_general_answer(
    question: str,
    settings: LLMSettings,
    *,
    default_answer: str,
    client: OllamaChatClient | None = None,
) -> CasualResult:
    """Return optional LLM prose with honest fallback; non-grounded on purpose."""
    if not settings.enabled:
        return CasualResult(answer=default_answer, fallback_reason="llm_desactivado")
    provider = client or OllamaChatClient(settings.base_url, settings.model, settings.timeout_s,
                                                keep_alive=settings.keep_alive)
    try:
        response = provider.chat(
            [
                {"role": "system", "content": CASUAL_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2, seed=settings.seed, num_predict=min(settings.max_output_tokens, 180),
        )
    except LLMError as exc:
        return CasualResult(answer=default_answer, fallback_reason=exc.reason)
    except Exception:
        # Keep the general chat usable even if a provider returns an
        # unexpected object; unlike policy answers, this is not fact-verified.
        return CasualResult(answer=default_answer, fallback_reason="error_inesperado_llm")
    candidate = response.content.strip()
    if not candidate or len(candidate) > 1200 or re.search(r"\[(?:[^\]]+)\.pdf,\s*p\.", candidate):
        return CasualResult(
            answer=default_answer, model=response.model, llm_calls=1,
            input_tokens=response.prompt_tokens, output_tokens=response.completion_tokens,
            latency_ms=response.latency_ms, fallback_reason="validacion_fallida",
        )
    return CasualResult(
        answer=candidate, generated=True, model=response.model, llm_calls=1,
        input_tokens=response.prompt_tokens, output_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
    )
