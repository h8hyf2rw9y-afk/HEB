"""Minimal client for Ollama's local ``POST /api/chat`` endpoint."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    """Base class; ``reason`` is a short machine-readable code for metrics."""

    reason = "error"


class LLMUnavailable(LLMError):
    reason = "ollama_no_disponible"


class LLMTimeout(LLMError):
    reason = "timeout"


class LLMBadResponse(LLMError):
    reason = "respuesta_invalida"


class LLMEmptyResponse(LLMError):
    reason = "respuesta_vacia"


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int  # Ollama prompt_eval_count
    completion_tokens: int  # Ollama eval_count
    latency_ms: float  # wall clock, measured by us
    load_ms: float = 0.0  # Ollama load_duration (model load into memory)
    prompt_eval_ms: float = 0.0
    eval_ms: float = 0.0

    @property
    def tokens_per_s(self) -> float | None:
        return round(self.completion_tokens / (self.eval_ms / 1000), 1) if self.eval_ms > 0 else None


def _ms(ns: object) -> float:
    """Ollama durations are nanoseconds; anything else counts as 0 (informational field)."""
    if isinstance(ns, bool) or not isinstance(ns, (int, float)) or ns < 0 or ns != ns:
        return 0.0
    return round(float(ns) / 1e6, 1)


def _token_count(data: dict, field: str) -> int:
    """Validate an Ollama token counter: absent/null → 0; otherwise a non-negative int.

    Token counts are reported to the user and summed into costs, so a value that
    is not a plain non-negative integer (strings, lists, floats, booleans,
    negatives) is a malformed response, not something to coerce.
    """
    value = data.get(field)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMBadResponse(f"Contador de tokens inválido en Ollama: {field}={value!r}")
    if value < 0:
        raise LLMBadResponse(f"Contador de tokens negativo en Ollama: {field}={value}")
    return value


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_s: float, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport  # injectable for tests (httpx.MockTransport)

    def chat(self, messages: list[dict], *, temperature: float = 0.0, seed: int = 42, num_predict: int = 700) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict},
        }
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"Ollama no respondió en {self.timeout_s} s") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"No se pudo conectar con Ollama en {self.base_url}: {type(exc).__name__}") from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if response.status_code != 200:
            raise LLMBadResponse(f"Ollama devolvió HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMBadResponse("Ollama devolvió JSON malformado") from exc
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise LLMBadResponse("Respuesta de Ollama sin campo 'message'")
        content = data["message"].get("content")
        if not isinstance(content, str):
            raise LLMBadResponse("Respuesta de Ollama sin texto")
        if not content.strip():
            raise LLMEmptyResponse("Ollama devolvió una respuesta vacía")
        model = data.get("model")
        return ChatResult(
            content=content.strip(),
            model=model if isinstance(model, str) and model.strip() else self.model,
            prompt_tokens=_token_count(data, "prompt_eval_count"),
            completion_tokens=_token_count(data, "eval_count"),
            latency_ms=latency_ms,
            load_ms=_ms(data.get("load_duration")),
            prompt_eval_ms=_ms(data.get("prompt_eval_duration")),
            eval_ms=_ms(data.get("eval_duration")),
        )
