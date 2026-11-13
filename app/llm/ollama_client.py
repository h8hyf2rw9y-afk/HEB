"""Minimal client for Ollama's local ``POST /api/chat`` endpoint."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

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
    # Tools the model asked to run: [{"name": str, "arguments": dict}] (only with ``tools=``).
    tool_calls: list[dict] = field(default_factory=list)

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


def _tool_calls(raw: object) -> list[dict]:
    """Normalise Ollama ``message.tool_calls``; a malformed shape is a bad response."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LLMBadResponse("tool_calls de Ollama no es una lista")
    calls = []
    for item in raw:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
            raise LLMBadResponse(f"tool_call de Ollama mal formado: {item!r}"[:200])
        args = fn.get("arguments", {})
        if isinstance(args, str):  # some Ollama versions/models send a JSON string
            try:
                args = json.loads(args) if args.strip() else {}
            except (json.JSONDecodeError, ValueError) as exc:
                raise LLMBadResponse("Argumentos de tool_call no son JSON válido") from exc
        if not isinstance(args, dict):
            raise LLMBadResponse("Argumentos de tool_call no son un objeto")
        calls.append({"name": fn["name"].strip(), "arguments": args})
    return calls


_CALL_LOG: ContextVar[list | None] = ContextVar("heb_llm_call_log", default=None)


@contextmanager
def record_llm_calls():
    """Collect stage timings of every Ollama /api/chat call made inside the block
    (one list per request; /chat wraps each request with it)."""
    token = _CALL_LOG.set([])
    try:
        yield _CALL_LOG.get()
    finally:
        _CALL_LOG.reset(token)


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_s: float, transport: httpx.BaseTransport | None = None,
                 keep_alive: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self._transport = transport  # injectable for tests (httpx.MockTransport)

    def chat(self, messages: list[dict], *, temperature: float = 0.0, seed: int = 42, num_predict: int = 700,
             tools: list[dict] | None = None) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict},
        }
        if tools:
            payload["tools"] = tools
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
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
        if content is None and tools:
            content = ""
        if not isinstance(content, str):
            raise LLMBadResponse("Respuesta de Ollama sin texto")
        tool_calls = _tool_calls(data["message"].get("tool_calls")) if tools else []
        if not content.strip() and not tool_calls:
            raise LLMEmptyResponse("Ollama devolvió una respuesta vacía")
        model = data.get("model")
        result = ChatResult(
            content=content.strip(),
            model=model if isinstance(model, str) and model.strip() else self.model,
            prompt_tokens=_token_count(data, "prompt_eval_count"),
            completion_tokens=_token_count(data, "eval_count"),
            latency_ms=latency_ms,
            load_ms=_ms(data.get("load_duration")),
            prompt_eval_ms=_ms(data.get("prompt_eval_duration")),
            eval_ms=_ms(data.get("eval_duration")),
            tool_calls=tool_calls,
        )
        log = _CALL_LOG.get()
        if log is not None:
            log.append({"latency_ms": result.latency_ms, "load_ms": result.load_ms,
                        "prompt_eval_ms": result.prompt_eval_ms, "eval_ms": result.eval_ms,
                        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                        "tokens_per_s": result.tokens_per_s, "used_tools": bool(tools)})
        return result
