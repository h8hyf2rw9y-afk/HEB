"""Embedding providers. Ollama access is isolated behind this adapter.

Uses only the standard library for HTTP so no new dependency is required.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingUsage:
    calls: int = 0
    input_tokens: int = 0
    latency_ms: float = 0.0

    def add(self, other: "EmbeddingUsage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.latency_ms += other.latency_ms


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingUsage]: ...


@dataclass
class OllamaEmbedder:
    """Calls ``POST {base_url}/api/embed`` (Ollama >= 0.3)."""

    model: str = "bge-m3"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 30.0
    batch_size: int = 32
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Ollama no disponible en {self.base_url}: {exc}") from exc

    def is_available(self) -> bool:
        try:
            self.embed(["prueba"])
            return True
        except EmbeddingError:
            return False

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingUsage]:
        vectors: list[list[float]] = []
        usage = EmbeddingUsage()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            t0 = time.perf_counter()
            data = self._post("/api/embed", {"model": self.model, "input": batch})
            usage.latency_ms += (time.perf_counter() - t0) * 1000
            usage.calls += 1
            usage.input_tokens += int(data.get("prompt_eval_count") or 0)
            batch_vectors = data.get("embeddings")
            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
                raise EmbeddingError(f"Respuesta inesperada de Ollama: {str(data)[:200]}")
            vectors.extend(batch_vectors)
        self.usage.add(usage)
        return vectors, usage


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def cosine_similarities(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Pure-Python cosine similarity (corpora here are small; avoids a numpy dependency)."""
    qn = _norm(query)
    return [sum(a * b for a, b in zip(query, row)) / (qn * _norm(row)) for row in matrix]
