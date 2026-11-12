"""Test doubles for the embedding provider (stand-ins for Ollama)."""

import hashlib
import math

from app.rag.embeddings import EmbeddingError, EmbeddingUsage
from app.rag.text import tokenize


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (stands in for Ollama)."""

    def __init__(self, model: str = "fake-embed", dim: int = 128) -> None:
        self.model = model
        self.dim = dim
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out, EmbeddingUsage(calls=1, input_tokens=sum(len(t.split()) for t in texts), latency_ms=1.0)


class DownEmbedder:
    model = "fake-embed"

    def embed(self, texts):
        raise EmbeddingError("Ollama no disponible (simulado)")
