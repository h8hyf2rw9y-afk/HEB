"""RAG configuration, read from environment variables (no secrets involved)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EmbeddingsMode = Literal["auto", "off", "required"]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class RagSettings:
    pdf_dir: Path = Path("data/policies")
    index_dir: Path = Path("data/rag_index")
    # Chunking
    chunk_max_words: int = 160
    chunk_overlap_words: int = 40
    # Retrieval
    top_k: int = 5
    min_coverage: float = 0.45
    min_embedding_similarity: float = 0.6
    # Embeddings via Ollama
    embeddings: EmbeddingsMode = "auto"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    ollama_timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "RagSettings":
        mode = _env("HEB_RAG_EMBEDDINGS", "auto").lower()
        if mode not in ("auto", "off", "required"):
            raise ValueError(f"HEB_RAG_EMBEDDINGS inválido: {mode!r} (auto|off|required)")
        return cls(
            pdf_dir=Path(_env("HEB_RAG_PDF_DIR", "data/policies")),
            index_dir=Path(_env("HEB_RAG_INDEX_DIR", "data/rag_index")),
            chunk_max_words=int(_env("HEB_RAG_CHUNK_MAX_WORDS", "160")),
            chunk_overlap_words=int(_env("HEB_RAG_CHUNK_OVERLAP_WORDS", "40")),
            top_k=int(_env("HEB_RAG_TOP_K", "5")),
            min_coverage=float(_env("HEB_RAG_MIN_COVERAGE", "0.45")),
            min_embedding_similarity=float(_env("HEB_RAG_MIN_EMBED_SIM", "0.6")),
            embeddings=mode,  # type: ignore[arg-type]
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            embed_model=_env("HEB_RAG_EMBED_MODEL", "bge-m3"),
            ollama_timeout_s=float(_env("HEB_RAG_OLLAMA_TIMEOUT_S", "30")),
        )
