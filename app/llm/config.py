"""LLM settings from environment variables (no secrets, no machine-specific values)."""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUE = {"1", "true", "yes", "si", "sí", "on"}


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout_s: float = 60.0
    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int = 700
    # "agente": the model chooses and runs the verified tools (DuckDB/RAG) and writes the answer.
    # "redaccion": the deterministic router queries and the model only rewrites the result.
    mode: str = "agente"
    # Agent mode: when the deterministic router already returns a verified answer
    # (DuckDB or RAG, with evidence), skip the model's tool-selection call and let
    # the model only write the answer (one call). Ambiguous questions still go to
    # the tool-calling agent.
    verified_first: bool = True
    # How long Ollama keeps the model in memory after a call. Reloading a 4B model
    # adds seconds to the next question; "30m" keeps it warm during a demo.
    keep_alive: str = "30m"

    @classmethod
    def from_env(cls) -> "LLMSettings":
        def env(name: str, default: str) -> str:
            value = os.getenv(name)
            return value.strip() if value and value.strip() else default

        return cls(
            enabled=env("HEB_LLM_ENABLED", "false").lower() in _TRUE,
            base_url=env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            model=env("HEB_LLM_MODEL", "llama3.2"),
            timeout_s=float(env("HEB_LLM_TIMEOUT_S", "60")),
            max_output_tokens=int(env("HEB_LLM_MAX_OUTPUT_TOKENS", "700")),
            keep_alive=env("HEB_LLM_KEEP_ALIVE", "30m"),
            verified_first=env("HEB_LLM_VERIFIED_FIRST", "true").lower() in _TRUE,
            mode="redaccion" if env("HEB_LLM_MODE", "agente").lower() in {"redaccion", "redacción", "rewrite"} else "agente",
        )
