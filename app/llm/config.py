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
        )
