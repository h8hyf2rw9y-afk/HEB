"""Grounded rewriting: deterministic answer in, validated natural-language answer out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import LLMSettings
from .ollama_client import ChatResult, LLMError, OllamaChatClient
from .prompts import build_messages
from .validation import has_injection, validate_rewrite

logger = logging.getLogger(__name__)

REWRITABLE_STATUSES = {"answered"}


@dataclass
class GenerationOutcome:
    """What the API needs: the answer to show plus LLM metrics and fallback info."""

    answer: str
    generated: bool = False  # True if the shown answer came from the model
    model: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    generation_latency_ms: float = 0.0
    fallback_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GroundedWriter:
    def __init__(self, settings: LLMSettings, client: OllamaChatClient | None = None) -> None:
        self.settings = settings
        self.client = client or OllamaChatClient(settings.base_url, settings.model, settings.timeout_s)

    def rewrite(
        self,
        *,
        question: str,
        kind: str,
        status: str,
        verified_answer: str,
        evidence: list[dict],
        conflicts: list[dict] | None = None,
    ) -> GenerationOutcome:
        """Return the validated rewrite or the original answer with a fallback reason.

        Never raises. Only ``answered`` responses are rewritten: abstentions,
        clarifications and errors are returned verbatim, so an
        ``insufficient_data`` answer can never become an invented one.
        """
        fallback = GenerationOutcome(answer=verified_answer, model=self.settings.model)
        if not self.settings.enabled:
            fallback.fallback_reason = "llm_desactivado"
            return fallback
        if status not in REWRITABLE_STATUSES:
            fallback.fallback_reason = f"estado_{status}_no_se_redacta"
            return fallback
        if has_injection(verified_answer):
            # Suspicious instructions inside the evidence: do not expose them to the model.
            fallback.fallback_reason = "evidencia_con_instrucciones_sospechosas"
            fallback.warnings.append("La evidencia contiene texto con forma de instrucciones; se devuelve sin redactar.")
            return fallback

        start = time.perf_counter()
        try:
            result: ChatResult = self.client.chat(
                build_messages(question, kind, verified_answer, conflicts),
                temperature=self.settings.temperature,
                seed=self.settings.seed,
                num_predict=self.settings.max_output_tokens,
            )
        except LLMError as exc:
            fallback.fallback_reason = exc.reason
            fallback.generation_latency_ms = round((time.perf_counter() - start) * 1000, 1)
            fallback.warnings.append(f"Respuesta determinista; el modelo no se usó ({exc.reason}): {exc}")
            logger.info("LLM fallback (%s): %s", exc.reason, exc)
            return fallback

        outcome = GenerationOutcome(
            answer=verified_answer,
            model=result.model,
            llm_calls=1,
            input_tokens=result.prompt_tokens,
            output_tokens=result.completion_tokens,
            generation_latency_ms=result.latency_ms,
        )
        sources = {e.get("source", "") for e in evidence if isinstance(e, dict)}
        check = validate_rewrite(
            kind=kind,
            question=question,
            verified=verified_answer,
            candidate=result.content,
            allowed_sources=sources,
            conflicts=conflicts,
        )
        if not check.ok:
            outcome.fallback_reason = "validacion_fallida"
            outcome.warnings.append("Respuesta determinista; la redacción del modelo no pasó la validación: " + check.reason)
            logger.info("LLM rewrite rejected: %s", check.reason)
            return outcome
        outcome.answer = result.content
        outcome.generated = True
        return outcome
