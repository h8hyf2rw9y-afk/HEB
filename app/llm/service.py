"""Grounded rewriting: deterministic answer in, validated natural-language answer out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import LLMSettings
from .ollama_client import ChatResult, LLMError, OllamaChatClient
from .prompts import build_messages, build_summary_messages
from .validation import compose_grounded_answer, has_injection, validate_data_summary, validate_rewrite

SUMMARY_MAX_TOKENS = 220
POLICY_MAX_TOKENS = 1600


def policy_output_budget(verified_answer: str, configured: int) -> int:
    """A policy rewrite must reproduce every condition and citation, so a fixed
    cap can cut a long answer mid-citation (then validation fails after the whole
    wait). Budget about one token per 3 characters of the verified text plus room
    for connecting prose, never below the configured value or above 1600."""
    return max(configured, min(POLICY_MAX_TOKENS, len(verified_answer) // 3 + 150))

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
        self.client = client or OllamaChatClient(settings.base_url, settings.model, settings.timeout_s,
                                                keep_alive=settings.keep_alive)

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
            summary_mode = kind == "datos" and not conflicts
            result: ChatResult = self.client.chat(
                build_summary_messages(question, verified_answer) if summary_mode
                else build_messages(question, kind, verified_answer, conflicts),
                temperature=self.settings.temperature,
                seed=self.settings.seed,
                # A data summary is 2-4 sentences: cap generation, which is the
                # slowest stage on a laptop (about 5 tokens/s observed with Qwen3 4B).
                num_predict=min(self.settings.max_output_tokens, SUMMARY_MAX_TOKENS) if summary_mode
                else policy_output_budget(verified_answer, self.settings.max_output_tokens),
            )
        except LLMError as exc:
            fallback.fallback_reason = exc.reason
            fallback.generation_latency_ms = round((time.perf_counter() - start) * 1000, 1)
            fallback.warnings.append(f"Respuesta determinista; el modelo no se usó ({exc.reason}): {exc}")
            logger.info("LLM fallback (%s): %s", exc.reason, exc)
            return fallback
        except Exception as exc:  # noqa: BLE001 - last line of defence for /chat
            # Anything the provider path raises that we did not classify (a new
            # httpx error type, an unexpected payload shape) must not break the
            # user's answer. It is logged as an error so it gets fixed, and the
            # metrics name it as an unexpected failure rather than a known one.
            fallback.fallback_reason = "error_inesperado_llm"
            fallback.generation_latency_ms = round((time.perf_counter() - start) * 1000, 1)
            fallback.warnings.append(
                f"Respuesta determinista; fallo inesperado al llamar al modelo: {type(exc).__name__}"
            )
            logger.exception("Unexpected LLM failure; deterministic answer kept")
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
        if summary_mode:
            return self._data_summary(outcome, question, verified_answer, result.content, sources)
        composed = compose_grounded_answer(
            kind=kind,
            question=question,
            verified=verified_answer,
            candidate=result.content,
            allowed_sources=sources,
            conflicts=conflicts,
        )
        if not composed.ok:
            outcome.fallback_reason = "validacion_fallida"
            reason = "; ".join(composed.failures)
            outcome.warnings.append("Respuesta determinista; la redacción del modelo no pasó la validación: " + reason)
            logger.info("LLM rewrite rejected: %s", reason)
            return outcome
        outcome.answer = composed.answer
        outcome.generated = True
        if composed.evidence_appended:
            outcome.warnings.append("El modelo redactó el resumen; se adjuntaron todas las cifras verificadas para no omitir resultados.")
        return outcome

    @staticmethod
    def _data_summary(outcome: GenerationOutcome, question: str, verified: str, summary: str,
                      sources: set[str]) -> GenerationOutcome:
        """Data answers: the model's summary plus the verified table, unless the
        summary already carries every verified figure (then the table would repeat it)."""
        check = validate_data_summary(question=question, verified=verified, summary=summary,
                                      allowed_sources=sources)
        if not check.ok:
            outcome.fallback_reason = "validacion_fallida"
            outcome.warnings.append("Respuesta determinista; el resumen del modelo no pasó la validación: "
                                    + "; ".join(check.failures))
            logger.info("LLM summary rejected: %s", check.failures)
            return outcome
        # Omit the table only when the summary alone passes the strict full check
        # (every row, figure, store code and entity present and correctly paired).
        complete = validate_rewrite(kind="datos", question=question, verified=verified,
                                    candidate=summary, allowed_sources=sources).ok
        outcome.answer = summary.strip() if complete else (
            summary.strip() + "\n\n**Detalle verificado (DuckDB):**\n" + verified)
        outcome.generated = True
        return outcome
