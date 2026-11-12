"""Prompt template for optional LLM generation over retrieved evidence.

The RAG module does not call the LLM itself (``app/llm`` is a shared module that
needs an agreed Issue). The orchestrator can pass this prompt to the Ollama adapter.
"""

from __future__ import annotations

from .models import PolicyAnswer

SYSTEM_PROMPT = (
    "Eres un asistente de operaciones de tienda. Responde en español usando EXCLUSIVAMENTE "
    "los pasajes numerados. Cita cada afirmación con [n]. Si los pasajes no contienen la "
    "respuesta, responde que no hay información suficiente. Si hay contradicciones, "
    "menciónalas; solo indica qué documento prevalece si la sección 'Conflictos' lo dice."
)


def build_grounded_prompt(answer: PolicyAnswer) -> str:
    parts = [f"Pregunta: {answer.question}", "", "Pasajes:"]
    for i, e in enumerate(answer.evidence, start=1):
        section = f", sección: {e.section}" if e.section else ""
        parts.append(f"[{i}] ({e.document}, p. {e.page}{section}) {e.passage}")
    if answer.conflicts:
        parts += ["", "Conflictos detectados:"]
        for c in answer.conflicts:
            status = (
                f"prevalece {c.preferred_document}" if c.preferred_document else "sin resolver"
            )
            parts.append(f"- {c.topic}: {status}. {c.basis}")
    parts += ["", "Respuesta:"]
    return "\n".join(parts)
