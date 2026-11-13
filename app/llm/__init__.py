"""Local generative layer (Ollama).

Two modes (``HEB_LLM_MODE``):
- ``agente`` (default): the model decides which verified tool to call (fixed-SQL
  DuckDB analytics or the policy RAG), then writes the answer from the tool output.
- ``redaccion``: the deterministic router queries and the model only rewrites.

In both, the model never writes SQL, never computes figures and never decides which
policy prevails; its text is validated against verified output (``validation.py``)
and any failure falls back to that verified output.
"""

from .agent import AgentOutcome, ToolAgent, ToolResult
from .config import LLMSettings
from .service import GenerationOutcome, GroundedWriter

__all__ = ["AgentOutcome", "GenerationOutcome", "GroundedWriter", "LLMSettings", "ToolAgent", "ToolResult"]
