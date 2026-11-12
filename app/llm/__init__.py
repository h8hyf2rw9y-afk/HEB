"""Local generative layer (Ollama) that rewrites verified answers.

The model never computes figures, chooses SQL or decides which policy prevails:
it only rewrites an answer that DuckDB or the RAG already produced, and the
result is validated before it reaches the user (``validation.py``). Any failure
falls back to the original deterministic answer.
"""

from .config import LLMSettings
from .service import GenerationOutcome, GroundedWriter

__all__ = ["GenerationOutcome", "GroundedWriter", "LLMSettings"]
