"""RAG module for H-E-B internal policy PDFs (owner: Claude Code).

Public API:
    retrieve_policy(query, top_k) -> list[Evidence]      # contract v0
    answer_policy_question(question, top_k) -> PolicyAnswer
    PolicyRetriever, PolicyIndex, RagSettings
"""

from .config import RagSettings
from .index import IndexNotFoundError, PolicyIndex
from .models import Evidence, PolicyAnswer, PolicyConflict
from .service import PolicyRetriever, answer_policy_question, retrieve_policy

__all__ = [
    "Evidence",
    "IndexNotFoundError",
    "PolicyAnswer",
    "PolicyConflict",
    "PolicyIndex",
    "PolicyRetriever",
    "RagSettings",
    "answer_policy_question",
    "retrieve_policy",
]
