"""Bounded, volatile chat context for the local Streamlit/FastAPI demo.

Only user questions are retained. History is conversation context, NEVER policy
evidence. No disk or external service is involved. State is lost on restart.
This is single-process memory; a production deployment needs an authenticated,
external session store.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

MAX_TURNS = 6
MAX_SESSIONS = 128
TTL_SECONDS = 1800.0
MAX_QUESTION_CHARS = 400
MAX_ANSWER_CHARS = 1800
MAX_EVIDENCE_ITEMS = 12


@dataclass(frozen=True)
class VerifiedTurn:
    question: str
    answer: str
    evidence: list[dict] = field(default_factory=list)
    kind: str = ""

    @property
    def is_grounded(self) -> bool:
        return bool(self.evidence) and self.kind in {"grounded_data", "grounded_policy"}



class VolatileConversationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[float, list[str]]] = {}
        self._verified: dict[str, VerifiedTurn] = {}

    def _prune(self, now: float) -> None:
        for key, (seen, _) in list(self._sessions.items()):
            if now - seen >= TTL_SECONDS:
                del self._sessions[key]
                self._verified.pop(key, None)
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda k: self._sessions[k][0])
            del self._sessions[oldest]
            self._verified.pop(oldest, None)

    def history(self, conversation_id: str | None) -> list[str]:
        if not conversation_id:
            return []
        now = time.monotonic()
        with self._lock:
            pair = self._sessions.get(conversation_id)
            if pair is None:
                return []
            last_seen, messages = pair
            if now - last_seen >= TTL_SECONDS:
                del self._sessions[conversation_id]
                self._verified.pop(conversation_id, None)
                return []
            self._sessions[conversation_id] = (now, messages)
            return list(messages)

    def latest_verified(self, conversation_id: str | None) -> VerifiedTurn | None:
        if not conversation_id:
            return None
        # Reuse history() to apply the same expiration and refresh the TTL.
        if not self.history(conversation_id):
            return None
        with self._lock:
            return self._verified.get(conversation_id)

    def add(self, conversation_id: str | None, question: str, *,
            answer: str = "", evidence: list[dict] | None = None,
            kind: str = "", status: str = "") -> None:
        if not conversation_id:
            return
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            previous = self._sessions.get(conversation_id, (now, []))[1]
            recent = (previous + [question[:MAX_QUESTION_CHARS]])[-MAX_TURNS:]
            self._sessions[conversation_id] = (now, recent)
            # Only real, source-backed successful answers can update factual
            # continuity. General chat and abstentions cannot poison the cache.
            if status == "answered" and evidence and kind in {"grounded_data", "grounded_policy"}:
                safe_sources = [dict(item) for item in evidence[:MAX_EVIDENCE_ITEMS] if isinstance(item, dict)]
                if safe_sources:
                    self._verified[conversation_id] = VerifiedTurn(
                        question=question[:MAX_QUESTION_CHARS], answer=answer[:MAX_ANSWER_CHARS],
                        evidence=safe_sources, kind=kind,
                    )

    def clear(self, conversation_id: str | None) -> None:
        if conversation_id:
            with self._lock:
                self._sessions.pop(conversation_id, None)
                self._verified.pop(conversation_id, None)

    def clear_all(self) -> None:
        """Test helper only."""
        with self._lock:
            self._sessions.clear()
            self._verified.clear()


SESSIONS = VolatileConversationStore()
