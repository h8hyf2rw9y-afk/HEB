"""Bounded, volatile chat context for the local Streamlit/FastAPI demo.

Only user questions are retained. History is conversation context, NEVER policy
evidence. No disk or external service is involved. State is lost on restart.
This is single-process memory; a production deployment needs an authenticated,
external session store.
"""
from __future__ import annotations

import threading
import time

MAX_TURNS = 6
MAX_SESSIONS = 128
TTL_SECONDS = 1800.0
MAX_QUESTION_CHARS = 400


class VolatileConversationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[float, list[str]]] = {}

    def _prune(self, now: float) -> None:
        for key, (seen, _) in list(self._sessions.items()):
            if now - seen >= TTL_SECONDS:
                del self._sessions[key]
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda k: self._sessions[k][0])
            del self._sessions[oldest]

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
                return []
            self._sessions[conversation_id] = (now, messages)
            return list(messages)

    def add(self, conversation_id: str | None, question: str) -> None:
        if not conversation_id:
            return
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            previous = self._sessions.get(conversation_id, (now, []))[1]
            recent = (previous + [question[:MAX_QUESTION_CHARS]])[-MAX_TURNS:]
            self._sessions[conversation_id] = (now, recent)

    def clear(self, conversation_id: str | None) -> None:
        if conversation_id:
            with self._lock:
                self._sessions.pop(conversation_id, None)

    def clear_all(self) -> None:
        """Test helper only."""
        with self._lock:
            self._sessions.clear()


SESSIONS = VolatileConversationStore()
