"""Simulated Ollama helpers for tests/llm (unique module name: tests/rag also has a conftest)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

# tests/llm deliberately has NO conftest.py: tests/rag does `from conftest import ...`
# and a second conftest module would shadow it depending on collection order.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm.config import LLMSettings  # noqa: E402
from app.llm.ollama_client import OllamaChatClient  # noqa: E402
from app.llm.service import GroundedWriter  # noqa: E402


def ollama_reply(content: str, *, prompt_tokens: int = 420, completion_tokens: int = 85, model: str = "llama3.2") -> dict:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "load_duration": 120_000_000,
        "prompt_eval_duration": 900_000_000,
        "eval_duration": 1_500_000_000,
    }


class FakeOllama:
    """httpx transport that answers /api/chat with a scripted behaviour."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        b = self.behaviour
        if callable(b):
            b = b(self.requests[-1])
        if isinstance(b, Exception):
            raise b
        if isinstance(b, httpx.Response):
            return b
        return httpx.Response(200, json=b)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def make_writer(behaviour, *, enabled: bool = True, timeout_s: float = 5.0) -> tuple[GroundedWriter, FakeOllama]:
    fake = FakeOllama(behaviour)
    settings = LLMSettings(enabled=enabled, model="llama3.2", timeout_s=timeout_s)
    client = OllamaChatClient(settings.base_url, settings.model, settings.timeout_s, transport=fake.transport)
    return GroundedWriter(settings, client), fake


@pytest.fixture(scope="module")
def policy_index_dir(tmp_path_factory) -> Path:
    """Build the RAG index from the 4 exercise PDFs into a temp dir (data/rag_index is
    git-ignored, so CI has no prebuilt index)."""
    from app.rag.config import RagSettings
    from app.rag.extraction import discover_pdfs
    from app.rag.index import PolicyIndex

    pdfs = discover_pdfs(ROOT / "candidatos" / "politicas")
    if len(pdfs) != 4:
        pytest.skip("candidatos/politicas no disponible")
    target = tmp_path_factory.mktemp("rag_index")
    PolicyIndex.build(pdfs, RagSettings(embeddings="off")).save(target)
    return target


@pytest.fixture
def policy_index(policy_index_dir, monkeypatch):
    """Point the shared retriever used by /chat at the temp index for one test."""
    from app.rag import service

    monkeypatch.setenv("HEB_RAG_INDEX_DIR", str(policy_index_dir))
    monkeypatch.setenv("HEB_RAG_EMBEDDINGS", "off")
    service.reset_retriever()
    yield policy_index_dir
    service.reset_retriever()
