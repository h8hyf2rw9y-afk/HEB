"""The conversation benchmark must not count a correct deterministic route as a failure."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_agent_conversations as bench  # noqa: E402


def test_deterministic_route_with_right_source_counts_as_correct(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/reset"):
            return httpx.Response(200, json={"reset": True})
        question = request.read().decode()
        is_data = any(w in question for w in ("vend", "ventas", "tienda", "Cumbres", "productos"))
        ext, mode = (".csv", "grounded_data") if is_data else (".pdf", "grounded_policy")
        return httpx.Response(200, json={
            "answer": "ok", "status": "answered", "conflicts": [],
            "evidence": [{"source_type": ext[1:], "source": "fuente" + ext}],
            "metrics": {"answer_mode": mode, "tool_calls": [], "total_tokens": 0,
                        "input_tokens": 0, "output_tokens": 0, "latency_ms": 1.0},
        })

    real_client = httpx.Client
    monkeypatch.setattr(bench.httpx, "Client",
                        lambda timeout=None: real_client(timeout=timeout, transport=httpx.MockTransport(handler)))
    result = bench.evaluate("http://fake", timeout=5)
    turns = [t for ep in result["episodes"] for t in ep["turns"]]
    assert turns and all(t["planner_action"] == "deterministic_route" for t in turns)
    assert result["summary"]["tool_selection_rate"] == 0.0  # the model chose no tool...
    correct = sum(t["route_correct"] for t in turns)
    assert result["summary"]["route_correct_rate"] == round(correct / len(turns), 3)
    assert correct > 0  # ...but routes that read the right kind of source count as correct
