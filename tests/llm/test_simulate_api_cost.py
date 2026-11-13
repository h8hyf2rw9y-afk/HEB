"""API cost simulation: arithmetic and report parsing (no network, no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import simulate_api_cost as sim  # noqa: E402


def test_cost_per_model_matches_list_prices():
    # 1,000 input + 200 output tokens
    assert sim.cost_usd(1000, 200, "claude-opus-5-5") == pytest.approx(0.004 + 0.004)
    assert sim.cost_usd(1000, 200, "claude-sonnet-5-5") == pytest.approx(0.002 + 0.002)
    assert sim.cost_usd(1000, 200, "claude-haiku-5-5") == pytest.approx(0.0001 + 0.0001)


def test_reads_mode_report_and_ignores_deterministic_rows_in_summary(tmp_path):
    report = {"modes": {"A_determinista": {"rows": [{"question": "q", "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}]},
                        "C_agente": {"rows": [{"question": "q", "input_tokens": 900, "output_tokens": 100, "llm_calls": 2}]}}}
    path = tmp_path / "eval_llm_modes.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = sim.simulate(sim.load_interactions(path), ["claude-opus-5-5"])
    s = result["summary"]
    assert s["interactions"] == 1  # the 0-token deterministic answer is excluded, not priced at $0
    assert result["excluded_without_tokens"] == [{"source": "eval_llm_modes.json:A_determinista", "question": "q"}]
    assert list(result["by_source"]) == ["eval_llm_modes.json:C_agente"]
    assert s["per_model"]["claude-opus-5-5"]["median_usd_per_interaction"] == pytest.approx(900 * 4e-6 + 100 * 20e-6)


def test_reads_conversation_report_and_rejects_totals_without_split(tmp_path):
    ok = {"episodes": [{"name": "ventas", "turns": [{"question": "q", "input_tokens": 500, "output_tokens": 50, "llm_calls": 1},
                                                     {"question": "err", "request_error": "timeout"}]}]}
    path = tmp_path / "conv.json"
    path.write_text(json.dumps(ok), encoding="utf-8")
    assert [r["input_tokens"] for r in sim.load_interactions(path)] == [500]
    old = {"episodes": [{"name": "x", "turns": [{"question": "q", "tokens": 550}]}]}
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit):
        sim.load_interactions(path)


def test_cli_manual_runs(capsys):
    assert sim.main(["--manual", "1000", "200", "--models", "claude-opus-5-5"]) == 0
    out = capsys.readouterr().out
    assert "SIMULACIÓN" in out and "8.00" in out  # $0.008 × 1,000 interactions
