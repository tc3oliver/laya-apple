"""Unit tests for benchmarks/serve/analyze.py on a synthetic campaign (analysis only)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "serve" / "analyze.py"
_spec = importlib.util.spec_from_file_location("serve_load_analyze", PATH)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

S = 1_000_000_000  # ns


def llm(tok_s: float, start: int, end: int) -> dict:
    """One request streaming evenly over the window at tok_s (1 char per token)."""
    n = int(tok_s * (end - start) / S)
    chunks = [[start + i * (end - start) // n, 1] for i in range(n)]
    usage = {"completion_tokens": n, "generation_tokens_per_second": tok_s, "time_to_first_token": 0.1}
    return {"requests": [{"status": 200, "chunks": chunks, "usage": usage}], "drain_aborted": 0}


def decisions(short_ms: float, mixed_ms: float, config: str, *, hard=0, prob_err=0.0) -> list[dict]:
    out = []
    for i in range(100):
        cls = "short_1q" if i % 5 else "mixed_3q"
        dev = "ane" if (config == "auto" and cls == "short_1q") else "gpu"
        lat = short_ms if cls == "short_1q" else mixed_ms
        out.append(
            {
                "cls": cls,
                "variant": 0,
                "t": i * 0.1,
                "warmup": False,
                "sched_ns": i * S // 10,
                "sent_ns": i * S // 10 + 100_000,
                "done_ns": i * S // 10 + int(lat * 1e6),
                "status": 200,
                "device": dev,
                "ref_device": dev,
                "prob_err": prob_err,
                "act_err": 0.0,
                "hard": ["q0"] if (hard and i == 1) else [],
                "flips": [],
            }
        )
    return out


def status(total: int) -> dict:
    return {"active_requests": 0, "waiting_requests": 0, "total_requests": total}


def make_campaign(tmp_path, *, auto_tok=39.0, gpu_tok=34.0, auto_loaded_ms=20.0, hard=0) -> Path:
    raw = tmp_path / "c" / "raw"
    raw.mkdir(parents=True)
    (raw / "campaign.json").write_text(
        json.dumps({"smoke": False, "args": {"clients": 8, "rate": 1.0, "llm_model": "m"}, "arrivals": "open loop"})
    )
    windows = [("llm_alone", None, 40.0, None), ("llm_alone", None, 40.4, None)]
    for cfg, tok, loaded in (("gpu", gpu_tok, 60.0), ("auto", auto_tok, auto_loaded_ms)):
        windows += [("decisions", cfg, None, 15.0), ("decisions_llm", cfg, tok, loaded)]
    total = 0
    for i, (kind, cfg, tok, short_ms) in enumerate(windows):
        w = {"index": i, "kind": kind, "config": cfg, "window_ns": [0, 10 * S], "llm_status_before": status(total)}
        if tok is not None:
            w["llm"] = llm(tok, 0, 10 * S)
            total += 1
        if kind != "llm_alone":
            w["decisions"] = decisions(short_ms, 30.0, cfg, hard=hard if cfg == "auto" else 0)
        w["llm_status_after"] = status(total)
        (raw / f"{i:03d}-{kind}.json").write_text(json.dumps(w))
    for cfg in ("gpu", "auto"):
        refs = {
            "short_1q": [{"device": "ane" if cfg == "auto" else "gpu"}],
            "mixed_3q": [{"device": "gpu"}],
        }
        (raw / f"refs-step0{cfg == 'auto'}-{cfg}.json").write_text(json.dumps({"config": cfg, "references": refs}))
    return tmp_path / "c"


def test_passing_campaign_reports_each_result_separately(tmp_path):
    res = analyze.summarise(make_campaign(tmp_path))
    assert res["valid"]
    assert res["llm_tok_s_drop"]["auto"] == pytest.approx(1 - 39.0 / 40.2, abs=1e-3)
    r = res["results"]
    assert r["gpu_free"]["G1_llm_cost_auto"]["verdict"] == "pass"
    assert r["gpu_free"]["G2_auto_below_gpu_only"]["verdict"] == "pass"
    assert r["decision_latency"]["L1_short_p99_loaded_abs_ms"]["value"] == pytest.approx(20.0, abs=0.01)
    assert r["decision_latency"]["L2_short_p99_loaded_vs_unloaded"]["value"] == pytest.approx(20 / 15, abs=1e-3)
    assert r["correctness"]["C1_fp16_gate"]["verdict"] == "pass"
    assert "passed" not in res  # no overall verdict
    assert "Criteria" in analyze.tables(res)


def test_llm_cost_and_no_gain_over_gpu_only_fail(tmp_path):
    res = analyze.summarise(make_campaign(tmp_path, auto_tok=36.0, gpu_tok=36.0))
    g = res["results"]["gpu_free"]
    assert g["G1_llm_cost_auto"]["verdict"] == "fail"
    assert g["G2_auto_below_gpu_only"]["verdict"] == "fail"
    # the other results are unaffected
    assert res["results"]["correctness"]["C1_fp16_gate"]["verdict"] == "pass"


def test_latency_and_correctness_fail_independently(tmp_path):
    res = analyze.summarise(make_campaign(tmp_path, auto_loaded_ms=80.0, hard=1))
    assert res["results"]["decision_latency"]["L1_short_p99_loaded_abs_ms"]["verdict"] == "fail"
    assert res["results"]["decision_latency"]["L2_short_p99_loaded_vs_unloaded"]["verdict"] == "fail"
    assert res["results"]["correctness"]["C1_fp16_gate"]["hard_mismatches"] == 2  # one per auto window
    assert res["results"]["correctness"]["C1_fp16_gate"]["verdict"] == "fail"
    assert res["results"]["gpu_free"]["G1_llm_cost_auto"]["verdict"] == "pass"


def test_foreign_llm_traffic_makes_results_invalid(tmp_path):
    campaign = make_campaign(tmp_path)
    p = sorted((campaign / "raw").glob("000-*.json"))[0]
    w = json.loads(p.read_text())
    w["llm_status_after"]["total_requests"] += 1  # someone else's request during the window
    p.write_text(json.dumps(w))
    res = analyze.summarise(campaign)
    assert not res["valid"] and res["validity"]["V3_llm_exclusive"]["windows_failed"] == [0]
    assert {r["verdict"] for g in res["results"].values() for r in g.values()} == {"invalid"}
