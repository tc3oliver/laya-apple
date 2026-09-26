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


# ---- run 2 criteria: V2 busy coverage, V4 policy short path ----------------------------

R2 = PATH.parent / "criteria-r2.json"


def test_busy_coverage_counts_prefill_between_streamed_tokens():
    # two requests back to back; each streams only after its prefill
    reqs = [
        {"start_ns": 0, "chunks": [[3 * S, 1], [5 * S, 1]]},
        {"start_ns": 5 * S, "chunks": [[8 * S, 1], [9 * S, 1]]},
    ]
    w = {"window_ns": [0, 10 * S], "llm": {"requests": reqs}}
    assert analyze.busy_coverage(w) == pytest.approx(0.9)
    gen = analyze.bench.llm_window_summary(w["llm"], 0, 10 * S)["generating_coverage"]
    assert gen == pytest.approx(0.3)  # tokens streaming: 3-5 s and 8-9 s


def test_busy_coverage_gap_between_requests_is_idle():
    reqs = [
        {"start_ns": 0, "chunks": [[1 * S, 1], [4 * S, 1]]},
        {"start_ns": 7 * S, "chunks": [[8 * S, 1]]},
        {"start_ns": 9 * S, "chunks": []},  # no chunk: not counted (an error, gated separately)
    ]
    w = {"window_ns": [0, 10 * S], "llm": {"requests": reqs}}
    assert analyze.busy_coverage(w) == pytest.approx(0.5)  # 0-4 s and 7-8 s


def test_short_routing_counts_backlog_spill_as_policy_path():
    recs = [{"cls": "short_1q", "status": 200, "device": "ane", "reason": "validated_short_single_question_path"}] * 95
    recs += [{"cls": "short_1q", "status": 200, "device": "gpu", "reason": "ane_backlog_shorter_on_gpu"}] * 4
    recs += [{"cls": "short_1q", "status": 200, "device": "gpu", "reason": "ane_starting"}]
    recs += [{"cls": "mixed_3q", "status": 200, "device": "gpu", "reason": "multi_question"}] * 10
    recs += [{"cls": "short_1q", "status": 200, "device": "gpu", "reason": "x", "warmup": True}] * 10
    r = analyze.short_routing({"index": 3, "decisions": recs}, "ane_backlog_shorter_on_gpu")
    assert r["n"] == 100
    assert r["policy_share"] == pytest.approx(0.99)
    assert r["spill_share"] == pytest.approx(0.04)
    assert r["other"] == {"gpu:ane_starting": 1}


def edit_windows(campaign: Path, fn) -> None:
    for p in (campaign / "raw").glob("[0-9]*.json"):
        w = json.loads(p.read_text())
        fn(w)
        p.write_text(json.dumps(w))


def spill(share: float, reason: str = "ane_backlog_shorter_on_gpu"):
    """Move `share` of every auto window's short_1q answers to the GPU with `reason`."""

    def fn(w):
        if w.get("config") == "auto" and "decisions" in w:
            short = [r for r in w["decisions"] if r["cls"] == "short_1q"]
            for r in short[: round(share * len(short))]:
                r.update(device="gpu", reason=reason)

    return fn


def with_start_ns(w):
    for r in (w.get("llm") or {}).get("requests", []):
        r["start_ns"] = w["window_ns"][0]


def test_run2_criteria_accept_designed_spill_that_run1_rejects(tmp_path):
    campaign = make_campaign(tmp_path)
    edit_windows(campaign, with_start_ns)
    edit_windows(campaign, spill(0.05))
    run1 = analyze.summarise(campaign)  # no criteria.json in the campaign: run 1's criteria
    assert not run1["valid"] and not run1["validity"]["V4_auto_short_on_ane"]["ok"]
    assert "criteria_revision" not in run1

    (campaign / "criteria.json").write_text(R2.read_text())  # as run.sh records it
    run2 = analyze.summarise(campaign)
    v4 = run2["validity"]["V4_auto_short_policy_path"]
    assert v4["ok"] and v4["worst_policy_share"] == pytest.approx(1.0)
    assert list(v4["spill_share_windows"].values()) == pytest.approx([0.05, 0.05])
    assert "V4_auto_short_on_ane" not in run2["validity"]
    assert run2["validity"]["V2_llm_saturated"]["ok"]
    assert run2["valid"] and run2["criteria_revision"] == "r2"
    # L1 includes the spilled requests: they are still short_1q answers of the window
    assert run2["cells"]["auto/decisions_llm"]["classes"]["short_1q"]["devices"]["gpu"] > 0
    assert "Criteria revision `r2`" in analyze.tables(run2)


def test_run2_other_gpu_reason_counts_against_v4(tmp_path):
    campaign = make_campaign(tmp_path)
    edit_windows(campaign, with_start_ns)
    edit_windows(campaign, spill(0.05, reason="ane_starting"))
    res = analyze.summarise(campaign, R2)
    v4 = res["validity"]["V4_auto_short_policy_path"]
    assert not v4["ok"] and v4["worst_policy_share"] == pytest.approx(0.95)
    assert v4["other_routes"] == {"gpu:ane_starting": 8}  # 4 of 80 short answers, 2 auto windows
    assert {r["verdict"] for g in res["results"].values() for r in g.values()} == {"invalid"}


def test_run2_v2_gates_busy_not_generating_coverage(tmp_path):
    campaign = make_campaign(tmp_path)

    def prefill(w):  # 2 s of prefill, then tokens
        for r in (w.get("llm") or {}).get("requests", []):
            r["start_ns"] = w["window_ns"][0]
            r["chunks"] = [c for c in r["chunks"] if c[0] >= 2 * S]

    edit_windows(campaign, prefill)
    v2 = analyze.summarise(campaign, R2)["validity"]["V2_llm_saturated"]
    assert v2["worst_generating_coverage"] < 0.9 <= v2["worst_busy_coverage"]
    assert v2["ok"]
    assert not analyze.summarise(campaign)["validity"]["V2_llm_saturated"]["ok"]  # run 1's definition

    # a window where the harness had nothing in flight for the first 2 s fails run 2's V2
    p = sorted((campaign / "raw").glob("000-*.json"))[0]
    w = json.loads(p.read_text())
    w["llm"]["requests"][0]["start_ns"] = 2 * S
    p.write_text(json.dumps(w))
    assert not analyze.summarise(campaign, R2)["validity"]["V2_llm_saturated"]["ok"]


def test_run1_record_reproduces_with_its_own_criteria():
    campaign = PATH.parent / "m4-max"
    res = analyze.summarise(campaign)
    assert json.dumps(res, indent=1, sort_keys=True) + "\n" == (campaign / "results.json").read_text()
    assert analyze.tables(res) == (campaign / "tables.md").read_text()
    assert not res["valid"]


def test_run2_record_reproduces_with_its_own_criteria():
    campaign = PATH.parent / "m4-max-r2"
    res = analyze.summarise(campaign)
    assert json.dumps(res, indent=1, sort_keys=True) + "\n" == (campaign / "results.json").read_text()
    assert analyze.tables(res) == (campaign / "tables.md").read_text()
    assert res["valid"]


R3 = PATH.parent / "criteria-r3.json"


def with_handoff(disable_index=None):
    """Adaptive-execution records around every auto decision window, as bench_serve writes them."""

    def fn(w):
        if w.get("config") != "auto" or "decisions" not in w:
            return
        on = w["index"] != disable_index
        w["handoff"] = {
            "present": True,
            "enabled_before": True,
            "enabled_after": on,
            "consistent_before": True,
            "consistent_after": True,
            "state_before": "armed",
            "state_after": "async_healthy" if on else "disabled",
            "episodes": 1,
            "trips": int(w["kind"] == "decisions_llm"),
            "forwards_sync": 64,
            "forwards_async": 36,
            "async_share": 0.36,
            "disabled_reason": None if on else "handoff state corrupted: x",
        }

    return fn


def auto_window(campaign: Path, kind: str) -> int:
    for p in sorted((campaign / "raw").glob(f"*-{kind}.json")):
        w = json.loads(p.read_text())
        if w["config"] == "auto":
            return w["index"]
    raise AssertionError(kind)


def test_run3_gates_exactly_as_run2_and_records_adaptive_execution(tmp_path):
    campaign = make_campaign(tmp_path)
    edit_windows(campaign, with_start_ns)
    edit_windows(campaign, spill(0.04))
    edit_windows(campaign, with_handoff())
    r2, r3 = analyze.summarise(campaign, R2), analyze.summarise(campaign, R3)
    for key in ("valid", "validity", "results", "cells", "llm_tok_s_drop"):
        assert r2[key] == r3[key]
    assert "adaptive_execution" not in r2 and r3["criteria_revision"] == "r3"
    ae = r3["adaptive_execution"]
    assert ae["A1_adaptive_execution_in_use"] == {"ok": True, "windows_failed": []}
    assert ae["cells"]["auto/decisions_llm"] == {
        "episodes": 1,
        "trips": 1,
        "forwards_sync": 64,
        "forwards_async": 36,
        "async_share": pytest.approx(0.36),
        "windows": 1,
    }
    assert ae["gpu_windows_with_handoff"] == []
    assert "Adaptive ANE execution (recorded, not gated)" in analyze.tables(r3)


def test_run3_a1_failure_changes_no_verdict(tmp_path):
    campaign = make_campaign(tmp_path)
    edit_windows(campaign, with_start_ns)
    idx = auto_window(campaign, "decisions_llm")
    edit_windows(campaign, with_handoff(disable_index=idx))
    r2, r3 = analyze.summarise(campaign, R2), analyze.summarise(campaign, R3)
    assert r3["results"] == r2["results"] and r3["valid"] == r2["valid"]
    a1 = r3["adaptive_execution"]["A1_adaptive_execution_in_use"]
    assert not a1["ok"]
    assert a1["windows_failed"] == [{"index": idx, "disabled_reason": "handoff state corrupted: x"}]


def test_run3_windows_without_a_handoff_record_fail_a1(tmp_path):
    campaign = make_campaign(tmp_path)  # a serve without the /health field: nothing recorded
    a1 = analyze.summarise(campaign, R3)["adaptive_execution"]["A1_adaptive_execution_in_use"]
    assert not a1["ok"] and len(a1["windows_failed"]) == 2
