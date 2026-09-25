"""Unit tests for the research-only energy sampler's pure logic (research/energy-sampler/).

No IOKit, no model: integration of power and energy over time, idle-baseline subtraction,
ABBA ordering, powermetrics parsing and the analysis on synthetic runs.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "energy-sampler" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import energy  # noqa: E402

# Other research tracks also have an `analyze` module: load this one under its own name so
# sys.modules["analyze"] stays free for their tests.
_spec = importlib.util.spec_from_file_location("energy_sampler_analyze", SCRIPTS / "analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

# --------------------------------------------------------------------- energy over time


def test_cumulative_interpolates_linearly_between_samples():
    ts, cum = [0.0, 1.0, 2.0], [0.0, 10.0, 30.0]
    assert energy.cumulative_at(ts, cum, 0.5) == pytest.approx(5.0)
    assert energy.cumulative_at(ts, cum, 1.5) == pytest.approx(20.0)
    assert energy.cumulative_at(ts, cum, 2.0) == pytest.approx(30.0)
    assert energy.window_energy(ts, cum, 0.5, 1.5) == pytest.approx(15.0)


def test_cumulative_refuses_to_extrapolate():
    with pytest.raises(ValueError):
        energy.cumulative_at([0.0, 1.0], [0.0, 1.0], 1.5)
    with pytest.raises(ValueError):
        energy.window_energy([0.0, 1.0], [0.0, 1.0], 0.8, 0.2)


def test_constant_power_integrates_to_power_times_time():
    ts = [i * 0.5 for i in range(21)]
    assert energy.integrate_power(ts, [7.0] * 21, 1.25, 8.75) == pytest.approx(7.0 * 7.5)
    assert energy.integrate_power(ts, [7.0] * 21, 1.25, 8.75, hold=True) == pytest.approx(7.0 * 7.5)


def test_trapezoid_integrates_a_ramp_exactly():
    ts = [0.0, 1.0, 2.0, 3.0]
    assert energy.integrate_power(ts, [0.0, 1.0, 2.0, 3.0], 0.0, 3.0) == pytest.approx(4.5)
    assert energy.integrate_power(ts, [0.0, 1.0, 2.0, 3.0], 0.5, 2.5) == pytest.approx(3.0)


def test_zero_order_hold_uses_each_reading_until_the_next():
    # PSTR-style steps: 10 W from t=0, 20 W from t=1, 5 W from t=2 (held to the end)
    ts, w = [0.0, 1.0, 2.0], [10.0, 20.0, 5.0]
    assert energy.integrate_power(ts, w, 0.5, 1.5, hold=True) == pytest.approx(5.0 + 10.0)
    assert energy.integrate_power(ts, w, 0.0, 4.0, hold=True) == pytest.approx(10 + 20 + 10)
    with pytest.raises(ValueError):
        energy.integrate_power(ts, w, -0.5, 1.0, hold=True)


# --------------------------------------------------------------------- baseline subtraction


def test_baseline_interpolates_between_idle_windows_and_holds_outside():
    idle = [(10.0, 4.0), (30.0, 6.0)]
    assert energy.baseline_at(idle, 20.0) == pytest.approx(5.0)
    assert energy.baseline_at(idle, 0.0) == pytest.approx(4.0)
    assert energy.baseline_at(idle, 99.0) == pytest.approx(6.0)
    assert energy.baseline_at([(30.0, 6.0), (10.0, 4.0)], 25.0) == pytest.approx(5.5)  # order-free
    with pytest.raises(ValueError):
        energy.baseline_at([], 1.0)


def test_net_energy_subtracts_idle_power_times_duration():
    assert energy.net_energy(150.0, 10.0, 5.0) == pytest.approx(100.0)
    assert energy.per_decision(100.0, 400) == pytest.approx(0.25)
    assert energy.per_decision(100.0, 0) is None


# --------------------------------------------------------------------- ordering and load


def test_abba_order_puts_every_config_in_both_positions_between_idle_windows():
    assert energy.abba_order(["gpu", "ane", "auto"], 1) == [
        "idle",
        "gpu",
        "ane",
        "auto",
        "idle",
        "auto",
        "ane",
        "gpu",
        "idle",
    ]
    order = energy.abba_order(["a", "b"], 3)
    assert order.count("a") == order.count("b") == 6
    assert order[0] == order[-1] == "idle"
    with pytest.raises(ValueError):
        energy.abba_order([], 1)


def test_offered_arrivals_are_evenly_spaced_at_the_rate():
    arr = energy.offered_arrivals(20.0, 2.0)
    assert len(arr) == 40
    assert arr[0] == 0.0 and arr[1] == pytest.approx(0.05)
    assert all(t < 2.0 for t in arr)


# --------------------------------------------------------------------- powermetrics


def _pm_sample(t: datetime, elapsed_ms: float, cpu: int, gpu: int, ane: int) -> str:
    return (
        f"*** Sampled system activity ({t.strftime('%a %b %d %H:%M:%S %Y %z')}) ({elapsed_ms:.2f}ms elapsed) ***\n\n"
        "**** Processor usage ****\n\n"
        f"CPU Power: {cpu} mW\nGPU Power: {gpu} mW\nANE Power: {ane} mW\n"
        f"Combined Power (CPU + GPU + ANE): {cpu + gpu + ane} mW\n\n"
        "**** GPU usage ****\n\nGPU HW active frequency: 1000 MHz\n\n"
    )


def test_parse_powermetrics_text_reads_rails_and_stamps():
    t0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    text = _pm_sample(t0, 1003.5, 1500, 20000, 0) + _pm_sample(t0.replace(second=1), 998.0, 1200, 0, 2500)
    got = energy.parse_powermetrics_text(text)
    assert len(got) == 2
    assert got[0] == {
        "t_end": t0.timestamp(),
        "elapsed_s": pytest.approx(1.0035),
        "cpu_w": 1.5,
        "gpu_w": 20.0,
        "ane_w": 0.0,
        "combined_w": 21.5,
    }
    assert got[1]["ane_w"] == 2.5


def test_agreement_is_relative_above_the_floor_and_absolute_below():
    assert energy.agreement(10.4, 10.0, 0.05, 1.0, 0.1)["ok"]
    assert not energy.agreement(10.6, 10.0, 0.05, 1.0, 0.1)["ok"]
    small = energy.agreement(0.35, 0.3, 0.05, 1.0, 0.1)
    assert small["ok"] and small["rel"] is None


# --------------------------------------------------------------------- analysis, synthetic runs


def _sampler(rail_power, t_end: float, dt: float = 0.5, t0_unix: float = 1.79e9) -> dict:
    """Synthetic sampler output: rail_power(t) -> {rail: W}; cumulative by the midpoint rule."""
    samples, cum, t = [], dict.fromkeys(analyze.SOC, 0.0), 0.0
    samples.append([0, int(t0_unix * 1e9), dict(cum)])
    while t < t_end - 1e-9:
        p = rail_power(t + dt / 2)
        for k in cum:
            cum[k] += p.get(k, 0.0) * dt
        t += dt
        samples.append([int(round(t * 1e9)), int(round((t0_unix + t) * 1e9)), dict(cum)])
    return {"samples": samples, "pstr": [[0, 20.0]], "meta": {"sampler_loop_cpu_frac": 0.007}}


def test_campaign_net_j_per_decision_on_a_synthetic_run():
    # idle 5 W on the CPU rail; gpu windows add 10 W on the GPU rail; ane windows add 2 W on ANE
    windows, spans = [], []
    order = ["idle", "gpu", "ane", "idle"]
    rate, dur = 20.0, 10.0
    for i, cfg in enumerate(order):
        m0 = 1.0 + i * dur
        spans.append((m0, m0 + dur, cfg))
        reqs = []
        if cfg != "idle":
            for k in range(int(rate * dur)):
                s = m0 + k / rate
                reqs.append([k, 0, 1, int(s * 1e9), int(s * 1e9), int((s + 0.01) * 1e9), cfg, "x", {}, None])
        windows.append(
            {
                "shape": "short",
                "config": cfg,
                "index": i,
                "rate": None if cfg == "idle" else rate,
                "measure_ns": [int(m0 * 1e9), int((m0 + dur) * 1e9)],
                "requests": reqs,
            }
        )

    def power(t):
        p = {"cpu": 5.0}
        for a, b, cfg in spans:
            if a <= t < b and cfg == "gpu":
                p["gpu"] = 10.0
            if a <= t < b and cfg == "ane":
                p["ane"] = 2.0
        return p

    run = {"meta": {}, "args": {}, "windows": windows, "sampler": _sampler(power, 1.0 + 4 * dur + 1.0)}
    out = analyze.campaign(run)
    s = out["shapes"]["short"]
    assert s["idle"]["soc_w"]["median"] == pytest.approx(5.0)
    assert s["idle"]["ok"]
    gpu, ane = s["cells"]["gpu"], s["cells"]["ane"]
    assert gpu["valid_windows"] == 1 and ane["valid_windows"] == 1
    # 10 W above idle over 10 s for 200 decisions = 0.5 J/decision; ANE 2 W -> 0.1
    assert gpu["net_j_per_decision"]["median"] == pytest.approx(0.5)
    assert ane["net_j_per_decision"]["median"] == pytest.approx(0.1)
    assert gpu["gross_j_per_decision"]["median"] == pytest.approx(0.75)
    assert ane["net_rail_j_per_decision"]["ane"]["median"] == pytest.approx(0.1)
    assert ane["net_rail_j_per_decision"]["cpu"]["median"] == pytest.approx(0.0)
    cmp_ = s["comparisons"]["ane_vs_gpu:net_j_per_decision"]
    assert cmp_["verdict"] == "lower" and cmp_["median_ratio"] == pytest.approx(0.2)
    assert out["sampler_overhead"]["ok"]


def test_a_window_that_falls_behind_the_offered_load_is_invalid():
    m0, dur, rate = 1.0, 10.0, 20.0
    # the server completes only half of the offered requests inside the window
    reqs = []
    for k in range(int(rate * dur)):
        s = m0 + k / rate
        start = m0 + k / (rate / 2)
        reqs.append([k, 0, 1, int(s * 1e9), int(start * 1e9), int((start + 0.05) * 1e9), "gpu", "x", {}, None])
    w = {
        "shape": "short",
        "config": "gpu",
        "index": 1,
        "rate": rate,
        "measure_ns": [int(1e9), int(11e9)],
        "requests": reqs,
    }
    samp = _sampler(lambda t: {"cpu": 5.0}, 12.0)
    ts, cum = analyze.series(samp)
    row = analyze.window_row(w, ts, cum, [0.0], [20.0])
    assert row["valid"] is False
    assert row["achieved_decisions_per_s"] < 0.6 * row["offered_decisions_per_s"]


def test_crosscheck_aligns_powermetrics_and_reports_agreement(tmp_path):
    t0_unix = 1_790_000_000.0
    phases = [("idle", 0.0, 20.0), ("gpu", 20.0, 60.0), ("idle", 60.0, 80.0)]

    def power(t):
        return {"cpu": 1.0, "gpu": 20.0 if 20.0 <= t < 60.0 else 0.0}

    samp = _sampler(power, 81.0, t0_unix=t0_unix)
    run = {
        "meta": {},
        "phases": [
            {"phase": n, "unix_ns": [int((t0_unix + a) * 1e9), int((t0_unix + b) * 1e9)], "requests": 0}
            for n, a, b in phases
        ],
        "sampler": samp,
    }
    with gzip.open(tmp_path / "s.json.gz", "wt") as f:
        json.dump(run, f)
    # powermetrics: 1 s samples stamped at their end, 3% above the sampler on the GPU
    text = "".join(
        _pm_sample(
            datetime.fromtimestamp(t0_unix + k, tz=timezone.utc),
            1000.0,
            1000,
            int(20600 if 20 < k <= 60 else 0),
            0,
        )
        for k in range(1, 81)
    )
    (tmp_path / "pm.txt").write_text(text)
    got = analyze.crosscheck(tmp_path / "s.json.gz", tmp_path / "pm.txt")
    assert got["lag_s"] == 0
    gpu_phase = got["phases"][1]
    assert gpu_phase["pairs"] >= 30
    assert gpu_phase["gpu"]["rel"] == pytest.approx(20.0 / 20.6 - 1, abs=1e-6)
    assert got["ok"]
    assert analyze.crosscheck(tmp_path / "missing.json.gz", tmp_path / "pm.txt") is None
