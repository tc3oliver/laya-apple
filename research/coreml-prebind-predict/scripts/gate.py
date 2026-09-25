"""The paired statistical gate of criteria.md (from this preregistration on; not applied to #77).

#57's engineering budgets, unchanged (short and long P99 ≤ 1.05×, aggregate throughput ≥ 0.95×),
judged on matched window pairs instead of one pooled point estimate:
  - a pair is a candidate hetero window and the production hetero window of the same round and
    the same cycle index (pair_windows);
  - per pair, the ratio candidate / production of short P99, long P99 and aggregate req/s;
  - statistic: the geometric mean of the pair ratios; primary 95% CI: Student t on the log
    ratios, df = n - 1 (t_interval); sensitivity only: a seeded percentile bootstrap over pairs
    (bootstrap_interval), which never changes the verdict;
  - three-way verdict per criterion (upper_verdict / lower_verdict) and per model (combine).
Pure: NumPy and math only (no SciPy in the lock), unit-tested.
"""

from __future__ import annotations

import math

import numpy as np

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260925


def _t_two_sided_mass(t: float, df: int) -> float:
    """P(|T| < t) for Student t with integer df (Abramowitz & Stegun 26.7.3 / 26.7.4)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    theta = math.atan(t / math.sqrt(df))
    c2 = math.cos(theta) ** 2
    if df % 2 == 1:
        term, s = 1.0, 1.0 if df > 1 else 0.0
        for k in range(1, (df - 3) // 2 + 1):  # 1 + 2/3 c2 + (2*4)/(3*5) c2^2 + ...
            term *= (2 * k) / (2 * k + 1) * c2
            s += term
        return 2 / math.pi * (theta + (math.sin(theta) * math.cos(theta) * s if df > 1 else 0.0))
    term, s = 1.0, 1.0
    for k in range(1, (df - 2) // 2 + 1):  # 1 + 1/2 c2 + (1*3)/(2*4) c2^2 + ...
        term *= (2 * k - 1) / (2 * k) * c2
        s += term
    return math.sin(theta) * s


def t_critical(df: int, confidence: float = CONFIDENCE) -> float:
    """Two-sided critical value: P(|T| < t) = confidence, by bisection."""
    lo, hi = 0.0, 1.0
    while _t_two_sided_mass(hi, df) < confidence:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_two_sided_mass(mid, df) < confidence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_interval(ratios, confidence: float = CONFIDENCE) -> dict:
    """Geometric mean of the ratios and its t CI on the log scale, back-transformed."""
    x = np.log(np.asarray(ratios, dtype=np.float64))
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 pairs")
    m, sd = float(x.mean()), float(x.std(ddof=1))
    hw = t_critical(n - 1, confidence) * sd / math.sqrt(n)
    return {"n": n, "geomean": math.exp(m), "lo": math.exp(m - hw), "hi": math.exp(m + hw), "log_halfwidth": hw}


def bootstrap_interval(
    ratios, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED, confidence: float = CONFIDENCE
) -> dict:
    """Percentile bootstrap over pairs of the geometric mean (sensitivity only)."""
    x = np.log(np.asarray(ratios, dtype=np.float64))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    a = (1 - confidence) / 2
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    return {"resamples": resamples, "seed": seed, "lo": math.exp(lo), "hi": math.exp(hi)}


def upper_verdict(ci: dict, limit: float) -> str:
    """A quantity that must not exceed limit (P99 ratio ≤ 1.05)."""
    if ci["hi"] <= limit:
        return PASS
    if ci["lo"] > limit:
        return FAIL
    return INCONCLUSIVE


def lower_verdict(ci: dict, limit: float) -> str:
    """A quantity that must not fall below limit (throughput ratio ≥ 0.95)."""
    if ci["lo"] >= limit:
        return PASS
    if ci["hi"] < limit:
        return FAIL
    return INCONCLUSIVE


def combine(verdicts) -> str:
    v = list(verdicts)
    if all(x == PASS for x in v):
        return PASS
    if any(x == FAIL for x in v):
        return FAIL
    return INCONCLUSIVE


def hetero_by_cycle(run: dict) -> dict[int, dict]:
    """{cycle: hetero window} of one run's part_a (bench_concurrency's window records)."""
    out = {}
    for w in run["part_a"]["windows"]:
        if w["condition"] == "hetero":
            if w["cycle"] in out:
                raise ValueError(f"two hetero windows in cycle {w['cycle']}")
            out[w["cycle"]] = w
    return out


def pair_windows(prod_runs: list[dict], cand_runs: list[dict]) -> list[tuple[int, int, dict, dict]]:
    """(round, cycle, production window, candidate window) for every matched pair.

    prod_runs[i] and cand_runs[i] are round i+1 of each configuration. A cycle present in only
    one of the two runs is an error, not a silently dropped pair."""
    if len(prod_runs) != len(cand_runs):
        raise ValueError("production and candidate have different numbers of rounds")
    pairs = []
    for rnd, (p, c) in enumerate(zip(prod_runs, cand_runs), start=1):
        pw, cw = hetero_by_cycle(p), hetero_by_cycle(c)
        if set(pw) != set(cw):
            raise ValueError(f"round {rnd}: hetero cycles differ ({sorted(pw)} vs {sorted(cw)})")
        pairs += [(rnd, k, pw[k], cw[k]) for k in sorted(pw)]
    return pairs


def aggregate_req_s(w: dict) -> float:
    return sum(s["req_s"] for s in w["streams"].values())


def pair_ratios(pairs) -> dict[str, list[float]]:
    return {
        "short_p99": [c["streams"]["short"]["p99_ms"] / p["streams"]["short"]["p99_ms"] for _, _, p, c in pairs],
        "long_p99": [c["streams"]["long"]["p99_ms"] / p["streams"]["long"]["p99_ms"] for _, _, p, c in pairs],
        "aggregate": [aggregate_req_s(c) / aggregate_req_s(p) for _, _, p, c in pairs],
    }


def paired_verdict(prod_runs, cand_runs, limits: dict, correctness: bool, isolation: bool) -> dict:
    """The per-model, per-candidate verdict. correctness and isolation are computed by the caller
    exactly as in #77 (0 mismatches; GPU return P50 ≤ 1 ms, and ≥ 5× better than A)."""
    pairs = pair_ratios(pair_windows(prod_runs, cand_runs))
    stats = {k: {"ratios": v, "t": t_interval(v), "bootstrap": bootstrap_interval(v)} for k, v in pairs.items()}
    checks = {
        "correctness": PASS if correctness else FAIL,
        "aggregate_throughput": lower_verdict(stats["aggregate"]["t"], limits["aggregate_min_ratio"]),
        "short_p99": upper_verdict(stats["short_p99"]["t"], limits["p99_max_ratio"]),
        "long_p99": upper_verdict(stats["long_p99"]["t"], limits["p99_max_ratio"]),
        "gpu_completion_isolation": PASS if isolation else FAIL,
    }
    sensitivity = {
        "aggregate_throughput": lower_verdict(stats["aggregate"]["bootstrap"], limits["aggregate_min_ratio"]),
        "short_p99": upper_verdict(stats["short_p99"]["bootstrap"], limits["p99_max_ratio"]),
        "long_p99": upper_verdict(stats["long_p99"]["bootstrap"], limits["p99_max_ratio"]),
    }
    return {
        "pairs": len(pairs["short_p99"]),
        "stats": stats,
        "checks": checks,
        "verdict": combine(checks.values()),
        "bootstrap_sensitivity": sensitivity,
    }
