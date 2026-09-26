"""The paired statistical gate (window pairs, log-ratio t-interval, PASS/FAIL/INCONCLUSIVE).

The statistics are copied unchanged from research/coreml-prebind-predict/scripts/gate.py
(t_critical, t_interval, bootstrap_interval, upper_verdict), so this track does not import
another track's scripts. Pure NumPy and math, no SciPy.
"""

from __future__ import annotations

import math

import numpy as np

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260926


def _t_two_sided_mass(t: float, df: int) -> float:
    """P(|T| < t) for Student t with integer df (Abramowitz & Stegun 26.7.3 / 26.7.4)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    theta = math.atan(t / math.sqrt(df))
    c2 = math.cos(theta) ** 2
    if df % 2 == 1:
        term, s = 1.0, 1.0 if df > 1 else 0.0
        for k in range(1, (df - 3) // 2 + 1):
            term *= (2 * k) / (2 * k + 1) * c2
            s += term
        return 2 / math.pi * (theta + (math.sin(theta) * math.cos(theta) * s if df > 1 else 0.0))
    term, s = 1.0, 1.0
    for k in range(1, (df - 2) // 2 + 1):
        term *= (2 * k - 1) / (2 * k) * c2
        s += term
    return math.sin(theta) * s


def t_critical(df: int, confidence: float = CONFIDENCE) -> float:
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
    """Percentile bootstrap over pairs of the geometric mean (sensitivity only, never the verdict)."""
    x = np.log(np.asarray(ratios, dtype=np.float64))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    a = (1 - confidence) / 2
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    return {"resamples": resamples, "seed": seed, "lo": math.exp(lo), "hi": math.exp(hi)}


def upper_verdict(ci: dict, limit: float) -> str:
    """A ratio that must not exceed `limit`: PASS if the whole CI is <= limit."""
    if ci["hi"] <= limit:
        return PASS
    if ci["lo"] > limit:
        return FAIL
    return INCONCLUSIVE
