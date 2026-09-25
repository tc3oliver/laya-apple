"""Pure logic for the energy measurements: no IOKit, no model, no clock. Unit-tested in
tests/unit/test_energy_sampler.py.

Conventions: times are seconds on one clock (the caller converts ns); energy in J; power in W.
A *cumulative* series is a non-decreasing energy counter sampled at increasing times (what the
sampler records from IOReport). A *power* series is instantaneous readings (SMC PSTR).
"""

from __future__ import annotations

import bisect
import math
import re
from datetime import datetime

# ------------------------------------------------------------------------ energy over intervals


def cumulative_at(ts: list[float], cum: list[float], t: float) -> float:
    """Linear interpolation of a cumulative counter at time t. Raises outside the sampled span:
    extrapolating energy would invent it."""
    if len(ts) != len(cum) or len(ts) < 2:
        raise ValueError("need at least two samples of equal-length ts and cum")
    if t < ts[0] or t > ts[-1]:
        raise ValueError(f"t={t} outside the sampled span [{ts[0]}, {ts[-1]}]")
    i = bisect.bisect_right(ts, t)
    if i >= len(ts):
        return cum[-1]
    t0, t1 = ts[i - 1], ts[i]
    c0, c1 = cum[i - 1], cum[i]
    return c0 if t1 == t0 else c0 + (c1 - c0) * (t - t0) / (t1 - t0)


def window_energy(ts: list[float], cum: list[float], t0: float, t1: float) -> float:
    """Energy (J) of a cumulative counter between t0 and t1."""
    if t1 < t0:
        raise ValueError("t1 < t0")
    return cumulative_at(ts, cum, t1) - cumulative_at(ts, cum, t0)


def integrate_power(ts: list[float], watts: list[float], t0: float, t1: float, hold: bool = False) -> float:
    """Energy (J) of a power series over [t0, t1].

    hold=False: trapezoid between readings, with interpolated end points.
    hold=True: zero-order hold, each reading holds until the next one (for a source like SMC
    PSTR that updates in steps); the last reading holds to t1 and t0 must not precede the
    first reading.
    """
    if len(ts) != len(watts) or not ts:
        raise ValueError("need samples")
    if t1 < t0:
        raise ValueError("t1 < t0")
    if t0 < ts[0] or (not hold and t1 > ts[-1]):
        raise ValueError("interval outside the sampled span")
    if hold:
        e = 0.0
        i = bisect.bisect_right(ts, t0) - 1
        t = t0
        while t < t1:
            nxt = ts[i + 1] if i + 1 < len(ts) else math.inf
            end = min(nxt, t1)
            e += watts[i] * (end - t)
            t, i = end, i + 1
        return e
    pts = [(t0, _interp(ts, watts, t0))]
    pts += [(t, w) for t, w in zip(ts, watts) if t0 < t < t1]
    pts.append((t1, _interp(ts, watts, t1)))
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))


def _interp(ts, ys, t):
    i = bisect.bisect_right(ts, t)
    if i == 0:
        return ys[0]
    if i >= len(ts):
        return ys[-1]
    t0, t1 = ts[i - 1], ts[i]
    return ys[i - 1] if t1 == t0 else ys[i - 1] + (ys[i] - ys[i - 1]) * (t - t0) / (t1 - t0)


# ------------------------------------------------------------------------ idle baseline


def baseline_at(idle: list[tuple[float, float]], t: float) -> float:
    """Idle power (W) at time t from idle windows [(mid_time, mean_W)]: linear interpolation
    between the idle windows either side of t (so slow drift is removed), the nearest one
    outside their span."""
    if not idle:
        raise ValueError("no idle windows")
    pts = sorted(idle)
    ts = [p[0] for p in pts]
    ws = [p[1] for p in pts]
    return _interp(ts, ws, t)


def net_energy(energy_j: float, duration_s: float, baseline_w: float) -> float:
    """Energy above idle: E - P_idle * T."""
    return energy_j - baseline_w * duration_s


def per_decision(energy_j: float, decisions: int) -> float | None:
    return energy_j / decisions if decisions > 0 else None


# ------------------------------------------------------------------------ ordering


def abba_order(configs: list[str], repeats: int, idle: str = "idle") -> list[str]:
    """[idle, A, B, C, idle, C, B, A] x repeats, then a closing idle: every config appears in
    both positions, and every loaded window has an idle window on each side within one block."""
    if repeats < 1 or not configs:
        raise ValueError("need configs and repeats >= 1")
    out: list[str] = []
    for _ in range(repeats):
        out += [idle, *configs, idle, *reversed(configs)]
    return out + [idle]


def offered_arrivals(rate: float, seconds: float) -> list[float]:
    """Deterministic open-loop arrival offsets at a fixed rate: k / rate for k = 0.. in [0, s)."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    n = int(math.floor(seconds * rate - 1e-9)) + 1
    return [k / rate for k in range(n) if k / rate < seconds]


# ------------------------------------------------------------------------ powermetrics


_HDR = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\) \*\*\*")
_PWR = re.compile(r"^(CPU|GPU|ANE) Power:\s*([\d.]+)\s*mW", re.M)
_COMB = re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", re.M)


def parse_powermetrics_text(text: str) -> list[dict]:
    """Samples from `powermetrics --format text` with the cpu_power, gpu_power and ane_power
    samplers: [{t_end, elapsed_s, cpu_w, gpu_w, ane_w, combined_w}], t_end in unix seconds
    from the sample header (1 s resolution; treated as the end of the sample interval)."""
    heads = list(_HDR.finditer(text))
    out = []
    for i, h in enumerate(heads):
        body = text[h.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        stamp = datetime.strptime(h.group(1).strip(), "%a %b %d %H:%M:%S %Y %z").timestamp()
        s = {"t_end": stamp, "elapsed_s": float(h.group(2)) / 1e3}
        for m in _PWR.finditer(body):
            s[m.group(1).lower() + "_w"] = float(m.group(2)) / 1e3
        c = _COMB.search(body)
        if c:
            s["combined_w"] = float(c.group(1)) / 1e3
        if all(k in s for k in ("cpu_w", "gpu_w", "ane_w")):
            s.setdefault("combined_w", s["cpu_w"] + s["gpu_w"] + s["ane_w"])
            out.append(s)
    return out


def sampler_power_over(ts: list[float], rails: dict[str, list[float]], t0: float, t1: float) -> dict[str, float]:
    """Mean power per rail over [t0, t1] from the sampler's cumulative series."""
    return {k: window_energy(ts, cum, t0, t1) / (t1 - t0) for k, cum in rails.items()}


def align_lag(pm: list[dict], ts: list[float], rails: dict[str, list[float]], lags=(-2, -1, 0, 1, 2), key="gpu"):
    """The whole-second shift of powermetrics' timestamps that best matches the sampler's rail
    `key` (least mean squared difference). Header stamps have 1 s resolution and no stated
    start/end convention, so a small lag search makes the alignment explicit."""
    best = None
    for lag in lags:
        err, n = 0.0, 0
        for s in pm:
            t1 = s["t_end"] + lag
            t0 = t1 - s["elapsed_s"]
            if t0 < ts[0] or t1 > ts[-1]:
                continue
            ours = window_energy(ts, rails[key], t0, t1) / (t1 - t0)
            err += (ours - s[f"{key}_w"]) ** 2
            n += 1
        if n and (best is None or err / n < best[1]):
            best = (lag, err / n, n)
    if best is None:
        raise ValueError("powermetrics samples do not overlap the sampler's span")
    return best[0]


def agreement(ours: float, ref: float, rel_tol: float, abs_floor_w: float, abs_tol_w: float) -> dict:
    """Relative agreement where the reference is at least abs_floor_w, absolute below it."""
    diff = ours - ref
    if abs(ref) >= abs_floor_w:
        rel = diff / ref
        return {"ours_w": ours, "ref_w": ref, "diff_w": diff, "rel": rel, "ok": abs(rel) <= rel_tol}
    return {"ours_w": ours, "ref_w": ref, "diff_w": diff, "rel": None, "ok": abs(diff) <= abs_tol_w}
