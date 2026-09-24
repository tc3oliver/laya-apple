/* Switchyard replay. Plain JS + Canvas 2D, no dependencies, no network.
   Reads the replay JSON inlined into #switchyard-data (format "switchyard-replay" v1) and
   replays it as a rail yard: every train is a request, every junction a decision, and the
   time a train stands at a red signal is its recorded decision latency. Every number on
   screen comes from that JSON. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const STAGE_W = 1600;
  const STAGE_H = 900;
  const YARD_W = 1600;
  const YARD_H = 768;
  const TL_W = 1600;
  const TL_H = 206;
  const CARD_W = 1600;
  const CARD_H = 900;
  const CARD_CSS_W = 1408;
  const CARD_CSS_H = 792;

  const LABEL = { gpu_only: "GPU only", hybrid: "GPU + ANE" };
  const DESC = {
    gpu_only: "Every request, trains and background documents alike, is answered by the MLX GPU.",
    hybrid: "Trains are answered by the Neural Engine; background documents stay on the MLX GPU.",
  };
  const DEVICE = ["MLX GPU", "Neural Engine"];
  const DEV_KEY = ["gpu", "ane"];
  const PLATFORMS = ["A", "B", "C"];
  const BG_LABEL = { medium_1q: "Document", long_1q: "Long document", short_4q: "4-question request" };

  const WINDOW = 3000; // timeline history shown
  // Playback speeds (screen time = data time / speed, linear). 0.25x by default: at real
  // time a rush hour sends ~70 trains through in one second, too fast to follow.
  const SPEEDS = [1, 0.5, 0.25, 0.1];
  const DEFAULT_SPEED = 2;
  // Bursty arrivals (laya_apple.workload.arrivals): 1 s at 3x the rate, then 2 s off, from
  // the round start. workload.burst_period_s / burst_on_s override these when present.
  const BURST_PERIOD_S = 3;
  const BURST_ON_S = 1;
  const UNVERIFIED_LABEL = "GPU + ANE (Neural Engine not used — not comparable)";
  const UNVERIFIED_DESC =
    "The Neural Engine stopped being used during this round, so it is not compared with GPU only.";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const lightQuery = window.matchMedia("(prefers-color-scheme: light)");
  const nf = new Intl.NumberFormat("en-US");

  // ------------------------------------------------------------------ formatting

  function fmtInt(n) {
    return n == null || !isFinite(n) ? "—" : nf.format(Math.round(n));
  }
  function fmtMs(v) {
    if (v == null || !isFinite(v)) return "—";
    if (v < 100) return v.toFixed(1) + " ms";
    return nf.format(Math.round(v)) + " ms";
  }
  // Shows 0% or 100% only for exactly 0 or 1. Below 10% and above 99% it keeps one
  // decimal, and says "<0.1%" / ">99.9%" where even that would round to 0 or 100.
  function fmtPct(r) {
    if (r == null || !isFinite(r)) return "—";
    if (r === 0) return "0%";
    if (r === 1) return "100%";
    const p = r * 100;
    if (p < 0.05) return "<0.1%";
    if (p >= 99.95) return ">99.9%";
    if (p < 10 || p > 99) return p.toFixed(1) + "%";
    return Math.round(p) + "%";
  }
  function fmtS(ms) {
    return (Math.max(0, ms) / 1000).toFixed(1) + " s";
  }
  function label(config) {
    return LABEL[config] || config;
  }
  // A hybrid round whose Neural Engine use was not verified is shown, never compared.
  function unverified(P) {
    return P.config === "hybrid" && !!P.res && P.res.ane_verified === false;
  }
  function roundLabel(P) {
    return unverified(P) ? UNVERIFIED_LABEL : label(P.config);
  }
  function roundDesc(P) {
    return unverified(P) ? UNVERIFIED_DESC : DESC[P.config] || "";
  }
  function comparable() {
    return get(result, "comparison.available") === true;
  }
  function get(obj, path) {
    let o = obj;
    for (const k of path.split(".")) {
      if (o == null) return undefined;
      o = o[k];
    }
    return o;
  }
  function setText(el, s) {
    if (el.textContent !== s) el.textContent = s;
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  }

  // ------------------------------------------------------------------ search

  function upperBound(a, v, n) {
    let lo = 0;
    let hi = n === undefined ? a.length : n;
    while (lo < hi) {
      const m = (lo + hi) >>> 1;
      if (a[m] <= v) lo = m + 1;
      else hi = m;
    }
    return lo;
  }
  function lowerBound(a, v) {
    let lo = 0;
    let hi = a.length;
    while (lo < hi) {
      const m = (lo + hi) >>> 1;
      if (a[m] < v) lo = m + 1;
      else hi = m;
    }
    return lo;
  }
  // numpy.percentile's default (linear) on a sorted typed array.
  function pct(sorted, n, q) {
    if (n === 0) return null;
    const k = ((n - 1) * q) / 100;
    const lo = Math.floor(k);
    const hi = Math.ceil(k);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (k - lo);
  }

  // ------------------------------------------------------------------ colours

  const C = {};
  let FONT = "sans-serif";
  function readColors() {
    FONT = getComputedStyle(document.body).fontFamily;
    const cs = getComputedStyle(document.documentElement);
    for (const k of [
      "bg", "surface", "surface-2", "border", "track", "text", "text-dim", "text-faint",
      "gpu", "gpu-ink", "gpu-soft", "ane", "ane-ink", "ane-soft", "late", "late-ink", "late-soft",
      "on-late", "ok", "bgreq", "warn", "yard", "parked", "train", "train-edge",
    ]) {
      C[k.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = cs.getPropertyValue("--" + k).trim();
    }
    C.light = lightQuery.matches;
  }

  function hatch(ctx, stripe, soft) {
    const c = document.createElement("canvas");
    const s = 16; // drawn at 2x for crisp stripes; the pattern is scaled back by 0.5
    c.width = s;
    c.height = s;
    const g = c.getContext("2d");
    g.fillStyle = soft;
    g.fillRect(0, 0, s, s);
    g.strokeStyle = stripe;
    g.lineWidth = 3.2;
    g.beginPath();
    g.moveTo(-4, 4); g.lineTo(4, -4);
    g.moveTo(0, s); g.lineTo(s, 0);
    g.moveTo(s - 4, s + 4); g.lineTo(s + 4, s - 4);
    g.stroke();
    return ctx.createPattern(c, "repeat");
  }

  // ------------------------------------------------------------------ data

  let data = null;
  let result = {};
  const R = []; // prepared rounds, in run order

  function fatal(msg) {
    $("fatal-text").textContent = msg;
    $("fatal").hidden = false;
  }

  function loadData() {
    const el = $("switchyard-data");
    const raw = el ? el.textContent.trim() : "";
    // The build replaces the data marker with JSON; an unbuilt page still starts with "__".
    if (!raw || raw.charAt(0) !== "{") {
      fatal("This page has no replay data inlined. Build it with `laya-apple switchyard --replay DIR`.");
      return false;
    }
    try {
      data = JSON.parse(raw);
    } catch (e) {
      fatal("The inlined replay data is not valid JSON: " + e.message);
      return false;
    }
    if (data.format !== "switchyard-replay" || !Array.isArray(data.rounds) || data.rounds.length === 0) {
      fatal("Unexpected replay format (" + data.format + " v" + data.format_version + ").");
      return false;
    }
    result = data.result || {};
    const rounds = data.rounds.slice().sort((a, b) => a.index - b.index);
    for (const r of rounds) R.push(prepare(r));
    return true;
  }

  function prepare(round) {
    const trains = round.trains.slice().sort((a, b) => a.arrival_ms - b.arrival_ms);
    const n = trains.length;
    const deadline = round.deadline_ms != null ? round.deadline_ms : result.workload && result.workload.deadline_ms;
    const P = {
      index: round.index,
      config: round.config,
      duration: (round.duration_s || 0) * 1000,
      deadline: deadline,
      trains,
      n,
      arr: new Float64Array(n),
      resp: new Float64Array(n),
      lat: new Float64Array(n),
      line: new Uint8Array(n),
      dev: new Uint8Array(n),
      ans: new Uint8Array(n),
      outc: new Uint8Array(n), // 0 delivered, 1 late, 2 misrouted
      dispX: new Float32Array(n),
      maxLat: 0,
      end: 0,
    };
    const lateT = [];
    const delivT = [];
    for (let i = 0; i < n; i++) {
      const t = trains[i];
      P.arr[i] = t.arrival_ms;
      P.resp[i] = t.response_ms;
      P.lat[i] = t.latency_ms;
      P.line[i] = Math.min(6, Math.max(1, t.line | 0)) - 1;
      P.dev[i] = t.device === "ane" ? 1 : 0;
      P.ans[i] = Math.max(0, PLATFORMS.indexOf(t.answer));
      P.outc[i] = t.outcome === "late" ? 1 : t.outcome === "misrouted" ? 2 : 0;
      if (P.outc[i] === 1) lateT.push(t.arrival_ms + deadline);
      if (P.outc[i] === 0) delivT.push(t.response_ms);
      P.maxLat = Math.max(P.maxLat, t.response_ms - t.arrival_ms);
      P.end = Math.max(P.end, t.response_ms);
    }
    P.lateT = Float64Array.from(lateT).sort();
    P.delivT = Float64Array.from(delivT).sort();
    const byResp = Array.from({ length: n }, (_, i) => i).sort((a, b) => P.resp[a] - P.resp[b]);
    P.byResp = Int32Array.from(byResp);
    P.respSorted = Float64Array.from(byResp, (i) => P.resp[i]);
    P.latByResp = Float64Array.from(byResp, (i) => P.lat[i]);
    P.scratch = new Float64Array(n);

    // Timeline items: trains + background, by arrival, packed into rows per device lane.
    const items = [];
    for (let i = 0; i < n; i++) {
      const t = trains[i];
      items.push([t.arrival_ms, t.service_start_ms, t.service_end_ms, P.dev[i], 0, P.outc[i] === 1 ? 1 : 0]);
    }
    const bgClasses = [];
    for (const b of round.background || []) {
      let k = bgClasses.indexOf(b.class);
      if (k < 0) k = bgClasses.push(b.class) - 1;
      items.push([b.arrival_ms, b.service_start_ms, b.service_end_ms, b.device === "ane" ? 1 : 0, 1 + k, 0]);
      P.end = Math.max(P.end, b.response_ms);
    }
    items.sort((a, b) => a[0] - b[0]);
    const m = items.length;
    P.m = m;
    P.iArr = new Float64Array(m);
    P.iStart = new Float64Array(m);
    P.iEnd = new Float64Array(m);
    P.iLane = new Uint8Array(m);
    P.iKind = new Uint8Array(m);
    P.iLate = new Uint8Array(m);
    P.iMaxDur = 0;
    P.bgClasses = bgClasses;
    P.laneUsed = [false, false];
    // Queue depth per lane as step functions, stacked bottom-up in drawing order:
    // trains waiting past the deadline <= all trains waiting <= everything waiting.
    const ev = [[[], [], []], [[], [], []]];
    for (let j = 0; j < m; j++) {
      const it = items[j];
      P.iArr[j] = it[0];
      P.iStart[j] = it[1];
      P.iEnd[j] = it[2];
      P.iLane[j] = it[3];
      P.iKind[j] = it[4];
      P.iLate[j] = it[5];
      P.laneUsed[it[3]] = true;
      P.iMaxDur = Math.max(P.iMaxDur, it[2] - it[0]);
      if (it[1] <= it[0]) continue;
      const e = ev[it[3]];
      e[2].push(it[0], 1, it[1], -1);
      if (it[4] === 0) {
        e[1].push(it[0], 1, it[1], -1);
        if (it[1] - it[0] > deadline) e[0].push(it[0] + deadline, 1, it[1], -1);
      }
    }
    P.depth = ev.map((lane) => lane.map(stepSeries));
    P.maxDepth = Math.max(P.depth[0][2].max, P.depth[1][2].max);
    P.end = Math.max(P.end, P.duration) + 800;

    // The queue slot each train holds when its answer arrives: trains ahead of it on its
    // line (earlier arrival) still waiting. Nonzero means it was answered out of turn.
    P.slot0 = new Uint16Array(n);
    P.maxRun = 0;
    for (let i = 0; i < n; i++) {
      let ahead = 0;
      for (let j = i - 1; j >= 0 && P.arr[j] >= P.arr[i] - P.maxLat; j--) {
        if (P.line[j] === P.line[i] && P.resp[j] > P.resp[i]) ahead++;
      }
      P.slot0[i] = Math.min(ahead, MAX_SLOTS - 1);
      P.maxRun = Math.max(P.maxRun, runTime(i, P));
    }
    P.bursts = bursts(P);
    P.res = (result.rounds || []).find((r) => r.index === round.index) || null;
    return P;
  }

  // Flat [time, delta, time, delta, ...] -> {t: Float64Array, v: Int32Array, max}: v[i] holds from t[i].
  function stepSeries(flat) {
    const n = flat.length / 2;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => flat[2 * a] - flat[2 * b] || flat[2 * a + 1] - flat[2 * b + 1]);
    const t = new Float64Array(n);
    const v = new Int32Array(n);
    let d = 0;
    let max = 0;
    for (let q = 0; q < n; q++) {
      const i = order[q];
      d += flat[2 * i + 1];
      t[q] = flat[2 * i];
      v[q] = d;
      if (d > max) max = d;
    }
    return { t, v, max };
  }

  // ------------------------------------------------------------------ state

  const S = {
    ri: 0,
    t: 0,
    playing: false,
    speed: DEFAULT_SPEED,
    timeline: false,
    dirty: true,
    snap: true,
    lastDecision: -2,
    lastK: -1,
    pctWall: 0,
    overlay: null,
    betweenTimer: 0,
  };
  let scale = 1;
  const dpr = () => Math.max(1, window.devicePixelRatio || 1);

  // ------------------------------------------------------------------ layout / scaling

  const stage = $("stage");
  const mapCanvas = $("map");
  const tlCanvas = $("tl");
  const scrubCanvas = $("scrub-bg");
  const cardCanvas = $("card");
  const mapCtx = mapCanvas.getContext("2d");
  const tlCtx = tlCanvas.getContext("2d");
  const scrubCtx = scrubCanvas.getContext("2d");
  const cardCtx = cardCanvas.getContext("2d");
  const mapStatic = document.createElement("canvas");
  let pat = {};

  function sizeCanvas(cv, w, h) {
    const bw = Math.round(w * scale * dpr());
    const bh = Math.round(h * scale * dpr());
    if (cv.width !== bw || cv.height !== bh) {
      cv.width = bw;
      cv.height = bh;
    }
    return bw / w;
  }

  function fit() {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    scale = Math.min(vw / STAGE_W, vh / STAGE_H);
    const x = (vw - STAGE_W * scale) / 2;
    const y = (vh - STAGE_H * scale) / 2;
    stage.style.transform = "translate(" + x + "px," + y + "px) scale(" + scale + ")";
    sizeCanvas(mapCanvas, YARD_W, YARD_H);
    sizeCanvas(tlCanvas, TL_W, TL_H);
    const sw = scrubCanvas.parentElement.clientWidth || 1432;
    sizeCanvas(scrubCanvas, sw, 22);
    buildStatic();
    buildScrub();
    if (S.overlay === "result") drawCardOnScreen();
    S.dirty = true;
  }

  function themeChanged() {
    readColors();
    pat = {
      gpu: hatch(tlCtx, C.gpu, C.gpuSoft),
      ane: hatch(tlCtx, C.ane, C.aneSoft),
      late: hatch(tlCtx, C.late, C.lateSoft),
      bg: hatch(tlCtx, C.bgreq, "rgba(128,128,128,0.10)"),
    };
    buildStatic();
    buildScrub();
    if (S.overlay === "result") drawCardOnScreen();
    S.dirty = true;
  }


  // ------------------------------------------------------------------ yard geometry
  //
  // Six lines run left to right. Each line: entry -> queue -> signal -> switch -> three
  // platforms (A, B, C). A train reaches the back of its line's queue exactly at arrival_ms,
  // waits at the red signal until response_ms (the recorded decision latency), then the
  // switch throws to `answer` and it runs into that platform. Approach, run and dwell are
  // presentation distances in data time; the wait is the data.

  const ROW0 = 112;
  const ROW_H = 110;
  const ENTRY_X = 58;
  const STOP_X = 1000; // centre of the train standing at the signal
  const SIGNAL_X = 1030;
  const SWITCH_X = 1080;
  const BERTH_X0 = 1236;
  const BERTH_X1 = 1580;
  const PARK_X = 1420;
  const BERTH_DY = [-30, 0, 30];
  const TRL = 30; // train length
  const TRH = 12;
  const PITCH = 36;
  const BYPASS_DY = 15; // a train answered before the ones ahead of it passes them below
  const MAX_SLOTS = Math.floor((STOP_X - ENTRY_X - TRL / 2) / PITCH) + 1;
  const lineY = (l) => ROW0 + l * ROW_H;
  const slotX = (k) => STOP_X - k * PITCH;

  // Motion, in data milliseconds (so it scales with the playback speed).
  const APPROACH = 320; // entry -> back of the queue, ending exactly at arrival_ms
  const RUN_V = 1.6; // px per data ms, signal -> platform
  const DWELL = 300;
  const FADE = 200;
  const CREEP_V = 0.9; // px per data ms, moving up the queue
  const THROW = 70; // switch blade movement

  function bez(out, x0, y0, x1, y1, x2, y2, x3, y3, steps) {
    for (let s = 1; s <= steps; s++) {
      const u = s / steps;
      const v = 1 - u;
      out.push(
        v * v * v * x0 + 3 * v * v * u * x1 + 3 * v * u * u * x2 + u * u * u * x3,
        v * v * v * y0 + 3 * v * v * u * y1 + 3 * v * u * u * y2 + u * u * u * y3
      );
    }
  }
  // One sampled path per (line, platform): signal -> switch -> platform stop.
  const PATHS = [];
  for (let l = 0; l < 6; l++) {
    for (let p = 0; p < 3; p++) {
      const y = lineY(l);
      const yb = y + BERTH_DY[p];
      const pts = [STOP_X, y, SWITCH_X, y];
      bez(pts, SWITCH_X, y, SWITCH_X + 70, y, BERTH_X0 - 80, yb, BERTH_X0, yb, 24);
      pts.push(PARK_X, yb);
      const n = pts.length / 2;
      const xs = new Float32Array(n);
      const ys = new Float32Array(n);
      const cum = new Float32Array(n);
      for (let q = 0; q < n; q++) {
        xs[q] = pts[2 * q];
        ys[q] = pts[2 * q + 1];
        if (q) cum[q] = cum[q - 1] + Math.hypot(xs[q] - xs[q - 1], ys[q] - ys[q - 1]);
      }
      PATHS.push({ xs, ys, cum, len: cum[n - 1] });
    }
  }
  const pos = { x: 0, y: 0, a: 0 };
  function along(path, d) {
    d = Math.min(path.len, Math.max(0, d));
    const c = path.cum;
    let q = 1;
    while (q < c.length - 1 && c[q] < d) q++;
    const f = (d - c[q - 1]) / Math.max(1e-6, c[q] - c[q - 1]);
    pos.x = path.xs[q - 1] + (path.xs[q] - path.xs[q - 1]) * f;
    pos.y = path.ys[q - 1] + (path.ys[q] - path.ys[q - 1]) * f;
    pos.a = Math.atan2(path.ys[q] - path.ys[q - 1], path.xs[q] - path.xs[q - 1]);
  }
  // Where a departing train is `dt` data ms after its answer. It starts from the queue slot
  // it held at response_ms (0 = at the signal); trains answered out of turn use the bypass.
  function departPos(i, P, dt) {
    const k0 = P.slot0[i];
    const prefix = k0 * PITCH;
    const d = dt * RUN_V;
    const path = PATHS[P.line[i] * 3 + P.ans[i]];
    if (d < prefix) {
      pos.x = slotX(k0) + d;
      const left = prefix - d;
      pos.y = lineY(P.line[i]) + (k0 ? BYPASS_DY * Math.min(1, left / 40) : 0);
      pos.a = 0;
      return false;
    }
    along(path, d - prefix);
    return d - prefix >= path.len;
  }
  // Data ms from a train's answer until its tail has passed the signal.
  function clearTime(i, P) {
    return (P.slot0[i] * PITCH + SIGNAL_X - STOP_X + TRL) / RUN_V;
  }
  function runTime(i, P) {
    return (P.slot0[i] * PITCH + PATHS[P.line[i] * 3 + P.ans[i]].len) / RUN_V;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function buildStatic() {
    if (!C.bg) return;
    const k = scale * dpr();
    mapStatic.width = Math.round(YARD_W * k);
    mapStatic.height = Math.round(YARD_H * k);
    const g = mapStatic.getContext("2d");
    g.setTransform(k, 0, 0, k, 0, 0);
    g.clearRect(0, 0, YARD_W, YARD_H);
    g.lineCap = "round";
    g.lineJoin = "round";

    // column headings
    g.font = "700 11px " + FONT;
    g.fillStyle = C.textFaint;
    g.textBaseline = "alphabetic";
    g.textAlign = "left";
    g.fillText("LINE", 16, 44);
    g.fillText("QUEUE: TRAINS WAIT HERE FOR THEIR ANSWER", ENTRY_X + 30, 44);
    g.textAlign = "center";
    g.fillText("SIGNAL · JUNCTION", (SIGNAL_X + SWITCH_X) / 2 + 10, 44);
    g.fillText("(THE MODEL'S DECISION)", (SIGNAL_X + SWITCH_X) / 2 + 10, 58);
    g.fillText("PLATFORMS", (BERTH_X0 + BERTH_X1) / 2, 44);

    for (let l = 0; l < 6; l++) {
      const y = lineY(l);
      // row band
      g.fillStyle = C.surface;
      g.globalAlpha = l % 2 ? 0.35 : 0.6;
      g.fillRect(0, y - ROW_H / 2 + 4, YARD_W, ROW_H - 8);
      g.globalAlpha = 1;
      // platforms: a slab beside each platform track
      for (let p = 0; p < 3; p++) {
        const yb = y + BERTH_DY[p];
        g.beginPath();
        roundRect(g, BERTH_X0 + 30, yb + 8, BERTH_X1 - BERTH_X0 - 40, 6, 2);
        g.fillStyle = C.border;
        g.fill();
      }
      // sleepers along the queue
      g.strokeStyle = C.border;
      g.lineWidth = 1;
      g.beginPath();
      for (let x = ENTRY_X; x < STOP_X + 10; x += 12) {
        g.moveTo(x, y - 6);
        g.lineTo(x, y + 6);
      }
      g.stroke();
      // rails: main line, the three branches, the platform tracks
      g.strokeStyle = C.track;
      g.lineWidth = 3;
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(SWITCH_X, y);
      for (let p = 0; p < 3; p++) {
        const yb = y + BERTH_DY[p];
        g.moveTo(SWITCH_X, y);
        g.bezierCurveTo(SWITCH_X + 70, y, BERTH_X0 - 80, yb, BERTH_X0, yb);
        g.lineTo(BERTH_X1, yb);
      }
      g.stroke();
      // buffer stops
      g.strokeStyle = C.textFaint;
      g.lineWidth = 3;
      g.beginPath();
      for (let p = 0; p < 3; p++) {
        const yb = y + BERTH_DY[p];
        g.moveTo(BERTH_X1, yb - 7);
        g.lineTo(BERTH_X1, yb + 7);
      }
      g.stroke();
      // platform letters
      g.font = "700 12px " + FONT;
      g.fillStyle = C.textDim;
      g.textAlign = "right";
      g.textBaseline = "middle";
      for (let p = 0; p < 3; p++) g.fillText(PLATFORMS[p], BERTH_X0 - 4, y + BERTH_DY[p] - 9);
      // signal mast
      g.strokeStyle = C.textFaint;
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(SIGNAL_X, y - 6);
      g.lineTo(SIGNAL_X, y - 16);
      g.stroke();
      // line badge
      g.beginPath();
      g.arc(28, y, 13, 0, Math.PI * 2);
      g.fillStyle = C.surface2;
      g.fill();
      g.strokeStyle = C.track;
      g.lineWidth = 1.5;
      g.stroke();
      g.fillStyle = C.text;
      g.font = "700 13px " + FONT;
      g.textAlign = "center";
      g.fillText(String(l + 1), 28, y + 0.5);
    }
  }

  // A train facing right, centred on (0, 0) in the current transform. `body` fills it,
  // `nose` (optional) colours the front, `edge` (optional) outlines it, `mark` is a glyph.
  function train(ctx, body, nose, edge, mark, markColor) {
    ctx.beginPath();
    ctx.moveTo(-TRL / 2 + 3, -TRH / 2);
    ctx.lineTo(TRL / 2 - 6, -TRH / 2);
    ctx.quadraticCurveTo(TRL / 2, -TRH / 2, TRL / 2, 0);
    ctx.quadraticCurveTo(TRL / 2, TRH / 2, TRL / 2 - 6, TRH / 2);
    ctx.lineTo(-TRL / 2 + 3, TRH / 2);
    ctx.quadraticCurveTo(-TRL / 2, TRH / 2, -TRL / 2, TRH / 2 - 3);
    ctx.lineTo(-TRL / 2, -TRH / 2 + 3);
    ctx.quadraticCurveTo(-TRL / 2, -TRH / 2, -TRL / 2 + 3, -TRH / 2);
    ctx.closePath();
    ctx.fillStyle = body;
    ctx.fill();
    if (edge) {
      ctx.strokeStyle = edge;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    if (nose) {
      ctx.save();
      ctx.clip();
      ctx.fillStyle = nose;
      ctx.fillRect(TRL / 2 - 9, -TRH / 2, 9, TRH);
      ctx.restore();
    }
    if (mark) {
      ctx.fillStyle = markColor;
      ctx.fillText(mark, -3, 0.5);
    }
  }

  function barrier(ctx, x, y) {
    ctx.fillStyle = C.surface;
    ctx.fillRect(x - 3, y - 10, 6, 20);
    ctx.fillStyle = C.late;
    for (let q = 0; q < 4; q++) if (q % 2 === 0) ctx.fillRect(x - 3, y - 10 + q * 5, 6, 5);
    ctx.strokeStyle = C.late;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 3, y - 10, 6, 20);
  }

  const lineCount = new Int32Array(6);
  const waitCount = new Int32Array(6);
  const overflow = new Int32Array(6);
  const lastDep = new Int32Array(6); // latest departed train per line (by response), -1 none
  const prevDep = new Int32Array(6);
  const runningTr = new Int32Array(6); // latest train still running into a platform
  const dwelling = new Int32Array(6);
  const head = new Int32Array(6);
  const blade = new Uint8Array(6).fill(1);

  function drawYard(P, t, dtData) {
    const ctx = mapCtx;
    const k = mapCanvas.width / YARD_W;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
    ctx.drawImage(mapStatic, 0, 0);
    ctx.setTransform(k, 0, 0, k, 0, 0);
    ctx.font = "800 10px " + FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    lineCount.fill(0);
    waitCount.fill(0);
    overflow.fill(0);
    lastDep.fill(-1);
    prevDep.fill(-1);
    runningTr.fill(-1);
    dwelling.fill(-1);
    head.fill(-1);

    const still = reduceMotion.matches;
    const creep = S.snap || still ? 1e9 : CREEP_V * Math.max(0, dtData);
    const devColor = [C.gpu, C.ane];
    const lo = lowerBound(P.arr, t - P.maxLat - P.maxRun - DWELL - FADE);
    const hi = upperBound(P.arr, t + APPROACH);

    // Pass 1: trains queued or approaching, and who is where per line.
    for (let i = lo; i < hi; i++) {
      const a = P.arr[i];
      const r = P.resp[i];
      const l = P.line[i];
      if (t >= r) {
        // answered: remember the latest departures per line for the signal and switch
        if (lastDep[l] < 0 || r > P.resp[lastDep[l]]) {
          prevDep[l] = lastDep[l];
          lastDep[l] = i;
        } else if (prevDep[l] < 0 || r > P.resp[prevDep[l]]) {
          prevDep[l] = i;
        }
        const dt = t - r;
        const run = runTime(i, P);
        if (dt < run) {
          if (runningTr[l] < 0 || r > P.resp[runningTr[l]]) runningTr[l] = i;
        } else if (dt < run + DWELL + FADE) {
          if (dwelling[l] < 0 || r > P.resp[dwelling[l]]) dwelling[l] = i;
        }
        continue;
      }
      const s = lineCount[l]++;
      const y = lineY(l);
      if (t < a) {
        // approaching: from off the left edge to the back of the queue, arriving at arrival_ms
        if (still || s >= MAX_SLOTS) continue;
        const target = slotX(s);
        const p = 1 - (a - t) / APPROACH;
        const x = -TRL + (target + TRL) * (1 - (1 - p) * (1 - p));
        P.dispX[i] = x;
        ctx.globalAlpha = 0.45; // still on its way: it has not asked yet
        ctx.save();
        ctx.translate(x, y);
        train(ctx, C.train, null, C.trainEdge, null, null);
        ctx.restore();
        ctx.globalAlpha = 1;
        continue;
      }
      // waiting for its answer
      waitCount[l]++;
      if (s === 0) head[l] = i;
      if (s >= MAX_SLOTS) {
        overflow[l]++;
        continue;
      }
      const target = slotX(s);
      let x = P.dispX[i];
      if (!x || x > target || target - x <= creep) x = target;
      else x += creep;
      P.dispX[i] = x;
      ctx.save();
      ctx.translate(x, y);
      if (t - a > P.deadline) train(ctx, C.late, null, null, "!", C.onLate);
      else train(ctx, C.train, null, C.trainEdge, null, null);
      ctx.restore();
    }

    // Platforms: show the situation of the train the line is dealing with now: the one
    // running in, else the one standing in a platform, else the one at the signal.
    ctx.font = "800 10px " + FONT;
    for (let l = 0; l < 6; l++) {
      // a wrong-platform train standing in its platform keeps the stage, to show the conflict
      const wrong = dwelling[l] >= 0 && P.outc[dwelling[l]] === 2;
      const i = wrong ? dwelling[l] : runningTr[l] >= 0 ? runningTr[l] : dwelling[l] >= 0 ? dwelling[l] : head[l];
      if (i < 0) continue;
      const pl = P.trains[i].platforms || {};
      const y = lineY(l);
      for (let p = 0; p < 3; p++) {
        const st = pl[PLATFORMS[p]];
        const yb = y + BERTH_DY[p];
        if (st === "occupied") {
          ctx.save();
          ctx.translate(PARK_X + 70, yb);
          train(ctx, C.parked, null, null, null, null);
          ctx.restore();
        } else if (st === "closed") {
          barrier(ctx, BERTH_X0 + 22, yb);
        }
      }
    }

    // Pass 2: trains running into (or standing in) their platform, over the parked ones.
    for (let i = lo; i < hi; i++) {
      const r = P.resp[i];
      if (t < r) continue;
      const dt = t - r;
      const run = runTime(i, P);
      if (dt >= run + DWELL + FADE) continue;
      if (still) {
        along(PATHS[P.line[i] * 3 + P.ans[i]], 1e9);
      } else {
        departPos(i, P, dt);
      }
      const fade = dt > run + DWELL ? 1 - (dt - run - DWELL) / FADE : 1;
      ctx.globalAlpha = Math.max(0, fade);
      ctx.save();
      ctx.translate(pos.x, pos.y);
      ctx.rotate(pos.a);
      const o = P.outc[i];
      const dc = devColor[P.dev[i]];
      if (o === 1) train(ctx, C.late, dc, null, "!", C.onLate);
      else if (o === 2) train(ctx, dc, null, C.warn, "×", C.bg);
      else train(ctx, dc, null, null, null, null);
      ctx.restore();
      if (o === 2 && dt >= run) {
        // wrong platform: a conflict marker over it
        const yb = lineY(P.line[i]) + BERTH_DY[P.ans[i]];
        ctx.beginPath();
        ctx.arc(PARK_X + 40, yb - 14, 8, 0, Math.PI * 2);
        ctx.fillStyle = C.warn;
        ctx.fill();
        ctx.fillStyle = C.bg;
        ctx.fillText("×", PARK_X + 40, yb - 13.5);
      }
      ctx.globalAlpha = 1;
    }

    // Signals, switches and queue labels.
    for (let l = 0; l < 6; l++) {
      const y = lineY(l);
      const d = lastDep[l];
      // green only while the train that just got its answer clears the signal; red
      // whenever a train is waiting otherwise
      const green = d >= 0 && t - P.resp[d] < clearTime(d, P);
      // switch blade: points to the platform of the latest departure, thrown at its answer
      let to = blade[l];
      let from = to;
      if (d >= 0) {
        to = P.ans[d];
        from = prevDep[l] >= 0 ? P.ans[prevDep[l]] : to;
        blade[l] = to;
      }
      const u = d >= 0 && !still ? Math.min(1, (t - P.resp[d]) / THROW) : 1;
      const dy = BERTH_DY[from] + (BERTH_DY[to] - BERTH_DY[from]) * u;
      ctx.strokeStyle = green ? devColor[P.dev[d]] : C.text;
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(SWITCH_X, y);
      ctx.lineTo(SWITCH_X + 34, y + dy * 0.28);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(SWITCH_X, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = C.text;
      ctx.fill();
      // signal head: red while a train waits, green (ringed in the answering device's
      // colour) just after a departure
      const sy = y - 24;
      ctx.beginPath();
      roundRect(ctx, SIGNAL_X - 8, sy - 8, 16, 16, 8);
      ctx.fillStyle = C.surface2;
      ctx.fill();
      ctx.strokeStyle = green ? devColor[P.dev[d]] : C.track;
      ctx.lineWidth = green ? 3 : 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(SIGNAL_X, sy, 5, 0, Math.PI * 2);
      ctx.fillStyle = green ? C.ok : waitCount[l] ? C.late : C.border;
      ctx.fill();
      if (waitCount[l]) {
        ctx.font = "600 12px " + FONT;
        ctx.textAlign = "right";
        ctx.fillStyle = C.textDim;
        ctx.fillText(waitCount[l] + " waiting", STOP_X + 14, y + 22);
        ctx.textAlign = "center";
        ctx.font = "800 10px " + FONT;
      }
      if (overflow[l]) {
        ctx.font = "700 12px " + FONT;
        ctx.textAlign = "left";
        ctx.fillStyle = C.lateInk;
        ctx.fillText("+" + overflow[l] + " more", ENTRY_X, y - 18);
        ctx.textAlign = "center";
        ctx.font = "800 10px " + FONT;
      }
    }
  }

  // ------------------------------------------------------------------ timeline

  const TL = {
    x0: 180,
    xNow: 1500,
    lanes: [
      { top: 8, wait: 58, run: 24 },
      { top: 100, wait: 58, run: 24 },
    ],
  };
  TL.ppm = (TL.xNow - TL.x0) / WINDOW;
  const MAXQ = 8192;
  const buf = {};
  for (const k of ["rgpu", "rane", "rbg"]) buf[k] = { a: new Float32Array(MAXQ * 4), n: 0 };
  const laneWaiting = new Int32Array(2);
  const runLabels = new Float32Array(256);
  const patMatrix = new DOMMatrix([0.5, 0, 0, 0.5, 0, 0]);
  const running = new Int32Array(2);

  function push(b, x, y, w, h) {
    if (b.n >= MAXQ || w <= 0) return;
    const o = b.n * 4;
    b.a[o] = x;
    b.a[o + 1] = y;
    b.a[o + 2] = w;
    b.a[o + 3] = h;
    b.n++;
  }
  function flush(ctx, b, style) {
    if (!b.n) return;
    ctx.fillStyle = style;
    ctx.beginPath();
    for (let q = 0; q < b.n; q++) {
      const o = q * 4;
      ctx.rect(b.a[o], b.a[o + 1], b.a[o + 2], b.a[o + 3]);
    }
    ctx.fill();
    b.n = 0;
  }

  let globalDepth = 1;
  let bgLabels = [];
  const GRID = [10, 50, 100, 200, 500, 1000];

  // Square-root height so a queue of 1 is visible next to a queue of 100; gridlines give
  // the actual counts.
  function depthH(v, h) {
    return v <= 0 ? 0 : h * Math.sqrt(Math.min(1, v / globalDepth));
  }

  // Fill one queue-depth step series over [tMin, t]; returns its value at t.
  function area(ctx, s, t, tMin, base, h, X, style) {
    let i = upperBound(s.t, tMin) - 1;
    let v = i >= 0 ? s.v[i] : 0;
    ctx.beginPath();
    let x = X(tMin);
    ctx.moveTo(x, base);
    ctx.lineTo(x, base - depthH(v, h));
    for (i = i + 1; i < s.t.length && s.t[i] <= t; i++) {
      x = X(s.t[i]);
      ctx.lineTo(x, base - depthH(v, h));
      v = s.v[i];
      ctx.lineTo(x, base - depthH(v, h));
    }
    x = X(t);
    ctx.lineTo(x, base - depthH(v, h));
    ctx.lineTo(x, base);
    ctx.closePath();
    ctx.fillStyle = style;
    ctx.fill();
    return v;
  }

  function drawTimeline(P, t) {
    const ctx = tlCtx;
    const k = tlCanvas.width / TL_W;
    ctx.setTransform(k, 0, 0, k, 0, 0);
    ctx.clearRect(0, 0, TL_W, TL_H);
    const font = FONT;
    const X = (ms) => TL.xNow - (t - ms) * TL.ppm;
    const tMin = t - WINDOW - 40;

    // lane frames and labels
    for (let L = 0; L < 2; L++) {
      const ln = TL.lanes[L];
      ctx.fillStyle = C.surface2;
      ctx.fillRect(TL.x0, ln.top, TL.xNow - TL.x0, ln.wait);
      ctx.fillRect(TL.x0, ln.top + ln.wait + 3, TL.xNow - TL.x0, ln.run);
      ctx.fillStyle = L ? C.aneInk : C.gpuInk;
      ctx.font = "700 14px " + font;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(L ? "Neural Engine" : "GPU (MLX)", 24, ln.top + 20);
      ctx.fillStyle = C.textDim;
      ctx.font = "12px " + font;
      ctx.fillText("Waiting in queue", 24, ln.top + 42);
      ctx.fillText("Running on device", 24, ln.top + ln.wait + 19);
      ctx.fillStyle = L ? C.ane : C.gpu;
      ctx.fillRect(TL.x0 - 4, ln.top, 3, ln.wait + ln.run + 3);
    }

    // deadline marker: anything still waiting left of this line is late
    const xd = X(t - P.deadline);
    ctx.fillStyle = C.lateSoft;
    ctx.fillRect(xd, TL.lanes[0].top, TL.xNow - xd, TL.lanes[1].top + TL.lanes[1].wait + TL.lanes[1].run + 3 - TL.lanes[0].top);

    laneWaiting.fill(0);
    running.fill(0);
    const lo = lowerBound(P.iArr, t - WINDOW - P.iMaxDur - 50);
    const hi = upperBound(P.iArr, t);
    let nLabels = 0;
    for (let j = lo; j < hi; j++) {
      const s = P.iStart[j];
      const e = P.iEnd[j];
      const L = P.iLane[j];
      const ln = TL.lanes[L];
      const kind = P.iKind[j];
      // running
      if (t > s && e > tMin) {
        const x1 = X(s);
        const x2 = X(Math.min(e, t));
        if (t < e) running[L]++;
        const b = kind ? buf.rbg : L ? buf.rane : buf.rgpu;
        push(b, x1, ln.top + ln.wait + 3, Math.max(1, x2 - x1 - 0.8), ln.run);
        if (kind && x2 - x1 > 96 && nLabels < 256) {
          runLabels[nLabels++] = x1;
          runLabels[nLabels++] = ln.top + ln.wait + 3 + ln.run / 2;
          runLabels[nLabels++] = x2 - x1;
          runLabels[nLabels++] = kind;
        }
      }
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(TL.x0, 0, TL.xNow - TL.x0, TL_H);
    ctx.clip();
    const shift = -((t * TL.ppm) % 8);
    patMatrix.e = shift;
    pat.gpu.setTransform(patMatrix);
    pat.ane.setTransform(patMatrix);
    pat.late.setTransform(patMatrix);
    pat.bg.setTransform(patMatrix);

    // queue depth: everything waiting (grey), trains waiting (device colour), trains
    // waiting past the deadline (red), stacked from the lane floor
    for (let L = 0; L < 2; L++) {
      const ln = TL.lanes[L];
      const base = ln.top + ln.wait;
      const d = P.depth[L];
      laneWaiting[L] = area(ctx, d[2], t, tMin, base, ln.wait, X, pat.bg);
      area(ctx, d[1], t, tMin, base, ln.wait, X, L ? pat.ane : pat.gpu);
      area(ctx, d[0], t, tMin, base, ln.wait, X, pat.late);
      // gridlines with the real queue length, drawn over the areas
      if (!P.laneUsed[L]) continue;
      ctx.strokeStyle = C.textFaint;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      for (const g of GRID) {
        if (g > globalDepth) break;
        const y = Math.round(base - depthH(g, ln.wait)) + 0.5;
        ctx.moveTo(TL.x0, y);
        ctx.lineTo(TL.xNow, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.font = "600 11px " + font;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      for (const g of GRID) {
        if (g > globalDepth) break;
        const y = base - depthH(g, ln.wait);
        const label = g + " waiting";
        const w = ctx.measureText(label).width + 8;
        ctx.fillStyle = C.surface;
        ctx.fillRect(TL.x0 + 2, y - 7, w, 14);
        ctx.fillStyle = C.textDim;
        ctx.fillText(label, TL.x0 + 6, y + 0.5);
      }
    }
    flush(ctx, buf.rbg, C.bgreq);
    flush(ctx, buf.rgpu, C.gpu);
    flush(ctx, buf.rane, C.ane);
    ctx.font = "600 11px " + font;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillStyle = C.light ? "#fff" : "#0c0e12";
    for (let q = 0; q < nLabels; q += 4) {
      const x = Math.max(runLabels[q], TL.x0) + 6;
      const w = runLabels[q] + runLabels[q + 2] - x - 6;
      if (w > 60) ctx.fillText(bgLabels[runLabels[q + 3] - 1] || "", x, runLabels[q + 1] + 0.5, w);
    }
    ctx.restore();

    // deadline line + now line
    const yTop = TL.lanes[0].top - 4;
    const yBot = TL.lanes[1].top + TL.lanes[1].wait + TL.lanes[1].run + 3;
    ctx.strokeStyle = C.late;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(xd, yTop);
    ctx.lineTo(xd, yBot + 4);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(TL.xNow, yTop);
    ctx.lineTo(TL.xNow, yBot + 4);
    ctx.stroke();

    // axis
    ctx.fillStyle = C.textFaint;
    ctx.font = "12px " + font;
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "center";
    const yAx = TL_H - 4;
    for (let s = 1; s <= 3; s++) ctx.fillText("−" + s + " s", TL.xNow - s * 1000 * TL.ppm, yAx);
    ctx.fillStyle = C.text;
    ctx.font = "600 12px " + font;
    ctx.fillText("now", TL.xNow, yAx);
    ctx.fillStyle = C.lateInk;
    ctx.textAlign = "right";
    ctx.fillText(fmtInt(P.deadline) + " ms deadline", xd - 6, yAx);

    // right of now: live counts per lane
    ctx.textAlign = "left";
    for (let L = 0; L < 2; L++) {
      const ln = TL.lanes[L];
      const x = TL.xNow + 12;
      if (!P.laneUsed[L]) {
        ctx.fillStyle = C.textFaint;
        ctx.font = "13px " + font;
        ctx.textAlign = "center";
        ctx.fillText(
          "Not used in this round: every request runs on the GPU",
          (TL.x0 + TL.xNow) / 2,
          ln.top + (ln.wait + ln.run) / 2 + 6
        );
        ctx.textAlign = "left";
        continue;
      }
      ctx.fillStyle = laneWaiting[L] ? C.text : C.textFaint;
      ctx.font = "700 18px " + font;
      ctx.fillText(String(laneWaiting[L]), x, ln.top + 26);
      ctx.fillStyle = C.textDim;
      ctx.font = "12px " + font;
      ctx.fillText("waiting", x, ln.top + 42);
      ctx.fillText(running[L] ? "busy" : "idle", x, ln.top + ln.wait + 19);
    }
  }


  // ------------------------------------------------------------------ scrubber

  // Rush-hour windows of a bursty timetable, [start, end) in round ms; empty otherwise.
  function bursts(P) {
    const wl = result.workload || {};
    if (wl.arrivals !== "bursty") return [];
    const period = (wl.burst_period_s ?? BURST_PERIOD_S) * 1000;
    const on = (wl.burst_on_s ?? BURST_ON_S) * 1000;
    const out = [];
    for (let s = 0; s < P.duration; s += period) out.push([s, Math.min(s + on, P.duration)]);
    return out;
  }
  function inRush(P, t) {
    for (const [a, b] of P.bursts) if (t >= a && t < b) return true;
    return false;
  }
  // Start of the next rush hour after t, a moment early so the build-up is visible.
  function nextRush(P, t) {
    const lead = 250;
    for (const [a] of P.bursts) if (a - lead > t + 1) return Math.max(0, a - lead);
    return P.bursts.length ? Math.max(0, P.bursts[0][0] - lead) : t;
  }

  function buildScrub() {
    const P = R[S.ri];
    if (!P || !C.bg) return;
    const w = scrubCanvas.parentElement.clientWidth || 1432;
    const k = scrubCanvas.width / w;
    const ctx = scrubCtx;
    ctx.setTransform(k, 0, 0, k, 0, 0);
    ctx.clearRect(0, 0, w, 22);
    ctx.fillStyle = C.surface2;
    ctx.fillRect(0, 7, w, 8);
    // rush-hour windows as markers along the bottom
    ctx.fillStyle = C.warn;
    for (const [a, b] of P.bursts) ctx.fillRect((a / P.end) * w, 19, Math.max(2, ((b - a) / P.end) * w), 3);
    // arrivals per bin (grey) and late trains per bin (red)
    const bins = Math.floor(w / 3);
    const arr = new Float32Array(bins);
    const late = new Float32Array(bins);
    for (let i = 0; i < P.m; i++) arr[Math.min(bins - 1, Math.floor((P.iArr[i] / P.end) * bins))]++;
    for (let i = 0; i < P.lateT.length; i++) late[Math.min(bins - 1, Math.floor((P.lateT[i] / P.end) * bins))]++;
    let mx = 1;
    for (let b = 0; b < bins; b++) mx = Math.max(mx, arr[b]);
    for (let b = 0; b < bins; b++) {
      const h = (arr[b] / mx) * 17;
      ctx.fillStyle = C.track;
      ctx.fillRect(b * 3, 18 - h, 2, h);
      if (late[b]) {
        const hl = (late[b] / mx) * 17;
        ctx.fillStyle = C.late;
        ctx.fillRect(b * 3, 18 - hl, 2, hl);
      }
    }
  }

  // ------------------------------------------------------------------ HUD

  const el = {
    clock: $("clock"),
    clockEnd: $("clock-end"),
    scrub: $("scrubber"),
    rush: $("rush"),
    dec: $("dec-line"),
    late: $("m-late"),
    lateRate: $("m-late-rate"),
    p99: $("m-p99"),
  };

  function updateHud(P, t, wall) {
    setText(el.clock, fmtS(t));
    if (document.activeElement !== el.scrub) el.scrub.value = String(Math.round(t));

    // latest decision
    const k = upperBound(P.respSorted, t);
    const idx = k > 0 ? P.byResp[k - 1] : -1;
    if (idx !== S.lastDecision) {
      S.lastDecision = idx;
      if (idx < 0) {
        el.dec.textContent = "Waiting for the first answer";
      } else {
        const tr = P.trains[idx];
        const o = P.outc[idx];
        const dev = DEV_KEY[P.dev[idx]];
        el.dec.innerHTML =
          "<b>Train " + esc(tr.id) + " → " + esc(tr.answer) + "</b> · " +
          '<span class="' + dev + '">' + DEVICE[P.dev[idx]] + "</span> · " + fmtMs(tr.latency_ms) + " · " +
          (o === 1
            ? '<span class="late">! Late</span>'
            : o === 2
              ? '<span class="wrong">× Wrong platform</span>'
              : '<span class="ok">✓ On time</span>');
      }
    }

    // late so far: a train counts as late once it has waited past the deadline
    const arrived = upperBound(P.arr, t);
    const late = upperBound(P.lateT, t);
    setText(el.late, fmtInt(late));
    el.late.classList.toggle("has-late", late > 0);
    setText(el.lateRate, "of " + fmtInt(arrived) + " trains" + (arrived ? " · " + fmtPct(late / arrived) : ""));

    // P99 over the answers so far (throttled; it only moves when k does)
    if (k !== S.lastK && (wall - S.pctWall > 150 || !S.playing)) {
      S.lastK = k;
      S.pctWall = wall;
      const sub = P.scratch.subarray(0, k);
      sub.set(P.latByResp.subarray(0, k));
      sub.sort();
      setText(el.p99, fmtMs(pct(sub, k, 99)));
    }

    el.rush.hidden = !inRush(P, t);
  }

  function buildDetails(P) {
    const r = P.res;
    const sys = (r && r.systems) || {};
    const game = (r && r.game) || {};
    const sec = sys.secondary || {};
    const rows = [];
    const row = (k, v) => v != null && v !== "—" && rows.push("<dt>" + esc(k) + "</dt><dd>" + esc(v) + "</dd>");
    const grp = (t) => rows.push('<div class="grp">' + esc(t) + "</div>");
    if (!r) {
      $("details-list").innerHTML = '<dt>No totals for this round in result.json</dt><dd></dd>';
      return;
    }
    grp("Trains");
    row("Trains", fmtInt(game.trains));
    row("Late", fmtInt(get(sys, "late.count")) + " (" + fmtPct(get(sys, "late.rate")) + ")");
    row("Delivered on time", fmtInt(game.delivered));
    row("On time but wrong platform", fmtInt(game.misrouted));
    grp("Waiting and latency");
    row("P99 queue wait", fmtMs(get(sys, "queue_wait.p99_ms")));
    row("P99 decision latency", fmtMs(get(sys, "decision_latency.p99_ms")));
    row("Slowest decision", fmtMs(get(sys, "decision_latency.max_ms")));
    row("Model time per decision (P50)", fmtMs(get(sec, "inference.p50_ms")));
    row("Hand-off delay (P99)", fmtMs(get(sec, "submit_lag.p99_ms")));
    row("Other overhead (P99)", fmtMs(get(sec, "overhead.p99_ms")));
    const miss = sys.miss_rate_at_ms || {};
    const th = Object.keys(miss).sort((a, b) => a - b);
    if (th.length) {
      grp("Share of decisions slower than");
      for (const x of th) row(x + " ms", fmtPct(miss[x]));
    }
    const devices = sec.devices || {};
    if (Object.keys(devices).length) {
      grp("Where train decisions ran");
      for (const d of Object.keys(devices)) row(d === "ane" ? "Neural Engine" : d === "gpu" ? "MLX GPU" : d, fmtInt(devices[d]));
    }
    const bg = sec.background || {};
    if (Object.keys(bg).length) {
      grp("Background documents (P99 latency)");
      for (const c of Object.keys(bg)) row(bgName(c), fmtMs(get(bg[c], "decision_latency.p99_ms") ?? get(bg[c], "p99_ms")));
    }
    grp("Throughput");
    row("Decisions per second", sec.decisions_per_s != null ? sec.decisions_per_s.toFixed(1) : null);
    row("Requests completed per second", sec.completed_req_s != null ? sec.completed_req_s.toFixed(1) : null);
    row("Time to finish all requests", sec.makespan_s != null ? sec.makespan_s.toFixed(1) + " s" : null);
    $("details-list").innerHTML = rows.join("");
  }

  function bgName(cls) {
    const len = get(result, "workload.lengths." + cls);
    const base = BG_LABEL[cls] || cls;
    return len ? base + " (" + len + " tokens)" : base;
  }


  // ------------------------------------------------------------------ rounds / playback

  function buildTabs() {
    const nav = $("round-tabs");
    nav.innerHTML = "";
    R.forEach((P, i) => {
      const b = document.createElement("button");
      b.className = "tab";
      b.type = "button";
      b.setAttribute("role", "tab");
      b.title = "Switch round (← →)";
      const dots = P.config === "hybrid" ? '<i class="dot dot-gpu"></i><i class="dot dot-ane"></i>' : '<i class="dot dot-gpu"></i>';
      b.setAttribute("aria-label", "Round " + (i + 1) + ": " + roundLabel(P));
      b.innerHTML =
        '<span class="tab-n">' + (i + 1) + "</span>" + dots + esc(label(P.config)) +
        (unverified(P) ? '<span class="chip chip-warn tab-flag">not comparable</span>' : "");
      if (unverified(P)) b.title = UNVERIFIED_LABEL + ". Switch round (← →)";
      b.addEventListener("click", () => selectRound(i, true));
      nav.appendChild(b);
    });
  }

  function selectRound(i, play) {
    if (i < 0 || i >= R.length) return;
    S.ri = i;
    const P = R[i];
    globalDepth = Math.max(1, ...R.map((q) => q.maxDepth));
    blade.fill(1);
    bgLabels = P.bgClasses.map(bgName);
    Array.from($("round-tabs").children).forEach((b, j) => b.setAttribute("aria-selected", String(j === i)));
    $("round-title").innerHTML =
      '<span class="dim">Round ' + (i + 1) + " of " + R.length + ":</span> " + esc(roundLabel(P));
    setText($("round-desc"), roundDesc(P));
    el.scrub.max = String(Math.round(P.end));
    setText(el.clockEnd, fmtS(P.end));
    for (const d of document.querySelectorAll(".deadline-text")) d.textContent = fmtInt(P.deadline) + " ms";
    buildDetails(P);
    buildScrub();
    P.dispX.fill(0);
    seek(0);
    closeBetween();
    setPlaying(!!play);
  }

  function seek(t) {
    const P = R[S.ri];
    S.t = Math.max(0, Math.min(P.end, t));
    S.snap = true;
    S.lastDecision = -2;
    S.lastK = -1;
    S.dirty = true;
  }

  function setPlaying(on) {
    const P = R[S.ri];
    if (on && S.t >= P.end) seek(0);
    S.playing = on;
    const b = $("btn-play");
    b.classList.toggle("playing", on);
    b.setAttribute("aria-label", on ? "Pause" : "Play");
    S.dirty = true;
  }

  function setSpeed(k) {
    S.speed = k;
    Array.from($("speeds").children).forEach((b, j) => b.setAttribute("aria-pressed", String(j === k)));
    setText($("speed-label"), SPEEDS[k] + "× real time");
  }

  function roundEnded() {
    setPlaying(false);
    const P = R[S.ri];
    const late = P.res ? get(P.res, "systems.late.count") : P.lateT.length;
    const trains = P.res ? get(P.res, "game.trains") : P.n;
    setText($("map-status"), "Round " + (S.ri + 1) + " complete: " + late + " of " + trains + " trains late.");
    if (S.ri + 1 < R.length) openBetween(P, late, trains);
    else openResult();
  }

  // ------------------------------------------------------------------ overlays

  let returnFocus = null;
  function openOverlay(id, focusId) {
    returnFocus = document.activeElement;
    $(id).hidden = false;
    S.overlay = id;
    const f = focusId && $(focusId);
    if (f) f.focus();
  }
  function closeOverlay(id) {
    $(id).hidden = true;
    if (S.overlay === id) S.overlay = null;
    if (returnFocus && returnFocus.focus) returnFocus.focus();
  }

  function openBetween(P, late, trains) {
    const next = R[S.ri + 1];
    setText($("between-eyebrow"), "Round " + (S.ri + 1) + " complete · " + roundLabel(P));
    setText($("between-title"), fmtInt(late) + " of " + fmtInt(trains) + " trains were late");
    setText($("between-text"), "Next: the same timetable with " + roundLabel(next) + ". " + roundDesc(next));
    openOverlay("between", "btn-next");
    let n = 6;
    const btn = $("btn-next");
    const tick = () => {
      n--;
      btn.textContent = reduceMotion.matches ? "Continue" : "Continue (" + n + ")";
      if (n <= 0) nextRound();
    };
    btn.textContent = "Continue";
    clearInterval(S.betweenTimer);
    if (!reduceMotion.matches) {
      tick();
      S.betweenTimer = setInterval(tick, 1000);
    }
  }
  function closeBetween() {
    clearInterval(S.betweenTimer);
    if (!$("between").hidden) closeOverlay("between");
  }
  function nextRound() {
    closeBetween();
    selectRound(S.ri + 1, true);
  }

  function openIntro() {
    setPlaying(false);
    openOverlay("intro", "btn-start");
  }
  function closeIntro() {
    try {
      localStorage.setItem("switchyard.intro.v2", "seen");
    } catch (e) {
      /* storage may be unavailable on file:// or in private windows */
    }
    closeOverlay("intro");
    setPlaying(true);
  }

  function openResult() {
    closeBetween();
    setPlaying(false);
    openOverlay("result", "btn-save");
    drawCardOnScreen();
  }

  // ------------------------------------------------------------------ result card

  // Per config, pooled over every round of that config: configs.<label>.summary has the
  // shape of rounds[].systems, configs.<label>.game that of rounds[].game.
  function configStats(cfg) {
    const sys = get(result, "configs." + cfg + ".summary") || {};
    const game = get(result, "configs." + cfg + ".game") || {};
    return {
      label: label(cfg),
      config: cfg,
      trains: game.trains,
      late: sys.late || {},
      dl: sys.decision_latency || {},
      qw: sys.queue_wait || {},
      miss: sys.miss_rate_at_ms || {},
    };
  }

  function sentence(s) {
    if (!s) return "";
    s = String(s).trim();
    return s.charAt(0).toUpperCase() + s.slice(1) + (/[.!?]$/.test(s) ? "" : ".");
  }

  function wrap(ctx, text, maxW) {
    const words = String(text).split(" ");
    const lines = [];
    let cur = "";
    for (const w of words) {
      const t = cur ? cur + " " + w : w;
      if (ctx.measureText(t).width > maxW && cur) {
        lines.push(cur);
        cur = w;
      } else cur = t;
    }
    if (cur) lines.push(cur);
    return lines;
  }

  function textRuns(ctx, x, y, runs) {
    for (const [s, color] of runs) {
      ctx.fillStyle = color;
      ctx.fillText(s, x, y);
      x += ctx.measureText(s).width;
    }
    return x;
  }

  function cardModel() {
    const labels = Object.keys(result.configs || {});
    const order = ["gpu_only", "hybrid"].filter((c) => labels.indexOf(c) !== -1);
    for (const c of labels) if (order.indexOf(c) === -1) order.push(c);
    const wl = result.workload || {};
    // Two columns only when the run says the configs are comparable; otherwise GPU only.
    const shown = comparable() ? order : order.filter((c) => c === "gpu_only");
    return { cols: shown.map(configStats), wl, th: (wl.thresholds_ms || []).slice().sort((a, b) => a - b), ane: result.ane || {} };
  }

  function drawCard(ctx) {
    const W = CARD_W;
    const H = CARD_H;
    const font = FONT;
    const mono = getComputedStyle(document.documentElement).getPropertyValue("--mono");
    const M = cardModel();
    const deadline = M.wl.deadline_ms != null ? M.wl.deadline_ms : R[0].deadline;
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, W, H);
    if (result.synthetic) {
      ctx.save();
      ctx.translate(W / 2, H / 2 + 40);
      ctx.rotate(-0.2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = C.warn;
      ctx.globalAlpha = 0.08;
      ctx.font = "850 104px " + font;
      ctx.fillText("SYNTHETIC SAMPLE DATA", 0, 0);
      ctx.restore();
    }
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";

    // header
    ctx.fillStyle = C.textDim;
    ctx.font = "700 15px " + font;
    ctx.fillText("SWITCHYARD  ·  LAYA-APPLE", 80, 78);
    if (result.synthetic) {
      ctx.textAlign = "right";
      ctx.fillStyle = C.warn;
      ctx.font = "800 15px " + font;
      ctx.fillText("SYNTHETIC SAMPLE DATA · NOT A MEASUREMENT", W - 80, 78);
      ctx.textAlign = "left";
    }
    ctx.font = "750 50px " + font;
    textRuns(ctx, 80, 140, [
      ["Same timetable. ", C.text],
      ["GPU-only", C.gpuInk],
      [" vs ", C.text],
      ["GPU", C.gpuInk],
      [" + ", C.text],
      ["ANE", C.aneInk],
      [".", C.text],
    ]);
    ctx.font = "20px " + font;
    ctx.fillStyle = C.textDim;
    const trains = M.cols[0] && M.cols[0].trains;
    const sub =
      (trains != null ? fmtInt(trains) + " trains" : "Trains") +
      (M.wl.duration_s ? " in " + fmtInt(M.wl.duration_s) + " s" : "") +
      (M.wl.arrivals ? ", " + M.wl.arrivals + " arrivals" : "") +
      (M.wl.seed != null ? ", seed " + M.wl.seed : "") +
      ". A train is late when its decision takes more than " + fmtInt(deadline) + " ms, queueing included.";
    ctx.fillText(sub, 80, 182);

    // table
    const colX = [560, 1080];
    const colW = 440;
    const rows = { head: 252, late: 368, p99: 492, qw: 580, miss: 660 };
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    const hr = (y) => {
      ctx.beginPath();
      ctx.moveTo(80, y);
      ctx.lineTo(W - 80, y);
      ctx.stroke();
    };
    hr(274);
    hr(424);
    hr(530);
    hr(618);

    const rowLabel = (y, main, subl) => {
      ctx.textAlign = "left";
      ctx.fillStyle = C.text;
      ctx.font = "650 21px " + font;
      ctx.fillText(main, 80, y);
      ctx.fillStyle = C.textDim;
      ctx.font = "15px " + font;
      ctx.fillText(subl, 80, y + 24);
    };
    rowLabel(rows.late - 30, "Late trains", "decision later than " + fmtInt(deadline) + " ms");
    rowLabel(rows.p99 - 12, "P99 decision latency", "train arrival to answer");
    rowLabel(rows.qw - 12, "P99 queue wait", "time waiting before running");
    rowLabel(rows.miss - 4, "Slower than", "share of train decisions");

    const drawCol = (c, x) => {
      // header
      ctx.textAlign = "left";
      ctx.font = "700 24px " + font;
      let hx = x;
      if (c.config === "hybrid") {
        hx = textRuns(ctx, hx, rows.head, [["GPU", C.gpuInk], [" + ", C.text], ["ANE", C.aneInk]]);
      } else if (c.config === "gpu_only") {
        hx = textRuns(ctx, hx, rows.head, [["GPU only", C.gpuInk]]);
      } else {
        hx = textRuns(ctx, hx, rows.head, [[c.label, C.text]]);
      }
      // late (hero)
      ctx.fillStyle = C.text;
      ctx.font = "780 96px " + font;
      const lateN = c.late.count;
      const lx = textRuns(ctx, x, rows.late + 20, [[fmtInt(lateN), C.text]]);
      ctx.font = "18px " + font;
      ctx.fillStyle = C.textDim;
      const rate = c.late.rate != null ? c.late.rate : c.trains ? lateN / c.trains : null;
      ctx.fillText(fmtPct(rate) + " of " + fmtInt(c.trains), lx + 16, rows.late + 20);
      // p99 decision, p99 queue
      ctx.fillStyle = C.text;
      ctx.font = "650 44px " + font;
      ctx.fillText(fmtMs(c.dl.p99_ms), x, rows.p99 + 12);
      ctx.fillText(fmtMs(c.qw.p99_ms), x, rows.qw + 14);
      // miss-rate row
      const cw = colW / Math.max(1, M.th.length);
      M.th.forEach((ms, q) => {
        const cx = x + q * cw;
        const v = c.miss[String(ms)];
        ctx.fillStyle = C.textFaint;
        ctx.font = "13px " + font;
        ctx.fillText(fmtInt(ms) + " ms", cx, rows.miss - 12);
        ctx.fillStyle = C.text;
        ctx.font = "650 21px " + font;
        ctx.fillText(fmtPct(v), cx, rows.miss + 14);
        ctx.fillStyle = C.surface2;
        ctx.fillRect(cx, rows.miss + 24, cw - 14, 5);
        if (v) {
          ctx.fillStyle = C.late;
          ctx.fillRect(cx, rows.miss + 24, Math.max(1.5, (cw - 14) * v), 5);
        }
      });
    };
    M.cols.slice(0, 2).forEach((c, i) => drawCol(c, colX[i]));

    if (M.cols.length < 2) {
      // GPU + ANE did not run: say why, and how to set it up when that is possible
      const x = colX[1];
      ctx.textAlign = "left";
      ctx.font = "700 24px " + font;
      textRuns(ctx, x, rows.head, [["GPU", C.textFaint], [" + ", C.textFaint], ["ANE", C.textFaint]]);
      ctx.fillStyle = C.surface2;
      ctx.beginPath();
      roundRect(ctx, x - 20, 300, colW + 20, 400, 14);
      ctx.fill();
      ctx.strokeStyle = C.border;
      ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.font = "700 26px " + font;
      ctx.fillText(R.some(unverified) ? "GPU + ANE not comparable" : "GPU + ANE unavailable", x + 8, 352);
      ctx.fillStyle = C.textDim;
      ctx.font = "18px " + font;
      let y = 388;
      for (const line of wrap(ctx, sentence(M.ane.reason_text || M.ane.reason) || "The Neural Engine was not used on this Mac.", colW - 40)) {
        ctx.fillText(line, x + 8, y);
        y += 26;
      }
      if (M.ane.state === "setup_available" && M.ane.setup_command) {
        y += 18;
        ctx.fillStyle = C.text;
        ctx.font = "600 17px " + font;
        ctx.fillText("Set it up on this Mac, then run again:", x + 8, y);
        y += 16;
        ctx.font = "15px " + mono;
        const cmd = wrap(ctx, M.ane.setup_command, colW - 60);
        ctx.fillStyle = C.bg;
        ctx.beginPath();
        roundRect(ctx, x + 8, y, colW - 36, cmd.length * 22 + 22, 8);
        ctx.fill();
        ctx.fillStyle = C.text;
        let cy = y + 27;
        for (const line of cmd) {
          ctx.fillText(line, x + 22, cy);
          cy += 22;
        }
      }
    }

    // footer
    hr(806);
    const plat = get(result, "machine.platform") || {};
    const rev = get(result, "model.revision");
    const seq = get(result, "design.sequence") || R.map((P) => P.config);
    const parts = [
      plat.soc,
      plat.macos ? "macOS " + plat.macos : null,
      get(result, "model.name") ? get(result, "model.name") + (rev ? "@" + String(rev).slice(0, 7) : "") : null,
      M.wl.seed != null ? "seed " + M.wl.seed : null,
      seq.length ? "order " + seq.map(label).join(" → ") : null,
      result.laya_apple ? "laya-apple " + result.laya_apple : null,
    ].filter(Boolean);
    ctx.textAlign = "left";
    ctx.fillStyle = C.textDim;
    ctx.font = "15px " + font;
    ctx.fillText(parts.join("  ·  "), 80, 846, 1040);
    ctx.textAlign = "right";
    ctx.fillStyle = C.text;
    ctx.font = "650 15px " + font;
    ctx.fillText("github.com/tc3oliver/laya-apple", W - 80, 846);

  }

  function cardSummary() {
    const M = cardModel();
    const s = M.cols
      .map((c) => c.label + ": " + fmtInt(c.late.count) + " of " + fmtInt(c.trains) + " trains late, P99 decision latency " + fmtMs(c.dl.p99_ms) + ", P99 queue wait " + fmtMs(c.qw.p99_ms) + ".")
      .join(" ");
    const heading = R.some(unverified) ? "GPU + ANE not comparable" : "GPU + ANE unavailable";
    const extra = M.cols.length < 2 ? " " + heading + ". " + sentence(M.ane.reason_text || M.ane.reason) : "";
    return (result.synthetic ? "Synthetic sample data. " : "") + "Same timetable. " + s + extra;
  }

  function drawCardOnScreen() {
    const k = sizeCanvas(cardCanvas, CARD_CSS_W, CARD_CSS_H) * (CARD_CSS_W / CARD_W);
    cardCtx.setTransform(k, 0, 0, k, 0, 0);
    drawCard(cardCtx);
    cardCanvas.setAttribute("aria-label", cardSummary());
    const cmd = get(result, "ane.state") === "setup_available" && get(result, "ane.setup_command");
    $("btn-copy").hidden = !cmd;
  }

  function savePng() {
    const c = document.createElement("canvas");
    c.width = CARD_W * 2;
    c.height = CARD_H * 2;
    const g = c.getContext("2d");
    g.setTransform(2, 0, 0, 2, 0, 0);
    drawCard(g);
    const soc = (get(result, "machine.platform.soc") || "mac").toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const name = "switchyard-" + soc + "-seed" + (get(result, "workload.seed") ?? "x") + (result.synthetic ? "-synthetic" : "") + ".png";
    const done = (url) => {
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setText($("copy-status"), "Saved " + name);
    };
    if (c.toBlob) c.toBlob((b) => done(URL.createObjectURL(b)), "image/png");
    else done(c.toDataURL("image/png"));
  }

  function copyCommand() {
    const cmd = get(result, "ane.setup_command") || "";
    const ok = () => setText($("copy-status"), "Copied setup command");
    const fallback = () => {
      const ta = document.createElement("textarea");
      ta.value = cmd;
      ta.setAttribute("readonly", "");
      ta.style.position = "absolute";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        ok();
      } catch (e) {
        setText($("copy-status"), cmd);
      }
      ta.remove();
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(cmd).then(ok, fallback);
    else fallback();
  }


  // ------------------------------------------------------------------ loop

  let lastWall = 0;
  function frame(now) {
    const dt = lastWall ? Math.min(100, now - lastWall) : 16;
    lastWall = now;
    const P = R[S.ri];
    const dtData = S.playing ? dt * SPEEDS[S.speed] : 0;
    if (S.playing) {
      S.t += dtData;
      S.dirty = true;
      if (S.t >= P.end) {
        S.t = P.end;
        roundEnded();
      }
    }
    if (S.dirty) {
      drawYard(P, S.t, dtData);
      if (S.timeline) drawTimeline(P, S.t);
      updateHud(P, S.t, now);
      if (S.snap) S.snap = false;
      else S.dirty = S.playing;
    }
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------ input

  function onKey(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (S.overlay === "intro") {
      if (k === "Enter" || k === " " || k === "Escape") {
        e.preventDefault();
        closeIntro();
      }
      return;
    }
    if (S.overlay === "help") {
      if (k === "Escape" || k === "?" || k === "Enter") {
        e.preventDefault();
        closeOverlay("help");
      }
      return;
    }
    if (S.overlay === "result") {
      if (k === "Escape" || k === "r" || k === "R") {
        e.preventDefault();
        closeOverlay("result");
      }
      return;
    }
    if (S.overlay === "between") {
      if (k === "Escape") {
        e.preventDefault();
        closeBetween();
      } else if (k === "r" || k === "R") openResult();
      else if (k === "ArrowRight" || k === "Enter" || k === " ") {
        e.preventDefault();
        nextRound();
      }
      return;
    }
    const onScrub = document.activeElement === el.scrub;
    const key = k.length === 1 ? k.toLowerCase() : k;
    if (k === " ") {
      e.preventDefault();
      setPlaying(!S.playing);
    } else if ((k === "ArrowLeft" || k === "ArrowRight") && !onScrub) {
      e.preventDefault();
      selectRound(S.ri + (k === "ArrowRight" ? 1 : -1), S.playing);
    } else if (k === "," || k === ".") {
      seek(S.t + (k === "." ? 1000 : -1000));
    } else if (key === "n") {
      jumpToRush();
    } else if (key === "r") {
      openResult();
    } else if (key === "s") {
      setSpeed((S.speed + 1) % SPEEDS.length);
    } else if (key === "t") {
      toggleTimeline();
    } else if (key === "d") {
      toggleDetails();
    } else if (k === "?") {
      setPlaying(false);
      openOverlay("help", "btn-help-close");
    } else if (k === "Escape") {
      if (!$("details").hidden) toggleDetails();
      else if (S.timeline) toggleTimeline();
    }
  }

  function jumpToRush() {
    const P = R[S.ri];
    seek(nextRush(P, S.t));
  }

  function toggleTimeline() {
    S.timeline = !S.timeline;
    $("timeline").hidden = !S.timeline;
    $("btn-timeline").setAttribute("aria-expanded", String(S.timeline));
    S.dirty = true;
  }

  function toggleDetails() {
    const open = $("details").hidden;
    $("details").hidden = !open;
    $("btn-details").setAttribute("aria-expanded", String(open));
  }

  function buildSpeeds() {
    const box = $("speeds");
    box.innerHTML = "";
    SPEEDS.forEach((v, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "seg-btn";
      b.textContent = v + "×";
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => setSpeed(i));
      box.appendChild(b);
    });
  }

  function wire() {
    $("btn-play").addEventListener("click", () => setPlaying(!S.playing));
    $("btn-rush").addEventListener("click", jumpToRush);
    $("btn-timeline").addEventListener("click", toggleTimeline);
    $("btn-details").addEventListener("click", toggleDetails);
    $("btn-result").addEventListener("click", openResult);
    $("btn-help").addEventListener("click", () => {
      setPlaying(false);
      openOverlay("help", "btn-help-close");
    });
    $("btn-help-close").addEventListener("click", () => closeOverlay("help"));
    $("btn-help-intro").addEventListener("click", () => {
      closeOverlay("help");
      openIntro();
    });
    $("btn-start").addEventListener("click", closeIntro);
    $("btn-next").addEventListener("click", nextRound);
    $("btn-between-result").addEventListener("click", openResult);
    $("btn-save").addEventListener("click", savePng);
    $("btn-copy").addEventListener("click", copyCommand);
    $("btn-result-close").addEventListener("click", () => closeOverlay("result"));
    $("btn-replay").addEventListener("click", () => {
      closeOverlay("result");
      selectRound(0, true);
    });
    el.scrub.addEventListener("input", () => seek(Number(el.scrub.value)));
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", fit);
    const onScheme = () => themeChanged();
    if (lightQuery.addEventListener) lightQuery.addEventListener("change", onScheme);
    else lightQuery.addListener(onScheme);
  }

  // ------------------------------------------------------------------ boot

  function boot() {
    readColors();
    fit();
    if (!loadData()) return;
    const synthetic = !!result.synthetic;
    $("synthetic-chip").hidden = !synthetic;
    $("watermark").hidden = !synthetic;
    const seq = R.map((P) => label(P.config));
    let intro = "This run has one round, " + seq[0] + "; the result explains why GPU + ANE did not run.";
    if (seq.length > 1 && comparable()) {
      intro = "The same timetable is replayed twice, " + seq.join(", then ") + ", and the result compares them.";
    } else if (seq.length > 1) {
      intro =
        "The same timetable is replayed twice, " + seq.join(", then ") +
        ". GPU + ANE could not be compared on this Mac; the result says why.";
    }
    setText($("intro-rounds"), intro);
    buildTabs();
    buildSpeeds();
    themeChanged();
    wire();
    selectRound(0, false);
    setSpeed(DEFAULT_SPEED);

    // Deep links for previews and tests: #round=2&t=12.5&speed=1&timeline&details&result&nointro&paused
    const h = new URLSearchParams(location.hash.slice(1));
    if (h.has("round")) selectRound(Number(h.get("round")) - 1, false);
    if (h.has("t")) seek(Number(h.get("t")) * 1000);
    if (h.has("speed")) {
      const i = SPEEDS.indexOf(Number(h.get("speed")));
      if (i >= 0) setSpeed(i);
    }
    if (h.has("timeline")) toggleTimeline();
    if (h.has("details")) toggleDetails();
    let seen = false;
    try {
      seen = localStorage.getItem("switchyard.intro.v2") === "seen";
    } catch (e) {
      seen = false;
    }
    if (h.has("result")) openResult();
    else if (h.has("intro") || (!seen && !h.has("nointro"))) openIntro();
    else if (!h.has("paused")) setPlaying(true);
    requestAnimationFrame(frame);
  }

  boot();
})();
