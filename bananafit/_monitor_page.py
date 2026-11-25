"""The page served by ``python -m bananafit.monitor`` (one self-contained HTML file, no external
resources, so it also works on a cluster node without internet access)."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Run monitor</title>
<style>
:root {
  color-scheme: light dark;
  --page:    light-dark(#f9f9f7, #0d0d0d);
  --surface: light-dark(#fcfcfb, #1a1a19);
  --ink:     light-dark(#0b0b0b, #f0efec);
  --ink-2:   light-dark(#52514e, #c3c2b7);
  --muted:   #898781;
  --grid:    light-dark(#e1e0d9, #2c2c2a);
  --axis:    light-dark(#c3c2b7, #383835);
  --ring:    light-dark(rgba(11,11,11,0.10), rgba(255,255,255,0.10));
  --s1: light-dark(#2a78d6, #3987e5); --s2: light-dark(#eb6834, #d95926);
  --s3: light-dark(#1baf7a, #199e70); --s4: light-dark(#eda100, #c98500);
  --s5: light-dark(#e87ba4, #d55181); --s6: light-dark(#008300, #0ca30c);
  --s7: light-dark(#6250d6, #9085e9); --s8: light-dark(#e34948, #e66767);
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
}
:root[data-theme="light"] { color-scheme: light; }
:root[data-theme="dark"] { color-scheme: dark; }
@supports not (color: light-dark(#000, #fff)) {
  :root { --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --grid: #e1e0d9;
    --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
    --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948; }
  @media (prefers-color-scheme: dark) {
    :root { --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --grid: #2c2c2a;
      --axis: #383835; --ring: rgba(255,255,255,0.10); --s1: #3987e5; --s2: #d95926; --s3: #199e70;
      --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767; }
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 0.875rem/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 80rem; margin: 0 auto; padding: 1.25rem 16px 3rem; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.5rem 1rem; }
h1 { font-size: 1.25rem; font-weight: 600; margin: 0; }
h2 { font-size: 0.95rem; font-weight: 600; margin: 1.75rem 0 0.6rem; }
.sub { color: var(--ink-2); }
.pill { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; color: var(--ink-2); }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(9.5rem, 100%), 1fr)); gap: 10px; margin-top: 1rem; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 0.6rem 0.75rem; }
.tile .k { color: var(--ink-2); font-size: 0.75rem; }
.tile .v { font-size: 1.25rem; font-weight: 600; margin-top: 2px; }
.tile .v small { font-size: 0.75rem; font-weight: 400; color: var(--ink-2); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(19rem, 100%), 1fr)); gap: 10px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 0.6rem 0.75rem 0.4rem; min-width: 0; }
.card .t { font-size: 0.8rem; font-weight: 600; display: flex; justify-content: space-between; gap: 0.5rem; }
.card .t span { font-weight: 400; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.plot { min-height: 150px; }
svg { display: block; width: 100%; overflow: visible; }
svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.note { color: var(--ink-2); font-size: 0.8rem; margin: -0.25rem 0 0.6rem; }
.legend { display: flex; flex-wrap: wrap; gap: 0.25rem 0.9rem; font-size: 0.75rem; color: var(--ink-2); margin: -0.1rem 0 0.6rem; }
.legend i { display: inline-block; width: 14px; height: 2px; vertical-align: middle; margin-right: 5px; border-radius: 1px; }
.tablewrap { overflow-x: auto; border: 1px solid var(--ring); border-radius: 8px; }
.tablewrap:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
table { width: 100%; border-collapse: collapse; background: var(--surface); font-variant-numeric: tabular-nums; }
caption { text-align: left; color: var(--ink-2); font-size: 0.8rem; padding: 0.5rem 0.6rem; caption-side: top; }
th, td { text-align: right; padding: 0.3rem 0.6rem; border-bottom: 1px solid var(--grid); }
th { font-weight: 600; font-size: 0.75rem; }
th button { font: inherit; color: var(--ink-2); background: none; border: 0; padding: 0; cursor: pointer; }
th button:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; border-radius: 2px; }
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
td.flag { color: var(--critical); font-weight: 600; }
tr.extra { display: none; }
table.all tr.extra { display: table-row; }
.more { font: inherit; font-size: 0.8rem; color: var(--ink); background: var(--surface); border: 1px solid var(--ring);
  border-radius: 6px; padding: 0.25rem 0.6rem; margin-top: 0.5rem; cursor: pointer; }
.tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
  border-radius: 6px; padding: 0.3rem 0.5rem; font-size: 0.75rem; box-shadow: 0 2px 8px rgba(0,0,0,0.12); white-space: nowrap; z-index: 10; }
.tip .k { font-weight: 600; } .tip .m { color: var(--ink-2); }
.empty { color: var(--ink-2); padding: 2.5rem 0; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title">Run monitor</h1>
    <span class="sub" id="what"></span>
    <span class="pill" id="status" role="status"></span>
  </header>
  <p class="empty" id="empty">Waiting for the run to start…</p>
  <div id="body" hidden>
    <div class="tiles" id="tiles"></div>
    <section id="sec-derived" hidden><h2>Derived quantities</h2>
      <p class="note">Running mean over all chains; the band is ±2 Monte Carlo errors.</p><div class="grid" id="derived"></div></section>
    <section><h2>Convergence</h2>
      <p class="note">Over all draws so far, for every parameter. Common targets: R-hat below 1.01, ESS above 400.</p>
      <div class="grid" id="conv"></div></section>
    <section id="sec-sampler" hidden><h2>Sampler</h2><div class="grid" id="sampler"></div></section>
    <section id="sec-traces" hidden><h2>Traces</h2><div class="legend" id="legend"></div>
      <p class="note" id="trace-note"></p><div class="grid" id="traces"></div></section>
    <section><h2 id="ptitle">Parameters</h2>
      <div class="tablewrap" role="region" aria-labelledby="ptitle" tabindex="0">
        <table id="ptable">
          <caption>Sorted by ESS, lowest first. Red: R-hat of 1.01 or more, or ESS below 400.</caption>
          <thead><tr>
            <th scope="col" data-k="name"><button type="button" commandfor="ptable" command="--sort" data-k="name">parameter</button></th>
            <th scope="col" data-k="mean"><button type="button" commandfor="ptable" command="--sort" data-k="mean">mean</button></th>
            <th scope="col" data-k="sd"><button type="button" commandfor="ptable" command="--sort" data-k="sd">sd</button></th>
            <th scope="col" data-k="rhat"><button type="button" commandfor="ptable" command="--sort" data-k="rhat">R-hat</button></th>
            <th scope="col" data-k="ess" aria-sort="ascending"><button type="button" commandfor="ptable" command="--sort" data-k="ess">ESS</button></th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <button type="button" class="more" id="more" hidden></button>
      <div class="sr" role="status" id="sort-status"></div>
    </section>
  </div>
</main>
<div class="tip" id="tip" aria-hidden="true" hidden></div>
<script>
"use strict";
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
const tip = $("tip");
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
let META = null, SUM = null, TRACE = null, lastStamp = null, built = "";
const sort = { key: "ess", asc: true };
const charts = new Map();   // key -> {plot element, cfg}
const rows = new Map();     // parameter name -> <tr>

function fmt(x, p = 4) {
  if (x === null || x === undefined || !isFinite(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(2);
  return String(+x.toPrecision(p));
}
const fmtInt = x => (x === null || x === undefined) ? "–" : Math.round(x).toLocaleString("en-US");
function fmtDur(s) {
  if (s === null || s === undefined || !isFinite(s)) return "–";
  s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h} h ${String(m).padStart(2, "0")} min` : (m ? `${m} min ${String(r).padStart(2, "0")} s` : `${r} s`);
}
function svgEl(tag, attrs, parent, style) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (style) Object.assign(e.style, style);
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) { const d = Math.abs(lo) || 1; lo -= d * 0.05; hi += d * 0.05; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw) || 10 * mag;
  const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) t.push(+v.toPrecision(12));
  return { ticks: t, step, lo: Math.min(lo, t[0] ?? lo), hi: Math.max(hi, t[t.length - 1] ?? hi) };
}
function tickFmt(step, maxAbs) {
  if (maxAbs !== 0 && (maxAbs < 1e-3 || maxAbs >= 1e6)) {
    const d = Math.max(0, Math.ceil(Math.log10(maxAbs / step)));
    return v => v === 0 ? "0" : v.toExponential(Math.min(d, 6));
  }
  let dec = 0;   // enough decimals to write the step itself exactly (0.25 needs 2)
  while (dec < 8 && Math.abs(Math.round(step * 10 ** dec) - step * 10 ** dec) > 1e-6) dec++;
  return v => v.toFixed(dec);
}
const chainColor = (i, n) => n <= 8 ? `var(--s${i + 1})` : "var(--s1)";

// Line chart with crosshair and tooltip. cfg = {series: [{label, x, y, color, width, opacity}], band, ref, aria}
function drawChart(plot, cfg) {
  plot.replaceChildren();
  const W = Math.max(plot.clientWidth, 200), H = 150, m = { l: 48, r: 8, t: 6, b: 22 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": cfg.aria || "" }, plot);
  const xs = [], ys = [];
  for (const s of cfg.series) s.x.forEach((xv, i) => { const v = s.y[i]; if (v !== null && v !== undefined) { xs.push(xv); ys.push(v); } });
  if (cfg.band) cfg.band.x.forEach((_, i) => { if (cfg.band.lo[i] !== null) ys.push(cfg.band.lo[i], cfg.band.hi[i]); });
  if (cfg.ref) ys.push(cfg.ref.y);
  if (!xs.length) { svgEl("text", { x: m.l, y: H / 2 }, svg).textContent = "no data yet"; return; }
  const xlo = Math.min(...xs), xhi = Math.max(...xs), yt = niceTicks(Math.min(...ys), Math.max(...ys));
  const X = v => m.l + (xhi > xlo ? (v - xlo) / (xhi - xlo) : 0.5) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - yt.lo) / (yt.hi - yt.lo) * (H - m.t - m.b);
  const yf = tickFmt(yt.step, Math.max(Math.abs(yt.lo), Math.abs(yt.hi)));
  for (const t of yt.ticks) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), "stroke-width": 1 }, svg, { stroke: "var(--grid)" });
    svgEl("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = yf(t);
  }
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, "stroke-width": 1 }, svg, { stroke: "var(--axis)" });
  for (const t of niceTicks(xlo, xhi, 4).ticks.filter(t => t >= xlo && t <= xhi))
    svgEl("text", { x: X(t), y: H - 6, "text-anchor": "middle" }, svg).textContent = fmtInt(t);
  if (cfg.band) {
    const b = cfg.band, idx = b.x.map((_, i) => i).filter(i => b.lo[i] !== null && b.hi[i] !== null);
    if (idx.length > 1) {
      const d = idx.map((i, k) => `${k ? "L" : "M"}${X(b.x[i])},${Y(b.hi[i])}`).join("") +
                idx.slice().reverse().map(i => `L${X(b.x[i])},${Y(b.lo[i])}`).join("") + "Z";
      svgEl("path", { d }, svg, { fill: b.color, fillOpacity: 0.12, stroke: "none" });
    }
  }
  if (cfg.ref) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(cfg.ref.y), y2: Y(cfg.ref.y), "stroke-width": 1 }, svg, { stroke: "var(--ink-2)" });
    const ry = Y(cfg.ref.y), below = ry - 4 < m.t + 12;
    svgEl("text", { x: W - m.r, y: below ? ry + 13 : ry - 4, "text-anchor": "end" }, svg).textContent = cfg.ref.label;
  }
  const paths = cfg.series.map(s => {
    let d = "", pen = false;
    let prev = null;
    s.x.forEach((xv, i) => { const v = s.y[i]; if (v === null || v === undefined) { pen = false; return; }
      if (cfg.circular && prev !== null && Math.abs(v - prev) > Math.PI) pen = false;   // wrapped angle: no seam lines
      d += `${pen ? "L" : "M"}${X(xv).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; prev = v; });
    return svgEl("path", { d, fill: "none", "stroke-linejoin": "round", "stroke-linecap": "round" }, svg,
                 { stroke: s.color, strokeWidth: s.width || 2, strokeOpacity: s.opacity ?? 1 });
  });
  if (cfg.series.length === 1) {
    const s = cfg.series[0], i = s.x.length - 1;
    if (i >= 0 && s.y[i] !== null) svgEl("circle", { cx: X(s.x[i]), cy: Y(s.y[i]), r: 4, "stroke-width": 2 }, svg, { fill: s.color, stroke: "var(--surface)" });
  }
  const cross = svgEl("line", { y1: m.t, y2: H - m.b, "stroke-width": 1, visibility: "hidden" }, svg, { stroke: "var(--axis)" });
  const hit = svgEl("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  const ref0 = cfg.series[0].x;
  const restore = () => paths.forEach((p, k) => { const s = cfg.series[k]; p.style.strokeOpacity = s.opacity ?? 1; p.style.strokeWidth = s.width || 2; });
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width, py = (ev.clientY - r.top) * H / r.height;
    const xv = xlo + (px - m.l) / (W - m.l - m.r) * (xhi - xlo);
    let lo = 0, hi = ref0.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ref0[mid] < xv) lo = mid; else hi = mid; }
    const i = Math.abs(ref0[lo] - xv) <= Math.abs(ref0[hi] - xv) ? lo : hi;
    let best = -1, bd = Infinity;
    cfg.series.forEach((s, k) => { const v = s.y[i]; if (v === null || v === undefined) return; const dd = Math.abs(Y(v) - py); if (dd < bd) { bd = dd; best = k; } });
    cross.setAttribute("x1", X(ref0[i])); cross.setAttribute("x2", X(ref0[i])); cross.setAttribute("visibility", "visible");
    if (cfg.series.length > 1) {
      restore();
      paths.forEach((p, k) => { if (k !== best) p.style.strokeOpacity = Math.min(cfg.series[k].opacity ?? 1, 0.2); });
      if (best >= 0) { paths[best].style.strokeOpacity = 1; paths[best].style.strokeWidth = 2.5; svg.insertBefore(paths[best], cross); }
    }
    const s = cfg.series[best];
    const lines = [[`step ${fmtInt(ref0[i])}`, "k"]];
    if (s) lines.push([`${s.label ? s.label + ": " : ""}${fmt(s.y[i])}`, ""]);
    if (cfg.band && cfg.band.lo[i] !== null && cfg.band.lo[i] !== undefined)
      lines.push([`±2 MC errors: ${fmt(cfg.band.lo[i])} – ${fmt(cfg.band.hi[i])}`, "m"]);
    tip.replaceChildren(...lines.map(([t, c]) => { const d = document.createElement("div"); d.textContent = t; if (c) d.className = c; return d; }));
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8) + "px";
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; restore(); });
}

function chart(container, key, title) {
  if (!charts.has(key)) {
    const c = document.createElement("div"); c.className = "card";
    const t = document.createElement("div"); t.className = "t";
    const name = document.createElement("div"); name.textContent = title;
    const val = document.createElement("span");
    t.append(name, val);
    const plot = document.createElement("div"); plot.className = "plot";
    c.append(t, plot); container.appendChild(c);
    charts.set(key, { plot, val, cfg: null });
  }
  return charts.get(key);
}
function show(key, value, cfg) { const c = charts.get(key); c.val.textContent = value; c.cfg = cfg; drawChart(c.plot, cfg); }

const SAMPLER = [["acceptance", "Acceptance per chunk"], ["leapfrog_slowest", "Leapfrog steps, slowest chain"],
                 ["hop_acceptance", "Hop acceptance per chunk"], ["divergent", "Divergent transitions per chunk"]];

// Build the page structure once per set of panels; later polls only update it.
function build() {
  const H = SUM.history;
  const sampler = SAMPLER.filter(([k]) => H.some(r => r[k] !== undefined && r[k] !== null)).map(([k]) => k);
  const traceNames = TRACE ? Object.keys(TRACE.series) : [];
  const sig = JSON.stringify([SUM.derived.map(d => d.name), sampler, traceNames, META.n_chains]);
  if (sig === built) return;
  built = sig;
  for (const id of ["derived", "conv", "sampler", "traces"]) $(id).replaceChildren();
  charts.clear();
  $("sec-derived").hidden = !SUM.derived.length;
  for (const d of SUM.derived) chart($("derived"), "d:" + d.name, d.name);
  chart($("conv"), "rhat", "Largest R-hat"); chart($("conv"), "ess", "Smallest ESS");
  $("sec-sampler").hidden = !sampler.length;
  for (const k of sampler) chart($("sampler"), "s:" + k, SAMPLER.find(r => r[0] === k)[1]);
  $("sec-traces").hidden = !TRACE;
  const nc = META.n_chains;
  $("legend").replaceChildren(...(nc <= 8 ? Array.from({ length: nc }, (_, i) => {
    const s = document.createElement("span"), sw = document.createElement("i");
    sw.style.background = chainColor(i, nc); s.append(sw, `chain ${i}`); return s; }) : []));
  for (const n of traceNames.concat(["log density"])) chart($("traces"), "t:" + n, n);
}

function update() {
  if (!META) return;
  $("title").textContent = META.name || "Run monitor";
  document.title = (META.name || "Run") + " · monitor";
  $("what").textContent = `${META.algorithm.toUpperCase()} · ${META.n_chains} chains · ${fmtInt(META.n_samples)} draws per chain` +
    (META.thinning > 1 ? ` (1 in ${META.thinning} steps kept)` : "") + (META.physical ? "" : " · values in sampler coordinates");
  updateStatus();
  if (!SUM) { $("empty").textContent = "Warmup finished; waiting for the first chunk…"; return; }
  $("empty").hidden = true; $("body").hidden = false;
  build();
  const H = SUM.history, last = H[H.length - 1] || {}, steps = H.map(r => r.steps);
  const pct = 100 * SUM.draws_done / Math.max(SUM.draws_total, 1);
  const tiles = [["Progress", `${pct.toFixed(0)} %`, `${fmtInt(SUM.draws_done)} / ${fmtInt(SUM.draws_total)}`],
                 ["Elapsed", fmtDur(SUM.elapsed_s)], ["Time left", META.status === "running" ? fmtDur(SUM.eta_s) : "–"]];
  if (last.acceptance !== undefined) tiles.push(["Acceptance, last chunk", fmt(last.acceptance, 3)]);
  if (last.divergent !== undefined) tiles.push(["Divergent, total", fmtInt(SUM.divergent_total)]);
  if (META.tuned && META.tuned.step_size !== undefined) tiles.push(["Step size", fmt([].concat(META.tuned.step_size)[0], 3)]);
  tiles.push(["Largest R-hat", fmt(last.max_rhat, 4)], ["Smallest ESS", fmt(last.min_ess, 3)]);
  $("tiles").replaceChildren(...tiles.map(([k, v, small]) => {
    const t = document.createElement("div"); t.className = "tile";
    const kk = document.createElement("div"); kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("div"); vv.className = "v"; vv.textContent = v;
    if (small) { const s = document.createElement("small"); s.textContent = " " + small; vv.appendChild(s); }
    t.append(kk, vv); return t; }));

  for (const d of SUM.derived) {
    const mean = H.map(r => (r.derived[d.name] || [null])[0]), mc = H.map(r => (r.derived[d.name] || [null, null])[1]);
    show("d:" + d.name, `${fmt(d.mean)} ± ${fmt(d.mcse, 2)}`, { aria: `${d.name} against step`,
      series: [{ label: d.name, x: steps, y: mean, color: "var(--s1)" }],
      band: { x: steps, color: "var(--s1)", lo: mean.map((v, i) => mc[i] === null || v === null ? null : v - 2 * mc[i]),
              hi: mean.map((v, i) => mc[i] === null || v === null ? null : v + 2 * mc[i]) } });
  }
  show("rhat", fmt(last.max_rhat, 4), { aria: "largest R-hat against step", ref: { y: 1.01, label: "1.01" },
       series: [{ label: "largest R-hat", x: steps, y: H.map(r => r.max_rhat), color: "var(--s1)" }] });
  show("ess", fmt(last.min_ess, 3), { aria: "smallest ESS against step", ref: { y: 400, label: "400" },
       series: [{ label: "smallest ESS", x: steps, y: H.map(r => r.min_ess), color: "var(--s1)" }] });
  for (const [k, label] of SAMPLER) if (charts.has("s:" + k))
    show("s:" + k, fmt(last[k], 3), { aria: `${label} against step`, series: [{ label, x: steps, y: H.map(r => r[k] ?? null), color: "var(--s1)" }] });
  if (TRACE) {
    const nc = META.n_chains;
    $("trace-note").textContent = (nc > 8 ? `${nc} chains in one colour; hover a line to pick out one chain. ` : "") +
      (TRACE.x.length < SUM.draws_done ? `Thinned for display: ${fmtInt(TRACE.x.length)} of ${fmtInt(SUM.draws_done)} draws per chain.` : "");
    for (const n of Object.keys(TRACE.series).concat(["log density"])) {
      const Y = n === "log density" ? TRACE.log_density : TRACE.series[n], p = SUM.params.find(q => q.name === n);
      show("t:" + n, p ? `${fmt(p.mean)} ± ${fmt(p.sd, 3)}` : "", { aria: `trace of ${n}`,
        circular: (TRACE.circular || []).includes(n),
        series: Y.map((y, i) => ({ label: `chain ${i}`, x: TRACE.x, y, color: chainColor(i, nc),
                                   width: nc <= 8 ? 1.5 : 1.25, opacity: nc <= 8 ? 0.9 : 0.45 })) });
    }
  }
  updateTable();
}

function cell(text, flag) { const td = document.createElement("td"); td.textContent = text; if (flag) td.className = "flag"; return td; }
function updateTable() {
  const tbody = $("ptable").tBodies[0];
  for (const p of SUM.params) {
    const tr = document.createElement("tr");
    tr.dataset.name = p.name;
    tr.append(cell(p.name), cell(fmt(p.mean, 5)), cell(fmt(p.sd, 3)),
              cell(fmt(p.rhat, 4), p.rhat !== null && p.rhat >= 1.01), cell(fmt(p.ess, 4), p.ess !== null && p.ess < 400));
    const old = rows.get(p.name);
    if (old) old.replaceWith(tr); else tbody.appendChild(tr);
    rows.set(p.name, tr);
  }
  applySort(false);
}
function applySort(announce) {
  const tbody = $("ptable").tBodies[0], byName = new Map(SUM.params.map(p => [p.name, p]));
  const k = sort.key, dir = sort.asc ? 1 : -1;
  const ordered = [...rows.values()].sort((a, b) => {
    const pa = byName.get(a.dataset.name), pb = byName.get(b.dataset.name);
    if (k === "name") return dir * collator.compare(pa.name, pb.name);
    const va = pa[k] ?? Infinity, vb = pb[k] ?? Infinity; return dir * (va - vb); });
  ordered.forEach((tr, i) => {
    tr.classList.toggle("extra", i >= 20);
    if (tbody.moveBefore) tbody.moveBefore(tr, null); else tbody.appendChild(tr);
  });
  for (const th of $("ptable").tHead.rows[0].cells) {
    if (th.dataset.k === k) th.setAttribute("aria-sort", sort.asc ? "ascending" : "descending"); else th.removeAttribute("aria-sort");
  }
  const labels = { name: "name", mean: "mean", sd: "sd", rhat: "R-hat", ess: "ESS" };
  $("ptable").caption.textContent = `Sorted by ${labels[k]}, ${sort.asc ? "lowest" : "highest"} first. Red: R-hat of 1.01 or more, or ESS below 400.`;
  if (announce) $("sort-status").textContent = `Sorted by ${labels[k]}, ${sort.asc ? "ascending" : "descending"}.`;
  const more = $("more"), all = $("ptable").classList.contains("all");
  more.hidden = rows.size <= 20;
  more.textContent = all ? "Show the first 20" : `Show all ${rows.size}`;
}
function onSort(key) {
  if (sort.key === key) sort.asc = !sort.asc; else { sort.key = key; sort.asc = key !== "rhat"; }
  applySort(true);
}
const table = $("ptable");
if ("commandForElement" in HTMLButtonElement.prototype) {
  table.addEventListener("command", e => { if (e.command === "--sort") onSort(e.source.dataset.k); });
} else {
  table.tHead.addEventListener("click", e => { const b = e.target.closest("button"); if (b) onSort(b.dataset.k); });
}
$("more").addEventListener("click", () => { table.classList.toggle("all"); applySort(false); });

function updateStatus() {
  const s = $("status");
  let color = "var(--s1)", label = "running";
  if (META.status === "finished") { color = "var(--good)"; label = "finished"; }
  else if (META.status === "stopped") { color = "var(--critical)"; label = "stopped before the end"; }
  else if (SUM && SUM.history.length) {
    const H = SUM.history, per = H.length > 1 ? (H[H.length - 1].elapsed_s - H[0].elapsed_s) / (H.length - 1) : H[0].elapsed_s;
    const age = Date.now() / 1000 - SUM.updated_unix;
    if (age > Math.max(3 * per, 120)) { color = "var(--warning)"; label = `⚠ no update for ${fmtDur(age)}`; }
    else label = `running · updated ${fmtDur(age)} ago`;
  }
  const dot = document.createElement("span"); dot.className = "dot"; dot.style.background = color;
  s.replaceChildren(dot, label);
}

async function get(name) {
  if (window.SNAPSHOT) return window.SNAPSHOT[name.split(".")[0]] ?? null;   // static snapshot (--html)
  const r = await fetch(`${name}?t=${Date.now()}`, { cache: "no-store" });
  return r.ok ? r.json() : null;
}
async function poll() {
  if (document.visibilityState === "hidden") return;
  try {
    META = await get("meta.json");
    const s = await get("summary.json"), stamp = s ? s.updated_unix : null;
    if (stamp !== lastStamp) { SUM = s; TRACE = s ? await get("trace.json") : null; lastStamp = stamp; update(); }
    else if (META) updateStatus();
  } catch (e) { $("status").textContent = "cannot reach the monitor server"; }
}
let rz;
window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => charts.forEach(c => c.cfg && drawChart(c.plot, c.cfg)), 150); });
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") poll(); });
poll(); if (!window.SNAPSHOT) setInterval(poll, 4000);
</script>
</body>
</html>
"""
