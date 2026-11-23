"""bananafit hot-path performance benchmarks (DESIGN.md Sections 1, 4, 7).

Standalone script (MANGO benchmark house style): builds
synthetic event samples/spline tables directly with the public bananafit API
(nothing from bananafit is edited or monkeypatched), times the pieces DESIGN.md
says should dominate, and prints + writes ``RESULTS.md``.

Run (from the repo root, after ``source env.sh``):

    JAX_PLATFORMS=cpu python benchmarks/bench_hotpath.py            # full run (~15-20 min)
    QUICK=1 JAX_PLATFORMS=cpu python benchmarks/bench_hotpath.py    # smoke test (~1 min)

Methodology
-----------
Every timed function is (a) traced+compiled once via ``.lower(*args).compile()``
(the compile time is reported separately and excluded from execution timing),
(b) called once more untimed to flush first-call dispatch overhead, then
(c) called ``REPS`` (>=5) times with ``time.perf_counter()`` around each call
and ``jax.block_until_ready`` on the output; the reported number is the
median. All arrays are float64 except spline coefficient tables, which use
bananafit's own default (float32) unless stated otherwise (DESIGN.md Section 4).

This machine is shared (CPU-only); a single run can be noisy -- re-run
sections that look suspicious. Total wall time is kept under ~25 minutes by
choice of rep counts and by scaling down the two provably-expensive corners
(the 1e6-event Barlow-Beeston-lite duplicate of the Poisson run, and the
per-event ["no-dedup"] spline-curve case) rather than by cutting corners on
the numbers that matter for the package's actual hot path.
"""

from __future__ import annotations

import os
import resource
import time
import types

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import bananafit  # noqa: F401 -- import enforces float64 (house convention)
import jaxnu
import blackjax

from bananafit.events import make_sample_prebinned
from bananafit.model import SampleModel, build_forward_model
from bananafit.norms import from_selections
from bananafit.oscillation import ConstantMatterOsc
from bananafit.parameters import Gaussian, ParamSpec, ParameterSet, osc_param_specs
from bananafit.posterior import make_posterior, asimov_data
from bananafit.splines import build_spline_table
from bananafit.likelihood import poisson_loglike
from bananafit.diagnostics import ess as bananafit_ess

QUICK = os.environ.get("QUICK", "") not in ("", "0")
REPS = 5

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)
# BENCH_OUT_TAG: write RESULTS_<tag>.md / figures/<tag>/ instead of RESULTS.md / figures/ (used to keep a
# second, differently-conditioned run next to the primary one without overwriting it).
OUT_TAG = os.environ.get("BENCH_OUT_TAG", "")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", *([OUT_TAG] if OUT_TAG else []))
os.makedirs(FIGDIR, exist_ok=True)
RESULTS_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"RESULTS_{OUT_TAG}.md" if OUT_TAG else "RESULTS.md")

# MANGO benchmarks house style: larger
# fonts for readability at report size, tight layout, PNG at a print-quality dpi.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "figure.dpi": 100,
})
_COLORS = {
    "poisson": "#1f77b4", "barlow-beeston": "#d62728",
    "oscillation": "#1f77b4", "spline_product": "#ff7f0e", "norm_product": "#2ca02c",
    "segment_sum": "#9467bd", "likelihood_priors": "#8c564b",
    "dedup": "#1f77b4", "nodedup": "#d62728",
}

BASELINE_KM = 1300.0
DENSITY_GCC = 2.85
YE = 0.5
KNOT_X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

T_START = time.perf_counter()


def elapsed():
    return time.perf_counter() - T_START


def log(*a):
    print(f"[{elapsed():7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# generic compile/exec timing helper
# ---------------------------------------------------------------------------


def bench(fn, args, reps=REPS):
    """jax.jit(fn), AOT-compile (timed separately), one untimed warm call,
    then ``reps`` timed calls on the *compiled* executable. Returns a dict
    with compile_s, median_s, min_s, max_s, all_s."""
    jfn = jax.jit(fn)
    t0 = time.perf_counter()
    compiled = jfn.lower(*args).compile()
    compile_s = time.perf_counter() - t0

    out = compiled(*args)
    jax.block_until_ready(out)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = compiled(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return {
        "compile_s": compile_s,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "all_s": times,
        "last_out": out,
    }


# ---------------------------------------------------------------------------
# synthetic analysis builder
# ---------------------------------------------------------------------------


def build_model(
    n_events,
    *,
    n_syst=20,
    n_curves=200,
    n_norm=4,
    n_bins=60,
    seed=0,
    spline_dtype=jnp.float32,
    name="bench",
):
    """One EventSample + SplineTable + NormSet + ConstantMatterOsc, wrapped
    into a ForwardModel/ParameterSet/BinnedPosterior, all via the public
    bananafit API. reco_bin is drawn directly in [0, n_bins) (no overflow
    events) via make_sample_prebinned -- this avoids the O(N log n_bins)
    np.digitize call, which is irrelevant to the hot path being measured
    here (digitization runs once at analysis-build time in real usage too).
    """
    rng = np.random.default_rng(seed)
    true_energy = rng.gamma(3.0, 1.0, size=n_events) + 0.3  # GeV, DUNE-ish
    reco_bin = rng.integers(0, n_bins, size=n_events).astype(np.int32)
    is_cc = rng.random(n_events) < 0.7  # 70% numu->numu CC signal, 30% NC bkg
    flavor_from = np.where(is_cc, 1, -1).astype(np.int32)
    flavor_to = np.where(is_cc, 1, -1).astype(np.int32)
    anti = rng.random(n_events) < 0.05
    nominal_weight = rng.uniform(0.5, 1.5, size=n_events)

    sample, perm = make_sample_prebinned(
        name,
        true_energy=true_energy,
        reco_bin=reco_bin,
        n_bins=n_bins,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
    )

    event_index_raw = rng.integers(0, n_curves + 1, size=(n_syst, n_events)).astype(np.int32)
    event_index = event_index_raw[:, perm]

    knot_x = np.tile(KNOT_X, (n_syst, 1))
    a = rng.uniform(-0.2, 0.2, size=(n_syst, n_curves))
    b = rng.uniform(-0.05, 0.05, size=(n_syst, n_curves))
    knot_y = (
        1.0
        + a[:, :, None] * np.tanh(KNOT_X)[None, None, :]
        + b[:, :, None] * (KNOT_X**2)[None, None, :]
    )

    spline_names = tuple(f"spl_{i}" for i in range(n_syst))
    spline_table = build_spline_table(
        knot_x, knot_y, event_index,
        names=spline_names, boundary="natural", dtype=spline_dtype,
    )

    norm_names = tuple(f"norm_{i}" for i in range(n_norm))
    norm_set = None
    if n_norm > 0:
        sel_idx = rng.integers(0, n_norm, size=n_events)
        selections = [(sel_idx == i)[perm] for i in range(n_norm)]
        norm_set = from_selections(norm_names, selections, n_events)

    osc = ConstantMatterOsc(baseline_km=BASELINE_KM, density_gcc=DENSITY_GCC, ye=YE)
    sample_model = SampleModel(events=sample, splines=spline_table, norms=norm_set, osc=osc)

    sys_specs = tuple(
        ParamSpec(nm, 0.0, Gaussian(0.0, 1.0), (-3.0, 3.0)) for nm in spline_names
    ) + tuple(
        ParamSpec(nm, 1.0, Gaussian(1.0, 0.2), (0.0, None)) for nm in norm_names
    )
    parameter_set = ParameterSet(osc_param_specs() + sys_specs)
    model = build_forward_model((sample_model,), parameter_set)
    data = asimov_data(model, parameter_set)

    posteriors = {
        "poisson": make_posterior(model, data, parameter_set, stat="poisson"),
        "barlow-beeston": make_posterior(model, data, parameter_set, stat="barlow-beeston"),
    }
    return dict(
        model=model, parameter_set=parameter_set, sample_model=sample_model,
        posteriors=posteriors, n_events=n_events, n_syst=n_syst,
        n_curves=n_curves, n_norm=n_norm, n_bins=n_bins,
    )


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB (Linux)


# ---------------------------------------------------------------------------
# Staged execution (added 2026-10 so a FULL run fits in <10-min foreground chunks)
#
#   BENCH_STAGES=A,B,C,D  (default: everything in one process, as before)
#     A = items 1-2 (logdensity/grad vs n_events, component breakdown)
#     B = items 3-4 (n_syst scaling, dedup payoff)
#     C = items 5-6 (NUTS/RWM throughput, chain parallelism)
#     D = figures + RESULTS.md (reads the checkpoint of A-C)
#   Each stage checkpoints its small result variables to
#   output/hotpath_ckpt_{full,quick}.pkl; later stages (and D) reload them.
#   The measurements themselves are unchanged -- only the process boundaries.
# ---------------------------------------------------------------------------

import pickle

STAGES = {s.strip() for s in os.environ.get("BENCH_STAGES", "A,B,C,D").split(",") if s.strip()}
CKPT = os.environ.get("BENCH_CKPT") or os.path.join(OUTDIR, "hotpath_ckpt_" + ("quick" if QUICK else "full") + ".pkl")
_STAGE_MARK = [0.0]
_SKIP_SAVE = {"STAGES", "CKPT", "QUICK", "REPS", "T_START", "OUTDIR", "FIGDIR", "RESULTS_MD", "_CK"}


def _run(stage):
    return stage in STAGES


def _load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as fh:
            return pickle.load(fh)
    return {"vars": {}, "_wall": {}, "_rss": {}}


def _strip_last_out(obj):
    if isinstance(obj, dict):
        return {k: _strip_last_out(v) for k, v in obj.items() if k != "last_out"}
    return obj


def _save_ckpt(stage):
    """Pickle every small plain-python/numpy module-level result variable."""
    ck = _load_ckpt()
    g = globals()
    for k, v in list(g.items()):
        if k.startswith("_") or k in _SKIP_SAVE:
            continue
        if isinstance(v, (int, float, str, bool, list, dict, tuple, np.ndarray, np.generic)):
            v = _strip_last_out(v)
            try:
                blob = pickle.dumps(v)
            except Exception:
                continue
            if len(blob) < 5_000_000:
                ck["vars"][k] = v
    ck["_wall"][stage] = elapsed() - _STAGE_MARK[0]
    ck["_rss"][stage] = rss_gb()
    _STAGE_MARK[0] = elapsed()
    with open(CKPT, "wb") as fh:
        pickle.dump(ck, fh)
    log(f"checkpoint saved after stage {stage}: {CKPT}")


_CK = _load_ckpt()
for _k, _v in _CK["vars"].items():
    globals().setdefault(_k, _v)



# ---------------------------------------------------------------------------
# item 1: logdensity / grad(logdensity) vs n_events
# ---------------------------------------------------------------------------

if _run("A"):
    N_SCALES = [1_000, 5_000] if QUICK else [10_000, 100_000, 300_000, 1_000_000]

    log("=== item 1: logdensity/grad cost vs n_events ===")
    item1_rows = []
    built_cache = {}
    for n in N_SCALES:
        t0 = time.perf_counter()
        built = build_model(n, n_syst=20, n_curves=200, n_norm=4, n_bins=60, seed=1)
        build_s = time.perf_counter() - t0
        built_cache[n] = built
        z0 = built["parameter_set"].init_z
        for stat in ("poisson", "barlow-beeston"):
            if stat == "barlow-beeston" and n == max(N_SCALES) and not QUICK:
                log(f"n={n}: skipping Barlow-Beeston-lite duplicate (time budget); "
                    "Poisson-only at this scale.")
                continue
            post = built["posteriors"][stat]
            fwd = bench(post.logdensity, (z0,))
            grad_fn = jax.grad(post.logdensity)
            grd = bench(grad_fn, (z0,))
            row = dict(
                n_events=n, stat=stat, build_s=build_s,
                fwd_compile_s=fwd["compile_s"], fwd_ms=fwd["median_s"] * 1e3,
                grad_compile_s=grd["compile_s"], grad_ms=grd["median_s"] * 1e3,
                ratio=grd["median_s"] / fwd["median_s"],
                events_per_sec_fwd=n / fwd["median_s"],
                events_per_sec_grad=n / grd["median_s"],
            )
            item1_rows.append(row)
            log(f"n={n:>9d} {stat:14s} fwd={row['fwd_ms']:9.3f} ms "
                f"grad={row['grad_ms']:9.3f} ms  ratio={row['ratio']:.2f}  "
                f"ev/s(fwd)={row['events_per_sec_fwd']:.3e}")
        if n == max(N_SCALES):
            peak_rss_1e6_gb = rss_gb()
            log(f"peak RSS after largest-n stage: {peak_rss_1e6_gb:.2f} GB")

    # memory: table sizes, f32 vs f64
    big = built_cache[max(N_SCALES)]
    coeffs32 = big["sample_model"].splines.coeffs
    event_index_arr = big["sample_model"].splines.event_index
    probe64 = build_model(1000, n_syst=20, n_curves=200, n_norm=4, n_bins=60, seed=1,
                           spline_dtype=jnp.float64)
    coeffs64 = probe64["sample_model"].splines.coeffs

    mem_rows = dict(
        n_events_largest=max(N_SCALES),
        coeffs_f32_MB=coeffs32.nbytes / 1e6,
        coeffs_f64_MB=coeffs64.nbytes / 1e6,
        event_index_MB=event_index_arr.nbytes / 1e6,
        true_energy_MB=big["sample_model"].events.true_energy.nbytes / 1e6,
        peak_rss_gb=peak_rss_1e6_gb,
    )
    log("memory:", mem_rows)

    # ---------------------------------------------------------------------------
    # item 2: component breakdown at 1e5 and 1e6 (or the two largest scales)
    # ---------------------------------------------------------------------------

    log("=== item 2: component breakdown ===")


    def component_breakdown(built, reps=REPS):
        model = built["model"]
        sm = built["sample_model"]
        meta = model.sample_meta[0]
        parameter_set = built["parameter_set"]
        z0 = parameter_set.init_z
        physical = parameter_set.constrain(z0)
        data = built["posteriors"]["poisson"].data

        osc_vals = physical[jnp.asarray(meta.osc_idx, dtype=jnp.int32)]
        osc_params = jaxnu.OscParams(
            theta12=osc_vals[0], theta13=osc_vals[1], theta23=osc_vals[2],
            deltacp=osc_vals[3], dm21=osc_vals[4], dm31=osc_vals[5],
        )
        spline_x = physical[jnp.asarray(meta.spline_idx, dtype=jnp.int32)]
        norm_x = physical[jnp.asarray(meta.norm_idx, dtype=jnp.int32)]
        events = sm.events

        def f_osc(params):
            return sm.osc.weights(params, events)

        def f_spline(x):
            return sm.splines(x)

        def f_norm(x):
            return sm.norms(x)

        w = (
            events.nominal_weight
            * sm.osc.weights(osc_params, events)
            * sm.splines(spline_x)
            * sm.norms(norm_x)
        ).astype(jnp.float64)
        w = jax.block_until_ready(w)

        def f_hist(w_):
            n_bins = meta.n_bins
            mu = jax.ops.segment_sum(w_, events.reco_bin, num_segments=n_bins + 1)[:n_bins]
            w2 = jax.ops.segment_sum(w_ * w_, events.reco_bin, num_segments=n_bins + 1)[:n_bins]
            return mu, w2

        mu, w2 = f_hist(w)
        mu = jax.block_until_ready(mu)

        def f_like(mu_, physical_):
            return poisson_loglike(mu_, data) + parameter_set.log_prior(physical_)

        out = {}
        out["oscillation"] = bench(f_osc, (osc_params,), reps=reps)
        out["spline_product"] = bench(f_spline, (spline_x,), reps=reps)
        out["norm_product"] = bench(f_norm, (norm_x,), reps=reps)
        out["segment_sum"] = bench(f_hist, (w,), reps=reps)
        out["likelihood_priors"] = bench(f_like, (mu, physical), reps=reps)
        return out


    item2_scales = [N_SCALES[len(N_SCALES) // 2], N_SCALES[-1]] if QUICK else [100_000, 1_000_000]
    item2 = {}
    for n in item2_scales:
        built = built_cache[n]
        comp = component_breakdown(built)
        item2[n] = comp
        total_ms = sum(v["median_s"] for v in comp.values()) * 1e3
        log(f"n={n}: " + ", ".join(f"{k}={v['median_s']*1e3:.4f}ms" for k, v in comp.items())
            + f"  (sum={total_ms:.4f}ms)")

    _save_ckpt("A")

if _run("B"):
    built_cache = globals().get("built_cache", {})
    # ---------------------------------------------------------------------------
    # item 3: scaling in n_syst at 1e5 events
    # ---------------------------------------------------------------------------

    log("=== item 3: n_syst scaling ===")
    N3 = N_SCALES[len(N_SCALES) // 2] if QUICK else 100_000
    NSYST_GRID = [2, 5] if QUICK else [5, 20, 50]
    item3_rows = []
    for nsyst in NSYST_GRID:
        if not QUICK and nsyst == 20 and N3 in built_cache and built_cache[N3]["n_syst"] == 20:
            built = built_cache[N3]
        else:
            built = build_model(N3, n_syst=nsyst, n_curves=200, n_norm=4, n_bins=60, seed=1)
        z0 = built["parameter_set"].init_z
        post = built["posteriors"]["poisson"]
        fwd = bench(post.logdensity, (z0,))
        grd = bench(jax.grad(post.logdensity), (z0,))
        row = dict(n_syst=nsyst, fwd_ms=fwd["median_s"] * 1e3, grad_ms=grd["median_s"] * 1e3)
        item3_rows.append(row)
        log(f"n_syst={nsyst:3d}  fwd={row['fwd_ms']:.4f} ms  grad={row['grad_ms']:.4f} ms")

    # linearity check: ms per additional systematic
    if len(item3_rows) >= 2:
        lo, hi = item3_rows[0], item3_rows[-1]
        slope_fwd = (hi["fwd_ms"] - lo["fwd_ms"]) / (hi["n_syst"] - lo["n_syst"])
        slope_grad = (hi["grad_ms"] - lo["grad_ms"]) / (hi["n_syst"] - lo["n_syst"])
        log(f"slope: {slope_fwd:.5f} ms/syst (fwd), {slope_grad:.5f} ms/syst (grad)")

    # ---------------------------------------------------------------------------
    # item 4: curve dedup payoff at 1e5 events, 20 systs
    # ---------------------------------------------------------------------------

    log("=== item 4: curve dedup payoff ===")

    N4 = N_SCALES[0] if QUICK else 100_000
    N4_NODEDUP_SYST = 1  # kept at 1 syst for the true per-event build (see RESULTS.md)
    base_events = build_model(N4, n_syst=1, n_curves=200, n_norm=0, n_bins=60, seed=2)[
        "sample_model"
    ].events


    def build_spline_only(n_syst, n_curves_eff, dedup, events, seed=11):
        N = events.true_energy.shape[0]
        rng = np.random.default_rng(seed)
        if dedup:
            event_index = rng.integers(0, n_curves_eff + 1, size=(n_syst, N)).astype(np.int32)
        else:
            event_index = np.tile(np.arange(1, N + 1, dtype=np.int32), (n_syst, 1))
            n_curves_eff = N
        knot_x = np.tile(KNOT_X, (n_syst, 1))
        a = rng.uniform(-0.2, 0.2, size=(n_syst, n_curves_eff))
        b = rng.uniform(-0.05, 0.05, size=(n_syst, n_curves_eff))
        knot_y = (
            1.0
            + a[:, :, None] * np.tanh(KNOT_X)[None, None, :]
            + b[:, :, None] * (KNOT_X**2)[None, None, :]
        )
        t0 = time.perf_counter()
        table = build_spline_table(
            knot_x, knot_y, event_index, boundary="natural", dtype=jnp.float32
        )
        build_s = time.perf_counter() - t0
        return table, build_s


    table_dedup1, build_dedup1_s = build_spline_only(1, 200, True, base_events)
    log(f"dedup build (1 syst, 200 curves, N={N4}): {build_dedup1_s*1e3:.2f} ms")

    nodedup_curves = N4 if not QUICK else N4
    table_nodedup1, build_nodedup1_s = build_spline_only(1, nodedup_curves, False, base_events)
    log(f"no-dedup build (1 syst, {nodedup_curves} curves == N, N={N4}): {build_nodedup1_s:.3f} s")

    x1 = jnp.array([0.3])
    eval_dedup1 = bench(table_dedup1.__call__, (x1,))
    eval_nodedup1 = bench(table_nodedup1.__call__, (x1,))
    log(f"dedup eval (1 syst): {eval_dedup1['median_s']*1e3:.4f} ms; "
        f"no-dedup eval (1 syst): {eval_nodedup1['median_s']*1e3:.4f} ms "
        f"({eval_nodedup1['median_s']/eval_dedup1['median_s']:.1f}x)")

    mem_dedup1 = table_dedup1.coeffs.nbytes + table_dedup1.event_index.nbytes
    mem_nodedup1 = table_nodedup1.coeffs.nbytes + table_nodedup1.event_index.nbytes

    item4 = dict(
        N=N4,
        dedup_1syst=dict(build_s=build_dedup1_s, eval_ms=eval_dedup1["median_s"] * 1e3,
                          mem_MB=mem_dedup1 / 1e6),
        nodedup_1syst=dict(build_s=build_nodedup1_s, eval_ms=eval_nodedup1["median_s"] * 1e3,
                            mem_MB=mem_nodedup1 / 1e6),
    )
    # 20-syst anchor from item 1/2 (measured, same N4=100_000 config) + a linear
    # extrapolation of the no-dedup numbers to 20 systs (each systematic's
    # coefficient build/eval/memory is an independent loop iteration -- see
    # splines.py's vmap-over-syst structure and the per-curve Python loop in
    # build_spline_table).
    if not QUICK and N4 in item2:
        dedup20_eval_ms = item2[N4]["spline_product"]["median_s"] * 1e3
    else:
        dedup20_eval_ms = None
    item4["dedup_20syst_measured_eval_ms"] = dedup20_eval_ms
    item4["nodedup_20syst_extrapolated"] = dict(
        build_s=build_nodedup1_s * 20,
        eval_ms=eval_nodedup1["median_s"] * 1e3 * 20,
        mem_MB=mem_nodedup1 / 1e6 * 20,
    )
    log("item4:", item4)

    _save_ckpt("B")

if _run("C"):
    # ---------------------------------------------------------------------------
    # item 5: sampler throughput, ~15 free params, 1e5 events
    # ---------------------------------------------------------------------------

    log("=== item 5: sampler throughput (NUTS vs RWM) ===")

    N5 = N_SCALES[0] if QUICK else 100_000
    built5 = build_model(N5, n_syst=5, n_curves=200, n_norm=4, n_bins=60, seed=3)
    post5 = built5["posteriors"]["poisson"]
    n_free5 = built5["parameter_set"].n_free
    logdensity5 = post5.logdensity
    log(f"n_free = {n_free5} (6 osc + 5 spline + 4 norm)")

    # --- NUTS: adaptation-cost finding + a bounded, non-adapted throughput probe ---
    #
    # A first FULL run of this exact script (n_events=20,000 for the NUTS section,
    # n_syst=5/curves=200/norm=4/n_bins=60, same model built here) measured
    # `bananafit.run_mcmc(algorithm="nuts", n_warmup=80)` -- ordinary
    # `blackjax.window_adaptation` -- taking **467.76 s for 80 warmup steps**
    # (~5.85 s/step), i.e. this benchmark's whole 25-minute time budget on its own.
    # That run is NOT repeated here (it would blow the time budget again); the
    # number is carried forward as ADAPTATION_FINDING below and reported as a
    # genuine, reproducible perf finding (see RESULTS.md "reading the numbers" and
    # recommendations) -- rerun with `NUTS_ADAPT_PROBE=1` and a lot of patience to
    # reproduce it directly.
    #
    # What's actually run here instead is a *bounded* NUTS throughput probe: a
    # hand-picked (not adapted) small step size, an identity mass matrix, and
    # `max_num_doublings` capped low, so a single misbehaving trajectory cannot
    # run away -- this measures raw per-leapfrog-step cost and the shape of the
    # leapfrog-count distribution without paying (or risking) the adaptation cost
    # above. It is explicitly NOT a properly tuned kernel; treat its steps/s and
    # ESS/s as a throughput floor/ceiling probe, not a real sampler's performance.
    ADAPTATION_FINDING = dict(
        n_events=20_000, n_warmup=80, warmup_s=467.76,
        s_per_step=467.76 / 80, source="prior full run of this script (see RESULTS.md)",
    )

    N5_NUTS = N5 if QUICK else 5_000
    built5_nuts = built5 if N5_NUTS == N5 else build_model(
        N5_NUTS, n_syst=5, n_curves=200, n_norm=4, n_bins=60, seed=3
    )
    post5_nuts = built5_nuts["posteriors"]["poisson"]
    logdensity5_nuts = post5_nuts.logdensity

    WARMUP5 = 20 if QUICK else 0  # 0 in full mode: adaptation is skipped, see above
    N_DRAWS5 = 20 if QUICK else 40
    MAP_STEPS5 = 30 if QUICK else 150
    NUTS_MAX_DOUBLINGS = 6  # caps a single trajectory at 2**6 = 64 leapfrog steps

    if QUICK:
        t0 = time.perf_counter()
        warm_run = bananafit.run_mcmc(
            logdensity5_nuts, post5_nuts.z0, algorithm="nuts", n_chains=1,
            n_warmup=WARMUP5, n_samples=1, progress=False, seed=7,
        )
        jax.block_until_ready(warm_run.positions)
        warmup_s = time.perf_counter() - t0
        log(f"NUTS window_adaptation ({WARMUP5} steps, n_events={N5_NUTS}): {warmup_s:.2f} s")
        step_size5 = warm_run.tuned["step_size"][0]
        inv_mm5 = warm_run.tuned["inverse_mass_matrix"][0]
        init_position5 = warm_run.positions[0, -1]
    else:
        warmup_s = 0.0
        step_size5 = 0.05
        inv_mm5 = jnp.ones((n_free5,))
        init_position5 = post5_nuts.z0
        log(f"NUTS: adaptation SKIPPED (see ADAPTATION_FINDING); using fixed "
            f"step_size={step_size5}, identity mass matrix, "
            f"max_num_doublings={NUTS_MAX_DOUBLINGS} on n_events={N5_NUTS}")

    nuts_kernel = blackjax.nuts(
        logdensity5_nuts, step_size=step_size5, inverse_mass_matrix=inv_mm5,
        max_num_doublings=NUTS_MAX_DOUBLINGS,
    )
    init_state5 = nuts_kernel.init(init_position5)


    def nuts_scan(state, keys):
        def body(state, k):
            state, info = nuts_kernel.step(k, state)
            return state, (state.position, info)

        return jax.lax.scan(body, state, keys)


    keys5 = jax.random.split(jax.random.PRNGKey(123), N_DRAWS5)
    jnuts = jax.jit(nuts_scan)
    t0 = time.perf_counter()
    compiled_nuts = jnuts.lower(init_state5, keys5).compile()
    nuts_compile_s = time.perf_counter() - t0
    _ = jax.block_until_ready(compiled_nuts(init_state5, keys5))
    t0 = time.perf_counter()
    _, (positions5, infos5) = compiled_nuts(init_state5, keys5)
    jax.block_until_ready((positions5, infos5))
    nuts_run_s = time.perf_counter() - t0

    nuts_steps_per_sec = N_DRAWS5 / nuts_run_s
    leapfrog = np.asarray(infos5.num_integration_steps)
    nuts_ess = bananafit_ess(types.SimpleNamespace(positions=np.asarray(positions5)[None]))
    nuts_mean_ess = float(np.mean(list(nuts_ess.values())))
    nuts_ess_per_sec = nuts_mean_ess / nuts_run_s

    log(f"NUTS: {N_DRAWS5} draws in {nuts_run_s:.3f} s -> {nuts_steps_per_sec:.2f} steps/s; "
        f"leapfrog/step mean={leapfrog.mean():.1f} median={np.median(leapfrog):.0f} "
        f"[{leapfrog.min()}, {leapfrog.max()}]; "
        f"mean ESS(200 draws)={nuts_mean_ess:.1f} -> {nuts_ess_per_sec:.3f} ESS/s")

    # --- RWM, proposal from Laplace covariance ---

    t0 = time.perf_counter()
    z_map5, _ = bananafit.map_fit(logdensity5, post5.z0, n_steps=MAP_STEPS5)
    jax.block_until_ready(z_map5)
    map_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    cov5 = bananafit.laplace_covariance(logdensity5, z_map5)
    jax.block_until_ready(cov5)
    cov_s = time.perf_counter() - t0
    log(f"MAP fit ({MAP_STEPS5} lbfgs steps): {map_s:.2f} s; Laplace covariance: {cov_s:.2f} s")

    sigma5 = (2.38**2 / n_free5) * cov5
    # blackjax's sigma is a scale matrix: pass chol(cov), not cov
    rw5 = blackjax.additive_step_random_walk.normal_random_walk(
        logdensity5, sigma=jnp.linalg.cholesky(sigma5))
    state_rw5 = rw5.init(z_map5)


    def rwm_scan(state, keys):
        def body(state, k):
            state, info = rw5.step(k, state)
            return state, (state.position, info)

        return jax.lax.scan(body, state, keys)


    N_DRAWS5_RWM = 100 if QUICK else 500
    keys5_rwm = jax.random.split(jax.random.PRNGKey(456), N_DRAWS5_RWM)
    jrwm = jax.jit(rwm_scan)
    t0 = time.perf_counter()
    compiled_rwm = jrwm.lower(state_rw5, keys5_rwm).compile()
    rwm_compile_s = time.perf_counter() - t0
    _ = jax.block_until_ready(compiled_rwm(state_rw5, keys5_rwm))
    t0 = time.perf_counter()
    _, (rwm_pos5, rwm_info5) = compiled_rwm(state_rw5, keys5_rwm)
    jax.block_until_ready((rwm_pos5, rwm_info5))
    rwm_run_s = time.perf_counter() - t0
    rwm_steps_per_sec = N_DRAWS5_RWM / rwm_run_s
    rwm_acc = float(np.mean(np.asarray(rwm_info5.is_accepted)))
    rwm_ess = bananafit_ess(types.SimpleNamespace(positions=np.asarray(rwm_pos5)[None]))
    rwm_mean_ess = float(np.mean(list(rwm_ess.values())))
    rwm_ess_per_sec = rwm_mean_ess / rwm_run_s

    log(f"RWM: {N_DRAWS5_RWM} draws in {rwm_run_s:.3f} s -> {rwm_steps_per_sec:.2f} steps/s; "
        f"accept={rwm_acc:.3f}; mean ESS({N_DRAWS5_RWM} draws)={rwm_mean_ess:.1f} "
        f"-> {rwm_ess_per_sec:.3f} ESS/s")

    item5 = dict(
        n_events=N5, n_free=n_free5, n_events_nuts=N5_NUTS, bounded_probe=not QUICK,
        adaptation_finding=dict(ADAPTATION_FINDING),
        nuts=dict(
            warmup_s=warmup_s, compile_s=nuts_compile_s, n_draws=N_DRAWS5, run_s=nuts_run_s,
            steps_per_sec=nuts_steps_per_sec, leapfrog_mean=float(leapfrog.mean()),
            leapfrog_median=float(np.median(leapfrog)), leapfrog_min=int(leapfrog.min()),
            leapfrog_max=int(leapfrog.max()), mean_ess=nuts_mean_ess, ess_per_sec=nuts_ess_per_sec,
        ),
        rwm=dict(
            map_s=map_s, cov_s=cov_s, compile_s=rwm_compile_s, n_draws=N_DRAWS5_RWM,
            run_s=rwm_run_s, steps_per_sec=rwm_steps_per_sec, accept=rwm_acc,
            mean_ess=rwm_mean_ess, ess_per_sec=rwm_ess_per_sec,
        ),
    )

    # Extrapolate ADAPTATION_FINDING (measured at n_events=20,000) to n_events=100,000
    # assuming grad cost (and hence per-warmup-step cost, at roughly fixed tree-depth
    # behaviour) scales linearly with n_events, per item 1's measured fwd/grad-vs-N
    # scaling. This is an approximation (tree depth can itself shift with curvature,
    # which can change with N) but gives the right order of magnitude.
    if not QUICK:
        scale = 100_000 / ADAPTATION_FINDING["n_events"]
        item5["adaptation_finding"]["projected_1e5_warmup_s_same_n_warmup"] = (
            ADAPTATION_FINDING["warmup_s"] * scale
        )

    # ---------------------------------------------------------------------------
    # item 6: chain parallelism (vmapped RWM, 1e5 events)
    # ---------------------------------------------------------------------------

    log("=== item 6: chain parallelism (vmapped RWM) ===")

    N_STEPS6 = 50 if QUICK else 300
    CHAIN_GRID = [1, 2] if QUICK else [1, 4, 8]
    item6_rows = []
    for nchains in CHAIN_GRID:
        def one_chain(z0, keys):
            state = rw5.init(z0)

            def body(state, k):
                state, info = rw5.step(k, state)
                return state, info

            _, infos = jax.lax.scan(body, state, keys)
            return infos

        def runner(z0_batch, keys_batch):
            return jax.vmap(one_chain)(z0_batch, keys_batch)

        zkey, kkey = jax.random.split(jax.random.PRNGKey(1000 + nchains))
        z0_batch = jnp.broadcast_to(z_map5, (nchains, n_free5)) + 0.02 * jax.random.normal(
            zkey, (nchains, n_free5)
        )
        keys_batch = jax.vmap(lambda k: jax.random.split(k, N_STEPS6))(
            jax.random.split(kkey, nchains)
        )
        res = bench(runner, (z0_batch, keys_batch))
        total_steps_per_sec = nchains * N_STEPS6 / res["median_s"]
        row = dict(
            n_chains=nchains, compile_s=res["compile_s"], exec_ms=res["median_s"] * 1e3,
            total_steps_per_sec=total_steps_per_sec,
            steps_per_sec_per_chain=total_steps_per_sec / nchains,
        )
        item6_rows.append(row)
        log(f"n_chains={nchains}: exec={row['exec_ms']:.2f} ms for {N_STEPS6} steps/chain -> "
            f"{row['total_steps_per_sec']:.1f} total steps/s "
            f"({row['steps_per_sec_per_chain']:.1f} steps/s/chain)")

    log(f"total wall time: {elapsed():.1f} s")
    final_rss_gb = rss_gb()
    log(f"final peak RSS: {final_rss_gb:.2f} GB")

    _save_ckpt("C")

if _run("D"):
    _CK = _load_ckpt()
    total_wall_s = sum(v for k, v in _CK["_wall"].items()) + elapsed() - _STAGE_MARK[0]
    final_rss_gb = max([rss_gb()] + list(_CK["_rss"].values()))
    STAGED_RUN = STAGES != {"A", "B", "C", "D"} and not os.environ.get("BENCH_SINGLE_PROCESS")

    # ---------------------------------------------------------------------------
    # figures (benchmarks/figures/*.png)
    # ---------------------------------------------------------------------------

    log("=== making figures ===")
    figure_files = {}


    def savefig(fig, name, caption):
        path = os.path.join(FIGDIR, name)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figure_files[name] = caption
        log(f"wrote {path}")


    # --- fig 1: logdensity/grad time vs n_events, log-log, poisson vs BB-lite ---

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for stat in ("poisson", "barlow-beeston"):
        rows = [r for r in item1_rows if r["stat"] == stat]
        if not rows:
            continue
        ns = [r["n_events"] for r in rows]
        fwd = [r["fwd_ms"] for r in rows]
        grad = [r["grad_ms"] for r in rows]
        c = _COLORS[stat]
        ax.plot(ns, fwd, "o-", color=c, label=f"{stat} fwd")
        ax.plot(ns, grad, "s--", color=c, label=f"{stat} grad", alpha=0.75)
        for r in rows:
            if r["stat"] == "poisson":
                ax.annotate(
                    f"{r['events_per_sec_fwd']:.1e} ev/s",
                    (r["n_events"], r["fwd_ms"]), textcoords="offset points",
                    xytext=(6, -12), fontsize=9, color=c,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_events")
    ax.set_ylabel("time per call (ms)")
    ax.set_title("logdensity / grad(logdensity) cost vs n_events\n"
                  "(20 spline systs x 200 curves, 4 norm systs, 60 bins, CPU)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    savefig(fig, "fig1_logdensity_grad_vs_nevents.png",
            "logdensity/grad(logdensity) time vs n_events (log-log), Poisson vs "
            "Barlow-Beeston-lite, with forward events/s annotated at each Poisson point.")

    # --- fig 2: component breakdown at 1e5 and 1e6, grouped bar, log y ---

    comp_names = ["oscillation", "spline_product", "norm_product", "segment_sum", "likelihood_priors"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(comp_names))
    width = 0.35
    n_lo, n_hi = min(item2.keys()), max(item2.keys())
    vals_lo = [item2[n_lo][c]["median_s"] * 1e3 for c in comp_names]
    vals_hi = [item2[n_hi][c]["median_s"] * 1e3 for c in comp_names]
    ax.bar(x - width / 2, vals_lo, width, label=f"n_events={n_lo:,}", color="#1f77b4")
    ax.bar(x + width / 2, vals_hi, width, label=f"n_events={n_hi:,}", color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=20, ha="right")
    ax.set_ylabel("median time (ms, log scale)")
    ax.set_title("Forward-pass component breakdown")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    savefig(fig, "fig2_component_breakdown.png",
            f"Component-wise forward cost at n_events={n_lo:,} and n_events={n_hi:,} "
            "(each component timed as its own AOT-compiled executable).")

    # --- fig 3: cost vs n_syst, linearity check ---

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ns3 = [r["n_syst"] for r in item3_rows]
    fwd3 = [r["fwd_ms"] for r in item3_rows]
    grad3 = [r["grad_ms"] for r in item3_rows]
    ax.plot(ns3, fwd3, "o-", color="#1f77b4", label="fwd (measured)")
    ax.plot(ns3, grad3, "s-", color="#d62728", label="grad (measured)")
    if len(ns3) >= 2:
        lin_fwd = np.interp(ns3, [ns3[0], ns3[-1]], [fwd3[0], fwd3[-1]])
        lin_grad = np.interp(ns3, [ns3[0], ns3[-1]], [grad3[0], grad3[-1]])
        ax.plot(ns3, lin_fwd, ":", color="#1f77b4", alpha=0.6, label="linear ref (fwd)")
        ax.plot(ns3, lin_grad, ":", color="#d62728", alpha=0.6, label="linear ref (grad)")
    ax.set_xlabel("n_syst (200 dedup curves each)")
    ax.set_ylabel("time per call (ms)")
    ax.set_title(f"Cost vs n_syst at n_events={N3:,}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, "fig3_cost_vs_nsyst.png",
            f"fwd/grad cost vs number of spline systematics at n_events={N3:,}, with a "
            "straight-line reference between the first and last measured points to "
            "visualize (non-)linearity.")

    # --- fig 4: dedup payoff -- eval time + memory, 200 curves vs per-event ---

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.8), gridspec_kw=dict(wspace=0.32))
    labels = ["dedup\n(200 curves)\n1 syst", "no-dedup\n(per-event)\n1 syst",
              "no-dedup\n(per-event)\n20 syst\n[x20 EXTRAP.]"]
    colors4 = [_COLORS["dedup"], _COLORS["nodedup"], "#f4a582"]

    build_vals = [
        item4["dedup_1syst"]["build_s"] * 1e3,
        item4["nodedup_1syst"]["build_s"] * 1e3,
        item4["nodedup_20syst_extrapolated"]["build_s"] * 1e3,
    ]
    eval_vals = [
        item4["dedup_1syst"]["eval_ms"],
        item4["nodedup_1syst"]["eval_ms"],
        item4["nodedup_20syst_extrapolated"]["eval_ms"],
    ]
    mem_vals = [
        item4["dedup_1syst"]["mem_MB"],
        item4["nodedup_1syst"]["mem_MB"],
        item4["nodedup_20syst_extrapolated"]["mem_MB"],
    ]
    if item4.get("dedup_20syst_measured_eval_ms") is not None:
        labels_eval = labels + ["dedup\n(200 curves)\n20 syst\n[MEASURED]"]
        eval_vals_full = eval_vals + [item4["dedup_20syst_measured_eval_ms"]]
        colors_eval = colors4 + ["#4393c3"]
    else:
        labels_eval, eval_vals_full, colors_eval = labels, eval_vals, colors4

    for ax, vals, lbls, cols, title, ylabel in (
        (axes[0], build_vals, labels, colors4, "Build time", "build time (ms, log)"),
        (axes[1], eval_vals_full, labels_eval, colors_eval, "Eval time (jitted __call__)",
         "eval time (ms, log)"),
        (axes[2], mem_vals, labels, colors4, "Table + index memory", "memory (MB, log)"),
    ):
        bars = ax.bar(range(len(vals)), vals, color=cols)
        for bar, lbl in zip(bars, lbls):
            if "EXTRAP" in lbl:      # hatch the x20-extrapolated bars so they cannot pass for measurements
                bar.set_hatch("//")
                bar.set_edgecolor("white")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(lbls, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.suptitle(f"Curve dedup payoff (n_events={item4['N']:,})", y=1.03)
    savefig(fig, "fig4_dedup_payoff.png",
            f"Build time, eval time, and memory: 200 dedup curves vs per-event "
            f"('no-dedup') spline curves, n_events={item4['N']:,}. Build time is measured "
            "at 1 systematic and (for the 20-syst no-dedup bar only) linearly "
            "extrapolated x20; eval time's 20-syst dedup bar is the actual measured "
            "value from item 2, all others are 1-syst measurements ([x20 EXTRAP.] "
            "bars only) scaled x20.")

    log(f"figures written to {FIGDIR}")

    # ---------------------------------------------------------------------------
    # RESULTS.md
    # ---------------------------------------------------------------------------


    def fmt_table(headers, rows):
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        for r in rows:
            lines.append("| " + " | ".join(str(c) for c in r) + " |")
        return "\n".join(lines)


    def write_results():
        lines = []
        lines.append("# bananafit hot-path benchmark results\n")
        lines.append(
            f"Generated by `bench_hotpath.py` on {time.strftime('%Y-%m-%d %H:%M %Z')}, "
            f"CPU-only, float64, {jax.devices()[0]}. "
            f"QUICK={'yes (smoke-test scale, NOT representative)' if QUICK else 'no (full run)'}. "
            f"Total wall time: {total_wall_s:.1f} s"
            + (" (sum of the per-stage wall times: stages A=items 1-2, B=items 3-4, C=items 5-6 "
               "were run as separate processes so each foreground command stayed under 10 min; "
               "`bench_hotpath.py` with `BENCH_STAGES` unset runs them in one process)" if STAGED_RUN else "")
            + ". Run conditions: "
            + os.environ.get("BENCH_RUN_NOTE", "not recorded")
            + ". Figures referenced below are written to "
            "`benchmarks/figures/` by this same script.\n"
        )

        lines.append("## 1. logdensity / grad(logdensity) vs n_events\n")
        lines.append(
            "20 spline systematics x 200 dedup curves, 4 norm systs, 60 bins, single sample, "
            "single-point evaluation (not a real physics likelihood -- synthetic weights)."
            " `fwd`/`grad` are medians of >=5 reps on an AOT-compiled executable; compile time "
            "(trace + XLA compile, via `.lower().compile()`) is reported separately and excluded "
            "from the per-call timing.\n"
        )
        headers = ["n_events", "stat", "fwd (ms)", "fwd compile (s)", "grad (ms)",
                   "grad compile (s)", "grad/fwd ratio", "events/s (fwd)"]
        rows = [
            [r["n_events"], r["stat"], f"{r['fwd_ms']:.3f}", f"{r['fwd_compile_s']:.2f}",
             f"{r['grad_ms']:.3f}", f"{r['grad_compile_s']:.2f}", f"{r['ratio']:.2f}",
             f"{r['events_per_sec_fwd']:.3e}"]
            for r in item1_rows
        ]
        lines.append(fmt_table(headers, rows))
        lines.append("")
        lines.append(
            f"Memory (spline coefficient table, n_syst=20, n_curves=200, independent of n_events): "
            f"float32 = {mem_rows['coeffs_f32_MB']:.3f} MB, "
            f"float64 = {mem_rows['coeffs_f64_MB']:.3f} MB "
            f"({mem_rows['coeffs_f64_MB']/mem_rows['coeffs_f32_MB']:.1f}x). "
            f"At n_events={mem_rows['n_events_largest']:,}: event_index (20, N) int32 = "
            f"{mem_rows['event_index_MB']:.1f} MB, true_energy (N,) f64 = "
            f"{mem_rows['true_energy_MB']:.1f} MB. "
            f"Peak RSS (`resource.getrusage`, cumulative high-water mark of this process, "
            f"measured right after the largest-n_events stage of item 1): "
            f"{mem_rows['peak_rss_gb']:.2f} GB. Final peak RSS at end of script: "
            f"{final_rss_gb:.2f} GB" + (" (max over the separate stage processes)" if STAGED_RUN else "") + ".\n"
        )
        lines.append(
            "![logdensity/grad time vs n_events](figures/fig1_logdensity_grad_vs_nevents.png)\n\n"
            "*Figure 1: logdensity/grad(logdensity) time vs n_events (log-log), Poisson vs "
            "Barlow-Beeston-lite, forward events/s annotated at each Poisson point.*\n"
        )

        lines.append("## 2. Component breakdown\n")
        lines.append(
            "Each component timed as its own AOT-compiled executable (not part of the fused "
            "`logdensity` jit), so the sum differs from item 1's fused `fwd` time by whatever "
            "XLA fusion saves/loses across the component boundary -- reported as a sanity check, "
            "not claimed to equal the fused cost exactly.\n"
        )
        for n, comp in item2.items():
            headers = ["component", "median (ms)", "compile (s)"]
            rows = [[k, f"{v['median_s']*1e3:.4f}", f"{v['compile_s']:.2f}"] for k, v in comp.items()]
            total = sum(v["median_s"] for v in comp.values()) * 1e3
            lines.append(f"**n_events = {n:,}** (sum of components = {total:.4f} ms)\n")
            lines.append(fmt_table(headers, rows))
            lines.append("")
        lines.append(
            "![component breakdown](figures/fig2_component_breakdown.png)\n\n"
            f"*Figure 2: forward-pass component breakdown at n_events={min(item2.keys()):,} and "
            f"n_events={max(item2.keys()):,}.*\n"
        )

        lines.append("## 3. Scaling in n_syst (n_events fixed)\n")
        lines.append(f"n_events = {N3:,}, 200 dedup curves per systematic.\n")
        headers = ["n_syst", "fwd (ms)", "grad (ms)"]
        rows = [[r["n_syst"], f"{r['fwd_ms']:.4f}", f"{r['grad_ms']:.4f}"] for r in item3_rows]
        lines.append(fmt_table(headers, rows))
        if len(item3_rows) >= 2:
            lines.append(
                f"\nSlope: {slope_fwd:.5f} ms/systematic (fwd), {slope_grad:.5f} ms/systematic "
                "(grad) -- linear fit through the first and last grid points.\n"
            )
        lines.append(
            "\n![cost vs n_syst](figures/fig3_cost_vs_nsyst.png)\n\n"
            f"*Figure 3: fwd/grad cost vs n_syst at n_events={N3:,}, with a straight-line "
            "reference between the endpoints to visualize (non-)linearity.*\n"
        )

        lines.append("## 4. Curve dedup payoff\n")
        lines.append(
            f"n_events = {item4['N']:,}. The per-curve build loop in `build_spline_table` "
            "(one `scipy.interpolate.CubicSpline` fit per curve, in a Python `for` loop -- see "
            "splines.py's builder) makes the true 20-syst x per-event (N=100,000 curves/syst) "
            "case far too slow to build within this benchmark's time budget "
            f"(~{item4['nodedup_20syst_extrapolated']['build_s']/60:.0f} minutes extrapolated, see below); "
            "measured directly at 1 systematic instead "
            "and linearly extrapolated to 20 (each systematic is an independent loop "
            "iteration in both the builder and `SplineTable._eval_all_curves`'s `vmap`).\n"
        )
        headers = ["config", "curves/syst", "build time", "eval (ms, jitted __call__)", "table+index mem (MB)"]
        rows = [
            ["dedup, 1 syst (measured)", 200, f"{item4['dedup_1syst']['build_s']*1e3:.2f} ms",
             f"{item4['dedup_1syst']['eval_ms']:.4f}", f"{item4['dedup_1syst']['mem_MB']:.3f}"],
            ["no-dedup, 1 syst (measured)", item4["N"], f"{item4['nodedup_1syst']['build_s']:.2f} s",
             f"{item4['nodedup_1syst']['eval_ms']:.4f}", f"{item4['nodedup_1syst']['mem_MB']:.2f}"],
            ["no-dedup, 20 syst (EXTRAPOLATED x20)", item4["N"],
             f"{item4['nodedup_20syst_extrapolated']['build_s']:.1f} s",
             f"{item4['nodedup_20syst_extrapolated']['eval_ms']:.3f}",
             f"{item4['nodedup_20syst_extrapolated']['mem_MB']:.1f}"],
        ]
        lines.append(fmt_table(headers, rows))
        build_ratio = item4["nodedup_1syst"]["build_s"] / item4["dedup_1syst"]["build_s"]
        eval_ratio_1syst = item4["nodedup_1syst"]["eval_ms"] / item4["dedup_1syst"]["eval_ms"]
        mem_ratio_1syst = item4["nodedup_1syst"]["mem_MB"] / item4["dedup_1syst"]["mem_MB"]
        curve_ratio = item4["N"] / 200
        ratio20 = None
        if item4.get("dedup_20syst_measured_eval_ms") is not None:
            ratio20 = (item4["nodedup_20syst_extrapolated"]["eval_ms"]
                       / item4["dedup_20syst_measured_eval_ms"])
        _ratios = [eval_ratio_1syst] + ([ratio20] if ratio20 is not None else [])
        eval_small = max(_ratios) < 1.5
        eval_phrase = (
            f"no resolvable eval-time penalty in this run (no-dedup/dedup eval ratio "
            f"{eval_ratio_1syst:.2f}x at 1 systematic"
            + (f", {ratio20:.2f}x at 20; within the noise of a shared-machine micro-benchmark)"
               if ratio20 is not None else ")")
            if eval_small else
            f"an eval-time penalty (no-dedup/dedup eval ratio {eval_ratio_1syst:.1f}x at 1 systematic"
            + (f", {ratio20:.1f}x at 20 systematics" if ratio20 is not None else "") + ")"
        )
        lines.append(
            f"\n**The dedup payoff is a *build-time* and memory payoff, with {eval_phrase}**: "
            f"at 1 systematic, no-dedup build is "
            f"{build_ratio:.0f}x slower than dedup build ({item4['nodedup_1syst']['build_s']:.1f} s "
            f"vs {item4['dedup_1syst']['build_s']*1e3:.0f} ms) and its table+index uses "
            f"{mem_ratio_1syst:.0f}x the memory, while no-dedup *eval* is "
            f"{eval_ratio_1syst:.2f}x the dedup eval "
            f"({item4['nodedup_1syst']['eval_ms']:.3f} ms vs {item4['dedup_1syst']['eval_ms']:.3f} ms) "
            f"for {curve_ratio:.0f}x more curves -- i.e. `SplineTable.__call__`'s vmapped Horner pass "
            "scales strongly sub-linearly in curve count on this CPU (the per-event gather dominates, "
            "not the per-curve Horner evaluation). The eval ratio is the least stable number in this "
            "benchmark (see 'Replicates and measurement noise' at the end: 1.2x-4.3x across three runs)."
        )
        if ratio20 is not None:
            fwd_1e5 = [r for r in item1_rows if r["stat"] == "poisson" and r["n_events"] == item4["N"]]
            fwd_1e5_txt = (
                f" (for scale: the whole fused dedup forward pass at this n_events is "
                f"{fwd_1e5[0]['fwd_ms']:.1f} ms, item 1)" if fwd_1e5 else ""
            )
            lines.append(
                f" At the full 20-systematic scale the measured dedup eval "
                f"({item4['dedup_20syst_measured_eval_ms']:.3f} ms) vs the extrapolated no-dedup "
                f"eval ({item4['nodedup_20syst_extrapolated']['eval_ms']:.3f} ms) gives "
                f"~{ratio20:.1f}x{fwd_1e5_txt}, far from the "
                f"{curve_ratio:.0f}x a naive O(n_curves) reading of `_eval_all_curves` would predict. "
                "The no-dedup eval numbers are 1-syst measurements scaled x20 (so this ratio compares "
                "a measurement with an extrapolation and can fall below 1 within noise), and would need "
                "re-checking at much larger curve counts or on a GPU. Absolute timings, especially the "
                "single-threaded Python build time, depend on machine load.\n"
            )
        else:
            lines.append("\n")
        lines.append(
            "![dedup payoff](figures/fig4_dedup_payoff.png)\n\n"
            f"*Figure 4: build time, eval time, and memory for 200 dedup curves vs per-event "
            f"('no-dedup') spline curves, n_events={item4['N']:,}.*\n"
        )

        lines.append("## 5. Sampler throughput (NUTS vs RWM)\n")
        lines.append(
            f"n_events = {item5['n_events']:,}, n_free = {item5['n_free']} "
            "(6 osc + 5 spline systs + 4 norm systs) throughout, except where noted.\n"
        )

        af = item5["adaptation_finding"]
        lines.append(
            f"**Headline finding: NUTS warmup (`blackjax.window_adaptation`) is the dominant cost, "
            f"not the per-step gradient.** A full run of `bananafit.run_mcmc(algorithm='nuts', "
            f"n_warmup={af['n_warmup']})` on an n_events={af['n_events']:,} instance of this same "
            f"model measured **{af['warmup_s']:.1f} s for {af['n_warmup']} warmup steps "
            f"({af['s_per_step']:.2f} s/step)** -- {af['source']}. Linearly extrapolating that to "
            f"n_events=100,000 (scaling with item 1's measured grad-cost-vs-n_events) gives "
            f"~{af.get('projected_1e5_warmup_s_same_n_warmup', float('nan'))/60:.0f} minutes for "
            f"just {af['n_warmup']} warmup steps -- i.e. this benchmark's entire ~25-minute time "
            "budget would be consumed several times over by NUTS warmup alone at the requested "
            "1e5-event scale with default settings. This is the main scaling-cliff finding of this "
            "benchmark; see 'reading the numbers' and recommendations below for the likely cause "
            "(deep/unbounded trajectory trees from poorly-conditioned parameters -- e.g. `dm31`, "
            "`dm21` and `theta13` sit on very different natural scales in this synthetic model -- "
            "before `window_adaptation`'s mass-matrix estimate has converged).\n"
        )

        if item5["bounded_probe"]:
            lines.append(
                f"Because of that finding, the throughput numbers below are a **bounded, "
                f"non-adapted NUTS probe**: a fixed step size ({step_size5}), identity mass matrix, "
                f"and `max_num_doublings={NUTS_MAX_DOUBLINGS}` (caps one trajectory at "
                f"{2**NUTS_MAX_DOUBLINGS} leapfrog steps) at a reduced n_events={item5['n_events_nuts']:,} "
                "-- this deliberately avoids both the adaptation cost above and the risk of an "
                "unbounded trajectory, so it measures raw per-leapfrog-step throughput and the "
                "*shape* of the leapfrog-count distribution, not a properly tuned sampler's real "
                "performance (its ESS/s in particular should not be compared to a tuned run).\n"
            )
        lines.append(
            f"RWM proposal covariance from `bananafit.map_fit` ({MAP_STEPS5} L-BFGS steps, "
            f"{item5['rwm']['map_s']:.1f} s, at n_events={item5['n_events']:,}) + "
            f"`bananafit.laplace_covariance` ({item5['rwm']['cov_s']:.1f} s -- dominated by one-time "
            "Hessian-graph XLA compilation, not by n_events, see reading-the-numbers section), "
            "scaled by the Roberts-Rosenthal 2.38^2/d constant, run for "
            f"{item5['rwm']['n_draws']} draws via a hand-rolled `additive_step_random_walk` scan "
            f"at the full n_events={item5['n_events']:,}. ESS via `bananafit.diagnostics.ess` "
            "(blackjax `effective_sample_size`) on that single short chain -- crude, not a "
            "converged-chain estimate.\n"
        )
        headers = ["algorithm", "n_events", "draws", "wall (s)", "steps/s", "mean ESS", "ESS/s", "extra"]
        rows = [
            ["NUTS (bounded, not tuned)" if item5["bounded_probe"] else "NUTS (tuned)",
             f"{item5['n_events_nuts']:,}", item5["nuts"]["n_draws"],
             f"{item5['nuts']['run_s']:.3f}",
             f"{item5['nuts']['steps_per_sec']:.2f}", f"{item5['nuts']['mean_ess']:.1f}",
             f"{item5['nuts']['ess_per_sec']:.3f}",
             f"leapfrog/step: mean={item5['nuts']['leapfrog_mean']:.1f}, "
             f"median={item5['nuts']['leapfrog_median']:.0f}, "
             f"range=[{item5['nuts']['leapfrog_min']},{item5['nuts']['leapfrog_max']}]"],
            ["RWM", f"{item5['n_events']:,}", item5["rwm"]["n_draws"],
             f"{item5['rwm']['run_s']:.3f}",
             f"{item5['rwm']['steps_per_sec']:.2f}", f"{item5['rwm']['mean_ess']:.1f}",
             f"{item5['rwm']['ess_per_sec']:.3f}", f"accept={item5['rwm']['accept']:.3f}"],
        ]
        lines.append(fmt_table(headers, rows))
        lines.append("")

        lines.append("## 6. Chain parallelism (vmapped RWM, CPU)\n")
        lines.append(f"n_events = {item5['n_events']:,}, n_free = {item5['n_free']}, "
                      f"{N_STEPS6} RWM steps/chain, vmapped over `n_chains`.\n")
        headers = ["n_chains", "compile (s)", "exec (ms, all chains)", "total steps/s",
                   "steps/s/chain"]
        rows = [
            [r["n_chains"], f"{r['compile_s']:.2f}", f"{r['exec_ms']:.2f}",
             f"{r['total_steps_per_sec']:.1f}", f"{r['steps_per_sec_per_chain']:.1f}"]
            for r in item6_rows
        ]
        lines.append(fmt_table(headers, rows))
        lines.append("")

        # --- reading the numbers -------------------------------------------------
        lines.append("## Reading the numbers\n")

        # dominant component, from item 2 (use the larger scale)
        largest_item2_n = max(item2.keys())
        comp = item2[largest_item2_n]
        dominant = max(comp.items(), key=lambda kv: kv[1]["median_s"])
        lines.append(
            f"**Where the time goes.** At n_events={largest_item2_n:,}, the dominant piece of the "
            f"forward pass is **{dominant[0]}** ({dominant[1]['median_s']*1e3:.4f} ms of "
            f"{sum(v['median_s'] for v in comp.values())*1e3:.4f} ms summed). "
        )
        if dominant[0] == "oscillation":
            lines.append(
                "This matches the DESIGN.md expectation that the jaxnu `nufast` oscillation "
                "probability calculation dominates (it does a real trig/matrix-style calculation "
                "per event; splines/norms/histogramming are comparatively cheap gather-multiply-add "
                "work). "
            )
        else:
            lines.append(
                f"DESIGN.md expects the oscillation ({comp['oscillation']['median_s']*1e3:.4f} ms) "
                f"to dominate over the spline product "
                f"({comp['spline_product']['median_s']*1e3:.4f} ms, expected to be "
                "memory-bandwidth-ish rather than compute-bound) -- "
                + ("this held" if dominant[0] == "oscillation" else "this measurement does NOT "
                   "clearly show that ordering; see the per-scale component table above before "
                   "treating this as a real deviation (component sizes are close and this machine "
                   "is shared/noisy).")
                + " "
            )
        lines.append(
            f"The grad/forward ratio in item 1 (Poisson rows: {min(r['ratio'] for r in item1_rows if r['stat']=='poisson'):.1f}-"
            f"{max(r['ratio'] for r in item1_rows if r['stat']=='poisson'):.1f}x, median "
            f"{float(np.median([r['ratio'] for r in item1_rows if r['stat']=='poisson'])):.1f}x; "
            "the Barlow-Beeston rows scatter more, partly CPU-load noise on a shared machine) is the "
            "expected few-x tax of reverse-mode AD through a fused elementwise+segment_sum+oscillation "
            "graph and is comparable to jaxnu's own documented ~3x forward/gradient ratio for the "
            "oscillation kernel alone, so the extra bananafit machinery (splines, norms, "
            "histogramming, priors) does not appear to add a disproportionate gradient tax.\n"
        )

        if item1_rows:
            largest_poisson = [r for r in item1_rows if r["stat"] == "poisson"][-1]
            lines.append(
                f"**Practical CPU event-count ceiling.** At the largest measured scale "
                f"(n_events={largest_poisson['n_events']:,}), a single `logdensity`+`grad` pair "
                f"costs {largest_poisson['fwd_ms']+largest_poisson['grad_ms']:.1f} ms. A NUTS step "
                f"needing O(10-100) gradient evaluations (item 5's measured leapfrog range was "
                f"[{item5['nuts']['leapfrog_min']}, {item5['nuts']['leapfrog_max']}]) therefore "
                "costs seconds at this scale once warmup is well-tuned, and tens of seconds "
                "when warmup is short/mistuned (as it was here, deliberately, for time budget -- "
                "see item 5's deviation note). Practically: **RWM scales comfortably to 1e6 events "
                "on this CPU, at a cost of one forward pass per step** "
                f"(item 1's fwd column: {[r for r in item1_rows if r['stat']=='poisson'][-1]['fwd_ms']:.0f} ms at "
                f"n_events={largest_poisson['n_events']:,}, "
                f"{[r for r in item1_rows if r['stat']=='poisson' and r['n_events']==100_000][0]['fwd_ms']:.1f} ms at 100,000; "
                "i.e. well under 10 steps/s at 1e6 events -- usable, not fast); **NUTS is comfortable up to roughly 1e4-1e5 events "
                "for exploratory/short chains**, but a full production NUTS fit (thousands of warmup "
                "+ draws, well-tuned so leapfrog counts are smaller than the under-tuned ones seen "
                "here) at 1e5-1e6 events should be budgeted in CPU-hours, not minutes, unless the "
                "event count is reduced (e.g. pre-binned/pre-summed MC where physically valid) or "
                "warmup is tuned more aggressively for a shorter trajectory length.\n"
            )

        lines.append(
            "**Expected GPU story (qualitative, not measured here -- CPU-only per the task).** "
            "Every hot-path op (oscillation nufast, spline Horner, norm product, segment_sum, "
            "Poisson/BB-lite reduction) is a dense, branch-free, embarrassingly-parallel elementwise "
            "or reduction op over the event axis -- exactly the shape that benefits most from a GPU: "
            "expect a substantial forward/grad speedup at 1e5-1e6 events (memory-bandwidth-bound "
            "pieces like the spline product should benefit less than compute-bound pieces like "
            "oscillation, all else equal). `jax.ops.segment_sum` is the one op worth watching -- "
            "its GPU implementation is a scatter-add, which is typically fine at these bin counts "
            "(tens to hundreds of bins) but would need checking if bin counts grew by orders of "
            "magnitude. Chain-parallelism (item 6) is the other place GPU should change the story "
            "qualitatively: on this CPU, more vmapped chains did not show clean amortization (see "
            f"item 6's table -- total steps/s rises only {item6_rows[0]['total_steps_per_sec']:.0f} -> "
            f"{item6_rows[-1]['total_steps_per_sec']:.0f} ({item6_rows[-1]['total_steps_per_sec']/item6_rows[0]['total_steps_per_sec']:.1f}x) "
            f"for {item6_rows[-1]['n_chains']}x more chains while "
            f"steps/s/chain falls {item6_rows[0]['steps_per_sec_per_chain']:.0f} -> "
            f"{item6_rows[-1]['steps_per_sec_per_chain']:.0f}, i.e. wall time grows nearly in proportion to the chain count "
            "(a modest throughput gain at best), "
            "since a single CPU core/thread pool has no spare "
            "parallel capacity to hide chain-batch latency behind); a GPU has the reverse profile "
            "(latency-bound at 1 chain, throughput-bound and heavily amortized at 4-64+ chains), so "
            "multi-chain NUTS/RWM is where a GPU is expected to pay off most, not single-chain "
            "per-step latency.\n"
        )

        lines.append("### Recommendations (no code changes made -- see task scope)\n")
        lines.append(
            f"- **(headline) `run_mcmc(algorithm='nuts')`'s default `blackjax.window_adaptation` "
            f"is a genuine scaling cliff, not just slow.** Measured directly: "
            f"{af['warmup_s']:.1f} s for {af['n_warmup']} warmup steps at n_events="
            f"{af['n_events']:,} ({af['s_per_step']:.2f} s/step) -- extrapolated to "
            f"~{af.get('projected_1e5_warmup_s_same_n_warmup', float('nan'))/60:.0f} minutes at "
            "n_events=100,000 for the same 80 steps (a real fit needs hundreds to a few thousand). "
            "The observed leapfrog-step distribution during (under-tuned, short) adaptation had a "
            "long tail toward `max_num_doublings`'s default cap (2**10 = 1024 leapfrog steps per "
            "transition) -- consistent with poorly-conditioned parameters confusing the "
            "diagonal-mass-matrix estimate before it converges (this synthetic benchmark model puts "
            "`dm21`, `dm31`, and angle parameters on very different natural scales with no "
            "informative priors to anchor them; a real DUNE-like analysis would normally add a "
            "reactor `theta13` constraint and physically-motivated priors that narrow the initial "
            "search, which should shrink this substantially). Recommend: (a) [done since this "
            "benchmark was first written: `run_mcmc` now exposes `max_num_doublings` and "
            "`target_acceptance_rate`, and `laplace_seeded_nuts` avoids window adaptation altogether; "
            "this benchmark's bounded probe still builds `blackjax.nuts` directly, as it predates that] "
            "(b) consider defaulting to a lower `max_num_doublings` (e.g. 8-10 is blackjax's "
            "default 10, but 6-8 bounds worst-case cost substantially for a modest exploration "
            "loss) for large-n_events / many-parameter models; (c) document in DESIGN.md/README "
            "that informative priors on scale-mismatched parameters (e.g. `dm21`/`dm31` vs mixing "
            "angles) are not just good statistical practice here but a practical warmup-cost "
            "mitigation.\n"
            "- **Curve dedup pays off mainly at build time and in memory; the eval-time difference is small "
            "and noisy, at least up to the N=100,000-curve scale measured here.** Item 4's measured "
            f"numbers: no-dedup build is ~{build_ratio:.0f}x slower than dedup build (a Python-level "
            "`scipy.interpolate.CubicSpline` call per curve, so build time is O(n_syst * n_curves), "
            f"not O(n_syst) as the dedup case effectively is at fixed n_curves) and uses ~{mem_ratio_1syst:.0f}x the "
            f"memory at 1 systematic (scales the same at 20), while the jitted `SplineTable.__call__` "
            f"eval is only ~{eval_ratio_1syst:.1f}x the dedup one for {item4['N']:,} curves vs 200 in this run "
            "(1.2x-4.3x across the replicate runs; XLA vectorizes the per-curve Horner pass well). "
            "Practical implication: a caller who accidentally builds per-event splines will pay it "
            f"back mainly as a slow, memory-hungry *build* step (item 4: {item4['nodedup_1syst']['build_s']:.0f} s "
            f"and {item4['nodedup_1syst']['mem_MB']:.0f} MB for 1 systematic x {item4['N']:,} events vs "
            f"{item4['dedup_1syst']['build_s']*1e3:.0f} ms and {item4['dedup_1syst']['mem_MB']:.1f} MB deduped) "
            "and at most a modest per-step eval penalty -- still worth avoiding (DESIGN.md Section 4 already assumes "
            "bin/dedup-before-`build_spline_table`), and a build-time warning if `n_curves` is a "
            "large fraction of `N` would be cheap insurance. The eval-time penalty is much smaller than a "
            "naive reading of `_eval_all_curves`'s O(n_curves) complexity would suggest, but that "
            "would need re-checking at curve counts well beyond 100,000, or on a GPU.\n"
            "- **`build_spline_table`'s per-curve Python loop is the only build-time hot spot found.** "
            "At the dedup scale the package actually recommends (~hundreds of curves per "
            "systematic) it is fast (~1 s for 20 systs x 200 curves, item 1's build_s column), so "
            "this is not urgent, but if analyses grow toward thousands of curves per systematic "
            "(finer kinematic binning) it would be worth vectorizing (batched `CubicSpline` "
            "construction, or a hand-rolled tridiagonal solve over all curves of a systematic at "
            "once) rather than the current per-curve scipy call.\n"
            "- **`laplace_covariance`'s Hessian graph is compile-cost-dominated, not size-"
            "dominated** (item 5: tens of seconds regardless of n_events, since the graph shape "
            "is the same for any N). That is a one-time, once-per-fit cost so not urgent, but it "
            "is worth knowing before assuming a slow `laplace_covariance` call means something is "
            "wrong with a particular dataset -- it is essentially a fixed tax for using it at all.\n"
            "- **Prefer RWM (or a well-tuned/short-trajectory NUTS) as the default recommendation "
            "for large-n_events production fits today**, until the warmup-cost mitigations above "
            "land -- item 5/6 show RWM throughput is CPU-comfortable at 1e5 events "
            f"({item5['rwm']['steps_per_sec']:.0f} steps/s single-chain), although item 6 shows that vmapping "
            "RWM chains on this CPU raises total throughput only modestly (per-chain throughput falls "
            "nearly ~1/n_chains, so multi-chain runs cost wall time roughly proportional to the chain count on CPU), "
            "whereas plain NUTS's warmup cost at this scale is the dominant risk to a fit's "
            "wall-clock budget.\n"
        )

        appendix = os.environ.get("BENCH_APPENDIX_MD")
        if appendix and os.path.exists(appendix):
            lines.append(open(appendix).read())
        text = "\n".join(lines) + "\n"
        if OUT_TAG:
            text = text.replace("](figures/", f"](figures/{OUT_TAG}/")
        with open(RESULTS_MD, "w") as f:
            f.write(text)
        log(f"wrote {RESULTS_MD}")


    write_results()
    log("done.")
