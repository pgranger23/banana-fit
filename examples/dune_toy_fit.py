#!/usr/bin/env python3
"""End-to-end DUNE-like toy fit -- the demonstration this whole package was
built for (DESIGN.md §12.4): build a self-contained 4-sample long-baseline
analysis, fit it with gradient-based NUTS and gradient-free RWM, and compare.

Run the full version (defaults matching bananafit.toy's own defaults; tens of
minutes on CPU -- NUTS on a correlated multi-sample binned likelihood pays a
real per-draw cost, see ``_config``'s comment)::

    python3 examples/dune_toy_fit.py

Run a smoke version (small analysis, short chains; ~5-6 minutes on CPU here
-- most of that is the one-time eager Laplace-Hessian call plus NUTS's
per-draw tree search on this genuinely correlated posterior, not event
count; see ``_config``)::

    python3 examples/dune_toy_fit.py --quick

Either way the script:

1. Builds the 4-sample (FHC/RHC x numu-disappearance/nue-appearance) toy
   analysis via ``bananafit.toy.make_dune_like_analysis``.
2. Builds an Asimov dataset at truth and a Poisson-fluctuated dataset from
   it (``bananafit.toy.poisson_fluctuate``) -- the fluctuated dataset is what
   gets fit.
3. Runs a MAP pre-fit (``map_fit``), a Laplace covariance
   (``laplace_covariance``), a NUTS posterior, and an RWM posterior seeded
   with the Laplace covariance (MaCh3-style proposal).
4. Demonstrates ``save_analysis``/``load_analysis`` HDF5 round-tripping the
   built analysis.
5. Saves four figures to ``examples/figures/`` and prints a summary block.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax

import bananafit  # noqa: F401 -- enforces float64, per house style
from bananafit.diagnostics import ess, rhat, to_arviz
from bananafit.io import load_analysis, save_analysis
from bananafit.model import SampleModel, build_forward_model
from bananafit.oscillation import ConstantMatterOsc
from bananafit.parameters import Gaussian, VonMises, ParameterSet
from bananafit.posterior import asimov_data, make_posterior
from bananafit.sampling import laplace_covariance, map_fit, run_mcmc
from bananafit.toy import make_dune_like_analysis, poisson_fluctuate

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

_OSC_FREE_DEFAULT = ("theta13", "theta23", "deltacp", "dm31")

# theta12/dm21 are fixed at truth (dataclasses.replace(..., fixed=True)) and
# theta13/deltacp get external-constraint-style priors instead of plain
# Flat(). This mirrors real long-baseline practice, not a toy shortcut:
# accelerator experiments like DUNE/T2K/NOvA have essentially zero
# sensitivity to the solar parameters (theta12, dm21) at this L/E and always
# fix them to reactor/solar values, and always combine their own data with
# an external reactor theta13 constraint. It is also load-bearing for
# sampler stability -- verified empirically (see tests/test_endtoend.py's
# module docstring): with all six oscillation parameters left Flat(), NUTS's
# early-warmup trajectories wander into the theta13-deltacp degeneracy
# (only broken by combining FHC+RHC, which even this 4-sample analysis
# leaves only partially resolved once nuisance parameters are also free)
# and produce ~100% divergent transitions, making NUTS impractically slow.
# theta13/deltacp remain genuinely free/fit parameters -- deltacp's
# regularizing prior is far wider than its actual posterior width.
#
# deltacp is `circular=True` (sampled unbounded, identity transform,
# parameters.py), so its regularizing prior must be periodic too: a plain
# Gaussian(mu, sigma) breaks the 2*pi symmetry (it penalizes the
# physically-identical mu+2*pi replica relative to mu), which is exactly
# what let a real RWM chain spend 69% of its draws outside (-pi, pi] while
# a non-periodic prior term quietly fought that branch-hopping instead of
# being indifferent to it. VonMises(mu, kappa) is the periodic analogue;
# kappa = 1/sigma**2 reproduces an existing Gaussian(mu, sigma)
# regularizer's strength in the small-angle limit -- this keeps the same
# ~1.5 rad effective width used throughout this validation ladder.
_THETA13_REACTOR_SIGMA = 0.0075
_DELTACP_REGULARIZING_SIGMA = 1.5  # rad -- effective width, for reference/docs only
_DELTACP_REGULARIZING_KAPPA = 1.0 / _DELTACP_REGULARIZING_SIGMA**2  # ~= 0.444


def _harden_parameter_set(parameter_set: ParameterSet) -> ParameterSet:
    specs = []
    for spec in parameter_set.specs:
        if spec.name in ("theta12", "dm21"):
            specs.append(dataclasses.replace(spec, fixed=True))
        elif spec.name == "theta13":
            specs.append(dataclasses.replace(spec, prior=Gaussian(spec.init, _THETA13_REACTOR_SIGMA)))
        elif spec.name == "deltacp":
            specs.append(dataclasses.replace(spec, prior=VonMises(spec.init, _DELTACP_REGULARIZING_KAPPA)))
        else:
            specs.append(spec)
    return ParameterSet(tuple(specs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="smoke run: small analysis, short chains (~5-6 min)")
    p.add_argument("--seed", type=int, default=0, help="master seed for the toy generator and samplers")
    p.add_argument("--outdir", default=FIGDIR, help="directory for saved figures and the HDF5 round-trip demo")
    return p.parse_args()


def _config(quick: bool) -> dict:
    # Chain lengths are calibrated against measured per-draw cost on this
    # exact toy (correlated binned likelihood, not a textbook Gaussian):
    # NUTS ~1-1.5s/draw (2 chains vmapped, gradient + tree search through
    # the full forward model), RWM ~1-2ms/step (gradient-free), MAP
    # ~50-70ms/L-BFGS-step, and the (eager, unjitted) Laplace Hessian
    # ~5-8s/free-parameter. See tests/test_endtoend.py's module docstring
    # for the full measurement/rationale -- DESIGN.md's suggested "~300
    # warmup, ~500 draws" would take >15 minutes on a model this size.
    if quick:
        return dict(
            n_events_per_sample=1500, n_spline_systs=2, n_norm_systs=1,
            map_steps=250, n_warmup_nuts=25, n_samples_nuts=35,
            n_warmup_rwm=300, n_samples_rwm=300, thinning_rwm=5,
        )
    return dict(
        n_events_per_sample=20000, n_spline_systs=6, n_norm_systs=2,
        map_steps=800, n_warmup_nuts=80, n_samples_nuts=100,
        n_warmup_rwm=1000, n_samples_rwm=1000, thinning_rwm=8,
    )


# ---------------------------------------------------------------------------
# component classification, for the stacked-spectrum plot -- reconstructed
# from public EventSample fields alone (flavor codes + anti sorting), no
# extra metadata needed: matches bananafit.toy's own signal/wrongsign/
# intrinsic/nc convention (module docstring there).
# ---------------------------------------------------------------------------


def _classify_components(sm: SampleModel) -> np.ndarray:
    ev = sm.events
    flavor_from = np.asarray(ev.flavor_from)
    anti = np.zeros(ev.n_events, dtype=bool)
    anti[ev.anti_split:] = True

    is_intrinsic = flavor_from == 0
    is_numu_like = flavor_from == 1

    nominal_anti = bool(np.mean(anti[is_numu_like]) < 0.5) if np.any(is_numu_like) else False
    is_signal = is_numu_like & (anti == nominal_anti)
    is_wrongsign = is_numu_like & (anti != nominal_anti)

    comp = np.full(ev.n_events, "nc", dtype=object)
    comp[is_intrinsic] = "intrinsic"
    comp[is_wrongsign] = "wrongsign"
    comp[is_signal] = "signal"
    return comp


_COMPONENT_ORDER = ("nc", "wrongsign", "intrinsic", "signal")
_COMPONENT_COLORS = {
    "nc": "#9e9e9e", "wrongsign": "#f4a259", "intrinsic": "#8ecae6", "signal": "#219ebc",
}


def _component_spectra(model, sm: SampleModel, physical) -> dict:
    """``{component: (n_bins,) mu}`` for one sample at a physical point,
    computed from ``ForwardModel.event_weights`` + a plain numpy
    ``bincount`` -- independent of ``ForwardModel.expected``'s own
    ``segment_sum`` (a cheap cross-check as well as the plotting tool)."""
    w = np.asarray(model.event_weights(physical, sm.events.name))
    reco_bin = np.asarray(sm.events.reco_bin)
    n_bins = sm.events.n_bins
    comp = _classify_components(sm)
    out = {}
    for c in _COMPONENT_ORDER:
        mask = comp == c
        out[c] = np.bincount(reco_bin[mask], weights=w[mask], minlength=n_bins + 1)[:n_bins]
    return out


# ---------------------------------------------------------------------------
# figure (a): prefit vs postfit stacked spectra + data points, per sample
# ---------------------------------------------------------------------------


def plot_spectra(model, samples, parameter_set, prefit_physical, postfit_physical, data, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    offset = 0
    for ax, sm in zip(axes.ravel(), samples):
        n_bins = sm.events.n_bins
        edges = np.asarray(sm.events.bin_edges) if sm.events.bin_edges is not None else np.arange(n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        pre = _component_spectra(model, sm, prefit_physical)
        post = _component_spectra(model, sm, postfit_physical)
        d = np.asarray(data)[offset:offset + n_bins]
        offset += n_bins

        bottom = np.zeros(n_bins)
        for c in _COMPONENT_ORDER:
            ax.bar(centers, post[c], width=widths, bottom=bottom, color=_COMPONENT_COLORS[c],
                   label=c, alpha=0.85, edgecolor="none")
            bottom = bottom + post[c]

        prefit_total = sum(pre[c] for c in _COMPONENT_ORDER)
        ax.step(edges, np.append(prefit_total, prefit_total[-1]), where="post",
                color="black", linestyle="--", linewidth=1.3, label="prefit (nominal)")

        ax.errorbar(centers, d, yerr=np.sqrt(np.maximum(d, 1.0)), fmt="o", color="black",
                    markersize=3, linewidth=1, label="data (fluctuated)")

        ax.set_title(sm.events.name)
        ax.set_xlabel("reco energy [GeV]")
        ax.set_ylabel("events / bin")
    axes.ravel()[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("prefit (dashed) vs postfit (stacked) spectra, with fluctuated data")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# circular-parameter wrap helpers, shared by every plotting function below.
#
# A circular parameter (deltacp) is sampled unbounded (parameters.py) -- a
# branch-hopping chain's raw draws can straddle many +-2*pi replicas
# (measured directly on the large-scale RWM run: 69% of draws landed outside
# (-pi, pi]). Naively histogramming/plotting the raw values, or wrapping
# them into a *fixed* (-pi, pi] window, either smears one physical mode
# across the full x-range or splits a mode that happens to straddle the
# wrap boundary into two visually separate lobes. Wrapping into the window
# centered on the draws' own circular mean instead moves the cut to sit
# opposite the mode, giving a single clean peak -- see
# examples/figures/large/deltacp_wrapped_rwm.png for the reference figure.
# ---------------------------------------------------------------------------


def _circular_mean(x: np.ndarray) -> float:
    """Circular mean of angles ``x`` (radians, any shape), via
    ``atan2(mean(sin), mean(cos))`` -- the standard mean-resultant-vector
    definition, insensitive to which +-2*pi replica each draw happens to
    sit in."""
    return float(np.arctan2(np.sin(x).mean(), np.cos(x).mean()))


def _wrap_around(x, mu: float):
    """Wrap ``x`` (array or python scalar) into ``(mu - pi, mu + pi]`` --
    same recipe as ``bananafit.parameters._wrap_pi``, just re-centered on
    ``mu`` instead of ``0``."""
    return np.mod(np.asarray(x, dtype=float) - mu + np.pi, 2.0 * np.pi) - np.pi + mu


# ---------------------------------------------------------------------------
# figure (b): hand-rolled corner plot of the free oscillation parameters
# ---------------------------------------------------------------------------


def plot_corner(run, parameter_set, truth, names, path, label="NUTS"):
    constrain = jax.vmap(jax.vmap(parameter_set.constrain))
    physical = np.asarray(constrain(run.positions))  # (chains, samples, n_params)
    idx = [parameter_set.names.index(n) for n in names]
    pooled = physical.reshape(-1, physical.shape[-1])[:, idx].copy()

    # wrap circular parameters (e.g. deltacp) around their own posterior
    # circular mean before plotting, and wrap truth into the same window so
    # the dashed truth line stays on the visible mode.
    circular_names = set(getattr(parameter_set, "circular_free_names", ()))
    truth_wrapped = dict(truth)
    for j, name in enumerate(names):
        if name not in circular_names:
            continue
        mu = _circular_mean(pooled[:, j])
        pooled[:, j] = _wrap_around(pooled[:, j], mu)
        if name in truth_wrapped:
            truth_wrapped[name] = float(_wrap_around(truth_wrapped[name], mu))

    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), constrained_layout=True)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(pooled[:, i], bins=40, color="#219ebc", alpha=0.8)
                ax.axvline(truth_wrapped[names[i]], color="black", linestyle="--", linewidth=1)
            else:
                ax.scatter(pooled[:, j], pooled[:, i], s=2, alpha=0.15, color="#219ebc", edgecolor="none")
                ax.axvline(truth_wrapped[names[j]], color="black", linestyle="--", linewidth=0.8)
                ax.axhline(truth_wrapped[names[i]], color="black", linestyle="--", linewidth=0.8)
            if i == n - 1:
                ax.set_xlabel(names[j])
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))   # dm31 tick labels overlapped
                ax.tick_params(axis="x", labelrotation=30)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i])
    fig.suptitle(f"{label} posterior: free oscillation parameters\n(dashed = truth; circular "
                 "params wrapped around their posterior circular mean)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure (c): trace plots
# ---------------------------------------------------------------------------


def plot_trace(run, parameter_set, truth, names, path):
    constrain = jax.vmap(jax.vmap(parameter_set.constrain))
    physical = np.asarray(constrain(run.positions)).copy()  # (chains, samples, n_params)
    idx = [parameter_set.names.index(n) for n in names]

    circular_names = set(getattr(parameter_set, "circular_free_names", ()))
    truth_wrapped = dict(truth)
    circular_mus = {}
    for name, i in zip(names, idx):
        if name not in circular_names:
            continue
        mu = _circular_mean(physical[..., i])
        circular_mus[name] = mu
        physical[..., i] = _wrap_around(physical[..., i], mu)
        if name in truth_wrapped:
            truth_wrapped[name] = float(_wrap_around(truth_wrapped[name], mu))

    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, name, i in zip(axes, names, idx):
        for c in range(physical.shape[0]):
            ax.plot(physical[c, :, i], linewidth=0.6, alpha=0.8, label=f"chain {c}" if name == names[0] else None)
        ax.axhline(truth_wrapped[name], color="black", linestyle="--", linewidth=1)
        label = f"{name} (wrapped)" if name in circular_mus else name
        ax.set_ylabel(label)
    axes[-1].set_xlabel("draw")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("NUTS trace plots (dashed = truth; circular params wrapped around their "
                 "posterior circular mean)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure (d): NUTS vs RWM ESS/second bar chart -- the headline number
# ---------------------------------------------------------------------------


def plot_ess_per_second(nuts_ess, nuts_time, rwm_ess, rwm_time, names, path):
    x = np.arange(len(names))
    width = 0.35
    nuts_rate = np.array([nuts_ess[n] / nuts_time for n in names])
    rwm_rate = np.array([rwm_ess[n] / rwm_time for n in names])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(x - width / 2, nuts_rate, width, label="NUTS (gradient-based)", color="#219ebc")
    ax.bar(x + width / 2, rwm_rate, width, label="RWM (gradient-free)", color="#f4a259")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("effective samples / second")
    ax.set_yscale("log")
    ax.set_title("NUTS vs RWM: sampling efficiency (the differentiability payoff)")
    ax.legend()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_analysis / load_analysis round-trip demo
# ---------------------------------------------------------------------------


def demo_save_load_roundtrip(samples, parameter_set, model, truth, outdir):
    path = os.path.join(outdir, "dune_toy_analysis.h5")
    sample_order = [sm.events.name for sm in samples]

    to_save = {}
    for sm in samples:
        osc = sm.osc
        osc_attrs = {
            "baseline_km": float(osc.baseline_km), "density_gcc": float(osc.density_gcc), "ye": float(osc.ye),
        }
        to_save[sm.events.name] = (sm.events, sm.splines, sm.norms, osc_attrs)

    save_analysis(path, to_save)
    loaded, _yaml = load_analysis(path)

    rebuilt = []
    for name in sample_order:
        events, splines, norms, osc_attrs = loaded[name]
        osc = ConstantMatterOsc(**osc_attrs)
        rebuilt.append(SampleModel(events=events, splines=splines, norms=norms, osc=osc))
    rebuilt_model = build_forward_model(rebuilt, parameter_set)

    physical = np.asarray([truth[n] for n in parameter_set.names])
    mu_orig, _ = model.expected(physical)
    mu_rebuilt, _ = rebuilt_model.expected(physical)
    max_diff = float(np.max(np.abs(np.asarray(mu_orig) - np.asarray(mu_rebuilt))))
    print(f"[io round-trip] saved -> {path}, reloaded {len(loaded)} samples, "
          f"max|mu_orig - mu_reloaded| = {max_diff:.3e}")
    return path, max_diff


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = _config(args.quick)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== bananafit DUNE-like toy fit ({'quick' if args.quick else 'full'} mode) ===")
    print(f"config: {cfg}\n")

    t0 = time.perf_counter()
    samples, parameter_set, truth = make_dune_like_analysis(
        n_events_per_sample=cfg["n_events_per_sample"], n_spline_systs=cfg["n_spline_systs"],
        n_norm_systs=cfg["n_norm_systs"], seed=args.seed,
    )
    parameter_set = _harden_parameter_set(parameter_set)
    model = build_forward_model(samples, parameter_set)
    t_build = time.perf_counter() - t0

    n_events_total = sum(sm.events.n_events for sm in samples)
    per_sample_counts = {sm.events.name: sm.events.n_events for sm in samples}

    prefit_physical = np.asarray(parameter_set.constrain(parameter_set.init_z))
    data_asimov = asimov_data(model, parameter_set, overrides=truth)
    data_fluct = poisson_fluctuate(data_asimov, seed=args.seed + 1)
    posterior = make_posterior(model, data_fluct, parameter_set, stat="poisson")

    t0 = time.perf_counter()
    ld0 = float(jax.jit(posterior.logdensity)(posterior.z0).block_until_ready())
    t_compile = time.perf_counter() - t0
    print(f"build: {t_build:.2f}s | first-call (compile+eval): {t_compile:.2f}s | "
          f"logdensity(z0)={ld0:.2f}")

    t0 = time.perf_counter()
    z_map, ld_map = map_fit(posterior.logdensity, posterior.z0, n_steps=cfg["map_steps"])
    t_map = time.perf_counter() - t0
    postfit_physical = np.asarray(parameter_set.constrain(z_map))
    print(f"MAP fit: {t_map:.2f}s ({cfg['map_steps']} L-BFGS steps, "
          f"{1e3 * t_map / cfg['map_steps']:.2f} ms/step), logdensity(z_map)={float(ld_map):.2f}")

    laplace_cov = laplace_covariance(posterior.logdensity, z_map)
    laplace_sigma = np.sqrt(np.diag(np.asarray(laplace_cov)))

    t0 = time.perf_counter()
    nuts = run_mcmc(
        posterior.logdensity, z_map, algorithm="nuts", n_chains=2,
        n_warmup=cfg["n_warmup_nuts"], n_samples=cfg["n_samples_nuts"], seed=args.seed,
    )
    t_nuts = time.perf_counter() - t0
    n_nuts_steps = cfg["n_warmup_nuts"] + cfg["n_samples_nuts"]
    print(f"NUTS: {t_nuts:.2f}s for {nuts.positions.shape[0]} chains x {n_nuts_steps} steps "
          f"({1e3 * t_nuts / (nuts.positions.shape[0] * n_nuts_steps):.3f} ms/step)")

    t0 = time.perf_counter()
    rwm = run_mcmc(
        posterior.logdensity, z_map, algorithm="rwm", n_chains=2,
        n_warmup=cfg["n_warmup_rwm"], n_samples=cfg["n_samples_rwm"], thinning=cfg["thinning_rwm"],
        seed=args.seed, proposal_cov=laplace_cov,
    )
    t_rwm = time.perf_counter() - t0
    n_rwm_steps = cfg["n_warmup_rwm"] + cfg["n_samples_rwm"] * cfg["thinning_rwm"]
    print(f"RWM: {t_rwm:.2f}s for {rwm.positions.shape[0]} chains x {n_rwm_steps} steps "
          f"({1e3 * t_rwm / (rwm.positions.shape[0] * n_rwm_steps):.3f} ms/step)\n")

    free_names = list(parameter_set.free_names)
    r = rhat(nuts, names=free_names)
    e_nuts = ess(nuts, names=free_names)
    e_rwm = ess(rwm, names=free_names)
    n_divergent = int(np.sum(np.asarray(nuts.info["is_divergent"])))
    nuts_acc = float(np.mean(np.asarray(nuts.info["acceptance_rate"])))
    rwm_acc = float(np.mean(np.asarray(rwm.info["acceptance_rate"])))

    # -- figures ----------------------------------------------------------
    plot_spectra(model, samples, parameter_set, prefit_physical, postfit_physical, data_fluct,
                 os.path.join(args.outdir, "spectra_prefit_postfit.png"))

    osc_free = [n for n in _OSC_FREE_DEFAULT if n in free_names]
    plot_corner(nuts, parameter_set, truth, osc_free, os.path.join(args.outdir, "corner_osc_params.png"))
    plot_trace(nuts, parameter_set, truth, osc_free, os.path.join(args.outdir, "trace_osc_params.png"))
    plot_ess_per_second(e_nuts, t_nuts, e_rwm, t_rwm, free_names,
                         os.path.join(args.outdir, "ess_per_second.png"))
    print(f"figures saved to {args.outdir}/")

    # -- arviz round trip (sanity: to_arviz works on a real analysis run) --
    idata = to_arviz(nuts, parameter_set=parameter_set)
    print(f"to_arviz: built InferenceData with {len(list(idata.posterior.data_vars))} variables\n")

    # -- HDF5 save_analysis/load_analysis round trip -----------------------
    demo_save_load_roundtrip(samples, parameter_set, model, truth, args.outdir)

    # -- final summary block -----------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"events per sample: {per_sample_counts}  (total {n_events_total})")
    print(f"total bins: {model.total_bins}  |  free parameters: {parameter_set.n_free}")
    print(f"timings: build={t_build:.2f}s  compile+first-call={t_compile:.2f}s  "
          f"map={t_map:.2f}s  nuts={t_nuts:.2f}s  rwm={t_rwm:.2f}s")
    print(f"NUTS: acceptance={nuts_acc:.3f}  divergences={n_divergent}  "
          f"max rhat={max(r.values()):.4f}  min ess={min(e_nuts.values()):.1f}")
    print(f"RWM:  acceptance={rwm_acc:.3f}  min ess={min(e_rwm.values()):.1f}")
    total_nuts_ess_per_s = sum(e_nuts.values()) / t_nuts
    total_rwm_ess_per_s = sum(e_rwm.values()) / t_rwm
    print(f"HEADLINE ESS/second: NUTS={total_nuts_ess_per_s:.1f}  RWM={total_rwm_ess_per_s:.1f}  "
          f"(NUTS/RWM speedup = {total_nuts_ess_per_s / total_rwm_ess_per_s:.2f}x)")

    print(f"\n{'name':>15s} {'truth':>12s} {'prefit':>12s} {'MAP':>12s} "
          f"{'NUTS mean':>12s} {'NUTS std':>10s} {'laplace sig':>12s} {'rhat':>7s} {'ess':>8s}")
    constrain = jax.vmap(jax.vmap(parameter_set.constrain))
    physical_samples = np.asarray(constrain(nuts.positions)).reshape(-1, parameter_set.n_params)
    for name in parameter_set.names:
        i = parameter_set.names.index(name)
        row_truth = truth.get(name, float("nan"))
        row_pre = prefit_physical[i]
        row_map = postfit_physical[i]
        if name in free_names:
            j = free_names.index(name)
            mean = float(physical_samples[:, i].mean())
            std = float(physical_samples[:, i].std(ddof=1))
            lap = laplace_sigma[j]
            rhat_v = r[name]
            ess_v = e_nuts[name]
            print(f"{name:>15s} {row_truth:12.4g} {row_pre:12.4g} {row_map:12.4g} "
                  f"{mean:12.4g} {std:10.4g} {lap:12.4g} {rhat_v:7.4f} {ess_v:8.1f}")
        else:
            print(f"{name:>15s} {row_truth:12.4g} {row_pre:12.4g} {row_map:12.4g} "
                  f"{'(fixed)':>12s} {'':>10s} {'':>12s} {'':>7s} {'':>8s}")
    print("=" * 78)


if __name__ == "__main__":
    main()
