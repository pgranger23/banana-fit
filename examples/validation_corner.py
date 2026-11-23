#!/usr/bin/env python3
"""README validation figure: one posterior, three estimates (NUTS, random-walk Metropolis, Laplace).

The model is the end-to-end test's toy (``tests/test_endtoend.py``): the two FHC samples of
``make_dune_like_analysis`` (2,500 MC events each, 3 spline + 1 normalisation systematics, 8 free
parameters), a reactor prior on theta13, a wide von Mises prior on delta_CP, Asimov data at the
true point. The test runs short chains to stay within CI time; this script runs long ones, so the
figure shows converged posteriors:

  - NUTS, seeded with the MAP and the Laplace covariance (``warmup="step-size"``);
  - random-walk Metropolis with the Laplace covariance as proposal;
  - the Laplace approximation itself (a Gaussian at the MAP).

    python examples/validation_corner.py            # sample (about 20 minutes on CPU), then plot
    python examples/validation_corner.py --plot     # redraw from the saved chains

Chains go to ``examples/figures/validation/`` (not committed); R-hat and ESS to
``examples/figures/validation/stats.json``; the figure to ``examples/figures/readme_corner.png``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import bananafit  # noqa: F401,E402 -- float64
from bananafit.diagnostics import ess, rhat  # noqa: E402
from bananafit.model import build_forward_model  # noqa: E402
from bananafit.parameters import Gaussian, ParameterSet, VonMises  # noqa: E402
from bananafit.posterior import asimov_data, make_posterior  # noqa: E402
from bananafit.sampling import laplace_covariance, map_fit, run_mcmc  # noqa: E402
from bananafit.toy import make_dune_like_analysis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import corner_plot  # noqa: E402

OUT = os.path.join(HERE, "figures", "validation")
FIG = os.path.join(HERE, "figures", "readme_corner.png")
OSC = ("theta13", "theta23", "deltacp", "dm31")
N_CHAINS = 4
NUTS = dict(n_warmup=300, n_samples=2500)
RWM = dict(n_warmup=4000, n_samples=20000, thinning=5)


def build():
    """Same construction as ``tests/test_endtoend.py::_build_small_fhc_analysis``."""
    samples_all, pset_all, truth = make_dune_like_analysis(
        n_events_per_sample=2500, n_spline_systs=3, n_norm_systs=1, seed=0)
    fhc = [sm for sm in samples_all if sm.events.name.startswith("FHC_")]
    used = set()
    for sm in fhc:
        for block in (sm.splines, sm.norms):
            if block is not None:
                used.update(block.names)
    specs = []
    for spec in pset_all.specs:
        if spec.name in ("theta12", "dm21"):
            specs.append(dataclasses.replace(spec, fixed=True))
        elif spec.name == "theta13":
            specs.append(dataclasses.replace(spec, prior=Gaussian(spec.init, 0.0075)))
        elif spec.name == "deltacp":
            specs.append(dataclasses.replace(spec, prior=VonMises(spec.init, 1.0 / 1.5**2)))
        elif spec.name in ("theta23", "dm31") or spec.name in used:
            specs.append(spec)
    pset = ParameterSet(tuple(specs))
    model = build_forward_model(fhc, pset)
    data = asimov_data(model, pset, overrides=truth)
    return pset, model, make_posterior(model, data, pset, stat="poisson"), truth


def sample():
    os.makedirs(OUT, exist_ok=True)
    pset, model, post, truth = build()
    t0 = time.perf_counter()
    z_map, _ = map_fit(post.logdensity, post.z0, n_steps=800)
    cov = laplace_covariance(post.logdensity, z_map)
    np.save(os.path.join(OUT, "z_map.npy"), np.asarray(z_map))
    np.save(os.path.join(OUT, "cov.npy"), np.asarray(cov))
    print(f"MAP + Laplace: {time.perf_counter() - t0:.0f}s", flush=True)
    timing = {}
    t0 = time.perf_counter()
    run = run_mcmc(post.logdensity, z_map, algorithm="nuts", n_chains=N_CHAINS, seed=1,
                   inverse_mass_matrix=cov, warmup="step-size", **NUTS)
    np.save(os.path.join(OUT, "nuts_positions.npy"), np.asarray(run.positions))
    np.save(os.path.join(OUT, "nuts_divergent.npy"), np.asarray(run.info["is_divergent"]))
    timing["nuts_s"] = time.perf_counter() - t0
    print(f"NUTS: {timing['nuts_s']:.0f}s", flush=True)
    t0 = time.perf_counter()
    run = run_mcmc(post.logdensity, z_map, algorithm="rwm", n_chains=N_CHAINS, seed=2,
                   proposal_cov=cov, **RWM)
    np.save(os.path.join(OUT, "rwm_positions.npy"), np.asarray(run.positions))
    timing["rwm_s"] = time.perf_counter() - t0
    print(f"RWM: {timing['rwm_s']:.0f}s", flush=True)
    with open(os.path.join(OUT, "timing.json"), "w") as fh:
        json.dump(timing, fh, indent=1)


def plot():
    pset, _model, _post, truth = build()
    names = list(pset.free_names)
    idx = {nm: pset.names.index(nm) for nm in OSC}
    constrain = jax.jit(jax.vmap(pset.constrain))
    stats, phys = {}, {}
    for alg in ("nuts", "rwm"):
        pos = np.load(os.path.join(OUT, f"{alg}_positions.npy"))
        run = SimpleNamespace(positions=jnp.asarray(pos))
        stats[alg] = {"n_chains": int(pos.shape[0]), "n_draws": int(pos.shape[1]),
                      "rhat": {k: float(v) for k, v in rhat(run, names=names, parameter_set=pset).items()},
                      "ess": {k: float(v) for k, v in ess(run, names=names, parameter_set=pset).items()}}
        phys[alg] = np.asarray(constrain(jnp.asarray(pos.reshape(-1, pos.shape[-1]))))
    div = np.load(os.path.join(OUT, "nuts_divergent.npy"))
    stats["nuts"]["divergent_fraction"] = float(div.mean())
    z_map, cov = np.load(os.path.join(OUT, "z_map.npy")), np.load(os.path.join(OUT, "cov.npy"))
    draws = np.random.default_rng(3).multivariate_normal(z_map, cov, size=200_000)
    phys["laplace"] = np.asarray(constrain(jnp.asarray(draws)))
    with open(os.path.join(OUT, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    for alg in ("nuts", "rwm"):
        s = stats[alg]
        print(f"{alg}: {s['n_chains']} x {s['n_draws']} draws, max R-hat {max(s['rhat'].values()):.3f}, "
              f"min ESS {min(s['ess'].values()):.0f}", flush=True)

    mu = float(np.arctan2(np.sin(phys["nuts"][:, idx["deltacp"]]).mean(),
                          np.cos(phys["nuts"][:, idx["deltacp"]]).mean()))

    def show(x, nm):
        x = np.asarray(x, dtype=float)
        if nm in ("theta13", "theta23"):
            return np.sin(x) ** 2
        if nm == "deltacp":
            return np.mod(x - mu + np.pi, 2 * np.pi) - np.pi + mu
        return 1e3 * x if nm == "dm31" else x

    cols = {k: [show(v[:, idx[nm]], nm) for nm in OSC] for k, v in phys.items()}
    labels = [r"$\sin^2\theta_{13}$", r"$\sin^2\theta_{23}$", r"$\delta_\mathrm{CP}$ (rad)",
              r"$\Delta m^2_{31}$ ($10^{-3}$ eV$^2$)"]
    sn, sr = stats["nuts"], stats["rwm"]
    stats_text = (f"NUTS: {sn['n_chains']} x {sn['n_draws']:,} draws\n"
                  f"random walk: {sr['n_chains']} x {sr['n_draws']:,} draws (every {RWM['thinning']}th step)\n"
                  f"all 8 parameters, both samplers:\n"
                  f"R-hat <= {max(max(sn['rhat'].values()), max(sr['rhat'].values())):.3f}, "
                  f"ESS >= {min(min(sn['ess'].values()), min(sr['ess'].values())):,.0f}")
    layers = [
        dict(cols=cols["nuts"], label="NUTS", color="#2a78d6", style="filled"),
        dict(cols=cols["rwm"], label="random-walk Metropolis", color="#1baf7a", style="line"),
        dict(cols=cols["laplace"], label="Laplace approximation", color="#eb6834", style="line", linestyle="--"),
    ]
    corner_plot.corner(layers, [float(show(truth[nm], nm)) for nm in OSC], labels,
                       "DUNE-like toy (FHC samples, 8 free parameters), Asimov data: "
                       "three estimates of one posterior", FIG, stats_text=stats_text)
    print("wrote", FIG, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="only redraw from the saved chains")
    if not ap.parse_args().plot:
        sample()
    plot()
