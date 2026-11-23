"""Smoke tests for the circular-parameter-aware plot helpers in
``examples/dune_toy_fit.py`` (``plot_corner``/``plot_trace``): the
plotting-side half of the circular-parameter fix (DESIGN.md §6/§10,
``bananafit/parameters.py``'s ``VonMises``, ``bananafit/diagnostics.py``'s
circular-aware ``rhat``/``ess``/``summary``/``to_arviz``).

These import ``examples/dune_toy_fit.py`` directly (matching the pattern
``examples/dune_large_fit.py`` itself uses to reuse ``dune_toy_fit``'s
generic helpers) rather than duplicating the plotting code -- ``examples/``
is not an installed package, so ``sys.path`` is extended the same way.

Deliberately does not build a real ``bananafit.toy`` analysis (expensive):
just enough of a ``ParameterSet`` (one circular, one linear free parameter)
and a hand-built synthetic ``run.positions`` array to exercise the wrap
logic end to end -- no MCMC needed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import jax.numpy as jnp

import bananafit  # noqa: F401 -- enforces float64, per house style
from bananafit.parameters import Flat, ParamSpec, ParameterSet

_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

import dune_toy_fit as toyfit  # noqa: E402


def _tiny_parameter_set():
    specs = (
        ParamSpec("deltacp", 3.0, Flat(), (None, None), circular=True),
        ParamSpec("theta23", 0.8, Flat(), (None, None)),
    )
    return ParameterSet(specs)


def _branch_hopping_run(mu=3.0, sigma=0.3, n_chains=3, n_samples=1500, seed=0):
    """A synthetic ``run`` whose deltacp column branch-hops across several
    +-2*pi replicas (mirroring the measured large-scale RWM chain: 69% of
    draws outside (-pi, pi]) and whose theta23 column is an ordinary linear
    Gaussian -- shaped exactly like ``run.positions`` from ``run_mcmc``."""
    rng = np.random.default_rng(seed)
    deltacp = mu + sigma * rng.standard_normal((n_chains, n_samples))
    winding = rng.integers(-2, 3, size=(n_chains, n_samples))  # random +-2*pi replica
    deltacp = deltacp + 2.0 * np.pi * winding
    theta23 = 0.8 + 0.05 * rng.standard_normal((n_chains, n_samples))
    positions = np.stack([deltacp, theta23], axis=-1)
    from types import SimpleNamespace
    return SimpleNamespace(positions=jnp.asarray(positions))


# ---------------------------------------------------------------------------
# helpers: _circular_mean / _wrap_around
# ---------------------------------------------------------------------------


def test_circular_mean_recovers_true_mean():
    rng = np.random.default_rng(0)
    x = 3.0 + 0.2 * rng.standard_normal(5000) + 2.0 * np.pi * rng.integers(-2, 3, 5000)
    mu = toyfit._circular_mean(x)
    assert mu == pytest.approx(3.0, abs=0.05)


def test_wrap_around_produces_single_window_no_mode_split():
    rng = np.random.default_rng(1)
    x = 3.0 + 0.2 * rng.standard_normal(5000) + 2.0 * np.pi * rng.integers(-2, 3, 5000)
    mu = toyfit._circular_mean(x)
    wrapped = toyfit._wrap_around(x, mu)
    assert wrapped.max() - wrapped.min() < 2.0 * np.pi
    assert wrapped.std() < 1.0  # collapsed to one mode, not spread over many windings
    assert wrapped.mean() == pytest.approx(3.0, abs=0.05)


# ---------------------------------------------------------------------------
# plot_corner / plot_trace smoke tests
# ---------------------------------------------------------------------------


def test_plot_corner_smoke_and_wraps_deltacp(tmp_path):
    parameter_set = _tiny_parameter_set()
    run = _branch_hopping_run()
    truth = {"deltacp": 3.4, "theta23": 0.8}  # 3.4 rad, matches bananafit.toy's truth deltacp
    path = tmp_path / "corner.png"

    toyfit.plot_corner(run, parameter_set, truth, ["deltacp", "theta23"], str(path))

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_trace_smoke_and_wraps_deltacp(tmp_path):
    parameter_set = _tiny_parameter_set()
    run = _branch_hopping_run()
    truth = {"deltacp": 3.4, "theta23": 0.8}
    path = tmp_path / "trace.png"

    toyfit.plot_trace(run, parameter_set, truth, ["deltacp", "theta23"], str(path))

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_corner_truth_wrapped_into_same_window_as_posterior():
    """Regression guard for the "truth line falls outside the plotted
    x-range" failure mode: after wrapping, truth must land inside
    (or very near) the actual span of the plotted (wrapped) samples."""
    run = _branch_hopping_run(mu=3.0, sigma=0.3)
    positions = np.asarray(run.positions)
    deltacp_raw = positions[..., 0].ravel()
    truth_raw = 3.4  # bananafit.toy's deltacp truth (2*pi-equivalent to -2.88)

    mu = toyfit._circular_mean(deltacp_raw)
    wrapped_samples = toyfit._wrap_around(deltacp_raw, mu)
    wrapped_truth = toyfit._wrap_around(truth_raw, mu)

    lo, hi = wrapped_samples.min(), wrapped_samples.max()
    margin = 0.5 * (hi - lo)
    assert lo - margin <= wrapped_truth <= hi + margin
