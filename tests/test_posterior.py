"""Tests for bananafit.posterior (DESIGN.md Section 9).

Builds the same synthetic 2-sample DUNE-like analysis as test_model.py (via
tests/_toyfixture.py) and exercises the one function the whole package is
built to differentiate: ``BinnedPosterior.logdensity``.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.posterior import asimov_data, make_posterior

from _toyfixture import build_toy_analysis

STATS = ("poisson", "barlow-beeston")


def _toy_posterior(stat="poisson", seed=0):
    toy = build_toy_analysis(seed=seed)
    data = asimov_data(toy.model, toy.parameter_set)
    posterior = make_posterior(toy.model, data, toy.parameter_set, stat=stat)
    return toy, posterior


def _central_diff_grad(f, z, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    grad = np.zeros_like(z)
    for i in range(z.shape[0]):
        zp = z.copy(); zp[i] += eps
        zm = z.copy(); zm[i] -= eps
        fp = float(f(jnp.asarray(zp)))
        fm = float(f(jnp.asarray(zm)))
        grad[i] = (fp - fm) / (2.0 * eps)
    return grad


def _displaced_z(toy, seed=3, scale=0.3):
    rng = np.random.default_rng(seed)
    z0 = np.asarray(toy.parameter_set.init_z)
    return jnp.asarray(z0 + rng.normal(scale=scale, size=z0.shape[0]))


# --- finiteness / jit-vs-eager -----------------------------------------------


@pytest.mark.parametrize("stat", STATS)
def test_logdensity_finite_at_z0(stat):
    toy, posterior = _toy_posterior(stat=stat)
    val = posterior.logdensity(posterior.z0)
    assert np.isfinite(float(val))


@pytest.mark.parametrize("stat", STATS)
def test_jit_matches_eager(stat):
    toy, posterior = _toy_posterior(stat=stat)
    jitted = jax.jit(posterior.logdensity)

    for z in (posterior.z0, _displaced_z(toy)):
        eager = float(posterior.logdensity(z))
        jit_val = float(jitted(z))
        np.testing.assert_allclose(jit_val, eager, rtol=1e-12, atol=1e-12)


# --- gradient vs central finite differences ----------------------------------


@pytest.mark.parametrize("stat", STATS)
def test_grad_matches_finite_differences(stat):
    toy, posterior = _toy_posterior(stat=stat)
    grad_fn = jax.grad(posterior.logdensity)

    for z in (posterior.z0, _displaced_z(toy)):
        grad_ad = np.asarray(grad_fn(z))
        assert np.all(np.isfinite(grad_ad))
        grad_fd = _central_diff_grad(posterior.logdensity, z)
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("stat", STATS)
def test_grad_nonzero_where_physically_expected(stat):
    toy, posterior = _toy_posterior(stat=stat)
    z = _displaced_z(toy)
    grad_ad = np.asarray(jax.grad(posterior.logdensity)(z))

    free_names = toy.parameter_set.free_names
    # theta23 drives numu disappearance -- must move the posterior.
    i_theta23 = free_names.index("theta23")
    assert abs(grad_ad[i_theta23]) > 1e-6

    # every free spline/norm systematic in the toy is genuinely used by at
    # least one sample, so none should be exactly inert either.
    for name in ("flux_shape", "xsec_maqe", "det_smear_nue", "sig_norm_numu", "bkg_norm_numu"):
        i = free_names.index(name)
        assert abs(grad_ad[i]) > 1e-8, f"grad wrt {name!r} unexpectedly ~0"


# --- Asimov consistency -------------------------------------------------


def test_asimov_loglike_gradient_vanishes_at_truth_poisson():
    toy, posterior = _toy_posterior(stat="poisson")
    truth = toy.parameter_set.constrain(toy.parameter_set.init_z)
    grad_physical = np.asarray(jax.grad(posterior.loglike)(truth))
    # n_b == mu_b(truth) exactly (data IS the Asimov mu at truth), so
    # d(loglike)/d(physical) = sum_b (n_b/mu_b - 1) dmu_b/dphysical == 0
    # exactly, for every physical parameter, chain rule or not.
    np.testing.assert_allclose(grad_physical, 0.0, atol=1e-6)


def test_asimov_point_maximizes_logdensity_physical():
    toy, posterior = _toy_posterior(stat="poisson")
    names = toy.parameter_set.names
    truth = np.asarray(toy.parameter_set.constrain(toy.parameter_set.init_z))
    ld_truth = float(posterior.logdensity_physical(jnp.asarray(truth)))

    scale = {
        "theta12": 0.05, "theta13": 0.02, "theta23": 0.05, "deltacp": 0.1,
        "dm21": 1e-6, "dm31": 5e-5,
        "flux_shape": 0.3, "xsec_maqe": 0.3, "det_smear_nue": 0.3,
        "sig_norm_numu": 0.05, "bkg_norm_numu": 0.05,
    }
    scale_vec = np.array([scale[n] for n in names])

    rng = np.random.default_rng(7)
    for _ in range(8):
        perturbed = truth + rng.normal(size=len(names)) * scale_vec
        ld_pert = float(posterior.logdensity_physical(jnp.asarray(perturbed)))
        assert ld_truth >= ld_pert - 1e-8, (
            f"Asimov truth is not the max: truth={ld_truth}, perturbed={ld_pert}"
        )


# --- performance smoke test (informational) ----------------------------------


def test_jit_logdensity_and_grad_performance_smoke():
    toy, posterior = _toy_posterior(stat="poisson")
    logdensity_and_grad = jax.jit(jax.value_and_grad(posterior.logdensity))

    z = posterior.z0
    val, grad = logdensity_and_grad(z)
    val.block_until_ready()
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad)))

    n_calls = 200
    start = time.perf_counter()
    for _ in range(n_calls):
        val, grad = logdensity_and_grad(z)
    val.block_until_ready()
    elapsed = time.perf_counter() - start

    n_events_total = sum(sm.events.n_events for sm in toy.samples)
    print(
        f"\n[perf smoke] jitted logdensity+grad: {1e3 * elapsed / n_calls:.4f} ms/call "
        f"over {n_calls} calls ({n_events_total} events, {toy.model.total_bins} bins, "
        f"{toy.parameter_set.n_free} free params)"
    )
