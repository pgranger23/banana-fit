"""Tests for bananafit.likelihood (DESIGN.md Section 9)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as sp_poisson

import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.likelihood import barlow_beeston_lite_loglike, poisson_loglike


# --- poisson_loglike vs scipy (up to the n-only log(n!) constant) -------------


def test_poisson_matches_scipy_up_to_constant():
    rng = np.random.default_rng(0)
    n = jnp.asarray(rng.integers(0, 20, size=8).astype(np.float64))
    mu1 = jnp.asarray(rng.uniform(0.5, 15.0, size=8))
    mu2 = jnp.asarray(rng.uniform(0.5, 15.0, size=8))

    diff_ours = float(poisson_loglike(mu1, n) - poisson_loglike(mu2, n))
    diff_scipy = float(
        np.sum(sp_poisson.logpmf(np.asarray(n), np.asarray(mu1)))
        - np.sum(sp_poisson.logpmf(np.asarray(n), np.asarray(mu2)))
    )
    np.testing.assert_allclose(diff_ours, diff_scipy, rtol=1e-10, atol=1e-10)


def test_poisson_n_zero_bin_contributes_zero_log_term():
    mu = jnp.array([5.0])
    n = jnp.array([0.0])
    # d/dmu poisson_loglike at n=0 is exactly -1 (no log(mu) contribution).
    grad = jax.grad(lambda m: poisson_loglike(m, n))(mu)
    np.testing.assert_allclose(float(grad[0]), -1.0, atol=1e-12)


def test_poisson_grad_finite_at_tiny_mu():
    mu = jnp.array([0.0, 1e-10, 1e-5])
    n = jnp.array([0.0, 3.0, 0.0])
    grad = jax.grad(lambda m: poisson_loglike(m, n))(mu)
    assert np.all(np.isfinite(np.asarray(grad)))


# --- BB-lite with w2=0 equals poisson exactly ----------------------------------


def test_bb_lite_w2_zero_equals_poisson_exactly():
    rng = np.random.default_rng(1)
    mu = jnp.asarray(rng.uniform(1e-8, 20.0, size=10))
    n = jnp.asarray(rng.integers(0, 25, size=10).astype(np.float64))
    w2 = jnp.zeros(10)

    bb = barlow_beeston_lite_loglike(mu, w2, n)
    pl = poisson_loglike(mu, n)
    assert bb == pl  # exact (not merely close): the w2=0 branch reuses the
    # identical mu_floored/log/linear-term computation as poisson_loglike.


# --- beta solves the quadratic --------------------------------------------------


def _beta_from_formula(mu, w2, n):
    mu, w2, n = float(mu), float(w2), float(n)
    sigma2 = w2 / mu**2
    b = mu * sigma2 - 1.0
    D = b**2 + 4.0 * n * sigma2
    return (-b + np.sqrt(D)) / 2.0, sigma2


@pytest.mark.parametrize(
    "mu,w2,n",
    [(5.0, 2.0, 4.0), (1.0, 0.01, 0.0), (20.0, 50.0, 25.0), (0.3, 0.05, 1.0)],
)
def test_beta_solves_quadratic(mu, w2, n):
    beta, sigma2 = _beta_from_formula(mu, w2, n)
    resid = beta**2 + (mu * sigma2 - 1.0) * beta - n * sigma2
    assert abs(resid) < 1e-9
    assert beta >= 0.0


# --- BB-lite loglike vs independent numerical profile over beta ----------------


def _neg_loglike_beta(beta, mu, w2, n):
    if beta <= 0:
        return 1e10
    sigma2 = w2 / mu**2
    return -(
        n * np.log(beta * mu) - beta * mu - (beta - 1.0) ** 2 / (2.0 * sigma2)
    )


@pytest.mark.parametrize(
    "mu,w2,n",
    [
        (5.0, 2.0, 4.0),
        (20.0, 50.0, 25.0),
        (100.0, 5.0, 90.0),
        (2.0, 0.001, 5.0),
        (0.5, 0.2, 0.0),
    ],
)
def test_bb_lite_matches_numerical_beta_profile(mu, w2, n):
    res = minimize_scalar(
        _neg_loglike_beta,
        bounds=(1e-10, 1e3),
        args=(mu, w2, n),
        method="bounded",
        options={"xatol": 1e-13},
    )
    ll_num = -res.fun
    ll_ours = float(
        barlow_beeston_lite_loglike(jnp.array([mu]), jnp.array([w2]), jnp.array([n]))
    )
    np.testing.assert_allclose(ll_ours, ll_num, rtol=1e-6, atol=1e-6)


# --- gradients: finite everywhere, match finite differences --------------------


def _fd_grad_wrt_mu(f, mu, w2, n, i, h_rel=1e-6, h_abs_floor=1e-6):
    # Absolute floor on the step, not a further-scaled-down relative one:
    # for a tiny mu (e.g. 1e-8, well below mc_floor) a purely relative step
    # collapses to ~1e-12-1e-14, far below the double-precision noise floor
    # of an O(1-100) function value -- the finite difference would measure
    # cancellation error, not the derivative.
    mu = np.asarray(mu, dtype=np.float64)
    h = max(h_rel * abs(mu[i]), h_abs_floor)
    mup, mum = mu.copy(), mu.copy()
    mup[i] += h
    mum[i] -= h
    return (f(jnp.asarray(mup), w2, n) - f(jnp.asarray(mum), w2, n)) / (2 * h)


def test_bb_lite_grad_finite_including_n_zero_tiny_mu_tiny_w2():
    mu = jnp.array([3.2, 0.0, 1e-8, 50.0, 5.0])
    w2 = jnp.array([1.0, 0.0, 1e-10, 0.0, 1e-9])
    n = jnp.array([3.0, 0.0, 0.0, 48.0, 4.0])

    grad = jax.grad(lambda m: barlow_beeston_lite_loglike(m, w2, n))(mu)
    assert np.all(np.isfinite(np.asarray(grad)))

    for i in range(len(mu)):
        fd = float(_fd_grad_wrt_mu(barlow_beeston_lite_loglike, mu, w2, n, i))
        np.testing.assert_allclose(float(grad[i]), fd, rtol=1e-3, atol=1e-3)


def test_bb_lite_grad_matches_finite_differences_generic_points():
    rng = np.random.default_rng(2)
    mu = jnp.asarray(rng.uniform(0.5, 30.0, size=6))
    w2 = jnp.asarray(rng.uniform(0.01, 5.0, size=6))
    n = jnp.asarray(rng.integers(0, 20, size=6).astype(np.float64))

    grad = jax.grad(lambda m: barlow_beeston_lite_loglike(m, w2, n))(mu)
    for i in range(6):
        fd = float(_fd_grad_wrt_mu(barlow_beeston_lite_loglike, mu, w2, n, i))
        np.testing.assert_allclose(float(grad[i]), fd, rtol=1e-4, atol=1e-5)


def test_poisson_grad_matches_finite_differences():
    rng = np.random.default_rng(3)
    mu = jnp.asarray(rng.uniform(0.5, 30.0, size=6))
    n = jnp.asarray(rng.integers(0, 20, size=6).astype(np.float64))

    grad = jax.grad(lambda m: poisson_loglike(m, n))(mu)
    mu_np = np.asarray(mu)
    for i in range(6):
        h = 1e-6 * max(abs(mu_np[i]), 1e-6)
        mup, mum = mu_np.copy(), mu_np.copy()
        mup[i] += h
        mum[i] -= h
        fd = (
            float(poisson_loglike(jnp.asarray(mup), n))
            - float(poisson_loglike(jnp.asarray(mum), n))
        ) / (2 * h)
        np.testing.assert_allclose(float(grad[i]), fd, rtol=1e-4, atol=1e-5)


# --- jit -------------------------------------------------------------------------


def test_poisson_and_bb_lite_under_jit_match_eager():
    mu = jnp.array([3.2, 0.5, 10.0, 0.0])
    w2 = jnp.array([1.0, 0.2, 0.0, 0.0])
    n = jnp.array([3.0, 0.0, 11.0, 0.0])

    p_eager = poisson_loglike(mu, n)
    p_jit = jax.jit(poisson_loglike)(mu, n)
    np.testing.assert_allclose(float(p_jit), float(p_eager), rtol=1e-12)

    bb_eager = barlow_beeston_lite_loglike(mu, w2, n)
    bb_jit = jax.jit(barlow_beeston_lite_loglike)(mu, w2, n)
    np.testing.assert_allclose(float(bb_jit), float(bb_eager), rtol=1e-12)


def test_bb_lite_grad_under_jit_matches_eager():
    mu = jnp.array([3.2, 0.5, 10.0, 1e-6])
    w2 = jnp.array([1.0, 0.2, 0.0, 1e-8])
    n = jnp.array([3.0, 0.0, 11.0, 0.0])

    grad_eager = jax.grad(lambda m: barlow_beeston_lite_loglike(m, w2, n))(mu)
    grad_jit = jax.jit(jax.grad(lambda m: barlow_beeston_lite_loglike(m, w2, n)))(mu)
    np.testing.assert_allclose(np.asarray(grad_jit), np.asarray(grad_eager), rtol=1e-10)
