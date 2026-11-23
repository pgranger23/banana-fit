"""Tests for the ND correlated-bin likelihood:
bananafit.likelihood.gaussian_cov_loglike / build_nd_covariance."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.likelihood import build_nd_covariance, gaussian_cov_loglike


# --- gaussian_cov_loglike vs hand-computed 2x2 ------------------------------


def test_gaussian_cov_loglike_hand_computed_2x2():
    pred = jnp.array([5.0, 3.0])
    data = jnp.array([4.0, 3.5])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    cov_inv = np.linalg.inv(cov)

    resid = np.array([5.0 - 4.0, 3.0 - 3.5])
    expected = -0.5 * resid @ cov_inv @ resid

    got = float(gaussian_cov_loglike(pred, data, jnp.asarray(cov_inv)))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_gaussian_cov_loglike_diagonal_matches_independent_gaussians():
    # A diagonal cov_inv should reduce to a sum of independent -0.5*(r/sigma)^2 terms.
    pred = jnp.array([5.0, 3.0, 10.0])
    data = jnp.array([4.0, 3.5, 9.0])
    sigmas = np.array([1.0, 2.0, 0.5])
    cov_inv = jnp.asarray(np.diag(1.0 / sigmas**2))

    got = float(gaussian_cov_loglike(pred, data, cov_inv))
    resid = np.asarray(pred) - np.asarray(data)
    expected = -0.5 * np.sum((resid / sigmas) ** 2)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_gaussian_cov_loglike_zero_at_zero_residual():
    pred = jnp.array([1.0, 2.0, 3.0])
    cov_inv = jnp.eye(3)
    assert float(gaussian_cov_loglike(pred, pred, cov_inv)) == 0.0


# --- build_nd_covariance matches the note's formula -------------------------


def test_build_nd_covariance_matches_formula():
    rng = np.random.default_rng(0)
    n = 5
    nominal = rng.uniform(10.0, 100.0, size=n)
    # a small, valid fractional-covariance matrix (symmetric, diag dominant enough)
    a = rng.normal(scale=0.02, size=(n, n))
    frac_cov = 0.05 * np.eye(n) + 0.5 * (a + a.T)

    cov, cov_inv = build_nd_covariance(frac_cov, nominal)

    expected_cov = frac_cov * np.outer(nominal, nominal)
    expected_cov = 0.5 * (expected_cov + expected_cov.T)
    expected_cov[np.diag_indices(n)] += nominal

    np.testing.assert_allclose(np.asarray(cov), expected_cov, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(cov) @ np.asarray(cov_inv), np.eye(n), rtol=1e-8, atol=1e-8
    )


def test_build_nd_covariance_adds_extra_mc_stat_diagonal():
    n = 4
    nominal = np.array([10.0, 20.0, 30.0, 40.0])
    frac_cov = 0.01 * np.eye(n)
    extra = np.array([1.0, 2.0, 3.0, 4.0])

    cov_no_extra, _ = build_nd_covariance(frac_cov, nominal)
    cov_extra, _ = build_nd_covariance(frac_cov, nominal, add_mc_stat_var=extra)

    diag_diff = np.diag(np.asarray(cov_extra)) - np.diag(np.asarray(cov_no_extra))
    np.testing.assert_allclose(diag_diff, extra, rtol=1e-10, atol=1e-10)
    # off-diagonal entries are untouched by the extra diagonal term
    off_no_extra = np.asarray(cov_no_extra) - np.diag(np.diag(np.asarray(cov_no_extra)))
    off_extra = np.asarray(cov_extra) - np.diag(np.diag(np.asarray(cov_extra)))
    np.testing.assert_allclose(off_no_extra, off_extra, rtol=1e-10, atol=1e-10)


def test_build_nd_covariance_shape_mismatch_raises():
    nominal = np.array([1.0, 2.0, 3.0])
    bad_frac_cov = np.eye(2)
    with pytest.raises(ValueError, match="frac_cov"):
        build_nd_covariance(bad_frac_cov, nominal)


def test_build_nd_covariance_non_pd_raises():
    # A frac_cov with a large negative off-diagonal correlation swamping the
    # diagonal, on a nominal small enough that adding it back doesn't rescue
    # positive-definiteness.
    nominal = np.array([1.0, 1.0])
    frac_cov = np.array([[0.01, -5.0], [-5.0, 0.01]])
    with pytest.raises(ValueError, match="positive-definite"):
        build_nd_covariance(frac_cov, nominal)


# --- gradients are finite ---------------------------------------------------


def test_gaussian_cov_loglike_grad_finite():
    rng = np.random.default_rng(1)
    n = 6
    nominal = rng.uniform(5.0, 50.0, size=n)
    a = rng.normal(scale=0.01, size=(n, n))
    frac_cov = 0.02 * np.eye(n) + 0.5 * (a + a.T)
    _, cov_inv = build_nd_covariance(frac_cov, nominal)
    data = jnp.asarray(nominal * 1.05)

    def f(pred):
        return gaussian_cov_loglike(pred, data, cov_inv)

    grad = jax.grad(f)(jnp.asarray(nominal))
    assert np.all(np.isfinite(np.asarray(grad)))

    # finite-difference cross-check
    eps = 1e-6
    pred0 = np.asarray(nominal)
    for i in range(n):
        pp, pm = pred0.copy(), pred0.copy()
        pp[i] += eps
        pm[i] -= eps
        fd = (float(f(jnp.asarray(pp))) - float(f(jnp.asarray(pm)))) / (2 * eps)
        np.testing.assert_allclose(float(grad[i]), fd, rtol=1e-5, atol=1e-6)


def test_gaussian_cov_loglike_jit_matches_eager():
    rng = np.random.default_rng(2)
    n = 4
    nominal = rng.uniform(5.0, 50.0, size=n)
    frac_cov = 0.02 * np.eye(n)
    _, cov_inv = build_nd_covariance(frac_cov, nominal)
    data = jnp.asarray(nominal * 0.9)
    pred = jnp.asarray(nominal)

    eager = gaussian_cov_loglike(pred, data, cov_inv)
    jitted = jax.jit(gaussian_cov_loglike)(pred, data, cov_inv)
    np.testing.assert_allclose(float(jitted), float(eager), rtol=1e-12)
