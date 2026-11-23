"""Tests for bananafit.norms (DESIGN.md Section 5)."""

import jax
import jax.numpy as jnp
import numpy as np

from bananafit import norms


def test_masked_events_scaled_unmasked_untouched():
    mask = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    ns = norms.NormSet(names=("a", "b"), mask=jnp.asarray(mask))
    x = jnp.array([1.5, 0.7])
    w = np.asarray(ns(x))
    expect = np.array([1.5, 1.5, 0.7, 0.7, 1.0, 1.0])
    np.testing.assert_allclose(w, expect, rtol=1e-12)


def test_from_selections_matches_manual_mask():
    n = 5
    sel_a = np.array([True, True, False, False, False])
    sel_b = np.array([False, False, True, True, True])
    ns = norms.from_selections(["a", "b"], [sel_a, sel_b], n)
    assert ns.names == ("a", "b")
    assert ns.mask.shape == (2, n)

    x = jnp.array([2.0, 0.5])
    w = np.asarray(ns(x))
    expect = np.array([2.0, 2.0, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(w, expect, rtol=1e-12)


def test_fractional_mask():
    mask = np.array([[0.5, 1.0, 0.0]], dtype=np.float32)
    ns = norms.NormSet(names=("a",), mask=jnp.asarray(mask))
    x = jnp.array([3.0])
    w = np.asarray(ns(x))
    expect = np.array([1.0 + (3.0 - 1.0) * 0.5, 3.0, 1.0])
    np.testing.assert_allclose(w, expect, rtol=1e-6)


def test_prior_mean_one_is_identity():
    mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    ns = norms.NormSet(names=("a", "b"), mask=jnp.asarray(mask))
    w = np.asarray(ns(jnp.array([1.0, 1.0])))
    np.testing.assert_allclose(w, np.ones(3), rtol=1e-12)


def test_grad_matches_analytic():
    mask = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    ns = norms.NormSet(names=("a",), mask=jnp.asarray(mask))

    def f(x):
        return jnp.sum(ns(x))

    g = jax.grad(f)(jnp.array([1.3]))
    # d/dx sum_i (1 + (x-1)*mask_i) = sum_i mask_i, independent of x
    np.testing.assert_allclose(np.asarray(g), [float(np.sum(mask))], rtol=1e-10)

    eps = 1e-6
    fd = (float(f(jnp.array([1.3 + eps]))) - float(f(jnp.array([1.3 - eps])))) / (2 * eps)
    np.testing.assert_allclose(float(g[0]), fd, rtol=1e-6)


def test_multi_norm_grad_finite_and_correct():
    rng = np.random.default_rng(0)
    n_norm, n = 4, 20
    mask = rng.uniform(0.0, 1.0, size=(n_norm, n)).astype(np.float32)
    ns = norms.NormSet(names=tuple(f"n{i}" for i in range(n_norm)), mask=jnp.asarray(mask))
    x0 = jnp.array([1.1, 0.9, 1.05, 0.8])

    def f(x):
        return jnp.sum(ns(x))

    g = np.asarray(jax.grad(f)(x0))
    assert np.all(np.isfinite(g))

    eps = 1e-6
    x_np = np.asarray(x0)
    fd = np.zeros(n_norm)
    for i in range(n_norm):
        xp, xm = x_np.copy(), x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_jit_matches_unjitted():
    mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ns = norms.NormSet(names=("a", "b"), mask=jnp.asarray(mask))
    x = jnp.array([1.2, 0.8])
    w_ref = ns(x)
    w_jit = jax.jit(ns.__call__)(x)
    np.testing.assert_allclose(np.asarray(w_jit), np.asarray(w_ref), rtol=1e-12)


def test_wrong_length_x_raises():
    mask = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    ns = norms.NormSet(names=("a",), mask=jnp.asarray(mask))
    try:
        ns(jnp.array([1.0, 2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched x length")
