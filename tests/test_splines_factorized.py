"""Tests for bananafit.splines' factorized (grid-cell) index mode -- the
``event_cell`` + ``cell_to_curve`` alternative to the dense ``event_index``,
added for the DUNE ND ingestion (memory-frugal spline index at ~1e7-1e8
event scale, see splines.py module docstring "Factorized (grid-cell) index
mode").

Mirrors ``tests/test_splines.py``'s style/coverage but focuses on what's new:
equivalence to the dense path (the main correctness property -- factorized
is purely a storage optimization, not a different physical model), plus
jit/grad/error-path checks for the new builder and dataclass fields.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bananafit import splines

_KNOT_GRID = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def _make_dense_and_factorized(seed=0, n_syst=3, n_cells=5, n_events=37, weight_clip=(None, None)):
    """Build two tables -- one dense, one factorized -- that describe the
    *same* per-event curve assignment, so any downstream comparison of the
    two is a same-physics equivalence check, not just a shape check.

    Cell ``c``'s curve for systematic ``s`` is ``cell_to_curve[s, c]``; each
    event is randomly assigned to one of the ``n_cells`` cells, shared
    across all systematics (mirroring the real ND use case: a
    ``(mode, Etrue, Erec, Yrec)`` grid cell each event lands in once).
    """
    rng = np.random.default_rng(seed)
    knot_x = np.tile(_KNOT_GRID, (n_syst, 1))
    n_curves_in = n_cells  # up to n_cells distinct non-identity curves/syst
    knot_y = 1.0 + 0.3 * rng.standard_normal(size=(n_syst, n_curves_in, _KNOT_GRID.size))

    # cell -> curve lookup; include 0 (identity) as a real possibility, like
    # the real "this systematic doesn't touch this mode/bin" case.
    cell_to_curve = rng.integers(0, n_curves_in + 1, size=(n_syst, n_cells)).astype(np.int32)
    event_cell = rng.integers(0, n_cells, size=n_events).astype(np.int32)

    # dense event_index is the same mapping, just gathered out to (n_syst, N)
    # in plain numpy (independent of the jax gather the factorized path uses
    # at call time) -- an independent re-derivation, not the same code path.
    event_index = cell_to_curve[:, event_cell]  # (n_syst, N)

    dense = splines.build_spline_table(
        knot_x, knot_y, event_index, boundary="natural", dtype=jnp.float64,
        weight_clip=weight_clip,
    )
    factorized = splines.build_spline_table_factorized(
        knot_x, knot_y, cell_to_curve, event_cell, boundary="natural", dtype=jnp.float64,
        weight_clip=weight_clip,
    )
    return dense, factorized, knot_x, knot_y, cell_to_curve, event_cell


# --- equivalence to the dense path -------------------------------------------


def test_factorized_matches_dense_weights_per_syst():
    dense, factorized, *_ = _make_dense_and_factorized()
    x = jnp.array([0.4, -1.7, 2.3], dtype=jnp.float64)
    w_dense = np.asarray(dense.weights_per_syst(x))
    w_fact = np.asarray(factorized.weights_per_syst(x))
    np.testing.assert_allclose(w_fact, w_dense, rtol=1e-12, atol=1e-12)


def test_factorized_matches_dense_call():
    dense, factorized, *_ = _make_dense_and_factorized(weight_clip=(0.0, None))
    x = jnp.array([1.1, -0.3, 0.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(factorized(x)), np.asarray(dense(x)), rtol=1e-12, atol=1e-12
    )


def test_factorized_matches_dense_across_many_random_x():
    dense, factorized, *_ = _make_dense_and_factorized(seed=7, n_syst=4, n_cells=9, n_events=123)
    rng = np.random.default_rng(1)
    for _ in range(10):
        x = jnp.asarray(rng.uniform(-4, 4, size=4), dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(factorized(x)), np.asarray(dense(x)), rtol=1e-10, atol=1e-10
        )


# --- identity curve / curve-0 semantics preserved ----------------------------


def test_factorized_identity_curve_is_exactly_one():
    _, factorized, *_ = _make_dense_and_factorized()
    for xv in [-10.0, -3.0, 0.0, 2.2, 10.0]:
        # find an event whose cell maps to curve 0 for systematic 0
        cell_to_curve = np.asarray(factorized.cell_to_curve)
        event_cell = np.asarray(factorized.event_cell)
        identity_cells = np.nonzero(cell_to_curve[0] == 0)[0]
        if identity_cells.size == 0:
            pytest.skip("no identity cell for systematic 0 in this random draw")
        ev = np.nonzero(np.isin(event_cell, identity_cells))[0]
        if ev.size == 0:
            pytest.skip("no event landed on an identity cell for systematic 0")
        w = np.asarray(factorized.weights_per_syst(jnp.array([xv, 0.0, 0.0], dtype=jnp.float64)))
        np.testing.assert_allclose(w[0, ev[0]], 1.0, rtol=0, atol=0)


# --- jit / vmap / grad --------------------------------------------------------


def test_factorized_jit_matches_eager():
    dense, factorized, *_ = _make_dense_and_factorized()
    x = jnp.array([0.4, -0.7, 1.2], dtype=jnp.float64)
    w_eager = factorized(x)
    w_jit = jax.jit(factorized.__call__)(x)
    np.testing.assert_allclose(np.asarray(w_jit), np.asarray(w_eager), rtol=1e-12)

    def call(tbl, xv):
        return tbl(xv)

    w_jit2 = jax.jit(call)(factorized, x)
    np.testing.assert_allclose(np.asarray(w_jit2), np.asarray(w_eager), rtol=1e-12)


def test_factorized_grad_matches_dense_grad():
    dense, factorized, *_ = _make_dense_and_factorized(weight_clip=(None, None))
    x = jnp.array([0.4, -0.7, 1.9], dtype=jnp.float64)

    def f(tbl, xv):
        return jnp.sum(tbl(xv))

    g_dense = np.asarray(jax.grad(f, argnums=1)(dense, x))
    g_fact = np.asarray(jax.grad(f, argnums=1)(factorized, x))
    assert np.all(np.isfinite(g_fact))
    np.testing.assert_allclose(g_fact, g_dense, rtol=1e-10, atol=1e-10)


def test_factorized_grad_matches_finite_difference():
    _, factorized, *_ = _make_dense_and_factorized(weight_clip=(None, None))
    x = jnp.array([0.4, -0.7, 1.9], dtype=jnp.float64)

    def f(xv):
        return jnp.sum(factorized(xv))

    g = np.asarray(jax.grad(f)(x))
    assert np.all(np.isfinite(g))

    eps = 1e-6
    x_np = np.asarray(x)
    fd = np.zeros(3)
    for i in range(3):
        xp, xm = x_np.copy(), x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_factorized_vmap_safe():
    _, factorized, *_ = _make_dense_and_factorized(seed=3)
    rng = np.random.default_rng(2)
    xs = jnp.asarray(rng.uniform(-3, 3, size=(6, 3)), dtype=jnp.float64)
    w_vmap = np.asarray(jax.vmap(factorized)(xs))
    w_loop = np.stack([np.asarray(factorized(xs[i])) for i in range(xs.shape[0])])
    np.testing.assert_allclose(w_vmap, w_loop, rtol=1e-12)


# --- pytree structure: None fields carry no leaves ---------------------------


def test_factorized_and_dense_are_distinct_pytree_structures():
    dense, factorized, *_ = _make_dense_and_factorized()
    assert dense.event_cell is None and dense.cell_to_curve is None
    assert dense.event_index is not None
    assert factorized.event_index is None
    assert factorized.event_cell is not None and factorized.cell_to_curve is not None

    leaves_dense, _ = jax.tree_util.tree_flatten(dense)
    leaves_fact, _ = jax.tree_util.tree_flatten(factorized)
    # both have real leaves (not e.g. accidentally empty)
    assert len(leaves_dense) > 0
    assert len(leaves_fact) > 0


# --- errors --------------------------------------------------------------------


def test_rejects_out_of_range_cell_to_curve():
    knot_x = np.array([[0.0, 1.0, 2.0]])
    knot_y = np.array([[[1.0, 0.5, 0.05]]])
    with pytest.raises(ValueError):
        splines.build_spline_table_factorized(
            knot_x, knot_y,
            cell_to_curve=np.array([[0, 2]], dtype=np.int32),  # only curve 0/1 valid
            event_cell=np.array([0, 1], dtype=np.int32),
        )


def test_rejects_out_of_range_event_cell():
    knot_x = np.array([[0.0, 1.0, 2.0]])
    knot_y = np.array([[[1.0, 0.5, 0.05]]])
    with pytest.raises(ValueError):
        splines.build_spline_table_factorized(
            knot_x, knot_y,
            cell_to_curve=np.array([[0, 1]], dtype=np.int32),  # n_cells=2
            event_cell=np.array([0, 5], dtype=np.int32),  # 5 is out of range
        )


def test_effective_event_index_raises_if_neither_representation_set():
    knot_x = np.array([[0.0, 1.0, 2.0]])
    knot_y = np.array([[[1.0, 0.5, 0.05]]])
    coeffs = splines._fit_coeffs(knot_x, knot_y, "natural", ("s0",))
    empty = splines.SplineTable(
        names=("s0",),
        knots=jnp.asarray(knot_x),
        coeffs=jnp.asarray(coeffs, dtype=jnp.float32),
    )
    with pytest.raises(ValueError):
        empty.weights_per_syst(jnp.array([0.0], dtype=jnp.float64))


def test_standalone_import_exposes_factorized_builder():
    from bananafit import splines as _s

    assert hasattr(_s, "build_spline_table_factorized")


# --- factorized-path backward fix (never materializes (n_syst, N)) ----------
#
# See splines.py's "Factorized hot path" module-docstring section: the ND
# analysis's factorized SplineTable used to reconstruct a dense (n_syst, N)
# curve-index array at call time (both forward and backward), which is both
# the ND-grad pathology (55 simultaneous skewed scatters, same shape as the
# already-fixed dense path's curve-0 collision, but not covered by
# _curve_gather's fix since the composition never went through it at (n_syst,
# N) scale) and the joint FD+ND OOM (that reconstructed array alone is
# ~10.9GB at ND's 49.5M-event scale). These tests exercise the new
# `_factorized_event_weight`/`_cell_event_product` composition (used by
# `SplineTable.__call__` for factorized tables) directly against (a) a
# from-scratch numpy reference, (b) the dense path, and (c) central finite
# differences -- including a deliberately skewed cell distribution (mimicking
# the real ND data's "most events map to curve 0" pattern) and a case with an
# exact-zero curve value (the leave-one-out backward derivation is
# division-free specifically to stay correct there).


def _make_skewed_dense_and_factorized(
    seed=0, n_syst=12, n_cells=40, n_events=5000, identity_frac=0.85, weight_clip=(None, None)
):
    """Like `_make_dense_and_factorized`, but with a skewed cell_to_curve (most
    (syst, cell) pairs map to curve 0 = identity, matching the skew
    measured on realistic data, median ~72%/worst ~99.9%) -- the shape that
    used to be pathological for the factorized path once composed with the
    dense (n_syst, N) reconstruction.
    """
    rng = np.random.default_rng(seed)
    knot_x = np.tile(_KNOT_GRID, (n_syst, 1))
    n_curves_in = n_cells
    knot_y = 1.0 + 0.3 * rng.standard_normal(size=(n_syst, n_curves_in, _KNOT_GRID.size))

    cell_to_curve = np.where(
        rng.random(size=(n_syst, n_cells)) < identity_frac,
        0,
        rng.integers(1, n_curves_in + 1, size=(n_syst, n_cells)),
    ).astype(np.int32)
    event_cell = rng.integers(0, n_cells, size=n_events).astype(np.int32)
    event_index = cell_to_curve[:, event_cell]

    dense = splines.build_spline_table(
        knot_x, knot_y, event_index, boundary="natural", dtype=jnp.float64,
        weight_clip=weight_clip,
    )
    factorized = splines.build_spline_table_factorized(
        knot_x, knot_y, cell_to_curve, event_cell, boundary="natural", dtype=jnp.float64,
        weight_clip=weight_clip,
    )
    return dense, factorized


def test_factorized_skewed_call_matches_dense():
    dense, factorized = _make_skewed_dense_and_factorized()
    x = jnp.asarray(np.linspace(-2.5, 2.5, 12), dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(factorized(x)), np.asarray(dense(x)), rtol=1e-10, atol=1e-10
    )


def test_factorized_skewed_grad_matches_dense_grad():
    """The core regression test for the backward fix: gradient of the
    factorized path, on a skewed (mostly-curve-0) cell_to_curve, must match
    the dense path's gradient exactly (both are the same exact math -- see
    the module docstring's "Factorized hot path" derivation)."""
    dense, factorized = _make_skewed_dense_and_factorized(seed=3, n_syst=20, n_cells=64, n_events=8000)
    x = jnp.asarray(np.linspace(-1.8, 1.8, 20), dtype=jnp.float64)

    def f(tbl, xv):
        return jnp.sum(tbl(xv) ** 2)  # nonlinear reduction so every event's weight matters

    g_dense = np.asarray(jax.grad(f, argnums=1)(dense, x))
    g_fact = np.asarray(jax.grad(f, argnums=1)(factorized, x))
    assert np.all(np.isfinite(g_fact))
    np.testing.assert_allclose(g_fact, g_dense, rtol=1e-9, atol=1e-9)


def test_factorized_skewed_grad_matches_finite_difference():
    _, factorized = _make_skewed_dense_and_factorized(seed=5, n_syst=15, n_cells=50, n_events=4000)
    x0 = jnp.asarray(np.linspace(-1.5, 1.5, 15), dtype=jnp.float64)

    def f(xv):
        return jnp.sum(factorized(xv) ** 2)

    g = np.asarray(jax.grad(f)(x0))
    assert np.all(np.isfinite(g))

    eps = 1e-6
    x_np = np.asarray(x0)
    fd = np.zeros(x_np.shape[0])
    for i in range(x_np.shape[0]):
        xp, xm = x_np.copy(), x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_cell_event_product_matches_naive_einsum_reference():
    """Unit-level check of `_cell_event_product`/`_factorized_event_weight`'s
    forward value against a completely independent (no shared code) numpy
    reference -- catches a bug in the cell-product/gather restructuring
    itself, isolated from the surrounding SplineTable machinery."""
    rng = np.random.default_rng(11)
    n_syst, n_cells, n_events = 9, 30, 2000
    curve_val = rng.uniform(0.2, 2.0, size=(n_syst, n_cells))
    event_cell = rng.integers(0, n_cells, size=n_events).astype(np.int32)

    ref = np.prod(curve_val[:, event_cell], axis=0)  # naive (n_syst, N) reference
    got = np.asarray(
        splines._cell_event_product(jnp.asarray(curve_val), jnp.asarray(event_cell))
    )
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_cell_event_product_grad_matches_finite_difference_with_a_zero_curve_value():
    """The leave-one-out backward derivation is division-free specifically so
    it stays correct when a curve value is exactly zero (where W[c]/curve_val
    would be a 0/0). Force one such entry and check the gradient is still
    finite and matches central finite differences."""
    rng = np.random.default_rng(22)
    n_syst, n_cells, n_events = 6, 10, 500
    curve_val0 = rng.uniform(0.5, 1.5, size=(n_syst, n_cells))
    curve_val0[2, 3] = 0.0  # force an exact zero
    event_cell = rng.integers(0, n_cells, size=n_events).astype(np.int32)
    # make sure at least one event actually lands on the zero cell so it's
    # not a vacuous test
    event_cell[0] = 3

    def f(cv):
        w = splines._cell_event_product(cv, jnp.asarray(event_cell))
        return jnp.sum(w * w)

    g = np.asarray(jax.grad(f)(jnp.asarray(curve_val0)))
    assert np.all(np.isfinite(g))

    eps = 1e-6
    fd = np.zeros_like(curve_val0)
    for s in range(n_syst):
        for c in range(n_cells):
            cp, cm = curve_val0.copy(), curve_val0.copy()
            cp[s, c] += eps
            cm[s, c] -= eps
            fd[s, c] = (
                float(f(jnp.asarray(cp))) - float(f(jnp.asarray(cm)))
            ) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_factorized_call_never_builds_dense_index_at_call_time(monkeypatch):
    """Guard against a future regression silently reintroducing the
    (n_syst, N) reconstruction into the __call__ hot path: patch
    `_effective_event_index` (the dense-reconstruction helper) to raise if
    invoked, and confirm `__call__` on a factorized table still works
    (i.e. it never calls that helper) while `weights_per_syst` (the
    documented debug-only path, unaffected by this fix) still does."""
    _, factorized = _make_skewed_dense_and_factorized(seed=1, n_syst=5, n_cells=12, n_events=300)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 5), dtype=jnp.float64)

    def _boom(self):
        raise AssertionError("__call__ must not reconstruct the dense (n_syst, N) index")

    monkeypatch.setattr(splines.SplineTable, "_effective_event_index", _boom)
    w = factorized(x)  # must NOT raise
    assert np.all(np.isfinite(np.asarray(w)))

    monkeypatch.undo()
    # weights_per_syst is documented as still using the dense reconstruction
    # (debug-only path, not memory-frugal) -- sanity-check that claim too.
    ws = factorized.weights_per_syst(x)
    assert ws.shape == (5, 300)
