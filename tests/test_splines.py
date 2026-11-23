"""Tests for bananafit.splines (DESIGN.md Section 4)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PchipInterpolator

from bananafit import splines


# --- helpers -----------------------------------------------------------------

_KNOT_GRID = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def _make_curve_ys(rng, n_curves, monotone):
    if monotone:
        # strictly increasing per curve, suitable for PchipInterpolator
        steps = rng.uniform(0.05, 0.3, size=(n_curves, _KNOT_GRID.size))
        y = 0.5 + np.cumsum(steps, axis=1)
    else:
        y = 1.0 + 0.3 * rng.standard_normal(size=(n_curves, _KNOT_GRID.size))
    return y


def _build(boundary, dtype, n_curves=3, n_syst=1, seed=0, weight_clip=(None, None)):
    rng = np.random.default_rng(seed)
    knot_x = np.tile(_KNOT_GRID, (n_syst, 1))
    knot_y = np.stack(
        [_make_curve_ys(rng, n_curves, boundary == "monotone") for _ in range(n_syst)],
        axis=0,
    )  # (n_syst, n_curves, n_knots)
    n_events = n_curves + 1
    event_index = np.tile(np.arange(n_events), (n_syst, 1)).astype(np.int32)
    table = splines.build_spline_table(
        knot_x,
        knot_y,
        event_index,
        boundary=boundary,
        dtype=dtype,
        weight_clip=weight_clip,
    )
    return table, knot_x, knot_y


def _ref_splines(boundary, x_row, y_syst):
    refs = []
    for c in range(y_syst.shape[0]):
        if boundary == "monotone":
            refs.append(PchipInterpolator(x_row, y_syst[c]))
        else:
            refs.append(CubicSpline(x_row, y_syst[c], bc_type=boundary))
    return refs


_BOUNDARIES = ["natural", "not-a-knot", "monotone"]
_DTYPES = [(jnp.float32, 1e-6), (jnp.float64, 1e-12)]


# --- values & derivatives in-range, vs scipy ----------------------------------


@pytest.mark.parametrize("boundary", _BOUNDARIES)
@pytest.mark.parametrize("dtype,tol", _DTYPES)
def test_values_match_scipy_in_range(boundary, dtype, tol):
    table, knot_x, knot_y = _build(boundary, dtype)
    refs = _ref_splines(boundary, knot_x[0], knot_y[0])
    xs = np.linspace(-3.0, 3.0, 41)
    for xv in xs:
        w = np.asarray(table.weights_per_syst(jnp.array([xv], dtype=jnp.float64))[0])
        expect = np.array([1.0] + [float(r(xv)) for r in refs])
        np.testing.assert_allclose(w, expect, rtol=tol, atol=tol)


@pytest.mark.parametrize("boundary", _BOUNDARIES)
@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 2e-3), (jnp.float64, 1e-8)])
def test_derivatives_match_scipy_in_range(boundary, dtype, tol):
    table, knot_x, knot_y = _build(boundary, dtype)
    refs = _ref_splines(boundary, knot_x[0], knot_y[0])
    for c, r in enumerate(refs, start=1):

        def f(xv, c=c):
            return table.weights_per_syst(xv)[0, c]

        for xv in [-2.7, -1.3, -0.2, 0.9, 2.4]:
            g = jax.grad(f)(jnp.array([xv], dtype=jnp.float64))
            np.testing.assert_allclose(
                float(g[0]), float(r.derivative()(xv)), rtol=tol, atol=tol
            )


# --- out-of-range: manual linear continuation ---------------------------------


@pytest.mark.parametrize("boundary", _BOUNDARIES)
@pytest.mark.parametrize("dtype,tol", _DTYPES)
def test_out_of_range_linear_continuation(boundary, dtype, tol):
    table, knot_x, knot_y = _build(boundary, dtype)
    refs = _ref_splines(boundary, knot_x[0], knot_y[0])
    x_min, x_max = knot_x[0, 0], knot_x[0, -1]
    for xv in [-9.0, -6.0, -3.5, 3.5, 6.0, 9.0]:
        w = np.asarray(table.weights_per_syst(jnp.array([xv], dtype=jnp.float64))[0])
        expect = [1.0]
        for r in refs:
            if xv < x_min:
                v0, s0 = float(r(x_min)), float(r.derivative()(x_min))
                expect.append(v0 + s0 * (xv - x_min))
            else:
                v0, s0 = float(r(x_max)), float(r.derivative()(x_max))
                expect.append(v0 + s0 * (xv - x_max))
        np.testing.assert_allclose(w, expect, rtol=tol, atol=tol)


# --- identity curve & exact knot recovery -------------------------------------


def test_identity_curve_is_exactly_one():
    for boundary in _BOUNDARIES:
        table, knot_x, _ = _build(boundary, jnp.float32)
        for xv in [-10.0, -3.0, -1.0, 0.0, 1.7, 3.0, 10.0]:
            w = table.weights_per_syst(jnp.array([xv], dtype=jnp.float64))[0, 0]
            assert float(w) == 1.0


@pytest.mark.parametrize("dtype,tol", _DTYPES)
def test_value_at_knot_matches_knot_y(dtype, tol):
    table, knot_x, knot_y = _build("natural", dtype)
    for k in range(knot_x.shape[1]):
        xv = knot_x[0, k]
        w = np.asarray(table.weights_per_syst(jnp.array([xv], dtype=jnp.float64))[0])
        expect = np.concatenate([[1.0], knot_y[0, :, k]])
        np.testing.assert_allclose(w, expect, rtol=tol, atol=tol)


# --- product over multiple systematics ----------------------------------------


def test_product_over_systematics_equals_manual_product():
    n_syst = 3
    table, knot_x, knot_y = _build(
        "natural", jnp.float64, n_curves=2, n_syst=n_syst, weight_clip=(None, None)
    )
    x = jnp.array([0.3, -1.2, 2.0], dtype=jnp.float64)
    w = np.asarray(table(x))
    per_syst = np.asarray(table.weights_per_syst(x))  # (n_syst, N)
    manual = np.prod(per_syst, axis=0)
    np.testing.assert_allclose(w, manual, rtol=1e-12)


# --- jit / grad safety ---------------------------------------------------------


def test_jit_matches_unjitted():
    n_syst = 2
    table, _, _ = _build(
        "natural", jnp.float64, n_curves=3, n_syst=n_syst, weight_clip=(None, None)
    )
    x = jnp.array([0.4, -0.7], dtype=jnp.float64)
    w_ref = table(x)
    w_jit = jax.jit(table.__call__)(x)
    np.testing.assert_allclose(np.asarray(w_jit), np.asarray(w_ref), rtol=1e-12)

    # also jit with the table itself passed as a (pytree) argument
    def call(tbl, xv):
        return tbl(xv)

    w_jit2 = jax.jit(call)(table, x)
    np.testing.assert_allclose(np.asarray(w_jit2), np.asarray(w_ref), rtol=1e-12)


def test_grad_matches_central_finite_difference():
    n_syst = 2
    table, _, _ = _build(
        "natural", jnp.float64, n_curves=3, n_syst=n_syst, weight_clip=(None, None)
    )
    x = jnp.array([0.4, -0.7], dtype=jnp.float64)

    def f(xv):
        return jnp.sum(table(xv))

    g = np.asarray(jax.grad(f)(x))
    assert np.all(np.isfinite(g))

    eps = 1e-6
    fd = np.zeros(n_syst)
    x_np = np.asarray(x)
    for i in range(n_syst):
        xp, xm = x_np.copy(), x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "xv",
    [-100.0, -3.0, -1.0, 0.0, 1.0, 3.0, 100.0],
)
def test_grad_finite_at_knots_and_out_of_range(xv):
    table, knot_x, _ = _build(
        "natural", jnp.float64, n_curves=2, n_syst=1, weight_clip=(None, None)
    )

    def f(x1):
        return jnp.sum(table(x1))

    g = jax.grad(f)(jnp.array([xv], dtype=jnp.float64))
    assert np.all(np.isfinite(np.asarray(g)))

    eps = 1e-6
    fd = (float(f(jnp.array([xv + eps]))) - float(f(jnp.array([xv - eps])))) / (2 * eps)
    np.testing.assert_allclose(float(g[0]), fd, rtol=1e-4, atol=1e-6)


# --- weight clip ----------------------------------------------------------------


def test_weight_clip_default_floors_at_zero():
    # a steeply decreasing curve whose linear continuation goes negative
    # far outside the knot range must be clamped at 0 by the default clip.
    knot_x = np.array([[0.0, 1.0, 2.0]])
    knot_y = np.array([[[1.0, 0.5, 0.05]]])
    event_index = np.array([[1, 1]], dtype=np.int32)
    table = splines.build_spline_table(
        knot_x, knot_y, event_index, boundary="natural", dtype=jnp.float64
    )  # default weight_clip=(0.0, None)
    w = table(jnp.array([50.0], dtype=jnp.float64))
    assert float(w[0]) >= 0.0

    unclipped = splines.build_spline_table(
        knot_x,
        knot_y,
        event_index,
        boundary="natural",
        dtype=jnp.float64,
        weight_clip=(None, None),
    )
    w_raw = unclipped(jnp.array([50.0], dtype=jnp.float64))
    assert float(w_raw[0]) < 0.0  # sanity: the un-clamped value really is negative


# --- errors ----------------------------------------------------------------------


def test_rejects_out_of_range_event_index():
    knot_x = np.array([[0.0, 1.0, 2.0]])
    knot_y = np.array([[[1.0, 0.5, 0.05]]])
    with pytest.raises(ValueError):
        splines.build_spline_table(
            knot_x, knot_y, np.array([[0, 2]], dtype=np.int32)
        )


# --- mixed real-knot-count (ragged) systematics --------------------------------
#
# Regression tests for a bug found by the DUNE ingestion work: when a table
# mixes systematics with different real knot counts (NaN-padded to a common
# n_knots), evaluating the *shorter* systematic at/beyond its last real knot
# picked a padding coefficient slot -- built by build_spline_table as a
# verbatim replica of the last real segment's [a,b,c,d], valid only relative
# to *that* segment's start knot -- while pairing it with the (correct, real)
# k0 of the padding slot's own knot. The mismatched reference point corrupted
# both the value there and (via the boundary value/slope used for the linear
# continuation) everything beyond it.


def _build_ragged_pair(dtype=jnp.float64, n_knots_short=4, n_knots_long=6):
    """One systematic with ``n_knots_short`` real knots (NaN-padded to
    ``n_knots_long``), one with ``n_knots_long`` real knots, distinct
    (non-degenerate) curvatures so an indexing bug can't hide behind a
    coincidentally-flat response.
    """
    x_short = np.linspace(-3.0, 1.0, n_knots_short)
    x_long = np.linspace(-3.0, 2.0, n_knots_long)
    knot_x = np.full((2, n_knots_long), np.nan)
    knot_x[0, :n_knots_short] = x_short
    knot_x[1, :] = x_long

    y_short = 1.0 + 0.5 * np.sin(1.3 * x_short) + 0.15 * x_short**2
    y_long = 1.0 + 0.4 * np.cos(1.1 * x_long) - 0.1 * x_long**2

    knot_y = np.full((2, 1, n_knots_long), np.nan)
    knot_y[0, 0, :n_knots_short] = y_short
    knot_y[1, 0, :] = y_long

    event_index = np.array([[1, 1], [1, 1]], dtype=np.int32)
    table = splines.build_spline_table(
        knot_x, knot_y, event_index, boundary="natural", dtype=dtype
    )
    ref_short = CubicSpline(x_short, y_short, bc_type="natural")
    ref_long = CubicSpline(x_long, y_long, bc_type="natural")
    return table, x_short, x_long, ref_short, ref_long


def test_mixed_knot_counts_short_syst_matches_scipy_across_full_range():
    """Table with a 4-knot and a 6-knot systematic; evaluate the shorter
    one's full range, including exactly at and beyond its last real knot.
    This must FAIL pre-fix: the off-by-one clamp corrupts the shorter
    systematic right at/after its last real knot.
    """
    table, x_short, x_long, ref_short, ref_long = _build_ragged_pair()
    x_min, x_max = x_short[0], x_short[-1]

    xs = np.concatenate(
        [
            np.linspace(x_min, x_max, 25),  # full in-range, including x_max exactly
            [x_max, x_max + 1e-9, x_max + 0.5, x_max + 2.0, 5.0],  # at/beyond last real knot
        ]
    )
    for xv in xs:
        w = np.asarray(table.weights_per_syst(jnp.array([xv, 0.0], dtype=jnp.float64))[0])
        if xv <= x_max:
            expect = float(ref_short(xv))
        else:
            v0, s0 = float(ref_short(x_max)), float(ref_short.derivative()(x_max))
            expect = v0 + s0 * (xv - x_max)
        np.testing.assert_allclose(float(w[1]), expect, rtol=1e-10, atol=1e-10)

    # the longer systematic (real knots fill the whole padded width) must be
    # unaffected by the fix -- sanity check against the same evaluation path.
    for xv in np.linspace(x_long[0], x_long[-1], 13):
        w = np.asarray(table.weights_per_syst(jnp.array([0.0, xv], dtype=jnp.float64)))
        np.testing.assert_allclose(float(w[1, 1]), float(ref_long(xv)), rtol=1e-10, atol=1e-10)


def test_mixed_knot_counts_plus2_and_plus3_sigma_differ():
    """The specific symptom reported by the DUNE ingestion work: +2 sigma and
    +3 sigma (both beyond the short systematic's last real knot) must give
    *different* weights for a curved (non-flat-slope) response, not the
    bit-identical values the off-by-one clamp produced.
    """
    table, x_short, _x_long, ref_short, _ref_long = _build_ragged_pair()
    x_max = x_short[-1]
    assert x_max < 2.0  # +2 and +3 sigma are indeed beyond the short syst's real range

    w2 = float(table.weights_per_syst(jnp.array([2.0, 0.0], dtype=jnp.float64))[0, 1])
    w3 = float(table.weights_per_syst(jnp.array([3.0, 0.0], dtype=jnp.float64))[0, 1])
    assert w2 != pytest.approx(w3, rel=0, abs=0)

    v0, s0 = float(ref_short(x_max)), float(ref_short.derivative()(x_max))
    np.testing.assert_allclose(w2, v0 + s0 * (2.0 - x_max), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(w3, v0 + s0 * (3.0 - x_max), rtol=1e-10, atol=1e-10)


def test_mixed_knot_counts_weights_per_syst_and_call_agree_with_scipy():
    """``weights_per_syst`` and ``__call__`` share the same evaluation path
    (``_eval_all_curves``) -- check the product-form ``__call__`` also
    reflects the fix, not just the debug accessor.
    """
    table, x_short, x_long, ref_short, ref_long = _build_ragged_pair()
    x = jnp.array([2.5, 0.7], dtype=jnp.float64)
    w = np.asarray(table(x))

    x_max = x_short[-1]
    v0, s0 = float(ref_short(x_max)), float(ref_short.derivative()(x_max))
    expect_short = v0 + s0 * (2.5 - x_max)
    expect_long = float(ref_long(0.7))
    np.testing.assert_allclose(w[0], expect_short * expect_long, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "dtype,eps,rtol", [(jnp.float64, 1e-6, 1e-4), (jnp.float32, 1e-3, 2e-2)]
)
def test_mixed_knot_counts_grad_matches_finite_difference(dtype, eps, rtol):
    """Grads on both the short and long systematics must be finite and match
    central finite differences, exactly at, and past, the short
    systematic's last real knot -- the hard case for the fix (segment index
    now depends on a data-derived, not static, real-knot count).

    ``eps``/``rtol`` are widened for the float32-storage table: a coefficient
    table rounded to float32 combined with a float64-scale ``eps`` amplifies
    storage rounding into the finite-difference estimate (cancellation, not a
    gradient bug) -- the same reason ``test_derivatives_match_scipy_in_range``
    above uses a looser tolerance for float32 than float64.
    """
    table, x_short, x_long, ref_short, ref_long = _build_ragged_pair(dtype=dtype)
    x_max = x_short[-1]

    def f(xv):
        return jnp.sum(table(xv))

    for xv_short in [x_max - 0.1, x_max, x_max + 1e-9, x_max + 0.7, x_max + 3.0]:
        x = jnp.array([xv_short, 0.4], dtype=jnp.float64)
        g = np.asarray(jax.grad(f)(x))
        assert np.all(np.isfinite(g))

        x_np = np.asarray(x)
        fd = np.zeros(2)
        for i in range(2):
            xp, xm = x_np.copy(), x_np.copy()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * eps)
        np.testing.assert_allclose(g, fd, rtol=rtol, atol=1e-4)


def test_mixed_knot_counts_jit_and_vmap_safe():
    """The fixed clamp (data-derived n_real per systematic) must stay
    jit/vmap-friendly: no python-level branching on traced values.
    """
    table, x_short, x_long, ref_short, ref_long = _build_ragged_pair()
    x_max = x_short[-1]
    xs = jnp.array(
        [[x_max - 0.1, 0.0], [x_max, 0.0], [x_max + 0.001, 0.0], [x_max + 2.0, 0.0]],
        dtype=jnp.float64,
    )

    w_eager = np.asarray(jax.vmap(table)(xs))
    w_jit = np.asarray(jax.jit(jax.vmap(table))(xs))
    np.testing.assert_allclose(w_jit, w_eager, rtol=1e-12)

    long_factor = float(ref_long(0.0))  # second column of xs is fixed at 0.0
    for row, xv in zip(w_eager, xs[:, 0]):
        xv = float(xv)
        if xv <= x_max:
            expect_short = float(ref_short(xv))
        else:
            v0, s0 = float(ref_short(x_max)), float(ref_short.derivative()(x_max))
            expect_short = v0 + s0 * (xv - x_max)
        # both events share curve index 1 for both systematics -- row is a
        # duplicate pair, so any element (here the first) equals the scalar
        # per-systematic-product expectation.
        np.testing.assert_allclose(
            float(row[0]), expect_short * long_factor, rtol=1e-8, atol=1e-8
        )


def test_standalone_import():
    # module must import cleanly without going through bananafit's __init__
    # forcing x64 as a prerequisite of correctness (defensive .astype calls
    # inside splines.py make this true regardless of global x64 state).
    from bananafit import splines as _s

    assert hasattr(_s, "build_spline_table")
    assert hasattr(_s, "SplineTable")
