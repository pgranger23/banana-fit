"""Tests for ``bananafit.splines`` ``extrapolation="cubic"`` (MaCh3's end-segment cubic).

``tests/test_splines.py`` is unchanged and still pins the default (``"linear"``); this file adds the opt-in
mode that reproduces ROOT ``TSpline3`` / MaCh3 (``MaCh3/Splines/SplineStructs.h:674-720``: any x uses the
cubic of the end segment, ``segment = min(segment, nPoints - 2)``), together with ``boundary="not-a-knot"``
on the REAL, NaN-padded knots.
"""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from bananafit import io as jio
from bananafit import splines
from bananafit.events import make_sample_prebinned


def _ragged(real_counts, n_curves=4, dtype=jnp.float64, seed=0, extrapolation="cubic", boundary="not-a-knot",
            factorized=False):
    """One systematic per entry of ``real_counts`` (rows NaN-padded to the max), ``n_curves`` random curves each."""
    rng = np.random.default_rng(seed)
    k = max(real_counts)
    knot_x = np.full((len(real_counts), k), np.nan)
    knot_y = np.full((len(real_counts), n_curves, k), np.nan)
    xs, ys = [], []
    for s, n in enumerate(real_counts):
        x = -3.0 + np.concatenate([[0.0], np.cumsum(0.3 + rng.uniform(0, 1, n - 1))])   # strictly increasing, uneven
        y = 1.0 + 0.4 * rng.standard_normal((n_curves, n))
        knot_x[s, :n], knot_y[s, :, :n] = x, y
        xs.append(x)
        ys.append(y)
    n_syst = len(real_counts)
    event_index = np.tile(np.arange(n_curves + 1), (n_syst, 1)).astype(np.int32)
    if factorized:
        tab = splines.build_spline_table_factorized(
            knot_x, knot_y, event_index, np.arange(n_curves + 1), boundary=boundary, dtype=dtype,
            weight_clip=(None, None), extrapolation=extrapolation)
    else:
        tab = splines.build_spline_table(
            knot_x, knot_y, event_index, boundary=boundary, dtype=dtype, weight_clip=(None, None),
            extrapolation=extrapolation)
    return tab, xs, ys


def _scipy_nak(x, y, xv):
    return CubicSpline(x, y.T, bc_type="not-a-knot", extrapolate=True)(xv).T  # (n_curves,)


def _eval_syst(tab, s, xv, n_syst):
    xvec = np.zeros(n_syst)
    xvec[s] = xv
    return np.asarray(tab.weights_per_syst(jnp.asarray(xvec))[s])  # (n_curves + 1,)


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1e-11), (jnp.float32, 5e-5)])
def test_cubic_matches_scipy_not_a_knot_on_ragged_rows(dtype, tol):
    """Real-knot counts 2/3/5/6/7/8 in one table; in range, at the knots, below and far above/below the knots."""
    counts = [2, 3, 5, 6, 7, 8]
    tab, xs, ys = _ragged(counts, dtype=dtype)
    assert tab.extrapolation == "cubic"
    for s, (x, y) in enumerate(zip(xs, ys)):
        grid = np.concatenate([np.linspace(x[0] - 5, x[0], 11), x, np.linspace(x[0], x[-1], 33),
                               np.linspace(x[-1], x[-1] + 7, 11)])
        for xv in grid:
            w = _eval_syst(tab, s, xv, len(counts))
            assert w[0] == 1.0
            np.testing.assert_allclose(w[1:], _scipy_nak(x, y, xv), rtol=tol, atol=tol)


def test_cubic_mode_is_a_single_polynomial_across_the_boundary_gradient_finite():
    """Finite gradient (and a continuous second derivative) at, just inside and just outside both ends."""
    counts = [2, 5, 7]
    tab, xs, ys = _ragged(counts)
    f = lambda xv, s, c: tab.weights_per_syst(xv)[s, c]
    for s, (x, y) in enumerate(zip(xs, ys)):
        for edge in (x[0], x[-1]):
            for dx in (-1e-3, -1e-9, 0.0, 1e-9, 1e-3):
                xv = jnp.zeros(len(counts)).at[s].set(edge + dx)
                for c in (1, 2, 3):
                    g = jax.grad(f)(xv, s, c)
                    assert np.all(np.isfinite(np.asarray(g)))
                    h = jax.grad(lambda v: jax.grad(f)(v, s, c)[s])(xv)
                    assert np.all(np.isfinite(np.asarray(h)))
                    ref = CubicSpline(x, y[c - 1], bc_type="not-a-knot", extrapolate=True)
                    np.testing.assert_allclose(float(g[s]), float(ref(edge + dx, 1)), rtol=1e-8, atol=1e-8)
                    np.testing.assert_allclose(float(h[s]), float(ref(edge + dx, 2)), rtol=1e-6, atol=1e-6)


def test_default_extrapolation_is_linear_and_unchanged():
    """Both builders default to "linear"; with boundary="natural" the cubic and linear tables share every in-range value."""
    rng = np.random.default_rng(3)
    kx = np.tile(np.arange(-3.0, 4.0), (2, 1))
    ky = 1 + 0.3 * rng.standard_normal((2, 3, 7))
    ei = np.tile(np.arange(4), (2, 1)).astype(np.int32)
    t_def = splines.build_spline_table(kx, ky, ei)
    t_lin = splines.build_spline_table(kx, ky, ei, extrapolation="linear")
    t_cub = splines.build_spline_table(kx, ky, ei, extrapolation="cubic")
    assert t_def.extrapolation == t_lin.extrapolation == "linear" and t_cub.extrapolation == "cubic"
    assert splines.build_spline_table_factorized(kx, ky, ei, np.arange(4)).extrapolation == "linear"
    np.testing.assert_array_equal(np.asarray(t_def.coeffs), np.asarray(t_lin.coeffs))
    for xv in np.linspace(-3, 3, 13):
        x = jnp.array([xv, -xv])
        np.testing.assert_array_equal(np.asarray(t_def.weights_per_syst(x)), np.asarray(t_lin.weights_per_syst(x)))
        np.testing.assert_array_equal(np.asarray(t_lin.weights_per_syst(x)), np.asarray(t_cub.weights_per_syst(x)))
    # outside they differ (cubic bends away from the boundary tangent line)
    x = jnp.array([5.0, -5.0])
    assert np.abs(np.asarray(t_lin.weights_per_syst(x)) - np.asarray(t_cub.weights_per_syst(x))).max() > 1e-3
    with pytest.raises(ValueError, match="unknown extrapolation"):
        splines.build_spline_table(kx, ky, ei, extrapolation="quadratic")


def test_factorized_builder_cubic_matches_dense():
    t_d, _, _ = _ragged([2, 5, 7], seed=5)
    t_f, _, _ = _ragged([2, 5, 7], seed=5, factorized=True)
    assert t_f.extrapolation == "cubic"
    for xv in (-6.0, -1.3, 0.2, 2.9, 8.0):
        x = jnp.array([xv, 0.5 * xv, -xv])
        np.testing.assert_allclose(np.asarray(t_f(x)), np.asarray(t_d(x)), rtol=1e-12)


def test_jit_vmap_and_product_clip_in_cubic_mode():
    """Cubic extrapolation goes negative far outside the knots; like MaCh3 only the PRODUCT is floored (total <= 0 -> dropped)."""
    kx = np.array([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]])
    ky = np.array([[[1.0, 0.9, 0.8, 1.0, 1.4, 2.2, 3.5]]])
    ei = np.array([[1, 1]], dtype=np.int32)
    t = splines.build_spline_table(kx, ky, ei, boundary="not-a-knot", dtype=jnp.float64, extrapolation="cubic",
                                   weight_clip=(0.0, None))
    ref = CubicSpline(kx[0], ky[0, 0], bc_type="not-a-knot", extrapolate=True)
    xs = jnp.array([[-9.0], [-6.5], [-3.0], [0.0], [4.0], [9.0]])
    w = np.asarray(jax.jit(jax.vmap(t))(xs))[:, 0]
    np.testing.assert_allclose(w, np.maximum(ref(np.asarray(xs)[:, 0]), 0.0), rtol=1e-12, atol=1e-12)
    assert (ref(np.asarray(xs)[:, 0]) < 0).any() and (w >= 0).all()   # the case really exercises the floor


def test_io_roundtrip_persists_cubic_and_default_files_have_no_new_attr(tmp_path):
    kx = np.tile(np.arange(-3.0, 4.0), (1, 1))
    ky = 1 + 0.2 * np.random.default_rng(0).standard_normal((1, 3, 7))
    ei = np.array([[0, 1, 2, 3]], dtype=np.int32)
    ev = make_sample_prebinned("S", true_energy=np.ones(4), reco_bin=np.zeros(4, dtype=np.int32), n_bins=1,
                               flavor_from=np.zeros(4, dtype=np.int32), flavor_to=np.zeros(4, dtype=np.int32),
                               anti=np.zeros(4, dtype=bool), nominal_weight=np.ones(4), bin_edges=None)[0]
    for extr in ("linear", "cubic"):
        tab = splines.build_spline_table(kx, ky, ei, boundary="not-a-knot", extrapolation=extr, dtype=jnp.float64)
        path = tmp_path / f"{extr}.h5"
        with h5py.File(path, "w") as f:
            jio.save_sample(f.create_group("S"), ev, splines=tab, spline_knot_y=ky, spline_boundary="not-a-knot")
        with h5py.File(path, "r") as f:
            assert ("extrapolation" in f["S"]["splines"].attrs) == (extr == "cubic")
            for rebuild in ("from-coeffs", "from-knots"):
                _, t2, _, _ = jio.load_sample(f["S"], rebuild=rebuild)
                assert t2.extrapolation == extr
                x = jnp.array([5.0])
                np.testing.assert_allclose(np.asarray(t2.weights_per_syst(x)), np.asarray(tab.weights_per_syst(x)),
                                           rtol=1e-6, atol=1e-6)


def test_curve_floor_is_off_by_default_and_floors_each_curve_before_the_product(tmp_path):
    """MaCh3 <= v2.6.1 floors every spline weight at 0 before multiplying (``BinnedSplineHandler::CalcSplineWeights``), >= v2.7.0 only the
    product; they differ for an even number of negative curves."""
    kx = np.tile(np.arange(-3.0, 4.0), (2, 1))
    y = np.array([[[1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]], [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]]])    # one curve per systematic, goes negative at +3
    ei = np.array([[1], [1]], dtype=np.int32)
    t0 = splines.build_spline_table(kx, y, ei, boundary="not-a-knot", dtype=jnp.float64, extrapolation="cubic", weight_clip=(0.0, None))
    t1 = splines.build_spline_table(kx, y, ei, boundary="not-a-knot", dtype=jnp.float64, extrapolation="cubic", weight_clip=(0.0, None),
                                    curve_floor=0.0)
    assert t0.curve_floor is None and t1.curve_floor == 0.0
    both = jnp.array([3.0, 3.0])          # both curves = -0.2: product +0.04 without a per-curve floor, 0 with it
    one = jnp.array([3.0, 0.0])           # one negative: the product is negative -> clipped to 0 either way
    np.testing.assert_allclose(np.asarray(t0(both)), [0.04], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(t1(both)), [0.0], atol=1e-15)
    np.testing.assert_allclose(np.asarray(t0(one)), [0.0], atol=1e-15)
    np.testing.assert_allclose(np.asarray(t1(one)), [0.0], atol=1e-15)
    # identity curve untouched, positive curves untouched, gradient finite everywhere
    x = jnp.array([1.0, -1.0])
    np.testing.assert_array_equal(np.asarray(t0.weights_per_syst(x)), np.asarray(t1.weights_per_syst(x)))
    for v in (-3.0, 0.0, 2.9, 3.0, 3.1, 5.0):
        assert np.all(np.isfinite(np.asarray(jax.grad(lambda z: t1(z).sum())(jnp.array([v, v])))))
    # factorized builder + io round trip
    tf = splines.build_spline_table_factorized(kx, y, ei, np.array([0]), boundary="not-a-knot", dtype=jnp.float64, extrapolation="cubic",
                                               weight_clip=(0.0, None), curve_floor=0.0)
    np.testing.assert_allclose(np.asarray(tf(both)), [0.0], atol=1e-15)
    ev = make_sample_prebinned("S", true_energy=np.ones(1), reco_bin=np.zeros(1, dtype=np.int32), n_bins=1,
                               flavor_from=np.zeros(1, dtype=np.int32), flavor_to=np.zeros(1, dtype=np.int32),
                               anti=np.zeros(1, dtype=bool), nominal_weight=np.ones(1), bin_edges=None)[0]
    path = tmp_path / "f.h5"
    with h5py.File(path, "w") as f:
        jio.save_sample(f.create_group("S"), ev, splines=t1, spline_knot_y=y, spline_boundary="not-a-knot")
    with h5py.File(path, "r") as f:
        for rebuild in ("from-coeffs", "from-knots"):
            assert jio.load_sample(f["S"], rebuild=rebuild)[1].curve_floor == 0.0
