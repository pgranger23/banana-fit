"""Tests for bananafit.io (DESIGN.md Section 11)."""

import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from bananafit import io, norms as norms_mod
from bananafit.events import make_sample
from bananafit.splines import build_spline_table

_KNOT_GRID = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


# --- helpers: build a small, realistic (sample, splines, norms) trio ---------


def _toy_events(name, n, seed, n_bins_edges):
    rng = np.random.default_rng(seed)
    true_energy = rng.uniform(0.5, 8.0, size=n)
    reco_variable = rng.uniform(0.0, n_bins_edges[-1] + 1.0, size=n)
    flavor_from = rng.integers(0, 3, size=n)
    flavor_to = rng.integers(0, 3, size=n)
    anti = rng.integers(0, 2, size=n).astype(bool)
    nominal_weight = rng.uniform(0.1, 3.0, size=n)
    sample, perm = make_sample(
        name,
        true_energy=true_energy,
        reco_variable=reco_variable,
        bin_edges=n_bins_edges,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
    )
    return sample, perm


def _toy_splines(n, perm, seed, n_syst=3, n_curves_in=4, dtype=jnp.float32):
    """3 systematics incl. one with ragged (NaN-padded) knots."""
    rng = np.random.default_rng(seed)
    knot_x = np.tile(_KNOT_GRID, (n_syst, 1))
    # systematic 2 has only 5 real knots -- ragged support.
    knot_x[2, -2:] = np.nan

    knot_y = 1.0 + 0.25 * rng.standard_normal(size=(n_syst, n_curves_in, _KNOT_GRID.size))

    names = tuple(f"syst_{i}" for i in range(n_syst))

    # built against the ORIGINAL (pre-sort) event order, then reordered by
    # `perm` to stay aligned with the sorted EventSample -- see
    # events.make_sample's docstring.
    event_index_raw = rng.integers(0, n_curves_in + 1, size=(n_syst, n)).astype(np.int32)
    event_index = event_index_raw[:, perm]

    table = build_spline_table(
        knot_x,
        knot_y,
        event_index,
        names=names,
        boundary="natural",
        dtype=dtype,
        weight_clip=(0.0, None),
    )
    return table, knot_x, knot_y, event_index


def _toy_norms(n, perm, seed, n_norm=2):
    rng = np.random.default_rng(seed)
    names = tuple(f"norm_{i}" for i in range(n_norm))
    selections_raw = [rng.random(n) < 0.5 for _ in range(n_norm)]
    selections = [sel[perm] for sel in selections_raw]
    return norms_mod.from_selections(names, selections, n)


def _build_sample_trio(name, n=200, seed=0):
    edges = np.linspace(0.0, 10.0, 9)  # 8 bins
    sample, perm = _toy_events(name, n, seed, edges)
    table, knot_x, knot_y, event_index = _toy_splines(n, perm, seed + 1)
    nset = _toy_norms(n, perm, seed + 2)
    osc_attrs = {"baseline_km": 1300.0, "density_gcc": 2.848, "ye": 0.5}
    return sample, table, nset, osc_attrs, knot_y


# --- equality helpers ---------------------------------------------------------


def _assert_event_samples_equal(a, b):
    assert a.name == b.name
    assert a.n_bins == b.n_bins
    assert a.anti_split == b.anti_split
    assert a.bin_edges == b.bin_edges
    for field in ("true_energy", "reco_bin", "flavor_from", "flavor_to", "nominal_weight"):
        av, bv = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
        assert av.dtype == bv.dtype, field
        assert np.array_equal(av, bv), field


def _assert_spline_tables_bitexact(a, b):
    assert a.names == b.names
    assert a.weight_clip == b.weight_clip
    for field in ("knots", "coeffs", "event_index"):
        av, bv = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
        assert av.dtype == bv.dtype, field
        # NaN-padded knots: compare with equal_nan
        assert np.array_equal(av, bv, equal_nan=True), field


def _assert_norm_sets_equal(a, b):
    assert a.names == b.names
    av, bv = np.asarray(a.mask), np.asarray(b.mask)
    assert av.dtype == bv.dtype
    assert np.array_equal(av, bv)


# --- full round trip: save_analysis / load_analysis ---------------------------


def test_full_round_trip_bit_exact_and_evaluates_identically(tmp_path):
    sample_a, table_a, norms_a, osc_a, _ = _build_sample_trio("FHC_numu", n=200, seed=0)
    sample_b, _ = _toy_events("RHC_nue", 50, seed=100, n_bins_edges=np.linspace(0.0, 5.0, 6))

    path = tmp_path / "analysis.h5"
    io.save_analysis(
        path,
        {
            "FHC_numu": (sample_a, table_a, norms_a, osc_a),
            "RHC_nue": (sample_b, None, None, None),
        },
        parameter_yaml="oscillation:\n  theta23: 0.85\n",
    )

    samples, parameter_yaml = io.load_analysis(path)

    assert set(samples) == {"FHC_numu", "RHC_nue"}
    assert parameter_yaml == "oscillation:\n  theta23: 0.85\n"

    loaded_a, loaded_table, loaded_norms, loaded_osc = samples["FHC_numu"]
    _assert_event_samples_equal(sample_a, loaded_a)
    _assert_spline_tables_bitexact(table_a, loaded_table)
    _assert_norm_sets_equal(norms_a, loaded_norms)
    assert loaded_osc == osc_a

    # evaluations must match EXACTLY (same bit-exact coeffs/knots/masks)
    n_syst = len(table_a.names)
    x_spl = jnp.array([0.3, -1.7, 2.9][:n_syst], dtype=jnp.float64)
    w_orig = np.asarray(table_a(x_spl))
    w_loaded = np.asarray(loaded_table(x_spl))
    assert np.array_equal(w_orig, w_loaded)

    x_norm = jnp.array([1.2, 0.8], dtype=jnp.float64)
    n_orig = np.asarray(norms_a(x_norm))
    n_loaded = np.asarray(loaded_norms(x_norm))
    assert np.array_equal(n_orig, n_loaded)

    # a jitted function over the loaded EventSample works
    @jax.jit
    def total_weight(events):
        in_range = events.reco_bin < events.n_bins
        return jnp.sum(jnp.where(in_range, events.nominal_weight, 0.0))

    result = total_weight(loaded_a)
    expect = float(np.sum(np.where(np.asarray(sample_a.reco_bin) < sample_a.n_bins,
                                    np.asarray(sample_a.nominal_weight), 0.0)))
    assert float(result) == pytest.approx(expect, rel=1e-12)

    # RHC_nue: no splines/norms, but still a full, working EventSample
    loaded_b, table_b, nb, osc_b = samples["RHC_nue"]
    _assert_event_samples_equal(sample_b, loaded_b)
    assert table_b is None
    assert nb is None
    assert osc_b == {}


# --- ragged (NaN-padded) knots: loaded table must evaluate correctly, not --
# --- just bit-exactly reproduce the (possibly buggy) in-memory table -------
#
# io.py stores ``knots``/``coeffs`` verbatim, NaN-padding included, and
# ``_read_splines(rebuild="from-coeffs")`` reconstructs a ``SplineTable``
# directly from those arrays. That means a file written by an *old*,
# buggy bananafit (mis-indexing the last real segment of a ragged
# systematic -- see tests/test_splines.py's "mixed knot counts" tests) is
# byte-for-byte the same file a fixed bananafit would write: the fix lives
# entirely in ``SplineTable._eval_all_curves``, not in what gets persisted.
# So a loaded table is only as correct as *today's* evaluator, regardless
# of which bananafit wrote the file -- this test checks the loaded table
# against ground truth (scipy), not merely against the in-memory original
# (``_assert_spline_tables_bitexact`` + "evaluations match EXACTLY" above
# would pass identically whether or not the evaluator bug were present,
# since both tables share the same eval code path).


def test_loaded_ragged_spline_table_matches_scipy_beyond_last_real_knot(tmp_path):
    sample, table, nset, osc_attrs, knot_y = _build_sample_trio("ragged_eval", n=40, seed=11)
    path = tmp_path / "ragged.h5"
    io.save_analysis(path, {"ragged_eval": (sample, table, nset, osc_attrs)})

    samples, _ = io.load_analysis(path)
    _loaded_sample, loaded_table, _loaded_norms, _loaded_osc = samples["ragged_eval"]

    # systematic 2 (from _toy_splines) is the ragged one: real knots are
    # _KNOT_GRID[:5] == [-3, -2, -1, 0, 1], the last two (2, 3) are NaN pad.
    syst = 2
    x_real = _KNOT_GRID[:5]
    x_max = x_real[-1]
    curve_idx = 1  # knot_y row 0 -> event_index/curve 1 (curve 0 is identity)

    ref = CubicSpline(x_real, knot_y[syst, curve_idx - 1, :5], bc_type="natural")

    n_syst = len(loaded_table.names)
    for xv in [x_max - 0.3, x_max, x_max + 1e-9, x_max + 1.9]:  # incl. +2/+3-sigma-style pts
        x = jnp.zeros(n_syst, dtype=jnp.float64).at[syst].set(xv)
        # per curve, not per event: weights_per_syst gathers through a random
        # event_index (built by _toy_splines), so index by curve directly via
        # the same evaluation path it wraps.
        per_curve = loaded_table._eval_all_curves(x)[syst]  # (n_curves,)
        got = float(per_curve[curve_idx])
        if xv <= x_max:
            expect = float(ref(xv))
        else:
            v0, s0 = float(ref(x_max)), float(ref.derivative()(x_max))
            expect = v0 + s0 * (xv - x_max)
        np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-4)


# --- single-sample save_sample/load_sample, plain path -----------------------


def test_save_load_sample_standalone_path(tmp_path):
    sample, table, nset, osc_attrs, _ = _build_sample_trio("standalone", n=64, seed=7)
    path = tmp_path / "sample.h5"
    io.save_sample(path, sample, splines=table, norms=nset, osc_attrs=osc_attrs)

    loaded_sample, loaded_table, loaded_norms, loaded_osc = io.load_sample(path)
    _assert_event_samples_equal(sample, loaded_sample)
    _assert_spline_tables_bitexact(table, loaded_table)
    _assert_norm_sets_equal(nset, loaded_norms)
    assert loaded_osc == osc_attrs


# --- from-knots rebuild path ---------------------------------------------------


def test_from_knots_rebuild_matches_to_float32_tolerance(tmp_path):
    sample, table, nset, osc_attrs, knot_y = _build_sample_trio("rebuild_test", n=150, seed=3)
    path = tmp_path / "analysis.h5"
    io.save_analysis(
        path,
        {"rebuild_test": (sample, table, nset, osc_attrs)},
        spline_knot_y={"rebuild_test": knot_y},
        spline_boundary={"rebuild_test": "natural"},
    )

    samples, _ = io.load_analysis(path, rebuild="from-knots")
    _, rebuilt_table, _, _ = samples["rebuild_test"]

    assert rebuilt_table.names == table.names
    n_syst = len(table.names)
    for xv in [-2.9, -1.1, 0.0, 0.5, 1.9, 2.9, -5.0, 5.0]:
        x = jnp.full((n_syst,), xv, dtype=jnp.float64)
        w_orig = np.asarray(table(x))
        w_rebuilt = np.asarray(rebuilt_table(x))
        np.testing.assert_allclose(w_rebuilt, w_orig, rtol=1e-6, atol=1e-6)

    # also directly via load_sample with rebuild="from-knots"
    with h5py.File(path, "r") as f:
        _, direct_rebuilt, _, _ = io.load_sample(
            f["samples"]["rebuild_test"], rebuild="from-knots"
        )
    x = jnp.full((n_syst,), 1.234, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(direct_rebuilt(x)), np.asarray(table(x)), rtol=1e-6, atol=1e-6
    )


def test_from_knots_without_stored_knot_y_raises(tmp_path):
    sample, table, nset, osc_attrs, _knot_y = _build_sample_trio("no_knots", n=30, seed=9)
    path = tmp_path / "analysis.h5"
    # no spline_knot_y passed -> only the coefficient table is stored
    io.save_analysis(path, {"no_knots": (sample, table, nset, osc_attrs)})

    with pytest.raises(ValueError, match="from-knots"):
        io.load_analysis(path, rebuild="from-knots")


# --- None handling --------------------------------------------------------------


def test_sample_without_splines_or_norms_round_trips(tmp_path):
    sample, _perm = _toy_events("bare", 40, seed=42, n_bins_edges=np.linspace(0.0, 4.0, 5))
    path = tmp_path / "bare.h5"
    io.save_analysis(path, {"bare": (sample, None, None, None)})

    samples, parameter_yaml = io.load_analysis(path)
    assert parameter_yaml is None
    loaded_sample, table, nset, osc_attrs = samples["bare"]
    _assert_event_samples_equal(sample, loaded_sample)
    assert table is None
    assert nset is None
    assert osc_attrs == {}


# --- errors ----------------------------------------------------------------------


def test_wrong_format_version_raises(tmp_path):
    sample, _perm = _toy_events("x", 10, seed=1, n_bins_edges=np.linspace(0.0, 3.0, 4))
    path = tmp_path / "bad_version.h5"
    io.save_analysis(path, {"x": (sample, None, None, None)})

    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = 999

    with pytest.raises(ValueError, match="format_version"):
        io.load_analysis(path)


def test_missing_format_version_raises(tmp_path):
    path = tmp_path / "no_version.h5"
    with h5py.File(path, "w") as f:
        f.create_group("samples")

    with pytest.raises(ValueError, match="format_version"):
        io.load_analysis(path)


def test_truncated_file_raises_informatively(tmp_path):
    sample, table, nset, osc_attrs, _ = _build_sample_trio("trunc", n=30, seed=11)
    path = tmp_path / "trunc.h5"
    io.save_analysis(path, {"trunc": (sample, table, nset, osc_attrs)})

    with h5py.File(path, "a") as f:
        del f["samples"]["trunc"]["true_energy"]

    with pytest.raises(ValueError, match="true_energy"):
        io.load_analysis(path)


def test_shape_inconsistent_event_index_raises(tmp_path):
    sample, table, nset, osc_attrs, _ = _build_sample_trio("shapebad", n=30, seed=12)
    path = tmp_path / "shapebad.h5"
    io.save_analysis(path, {"shapebad": (sample, table, nset, osc_attrs)})

    with h5py.File(path, "a") as f:
        g = f["samples"]["shapebad"]["splines"]
        data = g["event_index"][...]
        del g["event_index"]
        # drop events -> second dim no longer matches n_events
        g.create_dataset("event_index", data=data[:, :-5])

    with pytest.raises(ValueError, match="event_index"):
        io.load_analysis(path)


def test_osc_attrs_key_collision_raises(tmp_path):
    sample, _perm = _toy_events("y", 10, seed=2, n_bins_edges=np.linspace(0.0, 3.0, 4))
    path = tmp_path / "collision.h5"
    with pytest.raises(ValueError, match="n_bins"):
        io.save_sample(path, sample, osc_attrs={"n_bins": 5})


# --- file size sanity: gzip actually applied -----------------------------------


def test_gzip_compression_applied(tmp_path, capsys):
    sample, table, nset, osc_attrs, _ = _build_sample_trio("compressed", n=5000, seed=21)
    path = tmp_path / "compressed.h5"
    io.save_analysis(path, {"compressed": (sample, table, nset, osc_attrs)})

    with h5py.File(path, "r") as f:
        g = f["samples"]["compressed"]
        assert g["true_energy"].compression == "gzip"
        assert g["true_energy"].shuffle is True
        assert g["splines"]["coeffs"].compression == "gzip"
        assert g["norms"]["mask"].compression == "gzip"

    raw_bytes = (
        sample.true_energy.nbytes + sample.reco_bin.nbytes + sample.flavor_from.nbytes
        + sample.flavor_to.nbytes + sample.nominal_weight.nbytes
        + table.coeffs.nbytes + table.knots.nbytes + table.event_index.nbytes
        + nset.mask.nbytes
    )
    file_bytes = os.path.getsize(path)
    print(
        f"\n[test_gzip_compression_applied] raw array bytes={raw_bytes} "
        f"file bytes={file_bytes} ratio={file_bytes / raw_bytes:.3f}"
    )
    captured = capsys.readouterr()
    assert "raw array bytes" in captured.out
