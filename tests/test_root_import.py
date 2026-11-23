"""Tests for bananafit.root_import (MaCh3-style ROOT TSpline3 import).

Fixture strategy (see bananafit/root_import.py's module docstring for the
full investigation writeup): uproot cannot *write* TSpline3 (not in its
writable-class allowlist), so real fixtures are written with PyROOT
(available in this environment, ROOT 6.40 via the LCG_110 view) and read
back through the actual reader (uproot). Tests that only exercise the
knots -> SplineTable assembly logic (not the ROOT-file walk itself) instead
monkeypatch ``read_tspline3_knots`` with a plain dict, so they run without
ROOT and stay fast; anything that touches a real ``.root`` file uses
``pytest.importorskip("ROOT")`` / ``pytest.importorskip("uproot")`` so the
suite degrades gracefully (skips, not fails) if either is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from bananafit import root_import

_KNOT_GRID = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


# --- fixture helper: write real TSpline3 objects with PyROOT -----------------


def _write_root_fixture(path, entries: dict) -> None:
    """``entries``: ``{"dir/keyname": (x, y)}`` (``""`` dir part = file root)."""
    ROOT = pytest.importorskip("ROOT")
    import array

    f = ROOT.TFile(str(path), "RECREATE")
    dircache = {"": f}
    try:
        for keyname, (x, y) in entries.items():
            dirpath, _, name = keyname.rpartition("/")
            if dirpath and dirpath not in dircache:
                dircache[dirpath] = f.mkdir(dirpath)
            dircache.get(dirpath, f).cd()
            xarr = array.array("d", [float(v) for v in x])
            yarr = array.array("d", [float(v) for v in y])
            sp = ROOT.TSpline3(name, xarr, yarr, len(x))
            sp.Write(name)
            f.cd()
        f.Write()
    finally:
        f.Close()


# --- default_name_parser (pure, no ROOT needed) -------------------------------


def test_default_name_parser_tokens():
    assert root_import.default_name_parser("MAQE") == ("MAQE", 0)
    assert root_import.default_name_parser("MAQE_2") == ("MAQE", 2)
    assert root_import.default_name_parser("MAQE_0_3") == ("MAQE", (0, 3))
    assert root_import.default_name_parser("CA5_numu_signal") == (
        "CA5",
        ("numu", "signal"),
    )


# --- read_tspline3_knots: real-file round trip --------------------------------


def test_read_tspline3_knots_basic(tmp_path):
    pytest.importorskip("ROOT")
    x1 = _KNOT_GRID.copy()
    y1 = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    x2 = np.array([-2.0, 0.0, 2.0])
    y2 = np.array([0.5, 1.0, 1.5])
    path = tmp_path / "f.root"
    _write_root_fixture(
        path,
        {"syst1_0_2": (x1, y1), "subdir/syst2_1_0": (x2, y2)},
    )

    knots = root_import.read_tspline3_knots(path)

    assert set(knots.keys()) == {"syst1", "syst2"}
    kx1, ky1 = knots["syst1"][(0, 2)]
    np.testing.assert_allclose(kx1, x1)
    np.testing.assert_allclose(ky1, y1)
    kx2, ky2 = knots["syst2"][(1, 0)]
    np.testing.assert_allclose(kx2, x2)
    np.testing.assert_allclose(ky2, y2)


def test_read_tspline3_knots_custom_parser(tmp_path):
    pytest.importorskip("ROOT")
    x = np.linspace(-3, 3, 7)
    y = 1.0 + 0.05 * x
    path = tmp_path / "f.root"
    _write_root_fixture(path, {"weird::name::A": (x, y)})

    def parser(objname, dirpath):
        parts = objname.split("::")
        return parts[0], (parts[1] if len(parts) > 1 else 0)

    knots = root_import.read_tspline3_knots(path, name_parser=parser)
    assert "weird" in knots
    assert "name" in knots["weird"]
    kx, ky = knots["weird"]["name"]
    np.testing.assert_allclose(kx, x)
    np.testing.assert_allclose(ky, y)


def test_read_tspline3_knots_duplicate_key_raises(tmp_path):
    pytest.importorskip("ROOT")
    x = np.linspace(-3, 3, 4)
    y = np.ones(4)
    path = tmp_path / "f.root"
    # default parser ignores dirpath, so "sig_0" at root and under "subdir/"
    # both parse to the same (syst="sig", curve_key=0) -- must collide.
    _write_root_fixture(path, {"sig_0": (x, y), "subdir/sig_0": (x, y)})

    with pytest.raises(ValueError, match="duplicate"):
        root_import.read_tspline3_knots(path)


def test_read_tspline3_knots_no_splines_raises(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "empty.root"
    with uproot.recreate(str(path)) as f:
        f["t"] = {"x": np.arange(5)}

    with pytest.raises(ValueError, match="no TSpline3"):
        root_import.read_tspline3_knots(path)


# --- spline_table_from_root: assembly logic (monkeypatched, no ROOT needed) --


def _patch_knots(monkeypatch, fake_knots):
    monkeypatch.setattr(
        root_import,
        "read_tspline3_knots",
        lambda path, name_parser=None: fake_knots,
    )


def test_spline_table_from_root_matches_scipy_natural(monkeypatch):
    rng = np.random.default_rng(0)
    x = _KNOT_GRID.copy()
    y_a = 1.0 + 0.1 * rng.standard_normal(7)
    y_b = 1.0 + 0.1 * rng.standard_normal(7)
    _patch_knots(monkeypatch, {"sigA": {0: (x, y_a), 1: (x, y_b)}})

    event_curve_map = {"sigA": np.array([0, 1, None, 0, 1], dtype=object)}
    table = root_import.spline_table_from_root(
        "dummy.root", event_curve_map, boundary="natural", dtype=np.float64
    )

    ref_a = CubicSpline(x, y_a, bc_type="natural")
    ref_b = CubicSpline(x, y_b, bc_type="natural")
    for xv in np.linspace(-3.0, 3.0, 25):
        w = np.asarray(table(np.array([xv])))
        expected = np.array(
            [ref_a(xv), ref_b(xv), 1.0, ref_a(xv), ref_b(xv)], dtype=np.float64
        )
        np.testing.assert_allclose(w, expected, rtol=1e-9, atol=1e-12)


def test_spline_table_from_root_events_without_curve_get_weight_one(monkeypatch):
    x = _KNOT_GRID.copy()
    y = 1.3 + 0.2 * np.sin(x)  # offset from 1.0 so "!= 1.0" below is meaningful
    _patch_knots(monkeypatch, {"sigA": {0: (x, y)}})

    event_curve_map = {"sigA": np.array([None, None, 0, None], dtype=object)}
    table = root_import.spline_table_from_root(
        "dummy.root", event_curve_map, dtype=np.float64
    )
    for xv in (-2.5, 0.3, 1.7):
        w = np.asarray(table(np.array([xv])))
        assert w[0] == 1.0 and w[1] == 1.0 and w[3] == 1.0
        assert w[2] != 1.0  # the one event actually wired to the curve


def test_spline_table_from_root_ragged_across_systematics(monkeypatch):
    x_a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # 5 knots
    y_a = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    x_b = _KNOT_GRID.copy()  # 7 knots
    y_b = 1.0 + 0.02 * x_b
    _patch_knots(monkeypatch, {"sigA": {0: (x_a, y_a)}, "sigB": {0: (x_b, y_b)}})

    event_curve_map = {"sigA": np.array([0, 0]), "sigB": np.array([0, 0])}
    table = root_import.spline_table_from_root(
        "dummy.root", event_curve_map, dtype=np.float64
    )
    assert table.knots.shape == (2, 7)
    assert np.all(np.isnan(np.asarray(table.knots[0, 5:])))

    ref_a = CubicSpline(x_a, y_a, bc_type="natural")
    ref_b = CubicSpline(x_b, y_b, bc_type="natural")
    for xv in (-1.5, 0.0, 1.5):
        w = np.asarray(table(np.array([xv, xv])))
        np.testing.assert_allclose(w[0], ref_a(xv) * ref_b(xv), rtol=1e-9)


def test_spline_table_from_root_empty_map_raises():
    with pytest.raises(ValueError, match="event_curve_map"):
        root_import.spline_table_from_root("dummy.root", {})


def test_spline_table_from_root_missing_systematic_raises(monkeypatch):
    x = _KNOT_GRID.copy()
    _patch_knots(monkeypatch, {"sigA": {0: (x, np.ones(7))}})
    with pytest.raises(ValueError, match="sigB"):
        root_import.spline_table_from_root("dummy.root", {"sigB": np.array([0, 0])})


def test_spline_table_from_root_mismatched_lengths_raises(monkeypatch):
    x = _KNOT_GRID.copy()
    _patch_knots(
        monkeypatch,
        {"sigA": {0: (x, np.ones(7))}, "sigB": {0: (x, np.ones(7))}},
    )
    with pytest.raises(ValueError, match="inconsistent lengths"):
        root_import.spline_table_from_root(
            "dummy.root",
            {"sigA": np.array([0, 0, 0]), "sigB": np.array([0, 0])},
        )


def test_spline_table_from_root_unknown_curve_key_raises(monkeypatch):
    x = _KNOT_GRID.copy()
    _patch_knots(monkeypatch, {"sigA": {0: (x, np.ones(7)), 1: (x, np.ones(7) * 2)}})
    with pytest.raises(ValueError, match="curve key"):
        root_import.spline_table_from_root("dummy.root", {"sigA": np.array([0, 5])})


def test_spline_table_from_root_mismatched_grid_within_syst_raises(monkeypatch):
    x1 = _KNOT_GRID.copy()
    x2 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])  # last knot differs
    _patch_knots(
        monkeypatch,
        {"sigA": {0: (x1, np.ones(7)), 1: (x2, np.ones(7))}},
    )
    with pytest.raises(ValueError, match="share a knot grid"):
        root_import.spline_table_from_root("dummy.root", {"sigA": np.array([0, 1])})


def test_spline_table_from_root_single_knot_raises(monkeypatch):
    _patch_knots(monkeypatch, {"sigA": {0: (np.array([0.0]), np.array([1.0]))}})
    with pytest.raises(ValueError, match="fewer than 2"):
        root_import.spline_table_from_root("dummy.root", {"sigA": np.array([0])})


# --- gold validation: bananafit eval vs real ROOT TSpline3.Eval ---------------


def test_matches_root_tspline3_eval_natural_boundary(tmp_path):
    """Cross-check the 'natural' boundary against a *real* ROOT TSpline3.

    Important nuance discovered while writing this test: ROOT's TSpline3
    default construction (no explicit ``opt``) does *not* use a natural
    (zero second-derivative) boundary condition -- empirically it disagrees
    with ``scipy.interpolate.CubicSpline(..., bc_type="natural")`` by O(0.01-
    0.1) near the ends for a 7-knot test spline. Requesting natural boundary
    conditions explicitly at both ends (``opt="b2e2"``, ROOT's own flag for
    "zero second derivative at knot 1" + "... at knot n") reproduces scipy's
    natural spline to machine precision (~1e-15) *in range*. This matters
    for real MaCh3 files: whatever boundary condition the *original* file's
    TSpline3 objects were built with is irrelevant to this reader, because
    ``spline_table_from_root`` only ever imports raw knot ``(x, y)`` values
    and always refits coefficients locally at its own ``boundary=`` choice
    (never trusts the file's own ``fB``/``fC``/`fD`` Hermite coefficients) --
    so this test validates the *interpolation formula* bananafit uses (an
    honest "b2e2" TSpline3 vs bananafit's "natural"), not "whatever a
    default-constructed ROOT spline happens to do".

    Also out of scope here: outside the knot range, ``SplineTable`` linearly
    continues the boundary segment's value+slope (DESIGN.md Section 4,
    gradient-safety), whereas ROOT's ``TSpline3::Eval`` extrapolates with the
    *full boundary cubic* (verified separately: they diverge substantially,
    e.g. by O(1) a few sigma outside a 7-knot [-3, 3] grid) -- a deliberate
    bananafit design choice, not a bug, so only in-range points are compared.
    """
    ROOT = pytest.importorskip("ROOT")
    import array

    rng = np.random.default_rng(42)
    x = _KNOT_GRID.copy()
    y = 1.0 + 0.15 * rng.standard_normal(7)

    path = tmp_path / "gold.root"
    _write_root_fixture(path, {"gold_0": (x, y)})

    table = root_import.spline_table_from_root(
        path, {"gold": np.array([0])}, boundary="natural", dtype=np.float64
    )

    xarr = array.array("d", x.tolist())
    yarr = array.array("d", y.tolist())
    ref = ROOT.TSpline3("ref", xarr, yarr, len(x), "b2e2", 0.0, 0.0)

    xs = np.linspace(float(x[0]), float(x[-1]), 61)  # in-range only
    for xv in xs:
        got = float(np.asarray(table(np.array([float(xv)])))[0])
        want = ref.Eval(float(xv))
        assert abs(got - want) < 1e-9, (xv, got, want)
