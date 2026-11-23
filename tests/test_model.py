"""Tests for bananafit.model (DESIGN.md Section 7).

Uses the synthetic 2-sample DUNE-like analysis from ``tests/_toyfixture.py``.
The core validation is ``ForwardModel.expected`` vs a from-scratch numpy
reference pipeline built directly off ``_toyfixture``'s raw per-event data --
independent of ``bananafit.splines``/``bananafit.norms``/``bananafit.oscillation``
implementation details (scipy ``CubicSpline`` instead of the Horner
coefficient table, ``np.bincount`` instead of ``segment_sum``, an explicit
per-event flavor gather instead of the vectorized where-guarded one) so that
an agreement failure means something in ``model.py`` itself, not a shared bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import jaxnu
import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.events import make_sample
from bananafit.model import SampleModel, build_forward_model
from bananafit.oscillation import ConstantMatterOsc
from bananafit.parameters import Flat, ParamSpec, ParameterSet, osc_param_specs

from _toyfixture import BASELINE_KM, DENSITY_GCC, YE, build_toy_analysis

# --- pure-numpy reference pipeline (independent of bananafit.model) ----------


def _osc_prob_reference(raw, physical_dict):
    params = jaxnu.OscParams(
        theta12=physical_dict["theta12"], theta13=physical_dict["theta13"],
        theta23=physical_dict["theta23"], deltacp=physical_dict["deltacp"],
        dm21=physical_dict["dm21"], dm31=physical_dict["dm31"],
    )
    n = raw.true_energy.shape[0]
    split = raw.anti_split
    out = np.empty(n, dtype=np.float64)
    for anti, sl in ((False, slice(0, split)), (True, slice(split, n))):
        idx = np.arange(sl.start, sl.stop)
        if idx.size == 0:
            continue
        E = raw.true_energy[idx]
        Pmat = np.asarray(jaxnu.probability_constant(
            params, E, BASELINE_KM, density=DENSITY_GCC, ye=YE, anti=anti,
        ))
        for k, i in enumerate(idx):
            ff = int(raw.flavor_from[i])
            ft = int(raw.flavor_to[i])
            out[i] = 1.0 if ff == -1 else Pmat[k, ft, ff]
    return out


def _spline_weight_reference(raw, physical_dict):
    n = raw.true_energy.shape[0]
    if not raw.spline_names:
        return np.ones(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    n_curves_in = raw.spline_knot_y.shape[1]
    for s, name in enumerate(raw.spline_names):
        x = float(physical_dict[name])
        knot_x = raw.spline_knot_x[s]
        xmin, xmax = knot_x[0], knot_x[-1]
        curve_vals = np.empty(n_curves_in + 1, dtype=np.float64)
        curve_vals[0] = 1.0
        for c in range(n_curves_in):
            cs = CubicSpline(knot_x, raw.spline_knot_y[s, c], bc_type="natural")
            if x < xmin:
                curve_vals[c + 1] = float(cs(xmin)) + float(cs(xmin, 1)) * (x - xmin)
            elif x > xmax:
                curve_vals[c + 1] = float(cs(xmax)) + float(cs(xmax, 1)) * (x - xmax)
            else:
                curve_vals[c + 1] = float(cs(x))
        w = w * curve_vals[raw.spline_event_index[s]]
    return w


def _norm_weight_reference(raw, physical_dict):
    n = raw.true_energy.shape[0]
    if not raw.norm_names:
        return np.ones(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    for i, name in enumerate(raw.norm_names):
        x = float(physical_dict[name])
        w = w * (1.0 + (x - 1.0) * raw.norm_mask[i])
    return w


def _reference_expected(toy, physical):
    physical_dict = dict(zip(toy.parameter_set.names, np.asarray(physical)))
    mus, w2s = [], []
    for sm in toy.samples:
        raw = toy.raw[sm.events.name]
        w = raw.nominal_weight * _osc_prob_reference(raw, physical_dict)
        w = w * _spline_weight_reference(raw, physical_dict)
        w = w * _norm_weight_reference(raw, physical_dict)
        mu = np.bincount(raw.reco_bin, weights=w, minlength=raw.n_bins + 1)[:raw.n_bins]
        w2 = np.bincount(raw.reco_bin, weights=w * w, minlength=raw.n_bins + 1)[:raw.n_bins]
        mus.append(mu)
        w2s.append(w2)
    return np.concatenate(mus), np.concatenate(w2s)


def _random_physical_points(toy, n_points=4, seed=1):
    rng = np.random.default_rng(seed)
    ps = toy.parameter_set
    points = [ps.constrain(ps.init_z)]
    for _ in range(n_points):
        z = rng.normal(scale=0.7, size=ps.n_free)
        points.append(ps.constrain(z))
    return points


# --- expected() vs reference --------------------------------------------


def test_expected_matches_numpy_reference():
    toy = build_toy_analysis(seed=0)
    for physical in _random_physical_points(toy):
        mu, w2 = toy.model.expected(physical)
        mu_ref, w2_ref = _reference_expected(toy, physical)
        np.testing.assert_allclose(np.asarray(mu), mu_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(w2), w2_ref, rtol=1e-10, atol=1e-10)


def test_expected_nontrivial():
    # guard against a vacuously-passing comparison (e.g. all-zero weights)
    toy = build_toy_analysis(seed=0)
    physical = toy.parameter_set.constrain(toy.parameter_set.init_z)
    mu, _ = toy.model.expected(physical)
    assert float(np.sum(mu)) > 100.0


# --- bin bookkeeping -------------------------------------------------------


def test_overflow_excluded_and_populated():
    toy = build_toy_analysis(seed=0)
    for raw in toy.raw.values():
        # the toy's bin edges are deliberately narrower than the spectrum:
        # the overflow slot (index n_bins) must be genuinely populated,
        # otherwise this test can't distinguish "dropped" from "empty".
        assert np.count_nonzero(raw.reco_bin == raw.n_bins) > 0
        assert raw.reco_bin.max() <= raw.n_bins
        assert raw.reco_bin.min() >= 0


def test_sample_offsets_and_total_bins():
    toy = build_toy_analysis(seed=0)
    n_bins_expected = [raw.n_bins for raw in (toy.raw["numu_dis"], toy.raw["nue_app"])]
    assert toy.model.total_bins == sum(n_bins_expected)
    offset = 0
    for meta, n_bins in zip(toy.model.sample_meta, n_bins_expected):
        assert meta.bin_offset == offset
        assert meta.n_bins == n_bins
        offset += n_bins


def test_expected_by_sample_concatenation_matches_expected():
    toy = build_toy_analysis(seed=0)
    physical = toy.parameter_set.constrain(toy.parameter_set.init_z)
    mu, w2 = toy.model.expected(physical)
    by_sample = toy.model.expected_by_sample(physical)

    mu_cat = np.concatenate([np.asarray(by_sample[sm.events.name][0]) for sm in toy.samples])
    w2_cat = np.concatenate([np.asarray(by_sample[sm.events.name][1]) for sm in toy.samples])
    np.testing.assert_allclose(mu_cat, np.asarray(mu), rtol=0, atol=0)
    np.testing.assert_allclose(w2_cat, np.asarray(w2), rtol=0, atol=0)


# --- shared systematic / norm masking ---------------------------------------


def test_shared_spline_systematic_moves_both_samples():
    toy = build_toy_analysis(seed=0)
    ps = toy.parameter_set
    names = ps.names

    base = {n: s.init for n, s in zip(names, ps.specs)}
    moved = dict(base)
    moved["flux_shape"] = 2.0

    phys_base = np.asarray([base[n] for n in names])
    phys_moved = np.asarray([moved[n] for n in names])

    by_base = toy.model.expected_by_sample(phys_base)
    by_moved = toy.model.expected_by_sample(phys_moved)

    for name in ("numu_dis", "nue_app"):
        mu_base = np.asarray(by_base[name][0])
        mu_moved = np.asarray(by_moved[name][0])
        assert not np.allclose(mu_base, mu_moved), (
            f"shared systematic flux_shape did not move sample {name!r}"
        )


def test_norm_moves_only_masked_events():
    toy = build_toy_analysis(seed=0)
    ps = toy.parameter_set
    names = ps.names
    raw = toy.raw["numu_dis"]
    is_cc = raw.flavor_from != -1  # sig_norm_numu's mask

    base = {n: s.init for n, s in zip(names, ps.specs)}

    moved_sig = dict(base)
    moved_sig["sig_norm_numu"] = 1.5
    phys_base = np.asarray([base[n] for n in names])
    phys_sig = np.asarray([moved_sig[n] for n in names])

    w_base = np.asarray(toy.model.event_weights(phys_base, "numu_dis"))
    w_sig = np.asarray(toy.model.event_weights(phys_sig, "numu_dis"))

    np.testing.assert_allclose(w_base[~is_cc], w_sig[~is_cc], rtol=0, atol=0)
    assert np.any(np.abs(w_base[is_cc] - w_sig[is_cc]) > 0)

    moved_bkg = dict(base)
    moved_bkg["bkg_norm_numu"] = 1.7
    phys_bkg = np.asarray([moved_bkg[n] for n in names])
    w_bkg = np.asarray(toy.model.event_weights(phys_bkg, "numu_dis"))

    np.testing.assert_allclose(w_base[is_cc], w_bkg[is_cc], rtol=0, atol=0)
    assert np.any(np.abs(w_base[~is_cc] - w_bkg[~is_cc]) > 0)

    # nue_app doesn't reference either norm at all -- structurally untouched.
    w_app_base = np.asarray(toy.model.event_weights(phys_base, "nue_app"))
    w_app_sig = np.asarray(toy.model.event_weights(phys_sig, "nue_app"))
    np.testing.assert_allclose(w_app_base, w_app_sig, rtol=0, atol=0)


# --- build_forward_model error path -----------------------------------------


def test_build_forward_model_errors_on_missing_spline_name():
    toy = build_toy_analysis(seed=0)
    # drop "xsec_maqe" from the ParameterSet -- numu_dis's SplineTable still
    # references it by name, so the build must fail loudly, not silently.
    specs = tuple(s for s in toy.parameter_set.specs if s.name != "xsec_maqe")
    bad_ps = ParameterSet(specs)
    with pytest.raises(ValueError, match="xsec_maqe"):
        build_forward_model(toy.samples, bad_ps)


def test_build_forward_model_errors_on_missing_osc_name():
    specs = tuple(s for s in osc_param_specs() if s.name != "theta23")
    ps = ParameterSet(specs)
    events, _ = make_sample(
        "s", true_energy=np.array([1.0, 2.0]), reco_variable=np.array([1.0, 2.0]),
        bin_edges=np.array([0.0, 1.5, 3.0]), flavor_from=np.array([1, 1]),
        flavor_to=np.array([1, 1]), anti=np.array([False, False]),
        nominal_weight=np.array([1.0, 1.0]),
    )
    sm = SampleModel(events=events, splines=None, norms=None,
                      osc=ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE))
    with pytest.raises(ValueError, match="theta23"):
        build_forward_model((sm,), ps)


# --- extra_weight functional-systematics hook -------------------------------


def test_extra_weight_hook_applied_and_receives_full_dict():
    events, _ = make_sample(
        "extra", true_energy=np.array([1.0, 2.0, 3.0]),
        reco_variable=np.array([1.0, 2.0, 3.0]),
        bin_edges=np.array([0.0, 1.5, 3.0, 4.5]),
        flavor_from=np.array([1, 1, 1]), flavor_to=np.array([1, 1, 1]),
        anti=np.array([False, False, False]),
        nominal_weight=np.array([1.0, 1.0, 1.0]),
    )
    seen = {}

    def extra_weight(sys_physical_by_name, ev):
        seen.update(sys_physical_by_name)
        import jax.numpy as jnp
        return jnp.full((ev.n_events,), sys_physical_by_name["scale"])

    ps = ParameterSet(osc_param_specs() + (ParamSpec("scale", 2.0, Flat(), (0.0, None)),))
    sm_plain = SampleModel(events=events, splines=None, norms=None,
                            osc=ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE))
    sm_extra = SampleModel(events=events, splines=None, norms=None,
                            osc=ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE),
                            extra_weight=extra_weight)

    model_plain = build_forward_model((sm_plain,), ps)
    model_extra = build_forward_model((sm_extra,), ps)

    physical = ps.constrain(ps.init_z)
    mu_plain, _ = model_plain.expected(physical)
    mu_extra, _ = model_extra.expected(physical)

    np.testing.assert_allclose(np.asarray(mu_extra), 2.0 * np.asarray(mu_plain), rtol=1e-12)
    assert "scale" in seen and "theta23" in seen  # full param dict, not just this sample's
