"""Tests for bananafit.toy (DESIGN.md §12.4 toy-generator validation).

Small ``n_events_per_sample`` throughout (the generator's calibration step
makes a handful of ``mango.probability_constant`` calls per sample, so build
time is dominated by one-time JIT compilation, not event count -- a few
thousand events per sample is plenty to exercise every code path while
keeping the whole test module fast).
"""

from __future__ import annotations

import numpy as np
import pytest

import bananafit  # noqa: F401 -- enforces float64, per house style
from bananafit.model import build_forward_model
from bananafit.posterior import asimov_data
from bananafit.toy import make_dune_like_analysis, poisson_fluctuate

_N_SMALL = 3000


@pytest.fixture(scope="module")
def toy():
    return make_dune_like_analysis(n_events_per_sample=_N_SMALL, seed=0)


@pytest.fixture(scope="module")
def toy_repeat():
    """Independently-built analysis with the same seed, for reproducibility
    checks (deliberately not the same fixture instance as ``toy``)."""
    return make_dune_like_analysis(n_events_per_sample=_N_SMALL, seed=0)


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_reproducible_same_seed(toy, toy_repeat):
    samples_a, pset_a, truth_a = toy
    samples_b, pset_b, truth_b = toy_repeat

    assert truth_a == truth_b
    assert pset_a.names == pset_b.names

    for sa, sb in zip(samples_a, samples_b):
        assert sa.events.name == sb.events.name
        np.testing.assert_array_equal(
            np.asarray(sa.events.true_energy), np.asarray(sb.events.true_energy)
        )
        np.testing.assert_array_equal(
            np.asarray(sa.events.reco_bin), np.asarray(sb.events.reco_bin)
        )
        np.testing.assert_array_equal(
            np.asarray(sa.events.flavor_from), np.asarray(sb.events.flavor_from)
        )
        np.testing.assert_array_equal(
            np.asarray(sa.events.nominal_weight), np.asarray(sb.events.nominal_weight)
        )
        assert sa.events.anti_split == sb.events.anti_split

        if sa.splines is None:
            assert sb.splines is None
        else:
            np.testing.assert_array_equal(np.asarray(sa.splines.coeffs), np.asarray(sb.splines.coeffs))
            np.testing.assert_array_equal(
                np.asarray(sa.splines.event_index), np.asarray(sb.splines.event_index)
            )

        if sa.norms is None:
            assert sb.norms is None
        else:
            np.testing.assert_array_equal(np.asarray(sa.norms.mask), np.asarray(sb.norms.mask))


def test_different_seed_differs():
    samples_a, _, _ = make_dune_like_analysis(n_events_per_sample=500, seed=0)
    samples_b, _, _ = make_dune_like_analysis(n_events_per_sample=500, seed=1)
    a = np.asarray(samples_a[0].events.true_energy)
    b = np.asarray(samples_b[0].events.true_energy)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# composition sanity
# ---------------------------------------------------------------------------


_EXPECTED_NOMINAL_ANTI = {
    "FHC_numu_dis": False,
    "FHC_nue_app": False,
    "RHC_numu_dis": True,
    "RHC_nue_app": True,
}


def test_each_sample_has_signal_and_background(toy):
    samples, _, _ = toy
    for sm in samples:
        flavor_from = np.asarray(sm.events.flavor_from)
        n = flavor_from.shape[0]
        n_nc = int(np.count_nonzero(flavor_from == -1))
        n_signal_like = n - n_nc
        assert n_nc > 0, f"{sm.events.name}: no NC background events generated"
        assert n_signal_like > 0, f"{sm.events.name}: no oscillating (signal-like) events generated"
        # NC background should be a substantial but not overwhelming fraction
        # (the toy's target composition fractions are 30-35% NC).
        assert 0.05 < n_nc / n < 0.6


def test_nu_antinu_split_matches_beam_mode(toy):
    """FHC samples are neutrino-dominated (anti_split close to N events);
    RHC samples are antineutrino-dominated (anti_split close to 0) -- the
    small opposite-polarity fraction is the "wrongsign" component."""
    samples, _, _ = toy
    for sm in samples:
        ev = sm.events
        nu_frac = ev.anti_split / ev.n_events
        if _EXPECTED_NOMINAL_ANTI[ev.name] is False:
            assert nu_frac > 0.85, f"{ev.name}: expected neutrino-dominated, got nu_frac={nu_frac}"
        else:
            assert nu_frac < 0.15, f"{ev.name}: expected antineutrino-dominated, got nu_frac={nu_frac}"


def test_flavor_from_to_pairing_convention(toy):
    """flavor_from == -1 iff flavor_to == -1 (EventSample/NC convention);
    every non-NC event's (from, to) matches one of the sample's known
    channels (signal/wrongsign share one pair, intrinsic is always e->e)."""
    samples, _, _ = toy
    for sm in samples:
        ff = np.asarray(sm.events.flavor_from)
        ft = np.asarray(sm.events.flavor_to)
        np.testing.assert_array_equal(ff == -1, ft == -1)
        non_nc = ff != -1
        pairs = set(zip(ff[non_nc].tolist(), ft[non_nc].tolist()))
        # exactly the signal channel's (from, to) and (0, 0) (intrinsic)
        assert pairs <= {(1, 1), (1, 0), (0, 0)}
        assert len(pairs) >= 1


# ---------------------------------------------------------------------------
# spline table exercised (curve dedup, not one curve per event)
# ---------------------------------------------------------------------------


def test_spline_curve_count_in_dedup_range(toy):
    samples, _, _ = toy
    for sm in samples:
        assert sm.splines is not None
        n_curves = sm.splines.coeffs.shape[1]  # includes the identity curve
        # spec: "~50-200 curves per systematic, NOT one curve per event"
        assert 50 <= n_curves <= 200
        assert n_curves < sm.events.n_events


def test_spline_identity_curve_used_by_nc_events(toy):
    """NC events must map to event_index == 0 (identity response) for every
    spline systematic -- the curve-dedup mechanism's "this systematic
    doesn't affect this event" convention (DESIGN.md §4)."""
    samples, _, _ = toy
    for sm in samples:
        ff = np.asarray(sm.events.flavor_from)
        is_nc = ff == -1
        assert np.any(is_nc)
        event_index = np.asarray(sm.splines.event_index)  # (n_syst, N)
        for row in event_index:
            assert np.all(row[is_nc] == 0)
            # and at least some non-NC events use a non-identity curve
            assert np.any(row[~is_nc] != 0)


def test_spline_names_shared_and_sample_specific(toy):
    samples, parameter_set, _ = toy
    name_sets = [set(sm.splines.names) for sm in samples]
    shared = set.intersection(*name_sets)
    assert len(shared) >= 1, "expected at least one systematic shared across all samples"
    for sm, names in zip(samples, name_sets):
        specific = names - shared
        assert len(specific) >= 1, f"{sm.events.name}: expected at least one sample-specific systematic"
    # every referenced spline name must resolve in the ParameterSet
    for sm in samples:
        for name in sm.splines.names:
            assert name in parameter_set.names


def test_configurable_spline_and_norm_counts():
    samples, pset, truth = make_dune_like_analysis(
        n_events_per_sample=500, n_spline_systs=3, n_norm_systs=1, seed=0
    )
    all_spline_names = set()
    for sm in samples:
        assert sm.splines is not None
        all_spline_names.update(sm.splines.names)
    assert len(all_spline_names) == 3
    for sm in samples:
        assert sm.norms is not None
        assert set(sm.norms.names) == {"nc_norm"}
    assert "nc_norm" in truth and truth["nc_norm"] == 1.0


def test_zero_systematics_gives_none():
    samples, pset, _ = make_dune_like_analysis(
        n_events_per_sample=500, n_spline_systs=0, n_norm_systs=0, seed=0
    )
    for sm in samples:
        assert sm.splines is None
        assert sm.norms is None
    assert pset.n_params == 6  # osc params only


# ---------------------------------------------------------------------------
# Asimov spectra: finite, DUNE-ballpark
# ---------------------------------------------------------------------------


def test_asimov_spectra_finite_and_ballpark(toy):
    samples, parameter_set, truth = toy
    model = build_forward_model(samples, parameter_set)
    data = asimov_data(model, parameter_set, overrides=truth)

    data_np = np.asarray(data)
    assert data_np.shape == (model.total_bins,)
    assert np.all(np.isfinite(data_np))
    assert np.all(data_np >= 0.0)

    by_sample = model.expected_by_sample(
        np.asarray([truth[n] for n in parameter_set.names])
    )
    # DUNE-ballpark: the generator calibrates each component's nominal
    # weight against the truth-point oscillation probability so the Asimov
    # total lands near `target_total` *independent* of n_events_per_sample
    # (fewer raw MC events just means a larger per-event weight) -- so the
    # targets below are NOT rescaled by _N_SMALL.
    targets = {
        "FHC_numu_dis": 4000.0, "FHC_nue_app": 400.0,
        "RHC_numu_dis": 2000.0, "RHC_nue_app": 180.0,
    }
    for name, (mu, _w2) in by_sample.items():
        total = float(np.sum(mu))
        assert total > 0.0
        # generous band: reco binning/overflow and multinomial noise shift
        # the realized total around the calibration target.
        assert 0.3 * targets[name] < total < 2.0 * targets[name], (
            f"{name}: total={total}, target~{targets[name]}"
        )


def test_poisson_fluctuate_reproducible_and_close(toy):
    samples, parameter_set, truth = toy
    model = build_forward_model(samples, parameter_set)
    data = asimov_data(model, parameter_set, overrides=truth)

    f1 = poisson_fluctuate(data, seed=7)
    f2 = poisson_fluctuate(data, seed=7)
    f3 = poisson_fluctuate(data, seed=8)
    np.testing.assert_array_equal(np.asarray(f1), np.asarray(f2))
    assert not np.array_equal(np.asarray(f1), np.asarray(f3))

    data_np = np.asarray(data)
    f1_np = np.asarray(f1)
    assert np.all(np.isfinite(f1_np))
    # per-bin fluctuation should be of order sqrt(mu), not wildly off
    diff = np.abs(f1_np - data_np)
    sigma = np.sqrt(np.maximum(data_np, 1.0))
    assert np.mean(diff / sigma) < 5.0


# ---------------------------------------------------------------------------
# parameter_set consistency
# ---------------------------------------------------------------------------


def test_parameter_set_names_resolve_and_forward_model_builds(toy):
    samples, parameter_set, truth = toy
    model = build_forward_model(samples, parameter_set)  # raises on any unresolved name
    assert model.total_bins == sum(sm.events.n_bins for sm in samples)

    # truth dict covers exactly the parameter_set's names
    assert set(truth.keys()) == set(parameter_set.names)


def test_truth_roundtrips_through_constrain(toy):
    _samples, parameter_set, truth = toy
    physical = np.asarray([truth[n] for n in parameter_set.names])
    z = parameter_set.unconstrain(physical)
    physical_back = np.asarray(parameter_set.constrain(z))
    np.testing.assert_allclose(physical_back, physical, rtol=1e-8, atol=1e-8)


def test_osc_params_match_nufit_no():
    import mango

    _samples, _parameter_set, truth = make_dune_like_analysis(n_events_per_sample=500, seed=0)
    nu = mango.nufit_no()
    for name in ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31"):
        assert truth[name] == pytest.approx(float(getattr(nu, name)))
