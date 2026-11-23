"""Tests for bananafit.shifts (FD
reco-energy shift systematics, and the differentiable soft/hard binning that
lets those shifts move an event across an analysis bin boundary inside a
jit+grad-safe ``logdensity``).

Covers: zero-parameter exactness, per-parameter direction/sign, gradient vs
finite differences (``EnergyScaleModel``); tent-kernel weight conservation,
soft/hard agreement at a controlled zero-shift point, soft smoothness vs hard
piecewise-constancy (``soft_bin_weights``/``hard_bin_index``); an end-to-end
micro fit wiring ``EnergyShift`` through ``SampleModel``/``ForwardModel``/
``BinnedPosterior``; and a brief sanity check of the (explicitly
approximate -- see ``shifts.py`` module docstring) CVN selection-efficiency
weight.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.events import make_sample
from bananafit.model import SampleModel, build_forward_model
from bananafit.oscillation import ConstantMatterOsc
from bananafit.parameters import Gaussian, ParamSpec, ParameterSet, osc_param_specs
from bananafit.posterior import asimov_data, make_posterior
from bananafit.shifts import (
    FD_DETECTOR_PARAM_NAMES,
    EnergyComponents,
    EnergyScaleModel,
    EnergyShift,
    cvn_efficiency_weight,
    hard_bin_index,
    soft_bin_weights,
)

BASELINE_KM = 1300.0
DENSITY_GCC = 2.85
YE = 0.5


# --- small helpers -----------------------------------------------------------


def _make_components(n, seed=0, *, res_positive=True):
    """``n``-event :class:`EnergyComponents` with strictly positive reco
    energies in every group and (if ``res_positive``) ``e_true > e_reco`` in
    every group -- so every one of the 19 FD detector parameters has an
    unambiguous, predictable sign of effect (see
    ``test_each_param_moves_energy_in_documented_direction``)."""
    rng = np.random.default_rng(seed)

    def _pair(lo, hi):
        reco = rng.uniform(lo, hi, size=n)
        bump = rng.uniform(0.05, 0.3, size=n) * (1 if res_positive else -1)
        true = reco + bump
        return jnp.asarray(reco), jnp.asarray(true)

    e_had_reco, e_had_true = _pair(0.3, 2.0)
    e_mu_reco, e_mu_true = _pair(0.5, 3.0)
    e_n_reco, e_n_true = _pair(0.05, 0.5)
    e_em_reco, e_em_true = _pair(0.1, 1.0)

    return EnergyComponents(
        e_had_reco=e_had_reco, e_had_true=e_had_true,
        e_mu_reco=e_mu_reco, e_mu_true=e_mu_true,
        e_n_reco=e_n_reco, e_n_true=e_n_true,
        e_em_reco=e_em_reco, e_em_true=e_em_true,
        is_ccnumu=jnp.ones(n), is_ccnue=jnp.zeros(n),
    )


def _param_vec(**overrides):
    x = np.zeros(19)
    for name, val in overrides.items():
        x[FD_DETECTOR_PARAM_NAMES.index(name)] = val
    return jnp.asarray(x)


# --- EnergyScaleModel: zero-parameter exactness -------------------------------


def test_zero_params_gives_exact_nominal_total():
    comp = _make_components(50)
    model = EnergyScaleModel()
    shifted = model.shift(comp, jnp.zeros(19))
    # rtol=1e-14, not bit-exact: EnergyScaleModel.shift sums the four group
    # energies in a different order (mu + (had + em + n)) than
    # EnergyComponents.e_total_reco's property (((mu + had) + em) + n) --
    # floating-point addition isn't associative, so the two sums can differ
    # in the last bit even though both are mathematically exact at x=0 (every
    # eq-12/13 term is identically 0.0, not merely close to it).
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(comp.e_total_reco), rtol=1e-14, atol=1e-14)


def test_zero_params_exact_even_with_zero_energy_groups():
    # A group with exactly zero reco/true energy (e.g. no true hadronic
    # activity) must not introduce a NaN/inf at the sqrt(0)/0-division guard,
    # and must still reproduce e_total_reco exactly at x=0.
    n = 5
    zeros = jnp.zeros(n)
    comp = EnergyComponents(
        e_had_reco=zeros, e_had_true=zeros,
        e_mu_reco=jnp.array([1.0, 2.0, 0.0, 3.0, 0.5]),
        e_mu_true=jnp.array([1.1, 2.1, 0.0, 3.2, 0.6]),
        e_n_reco=zeros, e_n_true=zeros,
        e_em_reco=zeros, e_em_true=zeros,
        is_ccnumu=jnp.ones(n), is_ccnue=jnp.zeros(n),
    )
    shifted = EnergyScaleModel().shift(comp, jnp.zeros(19))
    assert np.all(np.isfinite(np.asarray(shifted)))
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(comp.e_total_reco), rtol=1e-14, atol=1e-14)


# --- each of the 19 parameters moves energy in the documented direction -----


@pytest.mark.parametrize("name", FD_DETECTOR_PARAM_NAMES)
def test_each_param_moves_energy_in_documented_direction(name):
    # Every group's reco energy is > 0 and e_true > e_reco (see
    # _make_components), so the energy-scale formula (p0/p1/p2 all multiply a positive e_rec
    # or sqrt(e_rec)) and the resolution formula (r * (e_true - e_reco),
    # positive here) both
    # predict the SAME sign: increasing any of the 19 parameters from 0
    # strictly increases every event's shifted total energy, and decreasing
    # it strictly decreases it. This also catches a parameter accidentally
    # left unwired (shift would be a no-op / flat in that index).
    comp = _make_components(30, seed=1)
    model = EnergyScaleModel()
    base = model.shift(comp, jnp.zeros(19))

    up = model.shift(comp, _param_vec(**{name: 0.2}))
    down = model.shift(comp, _param_vec(**{name: -0.2}))

    assert np.all(np.asarray(up) > np.asarray(base) - 1e-12), f"{name}: +param did not increase energy"
    assert np.all(np.asarray(up) - np.asarray(base) > 1e-8), f"{name}: +param had no effect (unwired?)"
    assert np.all(np.asarray(down) < np.asarray(base) + 1e-12), f"{name}: -param did not decrease energy"


# --- grad vs finite differences ------------------------------------------


def test_energy_scale_model_grad_matches_finite_differences():
    comp = _make_components(20, seed=2)
    model = EnergyScaleModel()

    rng = np.random.default_rng(3)
    x0 = jnp.asarray(rng.normal(scale=0.3, size=19))

    def f(x):
        return jnp.sum(model.shift(comp, x))

    grad = np.asarray(jax.grad(f)(x0))
    assert np.all(np.isfinite(grad))

    x0_np = np.asarray(x0)
    eps = 1e-6
    for i in range(19):
        xp, xm = x0_np.copy(), x0_np.copy()
        xp[i] += eps
        xm[i] -= eps
        fd = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * eps)
        np.testing.assert_allclose(grad[i], fd, rtol=1e-4, atol=1e-6)


def test_energy_scale_model_grad_finite_at_zero_energy_group():
    # Gradient through the eq-12 1/sqrt(e_rec) guard must stay finite even
    # when a group's nominal energy is exactly 0 for some events.
    n = 8
    rng = np.random.default_rng(4)
    e_had = jnp.asarray(np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.1, 1.0, n)))
    comp = EnergyComponents(
        e_had_reco=e_had, e_had_true=e_had + 0.1,
        e_mu_reco=jnp.asarray(rng.uniform(0.5, 2.0, n)),
        e_mu_true=jnp.asarray(rng.uniform(0.5, 2.0, n)),
        e_n_reco=jnp.zeros(n), e_n_true=jnp.zeros(n),
        e_em_reco=jnp.zeros(n), e_em_true=jnp.zeros(n),
        is_ccnumu=jnp.ones(n), is_ccnue=jnp.zeros(n),
    )
    x0 = jnp.asarray(rng.normal(scale=0.2, size=19))
    grad = jax.grad(lambda x: jnp.sum(EnergyScaleModel().shift(comp, x)))(x0)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_energy_scale_model_jit_matches_eager():
    comp = _make_components(15, seed=5)
    model = EnergyScaleModel()
    x = jnp.asarray(np.random.default_rng(6).normal(scale=0.3, size=19))
    eager = model.shift(comp, x)
    jitted = jax.jit(model.shift)(comp, x)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-12)


# --- soft vs hard binning ------------------------------------------------


def _bin_edges(n_bins=10, lo=0.0, hi=10.0):
    return tuple(np.linspace(lo, hi, n_bins + 1))


def _centers(edges):
    edges = np.asarray(edges)
    return 0.5 * (edges[:-1] + edges[1:])


def _soft_mu(e, edges, weights, n_bins):
    bin_lo, frac_lo = soft_bin_weights(e, jnp.asarray(edges))
    bin_hi = bin_lo + 1
    mu_lo = jax.ops.segment_sum(weights * frac_lo, bin_lo, num_segments=n_bins + 1)[:n_bins]
    mu_hi = jax.ops.segment_sum(weights * (1.0 - frac_lo), bin_hi, num_segments=n_bins + 2)[:n_bins]
    return mu_lo + mu_hi


def _hard_mu(e, edges, weights, n_bins):
    idx = hard_bin_index(e, jnp.asarray(edges))
    return jax.ops.segment_sum(weights, idx, num_segments=n_bins + 1)[:n_bins]


def test_soft_bin_weights_fractions_sum_to_one_in_range():
    edges = _bin_edges()
    rng = np.random.default_rng(7)
    e = jnp.asarray(rng.uniform(edges[0], edges[-1] - 1e-9, size=200))
    bin_lo, frac_lo = soft_bin_weights(e, jnp.asarray(edges))
    assert np.all((np.asarray(frac_lo) >= 0.0) & (np.asarray(frac_lo) <= 1.0))
    n_bins = len(edges) - 1
    assert np.all(np.asarray(bin_lo) < n_bins)  # in-range events never land in overflow


def test_soft_bin_weights_overflow_matches_hard():
    edges = _bin_edges()
    n_bins = len(edges) - 1
    e = jnp.array([edges[0] - 5.0, edges[-1] + 5.0, edges[-1]])  # below, above, == upper edge (overflow too)
    bin_lo, frac_lo = soft_bin_weights(e, jnp.asarray(edges))
    hard = hard_bin_index(e, jnp.asarray(edges))
    np.testing.assert_array_equal(np.asarray(bin_lo), n_bins * np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(bin_lo), np.asarray(hard))
    np.testing.assert_allclose(np.asarray(frac_lo), 1.0)


def test_total_weight_conserved_by_tent_sharing():
    edges = _bin_edges()
    n_bins = len(edges) - 1
    rng = np.random.default_rng(8)
    e = jnp.asarray(rng.uniform(edges[0], edges[-1] - 1e-9, size=500))
    w = jnp.asarray(rng.uniform(0.5, 2.0, size=500))
    mu = _soft_mu(e, edges, w, n_bins)
    np.testing.assert_allclose(float(jnp.sum(mu)), float(jnp.sum(w)), rtol=1e-12, atol=1e-10)


def test_soft_and_hard_histograms_agree_at_bin_centers():
    # Construct events exactly AT their own bin's center: soft_bin_weights'
    # tent kernel is anchored at bin centers, so frac_lo collapses to exactly
    # 1 there (see shifts.py docstring) -- an honest, non-vacuous point where
    # soft and hard binning must coincide bin-for-bin, not just approximately.
    edges = _bin_edges(n_bins=8)
    n_bins = len(edges) - 1
    centers = _centers(edges)
    rng = np.random.default_rng(9)
    events_per_bin = 20
    e = jnp.asarray(np.repeat(centers, events_per_bin))
    w = jnp.asarray(rng.uniform(0.5, 1.5, size=e.shape[0]))

    mu_soft = np.asarray(_soft_mu(e, edges, w, n_bins))
    mu_hard = np.asarray(_hard_mu(e, edges, w, n_bins))
    np.testing.assert_allclose(mu_soft, mu_hard, rtol=1e-10, atol=1e-10)


def test_hard_binning_is_piecewise_constant_soft_is_smooth():
    # Place a handful of events near (but not on) an interior bin edge; shift
    # them all by the SAME small scalar `x` (a synthetic single-parameter
    # shift, isolating soft_bin_weights/hard_bin_index from EnergyScaleModel)
    # and compare how the binned mu responds to a small perturbation in `x`
    # that does not cross the boundary.
    edges = _bin_edges(n_bins=6, lo=0.0, hi=6.0)  # edge at 3.0 is interior
    n_bins = len(edges) - 1
    e0 = jnp.array([2.9, 2.95, 3.05, 3.1])  # straddling the edge at 3.0
    w = jnp.array([1.0, 1.0, 1.0, 1.0])

    def mu_soft(x):
        return _soft_mu(e0 + x, edges, w, n_bins)

    def mu_hard(x):
        return _hard_mu(e0 + x, edges, w, n_bins)

    x1, x2 = 0.01, 0.03  # both keep every event's *sign* relative to 3.0 fixed... but
    # actually we want a genuinely small perturbation that does NOT cross 3.0 for
    # any event: use two values within (0, 0.04), well short of moving 2.9 past 3.0.
    x1, x2 = 0.005, 0.015

    hard1, hard2 = np.asarray(mu_hard(x1)), np.asarray(mu_hard(x2))
    np.testing.assert_array_equal(hard1, hard2)  # piecewise-constant: no boundary crossed

    soft1, soft2 = np.asarray(mu_soft(x1)), np.asarray(mu_soft(x2))
    assert not np.allclose(soft1, soft2), "soft binning should respond continuously to any shift"

    # grad of a specific bin's soft count wrt x matches a central finite difference
    i_bin = int(np.argmax(np.abs(soft2 - soft1)))
    grad_fn = jax.grad(lambda x: mu_soft(x)[i_bin])
    grad_ad = float(grad_fn(0.01))
    eps = 1e-6
    fd = (float(mu_soft(0.01 + eps)[i_bin]) - float(mu_soft(0.01 - eps)[i_bin])) / (2 * eps)
    np.testing.assert_allclose(grad_ad, fd, rtol=1e-4, atol=1e-6)

    # hard binning's gradient through the same bin is exactly zero (no local
    # sensitivity to the shift away from the boundary) -- the piecewise-
    # constant claim, stated as a gradient rather than just an equality.
    grad_hard = jax.grad(lambda x: mu_hard(x)[i_bin])(0.01)
    assert float(grad_hard) == 0.0


# --- end-to-end micro: EnergyShift wired through SampleModel/BinnedPosterior -


def _build_shift_sample(seed=0, mode="soft"):
    rng = np.random.default_rng(seed)
    n = 400
    comp = _make_components(n, seed=seed)
    true_energy = np.asarray(comp.e_had_true) + np.asarray(comp.e_mu_true) \
        + np.asarray(comp.e_n_true) + np.asarray(comp.e_em_true)
    reco_energy = np.asarray(comp.e_total_reco)  # must match e_total_reco exactly (see EnergyShift docstring)

    bin_edges = tuple(np.linspace(float(reco_energy.min()) - 0.5, float(reco_energy.max()) + 0.5, 13))

    # Unoscillated (flavor_from == flavor_to == -1, EventSample's NC-background
    # convention -- see events.py): oscillation weight is then *exactly* 1 for
    # every event, so `nominal_weight` alone is the reference per-event weight
    # these tests bincount against. These tests are about the shift/binning
    # machinery, not oscillation physics -- isolating it keeps the reference
    # computation (plain np.bincount on nominal_weight) exactly right rather
    # than needing to reproduce mango's oscillation probability here too.
    flavor_from = np.full(n, -1, dtype=np.int32)
    flavor_to = np.full(n, -1, dtype=np.int32)
    anti = np.zeros(n, dtype=bool)
    nominal_weight = rng.uniform(0.5, 1.5, size=n)

    sample, perm = make_sample(
        "fd_numu_shift",
        true_energy=true_energy,
        reco_variable=reco_energy,
        bin_edges=bin_edges,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
    )
    # reorder EnergyComponents to match the sample's sorted event order,
    # exactly as SplineTable.event_index / NormSet.mask columns must be
    # (see events.py's make_sample docstring and EnergyComponents').
    comp_sorted = EnergyComponents(
        e_had_reco=comp.e_had_reco[perm], e_had_true=comp.e_had_true[perm],
        e_mu_reco=comp.e_mu_reco[perm], e_mu_true=comp.e_mu_true[perm],
        e_n_reco=comp.e_n_reco[perm], e_n_true=comp.e_n_true[perm],
        e_em_reco=comp.e_em_reco[perm], e_em_true=comp.e_em_true[perm],
        is_ccnumu=comp.is_ccnumu[perm], is_ccnue=comp.is_ccnue[perm],
    )
    energy_shift = EnergyShift(
        components=comp_sorted, model=EnergyScaleModel(),
        bin_edges=bin_edges, mode=mode,
    )
    osc = ConstantMatterOsc(baseline_km=BASELINE_KM, density_gcc=DENSITY_GCC, ye=YE)
    sm = SampleModel(events=sample, splines=None, norms=None, osc=osc, energy_shift=energy_shift)
    return sm


def _shift_parameter_set():
    shift_specs = tuple(
        ParamSpec(name, 0.0, Gaussian(0.0, 1.0), (-5.0, 5.0)) for name in FD_DETECTOR_PARAM_NAMES
    )
    return ParameterSet(osc_param_specs() + shift_specs)


def test_energy_shift_reproduces_reco_bin_at_zero_shift():
    # Consistency requirement documented on EnergyShift/EnergyComponents:
    # e_total_reco (= the shift model's output at x=0) must digitize to the
    # SAME reco_bin the EventSample was itself built with.
    sm = _build_shift_sample(mode="hard")
    ps = _shift_parameter_set()
    model = build_forward_model((sm,), ps)
    physical = ps.constrain(ps.init_z)  # every shift param at its nominal (0.0)

    mu_shifted, _ = model.expected(physical)
    n_bins = sm.events.n_bins
    mu_fixed = np.bincount(
        np.asarray(sm.events.reco_bin), weights=np.asarray(sm.events.nominal_weight),
        minlength=n_bins + 1,
    )[:n_bins]
    np.testing.assert_allclose(np.asarray(mu_shifted), mu_fixed, rtol=1e-10, atol=1e-10)


def test_logdensity_jit_and_grad_finite_with_energy_shift_active():
    sm = _build_shift_sample(mode="soft")
    ps = _shift_parameter_set()
    model = build_forward_model((sm,), ps)
    data = asimov_data(model, ps)
    posterior = make_posterior(model, data, ps, stat="poisson")

    val = posterior.logdensity(posterior.z0)
    assert np.isfinite(float(val))

    jitted = jax.jit(posterior.logdensity)
    np.testing.assert_allclose(float(jitted(posterior.z0)), float(val), rtol=1e-10)

    grad_z0 = jax.grad(posterior.logdensity)(posterior.z0)
    assert np.all(np.isfinite(np.asarray(grad_z0)))

    # Gradient at z0 itself is a poor probe for "genuinely nonzero": z0 IS
    # the Asimov point (data == mu(truth) exactly), where d(poisson loglike)
    # / d(physical) vanishes identically for *every* parameter (see
    # test_posterior.py's test_asimov_loglike_gradient_vanishes_at_truth_poisson),
    # and every shift param's Gaussian(0, 1) prior also has zero derivative
    # at its own mean -- both terms are trivially ~0 there regardless of
    # whether the shift machinery even works. Displace z first, exactly like
    # test_posterior.py's _displaced_z, to get a non-vacuous probe.
    rng = np.random.default_rng(11)
    z_displaced = jnp.asarray(np.asarray(posterior.z0) + rng.normal(scale=0.3, size=ps.n_free))
    grad = jax.grad(posterior.logdensity)(z_displaced)
    assert np.all(np.isfinite(np.asarray(grad)))

    # gradient wrt at least one shift parameter is genuinely nonzero (soft
    # binning is what makes this differentiable at all -- see module
    # docstring on hard binning's zero gradient almost everywhere).
    i_had = ps.free_names.index("HadEScaleFD")
    assert abs(float(grad[i_had])) > 1e-8


def test_hard_vs_soft_mu_close_at_nominal():
    sm_soft = _build_shift_sample(mode="soft")
    sm_hard = _build_shift_sample(mode="hard")
    ps = _shift_parameter_set()

    model_soft = build_forward_model((sm_soft,), ps)
    model_hard = build_forward_model((sm_hard,), ps)
    physical = ps.constrain(ps.init_z)

    mu_soft, _ = model_soft.expected(physical)
    mu_hard, _ = model_hard.expected(physical)

    total = float(jnp.sum(mu_hard))
    l1 = float(jnp.sum(jnp.abs(mu_soft - mu_hard)))
    assert l1 / total < 0.15, f"soft/hard disagree by {l1 / total:.3%} of total at nominal"


def test_fast_fixed_bin_path_unaffected_by_energy_shift_field_existing():
    # SampleModel with energy_shift=None must be bit-identical to the
    # pre-existing fixed-bin path (DESIGN.md/model.py contract) -- a plain
    # regression guard that adding the field did not disturb the default.
    sm = _build_shift_sample(mode="soft")
    sm_plain = dataclasses.replace(sm, energy_shift=None)
    ps = _shift_parameter_set()
    model_plain = build_forward_model((sm_plain,), ps)
    physical = ps.constrain(ps.init_z)
    mu, _ = model_plain.expected(physical)

    n_bins = sm.events.n_bins
    mu_ref = np.bincount(
        np.asarray(sm.events.reco_bin), weights=np.asarray(sm.events.nominal_weight),
        minlength=n_bins + 1,
    )[:n_bins]
    np.testing.assert_allclose(np.asarray(mu), mu_ref, rtol=1e-10, atol=1e-10)


# --- CVN efficiency weight (documented approximation) -------------------------


def test_cvn_efficiency_weight_monotonic_and_bounded():
    score = jnp.linspace(0.0, 1.0, 50)
    w = cvn_efficiency_weight(score, shift=0.0, cut=0.5, softness=0.02)
    w_np = np.asarray(w)
    assert np.all((w_np >= 0.0) & (w_np <= 1.0))
    assert np.all(np.diff(w_np) >= 0.0)  # monotonically increasing in score


def test_cvn_efficiency_weight_shift_moves_effective_cut():
    # A positive shift is equivalent to lowering the effective cut: an event
    # just below the nominal cut should pass with higher probability.
    score = jnp.asarray(0.49)  # 0-d, so cvn_efficiency_weight's output is a
    # scalar array -- float() requires ndim == 0 (unlike a length-1 array).
    w_base = float(cvn_efficiency_weight(score, shift=0.0, cut=0.5, softness=0.01))
    w_shifted = float(cvn_efficiency_weight(score, shift=0.05, cut=0.5, softness=0.01))
    assert w_shifted > w_base


def test_cvn_efficiency_weight_grad_finite():
    score = jnp.asarray(np.random.default_rng(10).uniform(0.0, 1.0, size=20))
    grad = jax.grad(lambda s: jnp.sum(cvn_efficiency_weight(score, s, cut=0.5, softness=0.02)))(0.0)
    assert np.isfinite(float(grad))
