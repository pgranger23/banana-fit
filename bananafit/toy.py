"""DUNE-like toy MC generator (DESIGN.md §12.4 / §1 "validation-first").

:func:`make_dune_like_analysis` builds a small, self-contained, four-sample
long-baseline analysis (FHC/RHC x numu-disappearance/nue-appearance) that
exercises every corner of the ``bananafit`` public API in one call: multiple
oscillation channels sharing a baseline/density, spline systematics both
shared-across-samples and sample-specific (built via the curve-dedup
``event_index`` mechanism with realistic curve counts, not one curve per
event), norm systematics masked onto specific background components, and a
``ParameterSet`` mixing the standard six oscillation parameters with both
kinds of systematic.

It is the "grown-up" sibling of ``tests/_toyfixture.py`` (a smaller, 2-sample
fixture used by ``tests/test_model.py`` / ``tests/test_posterior.py``) --
same conventions (flavor codes, ``anti`` sorting, NC-as-unoscillated), more
samples, more systematics, and calibrated to DUNE-ballpark event counts
rather than arbitrary toy numbers.

Design of the fake physics (no external data, everything analytic/numpy):

- **Flux**: each sample's true-energy spectrum is drawn from a Gamma
  distribution shaped to peak at ~2.5 GeV (DUNE's beam-flux peak, chosen so
  the analysis sits near the L/E of the first oscillation minimum/maximum --
  the same reason the real DUNE beam is designed that way).
- **Composition**: every sample is built from four event *components* --
  ``signal`` (the sample's nominal oscillation channel at its nominal beam
  polarity), ``wrongsign`` (the same channel at the *opposite* polarity --
  e.g. a small nu_mu_bar contamination in an FHC beam), ``intrinsic`` (beam
  intrinsic nu_e, flavor 0->0, a real survival probability not exactly 1),
  and ``nc`` (flavor -1/-1, unoscillated by the ``EventSample`` convention).
  Component weights are calibrated (see ``_calibrate_component_weight``)
  against ``mango.probability_constant`` evaluated at the truth point so the
  *post-oscillation* Asimov composition lands at chosen target fractions of
  a DUNE-ballpark total (thousands of numu, hundreds of nue per FHC sample).
- **Reco energy**: true energy smeared by a stochastic 15%/sqrt(E) term and a
  2% constant term, added in quadrature (standard calorimetric convention).
- **Spline systematics**: physics-motivated smooth response curves
  ``w(x) = 1 + s*tanh(x) + q*x^2`` (knots -3..3), with ``(s, q)`` varying
  smoothly per (component, true-energy bin) -- i.e. curves are shared across
  many events of similar kinematics/topology, dedup'd through
  ``SplineTable.event_index`` (tens to ~100 curves per systematic, matching
  MaCh3-scale spline monoliths, NOT a curve per event). A configurable number
  of systematics are shared by name across all four samples (one physical
  parameter moves every sample coherently, DESIGN.md §7) and the rest are
  sample-specific (round-robin assigned, one to each sample by default).
- **Norm systematics**: flat multiplicative weights masked onto the ``nc``
  and ``wrongsign`` components specifically (the components a real analysis
  would give freely-floating or loosely-constrained normalizations to),
  shared by name across every sample that has that component.

Everything here is pure-numpy generation seeded by a single ``seed`` (plus
``mango``'s float64 oscillation-probability calls used only to *calibrate*
weights at build time, never inside the returned model's hot path) --
identical ``seed`` reproduces an identical analysis bit-for-bit.
"""

from __future__ import annotations


import numpy as np
import jax.numpy as jnp

import mango

from .events import make_sample
from .model import SampleModel
from .norms import from_selections
from .oscillation import ConstantMatterOsc
from .parameters import Gaussian, ParamSpec, ParameterSet, osc_param_specs
from .splines import build_spline_table

__all__ = ["make_dune_like_analysis", "poisson_fluctuate"]

# ---------------------------------------------------------------------------
# fixed toy-geometry / detector constants
# ---------------------------------------------------------------------------

BASELINE_KM = 1300.0
DENSITY_GCC = 2.85
YE = 0.5

_ENERGY_LO, _ENERGY_HI = 0.5, 8.0
_N_RECO_BINS = 40
_RECO_BIN_EDGES = tuple(np.linspace(_ENERGY_LO, _ENERGY_HI, _N_RECO_BINS + 1))

_FLUX_PEAK_GEV = 2.5
_FLUX_GAMMA_SHAPE = 3.0
_FLUX_GAMMA_SCALE = _FLUX_PEAK_GEV / (_FLUX_GAMMA_SHAPE - 1.0)

# Component indices (fixed order, used throughout this module).
_SIGNAL, _WRONGSIGN, _INTRINSIC, _NC = 0, 1, 2, 3

# name, dominant-polarity anti flag, signal flavor_from/flavor_to,
# DUNE-ballpark total event count (at nominal osc params, exposure_scale=1),
# post-oscillation composition fractions (signal, wrongsign, intrinsic, nc)
# -- sum to 1 by construction, illustrative (not a real DUNE selection).
_SAMPLE_SPECS = (
    dict(name="FHC_numu_dis", anti=False, flavor_from=1, flavor_to=1,
         target_total=4000.0, fracs=(0.55, 0.03, 0.07, 0.35)),
    dict(name="FHC_nue_app", anti=False, flavor_from=1, flavor_to=0,
         target_total=400.0, fracs=(0.30, 0.05, 0.35, 0.30)),
    dict(name="RHC_numu_dis", anti=True, flavor_from=1, flavor_to=1,
         target_total=2000.0, fracs=(0.55, 0.03, 0.07, 0.35)),
    dict(name="RHC_nue_app", anti=True, flavor_from=1, flavor_to=0,
         target_total=180.0, fracs=(0.30, 0.05, 0.35, 0.30)),
)
_SAMPLE_NAMES = tuple(s["name"] for s in _SAMPLE_SPECS)

# --- spline-systematic curve grid (shared shape/formula for every syst) ----

_KNOT_X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
_N_EBIN_CURVE = 25  # x 3 oscillating components (signal/wrongsign/intrinsic)
_EBIN_EDGES = np.linspace(0.1, 9.0, _N_EBIN_CURVE + 1)
_EBIN_CENTERS = 0.5 * (_EBIN_EDGES[:-1] + _EBIN_EDGES[1:])
# Per-(non-NC)-component relative amplitude: signal is affected most,
# wrongsign/intrinsic less (loosely mirrors "different topology/energy
# reconstruction sensitivity" without claiming physical precision).
_COMPONENT_SPLINE_SCALE = {_SIGNAL: 1.0, _WRONGSIGN: 0.6, _INTRINSIC: 0.5}

# --- norm-systematic plan: shared-by-name candidates, in priority order ----

_NORM_CANDIDATES_SHARED = (
    ("nc_norm", _NC, 0.05),
    ("wrongsign_norm", _WRONGSIGN, 0.08),
)


# ---------------------------------------------------------------------------
# small numpy helpers
# ---------------------------------------------------------------------------


def _sample_true_energy(rng: np.random.Generator, n: int) -> np.ndarray:
    """Gamma-shaped "flux" spectrum peaked at ``_FLUX_PEAK_GEV`` -- sampling
    directly from this density is the usual weighted-MC shortcut (event
    density already tracks the flux shape, so a flat per-component weight
    reproduces the flux integral to within Monte Carlo noise)."""
    return rng.gamma(shape=_FLUX_GAMMA_SHAPE, scale=_FLUX_GAMMA_SCALE, size=n) + 0.05


def _reco_energy(rng: np.random.Generator, true_energy: np.ndarray) -> np.ndarray:
    """15%/sqrt(E) stochastic + 2% constant term, added in quadrature
    (standard calorimetric resolution convention)."""
    sigma_frac = np.sqrt((0.15 / np.sqrt(true_energy)) ** 2 + 0.02**2)
    reco = true_energy * (1.0 + rng.normal(0.0, sigma_frac))
    return np.clip(reco, 0.05, None)


def _calibrate_component_weight(
    true_energy: np.ndarray, *, anti: bool, flavor_from: int, flavor_to: int,
    target_count: float, truth_osc,
) -> float:
    """Flat per-event weight so that ``sum(weight * P_osc(truth))`` over
    ``true_energy`` equals ``target_count`` -- i.e. calibrate the *nominal*
    (oscillation-independent) weight against the truth-point oscillation
    probability so the Asimov composition lands at the requested target.
    Used only at toy-build time; never touches the returned model's hot
    path (``nominal_weight`` itself stays a plain constant array)."""
    if true_energy.size == 0:
        return 0.0
    P = mango.probability_constant(
        truth_osc, true_energy, BASELINE_KM, density=DENSITY_GCC, ye=YE, anti=anti,
    )
    p_event = np.asarray(P)[:, flavor_to, flavor_from]
    denom = max(float(np.sum(p_event)), 1e-12)
    return float(target_count) / denom


def _curve_values(a_amp: float, b_amp: float) -> np.ndarray:
    """``(3 * _N_EBIN_CURVE, n_knots)`` response curves for one systematic:
    one row per (component in {signal, wrongsign, intrinsic}, energy bin),
    ``w(x) = 1 + s*tanh(x) + q*x^2`` with ``s``, ``q`` varying smoothly with
    the bin's true-energy center (an "energy-dependent shape", per
    DESIGN.md's toy-generator brief) and with a fixed per-component scale.
    """
    f_shape = 0.5 + 0.5 * np.tanh((_EBIN_CENTERS - 2.5) / 2.0)  # rises with E
    g_shape = np.exp(-((_EBIN_CENTERS - 2.5) / 3.0) ** 2)       # bump at peak
    rows = []
    for comp in (_SIGNAL, _WRONGSIGN, _INTRINSIC):
        scale = _COMPONENT_SPLINE_SCALE[comp]
        s_vals = a_amp * scale * f_shape       # (n_ebin,)
        q_vals = b_amp * scale * g_shape       # (n_ebin,)
        for s_val, q_val in zip(s_vals, q_vals):
            rows.append(1.0 + s_val * np.tanh(_KNOT_X) + q_val * _KNOT_X**2)
    return np.stack(rows, axis=0)


def _curve_index(true_energy: np.ndarray, component_idx: np.ndarray) -> np.ndarray:
    """Per-event curve index into :func:`_curve_values`'s rows, 1-based (0 is
    always the identity curve prepended by ``build_spline_table``). NC events
    always map to 0 (identity) -- these physics-motivated response shapes
    are defined for the oscillating components only."""
    ebin = np.clip(
        np.searchsorted(_EBIN_EDGES, true_energy, side="right") - 1, 0, _N_EBIN_CURVE - 1
    )
    comp_for_curve = np.where(component_idx == _NC, 0, component_idx)
    idx = 1 + comp_for_curve * _N_EBIN_CURVE + ebin
    return np.where(component_idx == _NC, 0, idx).astype(np.int32)


def poisson_fluctuate(data, seed: int = 0):
    """Statistically fluctuate an Asimov (expectation-value) spectrum:
    ``data`` -> one Poisson draw per bin, seeded by ``seed`` for
    reproducibility. Returns a float64 ``jax.Array`` (same shape as
    ``data``), suitable for ``make_posterior(..., data=...)``.
    """
    data_np = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fluctuated = rng.poisson(np.maximum(data_np, 0.0)).astype(np.float64)
    return jnp.asarray(fluctuated, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# per-sample raw-event generation
# ---------------------------------------------------------------------------


def _generate_sample_events(rng: np.random.Generator, spec: dict, n_events: int, truth_osc):
    """Return, in a single (pre-sort) event order: ``true_energy``,
    ``reco_energy``, ``flavor_from``, ``flavor_to``, ``anti``,
    ``nominal_weight``, ``component_idx`` -- all ``(n_events,)`` numpy."""
    counts = rng.multinomial(n_events, spec["fracs"])
    n_sig, n_ws, n_int, n_nc = (int(c) for c in counts)

    target_total = spec["target_total"]
    fracs = spec["fracs"]
    nominal_anti = spec["anti"]
    from_f, to_f = spec["flavor_from"], spec["flavor_to"]

    true_energy_parts = []
    flavor_from_parts = []
    flavor_to_parts = []
    anti_parts = []
    weight_parts = []
    component_parts = []

    def _add(n, from_v, to_v, anti_v, comp, target_frac):
        e = _sample_true_energy(rng, n)
        true_energy_parts.append(e)
        flavor_from_parts.append(np.full(n, from_v, dtype=np.int32))
        flavor_to_parts.append(np.full(n, to_v, dtype=np.int32))
        anti_parts.append(np.full(n, anti_v, dtype=bool))
        component_parts.append(np.full(n, comp, dtype=np.int32))
        target_count = target_total * target_frac
        if comp == _NC:
            w = target_count / n if n > 0 else 0.0
            weight_parts.append(np.full(n, w, dtype=np.float64))
        else:
            w = _calibrate_component_weight(
                e, anti=anti_v, flavor_from=from_v, flavor_to=to_v,
                target_count=target_count, truth_osc=truth_osc,
            )
            weight_parts.append(np.full(n, w, dtype=np.float64))

    _add(n_sig, from_f, to_f, nominal_anti, _SIGNAL, fracs[0])
    _add(n_ws, from_f, to_f, not nominal_anti, _WRONGSIGN, fracs[1])
    _add(n_int, 0, 0, nominal_anti, _INTRINSIC, fracs[2])
    _add(n_nc, -1, -1, nominal_anti, _NC, fracs[3])

    true_energy = np.concatenate(true_energy_parts)
    flavor_from = np.concatenate(flavor_from_parts)
    flavor_to = np.concatenate(flavor_to_parts)
    anti = np.concatenate(anti_parts)
    nominal_weight = np.concatenate(weight_parts)
    component_idx = np.concatenate(component_parts)
    reco_energy = _reco_energy(rng, true_energy)

    return (
        true_energy, reco_energy, flavor_from, flavor_to, anti, nominal_weight, component_idx,
    )


# ---------------------------------------------------------------------------
# spline / norm systematic planning (names, sharing structure)
# ---------------------------------------------------------------------------


def _plan_splines(n_spline_systs: int) -> tuple[tuple[str, ...], dict]:
    """Returns ``(shared_names, specific_names_by_sample)``: the first
    ``min(2, n_spline_systs)`` systematics are shared across all four
    samples; the rest are round-robin assigned one-by-one to each sample
    (so with the default ``n_spline_systs=6`` every sample gets exactly one
    sample-specific systematic, per DESIGN.md's toy-generator brief)."""
    n_shared = min(2, n_spline_systs)
    shared_names = tuple(f"shared_spline_{i}" for i in range(n_shared))

    specific_by_sample: dict[str, list[str]] = {n: [] for n in _SAMPLE_NAMES}
    n_remaining = n_spline_systs - n_shared
    for i in range(n_remaining):
        sname = _SAMPLE_NAMES[i % len(_SAMPLE_NAMES)]
        specific_by_sample[sname].append(f"{sname}_spline_{i // len(_SAMPLE_NAMES)}")

    return shared_names, specific_by_sample


def _plan_norms(n_norm_systs: int) -> tuple[dict, list[tuple[str, float]]]:
    """Returns ``(plan, specs)``: ``plan[sample_name]`` is a list of
    ``(norm_name, component_idx)`` pairs that sample's own ``NormSet`` masks
    onto; ``specs`` is ``[(norm_name, prior_sigma), ...]`` for every distinct
    norm name created, in creation order. The first two candidates (NC norm,
    wrongsign norm) are shared by name across every sample; any further
    requested norms are per-sample-specific extra NC norms (round-robin)."""
    plan: dict[str, list[tuple[str, int]]] = {n: [] for n in _SAMPLE_NAMES}
    specs: list[tuple[str, float]] = []

    shared = _NORM_CANDIDATES_SHARED[:n_norm_systs]
    for name, comp_id, sigma in shared:
        for sname in _SAMPLE_NAMES:
            plan[sname].append((name, comp_id))
        specs.append((name, sigma))

    n_remaining = n_norm_systs - len(shared)
    for i in range(n_remaining):
        sname = _SAMPLE_NAMES[i % len(_SAMPLE_NAMES)]
        name = f"{sname}_nc_norm_extra{i // len(_SAMPLE_NAMES)}"
        plan[sname].append((name, _NC))
        specs.append((name, 0.1))

    return plan, specs


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def make_dune_like_analysis(
    *,
    n_events_per_sample: int = 20000,
    n_spline_systs: int = 6,
    n_norm_systs: int = 2,
    seed: int = 0,
    exposure_scale: float = 1.0,
    coeff_dtype=np.float64,
) -> tuple[list, ParameterSet, dict]:
    """Build a self-contained, four-sample DUNE-like toy analysis.

    Four samples -- ``FHC_numu_dis``, ``FHC_nue_app``, ``RHC_numu_dis``,
    ``RHC_nue_app`` -- each with ``n_events_per_sample`` raw MC events split
    into signal / wrongsign / intrinsic-nue / NC components (see module
    docstring), 1300 km baseline through 2.85 g/cc constant-density matter,
    ~40 reco-energy bins from 0.5-8 GeV, ``n_spline_systs`` spline
    systematics (a couple shared by name across all four samples, the rest
    sample-specific) and ``n_norm_systs`` norm systematics masked onto NC /
    wrongsign components. ``exposure_scale`` multiplies every sample's
    nominal weight uniformly (POT-equivalent scaling); ``coeff_dtype`` is the
    spline coefficient table's storage dtype (float64 by default here --
    unlike ``build_spline_table``'s own float32 default -- because this
    generator's primary use is float64 validation, DESIGN.md §12.4).

    Returns ``(samples, parameter_set, truth)``:

    - ``samples``: ``list[SampleModel]``, ready for
      ``build_forward_model(samples, parameter_set)``.
    - ``parameter_set``: the standard six oscillation parameters
      (:func:`~bananafit.parameters.osc_param_specs`) plus every spline
      systematic (``Gaussian(0, 1)`` prior, bounds ``(-3, 3)``) and every
      norm systematic (``Gaussian(1, sigma)`` prior, bounds ``(0, None)``).
    - ``truth``: ``{name: value}`` for every parameter -- NuFIT-like
      oscillation values at their nominal point, every systematic at its
      nominal (0 for splines, 1 for norms).

    Everything is generated from ``numpy.random.default_rng(seed)`` in a
    fixed, deterministic order: identical ``seed`` (and other kwargs)
    reproduces an identical analysis.
    """
    rng = np.random.default_rng(seed)
    truth_osc = mango.nufit_no()

    shared_spline_names, specific_spline_by_sample = _plan_splines(n_spline_systs)
    all_spline_names = list(shared_spline_names)
    for sname in _SAMPLE_NAMES:
        all_spline_names.extend(specific_spline_by_sample[sname])

    # Per-systematic (a_amp, b_amp), drawn once in the fixed name order above
    # so the whole analysis is reproducible from `seed` alone.
    syst_amp = {}
    for name in all_spline_names:
        syst_amp[name] = (rng.uniform(0.08, 0.20), rng.uniform(0.01, 0.03))

    norm_plan, norm_specs = _plan_norms(n_norm_systs)

    samples = []
    for spec in _SAMPLE_SPECS:
        (true_energy, reco_energy, flavor_from, flavor_to, anti,
         nominal_weight, component_idx) = _generate_sample_events(
            rng, spec, n_events_per_sample, truth_osc
        )
        nominal_weight = nominal_weight * exposure_scale

        sample, perm = make_sample(
            spec["name"],
            true_energy=true_energy,
            reco_variable=reco_energy,
            bin_edges=_RECO_BIN_EDGES,
            flavor_from=flavor_from,
            flavor_to=flavor_to,
            anti=anti,
            nominal_weight=nominal_weight,
        )
        component_idx_sorted = component_idx[perm]
        true_energy_sorted = np.asarray(sample.true_energy)

        # -- splines: shared names + this sample's specific names ----------
        this_spline_names = list(shared_spline_names) + specific_spline_by_sample[spec["name"]]
        if this_spline_names:
            knot_x_rows = [_KNOT_X for _ in this_spline_names]
            knot_y_rows = [_curve_values(*syst_amp[n]) for n in this_spline_names]
            event_index_rows = [
                _curve_index(true_energy_sorted, component_idx_sorted) for _ in this_spline_names
            ]
            spline_table = build_spline_table(
                np.stack(knot_x_rows, axis=0),
                np.stack(knot_y_rows, axis=0),
                np.stack(event_index_rows, axis=0),
                names=tuple(this_spline_names),
                boundary="natural",
                dtype=coeff_dtype,
            )
        else:
            spline_table = None

        # -- norms: this sample's plan entries -------------------------------
        this_norms = norm_plan[spec["name"]]
        if this_norms:
            norm_names = [n for n, _ in this_norms]
            selections = [
                (component_idx_sorted == comp_id).astype(np.float64) for _, comp_id in this_norms
            ]
            norm_set = from_selections(norm_names, selections, sample.n_events)
        else:
            norm_set = None

        osc = ConstantMatterOsc(baseline_km=BASELINE_KM, density_gcc=DENSITY_GCC, ye=YE)
        samples.append(SampleModel(events=sample, splines=spline_table, norms=norm_set, osc=osc))

    spline_specs = tuple(
        ParamSpec(name, 0.0, Gaussian(0.0, 1.0), (-3.0, 3.0)) for name in all_spline_names
    )
    norm_param_specs = tuple(
        ParamSpec(name, 1.0, Gaussian(1.0, sigma), (0.0, None)) for name, sigma in norm_specs
    )
    parameter_set = ParameterSet(osc_param_specs() + spline_specs + norm_param_specs)

    truth = {
        "theta12": float(truth_osc.theta12),
        "theta13": float(truth_osc.theta13),
        "theta23": float(truth_osc.theta23),
        "deltacp": float(truth_osc.deltacp),
        "dm21": float(truth_osc.dm21),
        "dm31": float(truth_osc.dm31),
    }
    truth.update({name: 0.0 for name in all_spline_names})
    truth.update({name: 1.0 for name, _ in norm_specs})

    return samples, parameter_set, truth
