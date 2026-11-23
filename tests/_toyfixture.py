"""Synthetic 2-sample DUNE-like analysis, shared by tests/test_model.py and
tests/test_posterior.py.

Builds a small but structurally complete toy analysis exercising every
feature model.py / posterior.py need to handle:

- 2 samples: "numu_dis" (~2000 events, numu->numu CC + NC background) and
  "nue_app" (~1000 events, numu->nue signal + intrinsic nue + NC background).
- DUNE-like true/reco energies (reco = true * (1 + Gaussian smearing)), with
  bin edges chosen so a meaningful fraction of events over/underflow (tests
  the overflow-slot bookkeeping).
- 1300 km baseline / 2.85 g/cc constant matter density.
- 3 spline systematics: "flux_shape" (SHARED -- appears in both samples'
  SplineTable, same ParameterSet entry moves both), "xsec_maqe" (numu_dis
  only, identity (event_index 0) on NC events), "det_smear_nue" (nue_app
  only, identity on NC events). Each curve is the smooth fake response
  w(x) = 1 + a*tanh(x) + b*x^2 knotted on [-3, 3].
- 2 norm parameters ("sig_norm_numu", "bkg_norm_numu"), both scoped to
  numu_dis, masking the CC and NC subsets respectively (nue_app has
  ``norms=None`` -- exercises the "norms may be absent" path).
- ParameterSet = osc_param_specs() + Gaussian-prior systematics.

In addition to the built ``bananafit`` objects (``parameter_set``, ``model``,
``samples``), this module returns a plain-numpy ``raw`` record per sample
carrying everything a fully independent (no bananafit.model/splines/norms
code reused) numpy reference pipeline needs: sorted per-event arrays, raw
spline knot positions/values (pre-Horner-coefficients) and event-curve
indices, and raw norm masks. Test files build their own reference
computation from ``raw`` -- this module does not itself implement one, to
keep it a true independent check.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from bananafit.events import make_sample
from bananafit.model import SampleModel, build_forward_model
from bananafit.norms import from_selections
from bananafit.oscillation import ConstantMatterOsc
from bananafit.parameters import Gaussian, ParamSpec, ParameterSet, osc_param_specs
from bananafit.splines import build_spline_table

BASELINE_KM = 1300.0
DENSITY_GCC = 2.85
YE = 0.5

_KNOT_X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def _curve(a: float, b: float) -> np.ndarray:
    """w(x) = 1 + a*tanh(x) + b*x^2 evaluated at the shared knot grid."""
    return 1.0 + a * np.tanh(_KNOT_X) + b * _KNOT_X**2


@dataclasses.dataclass(frozen=True)
class ToySampleRaw:
    """Plain-numpy, already-sorted per-event data for one sample, enough to
    build a fully independent reference pipeline against."""

    name: str
    true_energy: np.ndarray
    reco_bin: np.ndarray
    n_bins: int
    flavor_from: np.ndarray
    flavor_to: np.ndarray
    anti_split: int
    nominal_weight: np.ndarray
    spline_names: tuple[str, ...]
    spline_knot_x: np.ndarray | None       # (n_syst, n_knots)
    spline_knot_y: np.ndarray | None       # (n_syst, n_curves_in, n_knots), identity NOT included
    spline_event_index: np.ndarray | None  # (n_syst, N), 0 == identity
    norm_names: tuple[str, ...]
    norm_mask: np.ndarray | None           # (n_norm, N)


@dataclasses.dataclass(frozen=True)
class ToyAnalysis:
    parameter_set: ParameterSet
    model: object  # bananafit.model.ForwardModel
    samples: tuple  # tuple[SampleModel, ...], same order as model.samples
    raw: dict  # name -> ToySampleRaw


def _make_energies(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """DUNE-like true energy (Gamma-shaped beam spectrum) and a reco energy
    with 15% Gaussian smearing (clipped positive)."""
    true_energy = rng.gamma(shape=3.0, scale=1.0, size=n) + 0.3
    reco_energy = true_energy * (1.0 + rng.normal(0.0, 0.15, size=n))
    reco_energy = np.clip(reco_energy, 0.05, None)
    return true_energy, reco_energy


def _build_numu_dis(rng: np.random.Generator, n: int):
    true_energy, reco_energy = _make_energies(rng, n)

    is_cc = rng.random(n) < 0.70  # 70% numu->numu CC signal, 30% NC background
    flavor_from = np.where(is_cc, 1, -1).astype(np.int32)
    flavor_to = np.where(is_cc, 1, -1).astype(np.int32)

    anti = (rng.random(n) < 0.05)  # 5% antineutrino contamination
    nominal_weight = rng.uniform(0.5, 1.5, size=n)

    bin_edges = np.linspace(0.5, 6.0, 21)  # 20 bins, deliberately narrower than the spectrum

    sample, perm = make_sample(
        "numu_dis",
        true_energy=true_energy,
        reco_variable=reco_energy,
        bin_edges=bin_edges,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
    )

    # --- spline event-curve indices, computed in the ORIGINAL (pre-sort)
    #     event order, then permuted to match the sorted EventSample. ---
    low_e = true_energy < 2.5
    flux_idx = np.where(low_e, 1, 2).astype(np.int32)          # never identity
    maqe_idx = np.where(is_cc, 1, 0).astype(np.int32)          # NC -> identity
    event_index_raw = np.stack([flux_idx, maqe_idx], axis=0)   # (2, N) pre-sort
    event_index = event_index_raw[:, perm]

    knot_x = np.tile(_KNOT_X, (2, 1))
    knot_y = np.stack([
        np.stack([_curve(0.10, 0.02), _curve(-0.05, 0.01)], axis=0),   # flux_shape: low-E, high-E
        np.stack([_curve(0.15, -0.02), _curve(0.15, -0.02)], axis=0),  # xsec_maqe: real, padding dup
    ], axis=0)  # (2, 2, n_knots)

    sig_mask_raw = is_cc
    bkg_mask_raw = ~is_cc
    norm_mask = np.stack([sig_mask_raw[perm], bkg_mask_raw[perm]], axis=0).astype(np.float64)

    raw = ToySampleRaw(
        name="numu_dis",
        true_energy=np.asarray(sample.true_energy),
        reco_bin=np.asarray(sample.reco_bin),
        n_bins=sample.n_bins,
        flavor_from=np.asarray(sample.flavor_from),
        flavor_to=np.asarray(sample.flavor_to),
        anti_split=sample.anti_split,
        nominal_weight=np.asarray(sample.nominal_weight),
        spline_names=("flux_shape", "xsec_maqe"),
        spline_knot_x=knot_x,
        spline_knot_y=knot_y,
        spline_event_index=event_index,
        norm_names=("sig_norm_numu", "bkg_norm_numu"),
        norm_mask=norm_mask,
    )

    spline_table = build_spline_table(
        knot_x, knot_y, event_index,
        names=("flux_shape", "xsec_maqe"), boundary="natural", dtype=np.float64,
    )
    norm_set = from_selections(
        ("sig_norm_numu", "bkg_norm_numu"), (sig_mask_raw[perm], bkg_mask_raw[perm]), sample.n_events,
    )
    osc = ConstantMatterOsc(baseline_km=BASELINE_KM, density_gcc=DENSITY_GCC, ye=YE)
    sample_model = SampleModel(events=sample, splines=spline_table, norms=norm_set, osc=osc)
    return sample_model, raw


def _build_nue_app(rng: np.random.Generator, n: int):
    true_energy, reco_energy = _make_energies(rng, n)

    r = rng.random(n)
    is_signal = r < 0.40      # numu -> nue
    is_intrinsic = (r >= 0.40) & (r < 0.60)  # beam intrinsic nue -> nue
    # remaining 40% is NC background
    flavor_from = np.full(n, -1, dtype=np.int32)
    flavor_from[is_signal] = 1
    flavor_from[is_intrinsic] = 0
    flavor_to = np.full(n, -1, dtype=np.int32)
    flavor_to[is_signal] = 0
    flavor_to[is_intrinsic] = 0

    anti = (rng.random(n) < 0.05)
    nominal_weight = rng.uniform(0.3, 1.2, size=n)

    bin_edges = np.linspace(0.5, 6.0, 16)  # 15 bins

    sample, perm = make_sample(
        "nue_app",
        true_energy=true_energy,
        reco_variable=reco_energy,
        bin_edges=bin_edges,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
    )

    low_e = true_energy < 2.5
    flux_idx = np.where(low_e, 1, 2).astype(np.int32)  # never identity, same convention as numu_dis
    is_nue_id = flavor_to == 0
    smear_idx = np.where(is_nue_id, 1, 0).astype(np.int32)  # NC -> identity
    event_index_raw = np.stack([flux_idx, smear_idx], axis=0)
    event_index = event_index_raw[:, perm]

    knot_x = np.tile(_KNOT_X, (2, 1))
    knot_y = np.stack([
        np.stack([_curve(0.10, 0.02), _curve(-0.05, 0.01)], axis=0),   # flux_shape: SAME response as numu_dis
        np.stack([_curve(-0.10, 0.03), _curve(-0.10, 0.03)], axis=0),  # det_smear_nue: real, padding dup
    ], axis=0)

    raw = ToySampleRaw(
        name="nue_app",
        true_energy=np.asarray(sample.true_energy),
        reco_bin=np.asarray(sample.reco_bin),
        n_bins=sample.n_bins,
        flavor_from=np.asarray(sample.flavor_from),
        flavor_to=np.asarray(sample.flavor_to),
        anti_split=sample.anti_split,
        nominal_weight=np.asarray(sample.nominal_weight),
        spline_names=("flux_shape", "det_smear_nue"),
        spline_knot_x=knot_x,
        spline_knot_y=knot_y,
        spline_event_index=event_index,
        norm_names=(),
        norm_mask=None,
    )

    spline_table = build_spline_table(
        knot_x, knot_y, event_index,
        names=("flux_shape", "det_smear_nue"), boundary="natural", dtype=np.float64,
    )
    osc = ConstantMatterOsc(baseline_km=BASELINE_KM, density_gcc=DENSITY_GCC, ye=YE)
    sample_model = SampleModel(events=sample, splines=spline_table, norms=None, osc=osc)
    return sample_model, raw


def build_toy_analysis(seed: int = 0, n_dis: int = 2000, n_app: int = 1000) -> ToyAnalysis:
    rng = np.random.default_rng(seed)

    sm_dis, raw_dis = _build_numu_dis(rng, n_dis)
    sm_app, raw_app = _build_nue_app(rng, n_app)

    sys_specs = (
        ParamSpec("flux_shape", 0.0, Gaussian(0.0, 1.0), (-3.0, 3.0)),
        ParamSpec("xsec_maqe", 0.0, Gaussian(0.0, 1.0), (-3.0, 3.0)),
        ParamSpec("det_smear_nue", 0.0, Gaussian(0.0, 1.0), (-3.0, 3.0)),
        ParamSpec("sig_norm_numu", 1.0, Gaussian(1.0, 0.1), (0.0, None)),
        ParamSpec("bkg_norm_numu", 1.0, Gaussian(1.0, 0.2), (0.0, None)),
    )
    parameter_set = ParameterSet(osc_param_specs() + sys_specs)

    samples = (sm_dis, sm_app)
    model = build_forward_model(samples, parameter_set)

    return ToyAnalysis(
        parameter_set=parameter_set,
        model=model,
        samples=samples,
        raw={"numu_dis": raw_dis, "nue_app": raw_app},
    )
