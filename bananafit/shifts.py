"""FD reconstructed-energy shift systematics + differentiable binning.

Background
----------
Unlike the spline systematics in ``splines.py`` (a fixed multiplicative
response curve baked in at build time), far-detector (FD) energy-scale and
resolution systematics are *shift* systematics: the value of the nuisance
parameter moves each event's reconstructed energy before it is re-binned, at
every MCMC step, so an event that crosses an analysis bin boundary is counted
in its new bin. This is how MaCh3_DUNE implements them (``DUNE/MaCh3_DUNE``,
``Samples/SampleHandlerBeamFD.cpp``). This module implements the mechanism in
JAX: a pure, differentiable function from (nominal per-particle-group
energies, systematic parameter values) to (shifted total reco energy,
soft/hard bin assignment).

Equations
---------
Energy scale, per particle group::

    Delta E = E_rec * (p0 + p1 * sqrt(E_rec) + p2 / sqrt(E_rec))

Resolution, per particle group::

    Delta E = r * (E_true - E_rec)

Both are *additive* shifts, ``E_rec' = E_rec + sum(Delta E)``, and every
parameter's nominal (generated and pre-fit) value is 0, so the nominal point
leaves the spectrum unchanged. The resolution term interpolates between the
nominal smearing (``r = 0``: ``E_rec' = E_rec``) and perfect resolution
(``r = 1``: ``E_rec' = E_true``).

Group structure: the 19 parameters
----------------------------------
Five particle groups -- Total (calorimetric, everything but the muon),
Hadronic (p, pi+/-), mu, n, electromagnetic (e, gamma, pi0) -- each with 3
scale parameters (p0, p1, p2), plus a resolution parameter r for every group
except Total: 4 groups x 4 params + 1 group x 3 params = 19
(:data:`FD_DETECTOR_PARAM_NAMES`). The Total group's ``E_rec`` is the
calorimetric sum (hadronic + EM + neutron; the muon energy comes from range
or multiple scattering instead), shifted by an overall calibration term; each
particle-species group *additionally* shifts its own sub-component (a
calibration uncertainty and a per-species response uncertainty are different
effects, both active at once). All five group shifts are evaluated *in
parallel* against the nominal (unshifted) per-group energies, not
sequentially, which is also how every other systematic response in this
package (``splines.py``, ``norms.py``) is defined: relative to a fixed
nominal, not a running total.

MaCh3_DUNE's ``Configs/CovObjs/tdr_covs/DetSysts_FD.yaml`` (30 entries) holds
this 19-parameter core, plus 7 "NotCCNumu"/"CCNue" split-variant parameters
(a CAFAna-derived correction applied only to a kinematic subset -- see the
YAML's own comment citing ``CAFAna/Systs/EnergySysts.h``), 2 CVN
selection-efficiency parameters (``RecoCVNNumuFD``, ``RecoCVNNueFD``) and 2
fiducial-volume normalisations (``NumuFVFD``, ``NueFVFD``, expressible with
``norms.py`` and not handled here). :class:`EnergyScaleModel` implements the
19-parameter core; the split-variant parameters are not implemented (the
extension is an extra per-event kinematic mask with the same formulas, added
into the same group's ``Delta E``).

Cross-check against the MaCh3_DUNE C++ source (independent confirmation)
--------------------------------------------------------------------------
The additive-shift, parallel-composition reading above was checked
against the actual MaCh3_DUNE implementation, ``DUNE/MaCh3_DUNE`` @
``30336a89ed9a5dd2095d5d7d6c3fa7724b011911``, file ``Samples/SampleHandlerBeamFD.cpp``:

- ``ResetShifts`` (l. 325-329) sets ``rw_erec_shifted = rw_erec`` (the nominal
  total reco energy) at the start of every reweight -- exactly this module's
  ``E_rec' = E_rec + sum(Delta E)`` starting point.
- Every registered "functional parameter" callback then does
  ``rw_erec_shifted += (*par) * <some energy quantity>``, e.g. (l. 98-107,
  ``HadEScale``/``HadEScaleSqrt``/``HadEScaleInvSqrt``)::

      rw_erec_shifted += par * rw_sum_ehad;                        // p0 * E
      rw_erec_shifted += par * rw_sum_ehad * rw_sum_ehad_sqrt;      // p1 * E * sqrt(E)
      rw_erec_shifted += par * rw_sum_ehad_sqrt;                    // p2 * sqrt(E)  (== p2*E/sqrt(E))

  with ``rw_sum_ehad = eRecoP + eRecoPip + eRecoPim`` (l. 566, the "Hadronic
  (p, pi+/-)" group) and ``rw_sum_ehad_sqrt = sqrt(rw_sum_ehad)`` (l. 571) --
  i.e. exactly the energy-scale formula's three terms, added independently (confirming the
  *additive*, not replacement, reading), and confirming each group is scaled
  against its *own* nominal energy (parallel composition, not sequential --
  every callback reads a plain ``dunemcSamples[iEvent].rw_*`` field computed
  once at load time, l. 561-571, never ``rw_erec_shifted`` itself). Resolution
  callbacks (l. 164-185, ``HadRes``/``MuRes``/``NRes``/``EMRes``) are exactly
  the resolution formula, e.g. ``HadRes``: ``rw_erec_shifted += par * (rw_eP+rw_ePip+rw_ePim -
  rw_sum_ehad)``. ``TotalEScale`` (l. 65-70) scales ``rw_erec_had``
  (``RecoHadEnNumu``/``RecoHadEnNue``, the broader "everything but the primary
  lepton" reco energy -- *not* the same field as ``rw_sum_ehad``/``e_had_*``
  here, which is the narrower charged-hadron-only sum); this module's
  ``e_calo = e_had + e_em + e_n`` recombination is the closest available
  parallel given ``EnergyComponents``' 4-group (not 5-field) schema.
- The "NotCCNumu"/"CCNue" split-variant parameters this module does *not*
  implement (see "Group structure" above) are, in the C++, forced-equal-by-
  construction companions: e.g. ``TotalEScaleNotCCNumuFD`` is a *separate*
  registered parameter from ``TotalEScaleFD``, but ``DetSysts_FD.yaml`` gives
  it ``Correlations: {TotalEScaleFD: 1.0}`` with an *identical* ``Error`` --
  a degenerate (rank-deficient) correlated Gaussian prior that only supports
  the two parameters being numerically equal, so it is physically the same
  free parameter applied to a second energy component
  (``rw_erec_lep``/``RecoLepEnNumu``-or-``Nue``, gated by ``KinematicCuts:
  IsTrueCCnumu: [-0.1, 0.1]`` and restricted ``SampleNames`` -- l. 72-77,
  110-123, 141-161 of the C++; ``Configs/CovObjs/tdr_covs/DetSysts_FD.yaml``
  for the correlation/cut declarations; ``rw_trueccnumu``/``rw_trueccnue``
  computed at l. 582-583 as ``isCC && |nuPDG| == 14``/``12``, matching this
  module's ``EnergyComponents.is_ccnumu``/``is_ccnue`` contract exactly).
- The CVN selection cut this module's :func:`cvn_efficiency_weight`
  approximates is a genuine hard cut on the *shifted* score: ``KinPar ==
  kCVNNumu``/``kCVNNue`` resolve to ``rw_cvnnumu_shifted``/``rw_cvnnue_shifted``
  (``GetPointerToKinematicParameter``, l. 620-623), and
  ``Configs/Samples/SampleHandler_FD_Beam.yaml`` applies ``SelectionCuts:
  CVNNumu > 0.5`` (numu samples) / ``CVNNue > 0.85`` (nue samples) against
  that live pointer -- re-evaluated every reweight, so a shifted score can
  cross the cut and move the event's contribution into or out of that
  sample's spectrum. These are the ``cut``/threshold values a caller should
  pass to :func:`cvn_efficiency_weight`.

CVN selection-efficiency shifts (approximation, DEVIATION FLAGGED)
--------------------------------------------------------------------
In MaCh3_DUNE the CVN shift is literal event migration: shifting a CVN
classifier score can move an event
across the numu/nue selection boundary into a different sample entirely,
changing which sample's spectrum it contributes to. Implementing that exactly
would require restructuring events dynamically across two different
:class:`~bananafit.model.SampleModel` objects inside the hot ``logdensity``
path -- a materially larger architectural change than a per-event weight
(dynamic-shape data movement is exactly what this package's "static shapes,
precompute at build time" design (DESIGN.md §1) rules out). Instead,
:func:`cvn_efficiency_weight` implements the documented *approximation*: a
smooth (sigmoid) per-event selection-probability weight as a function of the
shifted CVN score, degenerating to a hard step at the selection cut as the
softness parameter shrinks. This changes each event's *effective weight*
within its own sample rather than moving it to another sample -- it will not
reproduce MaCh3's bin-to-bin migration between the numu- and nue-selected
spectra. Flagged here for later review/replacement with true migration if
that fidelity is needed.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

__all__ = [
    "EnergyComponents",
    "EnergyScaleModel",
    "EnergyShift",
    "FD_DETECTOR_PARAM_NAMES",
    "soft_bin_weights",
    "hard_bin_index",
    "cvn_efficiency_weight",
]


def _static_field(**kwargs):
    return dataclasses.field(metadata=dict(static=True), **kwargs)


# The 19 FD detector parameter names, in the fixed order EnergyScaleModel.shift
# unpacks its (19,) parameter subvector -- matching the FancyName strings in
# MaCh3_DUNE's Configs/CovObjs/tdr_covs/DetSysts_FD.yaml exactly
# (module docstring "Group structure" section) so that resolving these names
# against a ParameterSet, MaCh3-DUNE-config-derived or not, is unambiguous.
FD_DETECTOR_PARAM_NAMES = (
    "TotalEScaleFD", "TotalEScaleSqrtFD", "TotalEScaleInvSqrtFD",
    "HadEScaleFD", "HadEScaleSqrtFD", "HadEScaleInvSqrtFD", "HadResFD",
    "MuEScaleFD", "MuEScaleSqrtFD", "MuEScaleInvSqrtFD", "MuResFD",
    "NEScaleFD", "NEScaleSqrtFD", "NEScaleInvSqrtFD", "NResFD",
    "EMEScaleFD", "EMEScaleSqrtFD", "EMEScaleInvSqrtFD", "EMResFD",
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnergyComponents:
    """Per-event reconstructed and true energy, split by particle group
    (the ``eReco<Species>``/``e<Species>`` CAF branches, regrouped into 4
    particle-response groups).

    This is the ingestion contract: the analysis's FD-CAF-ingestion code
    is responsible for producing these
    arrays; this module only consumes them. All fields are ``(N,)`` f64,
    aligned 1:1 with the ``EventSample`` they accompany (same event order --
    build them from the same permutation ``make_sample``/``make_sample_prebinned``
    return, exactly as ``SplineTable.event_index``/``NormSet.mask`` columns
    must be).

    - ``e_had_reco``/``e_had_true``: hadronic group (p, pi+-) --
      CAF ``eRecoP + eRecoPip + eRecoPim`` / ``eP + ePip + ePim``.
    - ``e_mu_reco``/``e_mu_true``: muon group -- the reconstructed/true
      outgoing-muon energy (CAF ``RecoLepEnNumu``-derived / ``LepE`` for
      true-CC-numu events; 0 for events with no true/reco muon).
    - ``e_n_reco``/``e_n_true``: neutron group -- CAF ``eRecoN`` / ``eN``.
    - ``e_em_reco``/``e_em_true``: electromagnetic group (e, gamma, pi0) --
      CAF ``eRecoPi0 + eRecoOther`` / ``ePi0 + eOther`` (``eOther`` bundles
      the non-pi0 EM-like remainder; a true outgoing CC-nue electron's energy
      belongs here too if not already folded into ``eOther`` by the
      ingestion pipeline).
    - ``is_ccnumu``/``is_ccnue``: ``(N,)`` f64 in ``{0, 1}`` -- true-CCnumu /
      true-CCnue event flags (CAF-derived ``IsTrueCCnumu``/``IsTrueCCnue``
      kinematic cuts referenced in ``DetSysts_FD.yaml``), for the
      NotCCNumu/CCNue split-variant parameters *not yet implemented* by
      :class:`EnergyScaleModel` (see module docstring) -- carried here so
      that extension does not require re-touching the ingestion contract.

    Consistency requirement for exact zero-shift parity (tested in
    ``tests/test_shifts.py``): the reco energy an ``EventSample`` was binned
    on must equal ``e_mu_reco + e_had_reco + e_em_reco + e_n_reco`` exactly,
    since :class:`EnergyScaleModel` computes the shifted *total* from the sum
    of these four components, not from a separately-stored total.
    """

    e_had_reco: jax.Array
    e_had_true: jax.Array
    e_mu_reco: jax.Array
    e_mu_true: jax.Array
    e_n_reco: jax.Array
    e_n_true: jax.Array
    e_em_reco: jax.Array
    e_em_true: jax.Array
    is_ccnumu: jax.Array
    is_ccnue: jax.Array

    @property
    def n_events(self) -> int:
        return self.e_had_reco.shape[0]

    @property
    def e_total_reco(self) -> jax.Array:
        """Nominal total reco energy: sum of the four group components."""
        return self.e_mu_reco + self.e_had_reco + self.e_em_reco + self.e_n_reco


def _safe_scale_shift(e_rec: jax.Array, p0, p1, p2) -> jax.Array:
    """Eq 12 additive term: ``e_rec * (p0 + p1*sqrt(e_rec) + p2/sqrt(e_rec))``,
    double-where guarded (jaxnu ``_safe_sqrt`` idiom, see
    ``jaxnu.nufast._safe_sqrt``) so ``1/sqrt(e_rec)`` never sees ``e_rec <= 0``
    on the gradient path. Events with no energy in this group (``e_rec <= 0``,
    e.g. no true hadronic activity) get exactly zero shift -- there is no
    particle of that species to which a species-specific response
    uncertainty could apply.
    """
    pos = e_rec > 0.0
    e_safe = jnp.where(pos, e_rec, 1.0)
    sqrt_e = jnp.sqrt(e_safe)
    factor = p0 + p1 * sqrt_e + p2 / sqrt_e
    raw = e_safe * factor
    return jnp.where(pos, raw, 0.0)


def _res_shift(e_true: jax.Array, e_rec: jax.Array, r) -> jax.Array:
    """Eq 13 additive term: ``r * (e_true - e_rec)``. Polynomial in finite
    inputs -- no where-guarding needed (same reasoning as ``norms.py``)."""
    return r * (e_true - e_rec)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnergyScaleModel:
    """The 19-parameter FD energy-scale + resolution shift model
    (module docstring for the full derivation/composition rationale).

    Stateless (no array fields -- ``components`` and the parameter
    subvector are passed to :meth:`shift` at call time); registered as a
    pytree purely so it can sit inside a larger frozen-dataclass model
    without special-casing.

    ``floor_at_zero`` (static, default ``True`` = the historical behaviour):
    whether :meth:`shift` clamps the shifted total at 0.  With the clamp a
    negative shifted energy lands in bin 0 under hard binning;
    MaCh3 instead drops it
    (``FindXBin`` returns no bin outside the analysis range).  Pass
    ``floor_at_zero=False`` together with ``EnergyShift(mode="hard")`` for
    the MaCh3-faithful drop (``hard_bin_index`` maps ``E < edges[0]`` to the
    overflow slot).  Opt-in only: the soft path needs the clamp.
    """

    floor_at_zero: bool = _static_field(default=True)

    def shift(self, components: EnergyComponents, x: jax.Array) -> jax.Array:
        """``x``: ``(19,)`` f64, values for :data:`FD_DETECTOR_PARAM_NAMES` in
        that exact order (sigma-like units -- 0 is nominal). Returns the
        shifted total reco energy ``(N,)`` f64, floored at 0 (a shift can't
        be trusted to produce a physically negative energy; the floor's
        subgradient at exactly 0 is well-defined and only ever engaged for
        pathological/very-large parameter excursions well outside any
        sensible prior range) unless ``floor_at_zero=False``.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        (
            p0_tot, p1_tot, p2_tot,
            p0_had, p1_had, p2_had, r_had,
            p0_mu, p1_mu, p2_mu, r_mu,
            p0_n, p1_n, p2_n, r_n,
            p0_em, p1_em, p2_em, r_em,
        ) = (x[i] for i in range(19))

        e_had, e_had_t = components.e_had_reco, components.e_had_true
        e_mu, e_mu_t = components.e_mu_reco, components.e_mu_true
        e_n, e_n_t = components.e_n_reco, components.e_n_true
        e_em, e_em_t = components.e_em_reco, components.e_em_true

        e_calo = e_had + e_em + e_n  # "Total (except muons)" group's E_rec argument

        d_calo_total = _safe_scale_shift(e_calo, p0_tot, p1_tot, p2_tot)
        d_had = _safe_scale_shift(e_had, p0_had, p1_had, p2_had) + _res_shift(e_had_t, e_had, r_had)
        d_em = _safe_scale_shift(e_em, p0_em, p1_em, p2_em) + _res_shift(e_em_t, e_em, r_em)
        d_n = _safe_scale_shift(e_n, p0_n, p1_n, p2_n) + _res_shift(e_n_t, e_n, r_n)
        d_mu = _safe_scale_shift(e_mu, p0_mu, p1_mu, p2_mu) + _res_shift(e_mu_t, e_mu, r_mu)

        e_calo_shifted = e_calo + d_calo_total + d_had + d_em + d_n
        e_mu_shifted = e_mu + d_mu
        e_total_shifted = e_mu_shifted + e_calo_shifted
        if not self.floor_at_zero:
            return e_total_shifted
        return jnp.maximum(e_total_shifted, 0.0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnergyShift:
    """Bundles the per-sample energy-shift machinery for
    :class:`~bananafit.model.SampleModel`'s optional ``energy_shift`` field:
    the event-level components, the (stateless) shift model, and the static
    binning configuration.

    ``components``/``model`` are pytree children (``model`` carries no
    arrays, but is registered for uniformity); ``param_names``, ``bin_edges``
    and ``mode`` are static, exactly like ``SplineTable.names`` /
    ``NormSet.names`` -- resolved against a ``ParameterSet`` once, at
    :func:`~bananafit.model.build_forward_model` time.

    ``param_names``: which entries of the global physical parameter vector
    feed :meth:`EnergyScaleModel.shift`, in the order it expects them
    (default :data:`FD_DETECTOR_PARAM_NAMES`, the 19-parameter core).

    ``bin_edges``: the analysis bin edges to re-digitize the shifted energy
    against -- must match the edges the accompanying ``EventSample`` was
    itself binned with at zero shift (tested in ``tests/test_shifts.py``),
    since :meth:`EnergyScaleModel.shift` returns exactly ``e_total`` (the sum
    of ``components``' four group energies) when every parameter is 0, and
    that must digitize to the same ``reco_bin`` the ``EventSample`` already
    carries for the fast fixed-bin path to be recoverable as a special case.

    ``mode``: ``"soft"`` (default, differentiable tent-kernel binning via
    :func:`soft_bin_weights`) or ``"hard"`` (exact MaCh3-parity binning via
    :func:`hard_bin_index` -- matches MaCh3 bin-for-bin, but the binned
    prediction is then piecewise-constant in the shift parameters, so its
    gradient is zero almost everywhere and exactly undefined at a bin
    boundary; unsuitable for gradient-based samplers, useful for a
    MaCh3-comparison / random-walk-MCMC path).
    """

    components: EnergyComponents
    model: EnergyScaleModel
    param_names: tuple[str, ...] = _static_field(default=FD_DETECTOR_PARAM_NAMES)
    bin_edges: tuple[float, ...] = _static_field(default=None)
    mode: str = _static_field(default="soft")

    def __post_init__(self):
        if self.bin_edges is None:
            raise ValueError("EnergyShift.bin_edges is required")
        if self.mode not in ("soft", "hard"):
            raise ValueError(f"EnergyShift.mode must be 'soft' or 'hard', got {self.mode!r}")
        if len(self.param_names) != 19:
            raise ValueError(
                f"EnergyShift.param_names must have 19 entries (the "
                f"energy-scale/resolution core, see FD_DETECTOR_PARAM_NAMES), got "
                f"{len(self.param_names)}"
            )


# ---------------------------------------------------------------------------
# Differentiable binning
# ---------------------------------------------------------------------------


def hard_bin_index(e_shifted: jax.Array, bin_edges: jax.Array) -> jax.Array:
    """Exact MaCh3-parity binning: half-open bins ``[edges[k], edges[k+1])``,
    same convention as ``events.py``'s (numpy, build-time) ``_digitize`` --
    ``right=False`` digitization, both under- and overflow mapped to the
    overflow slot ``n_bins`` (``len(bin_edges) - 1``). Unlike ``_digitize``
    this is a pure ``jax.numpy`` function (jit/vmap-safe, usable inside the
    traced hot path since the *bin edges* are static but ``e_shifted`` is
    traced) -- there is no meaningful gradient of an integer bin index
    itself, this is the "hard" (non-smooth) path.

    Parameters
    ----------
    e_shifted : (N,) f64.
    bin_edges : (n_bins + 1,) f64, strictly increasing.
    """
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    n_bins = edges.shape[0] - 1
    # method="compare_all": the edge array is tiny (n_bins + 1 static edges), so one fused
    # compare-and-count per event beats the default binary search, which lowers to a
    # ceil(log2(n_bins + 2))-iteration XLA while loop over the (N,) event array.  Same
    # total-order comparator as the default method, so the index is identical for every
    # input, NaN included.
    idx = jnp.searchsorted(edges, e_shifted, side="right", method="compare_all") - 1
    out_of_range = (e_shifted < edges[0]) | (e_shifted >= edges[-1])
    idx = jnp.where(out_of_range, n_bins, idx)
    return idx.astype(jnp.int32)


def soft_bin_weights(e_shifted: jax.Array, bin_edges: jax.Array):
    """Differentiable binning via a tent (linear) kernel between adjacent
    bin *centers*: returns ``(bin_lo, frac_lo)`` such that an event's weight
    should be split ``frac_lo`` into bin ``bin_lo`` and ``(1 - frac_lo)``
    into bin ``bin_lo + 1``. Degenerates exactly to hard binning
    (``frac_lo in {0, 1}``) for an event at or beyond a bin center's own
    reach into its neighbor, and is smooth (grad-safe) everywhere in
    between -- this is what lets an MCMC step move an event's shifted energy
    across a bin boundary without a discontinuous jump in the binned
    prediction (the effect ``hard_bin_index`` alone cannot give a gradient
    through).

    Construction: bin centers ``c_k = (edges[k] + edges[k+1]) / 2``. For
    ``e`` between ``c_k`` and ``c_{k+1}``, ``frac_lo = (c_{k+1} - e) /
    (c_{k+1} - c_k)`` linearly interpolates from 1 (at ``c_k``) to 0 (at
    ``c_{k+1}``); clipping this to ``[0, 1]`` after locating the nearest
    interior segment automatically saturates to hard-bin-0/hard-bin-(n-1)
    membership for events beyond the first/last center (no explicit
    special-casing needed for the two edge bins). Total weight is
    conserved by construction (``frac_lo + (1 - frac_lo) = 1`` identically).

    Events outside ``[edges[0], edges[-1])`` (MaCh3/``events.py`` overflow
    convention) get ``bin_lo = n_bins`` (the overflow slot) and
    ``frac_lo = 1.0`` -- entirely in the overflow slot, consistent with
    :func:`hard_bin_index` and ``events.py``'s digitization.

    Parameters
    ----------
    e_shifted : (N,) f64.
    bin_edges : (n_bins + 1,) f64, strictly increasing, ``n_bins >= 2``
        (need at least 2 bins to have a neighbor to share weight with).

    Returns
    -------
    (bin_lo, frac_lo) : ``(N,) int32``, ``(N,) f64``.
    """
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    n_bins = edges.shape[0] - 1
    centers = 0.5 * (edges[:-1] + edges[1:])  # (n_bins,)

    e = jnp.asarray(e_shifted, dtype=jnp.float64)
    in_range = (e >= edges[0]) & (e < edges[-1])

    # Lower-neighbor-center index, clamped to [0, n_bins - 2] so c_lo/c_hi
    # always index two real, distinct centers -- clipping frac_lo (below)
    # then correctly saturates to hard membership for e outside [c_0, c_{-1}].
    k = jnp.clip(jnp.searchsorted(centers, e, side="right", method="compare_all") - 1, 0, n_bins - 2)
    c_lo = centers[k]
    c_hi = centers[k + 1]
    frac_interp = (c_hi - e) / (c_hi - c_lo)
    frac_clipped = jnp.clip(frac_interp, 0.0, 1.0)

    bin_lo = jnp.where(in_range, k, n_bins).astype(jnp.int32)
    frac_lo = jnp.where(in_range, frac_clipped, 1.0)
    return bin_lo, frac_lo


# ---------------------------------------------------------------------------
# CVN selection-efficiency shift (approximation -- see module docstring)
# ---------------------------------------------------------------------------


def cvn_efficiency_weight(cvn_score: jax.Array, shift, cut: float, *, softness: float = 0.01):
    """DEVIATION FROM EXACT MaCh3 BEHAVIOR -- see module docstring
    "CVN selection-efficiency shifts" section: this is a per-event soft
    selection-probability *weight*, not the literal cross-sample event
    migration MaCh3_DUNE performs.

    Models "shift the CVN score by ``shift``, then re-apply the selection
    cut" as a smooth sigmoid rather than a hard pass/fail, so it stays
    differentiable: ``weight = sigmoid((cvn_score + shift - cut) / softness)``,
    which -> the hard step function ``1[cvn_score + shift > cut]`` as
    ``softness -> 0``. Multiply this into a sample's per-event weight (e.g.
    via ``SampleModel.extra_weight``) to approximate the selection-efficiency
    penalty; it does *not* move the event's contribution into a different
    sample's spectrum.

    Parameters
    ----------
    cvn_score : (N,) f64 -- nominal CVN classifier score for this event.
    shift : scalar f64 -- the RecoCVNNumuFD/RecoCVNNueFD parameter value
        (sigma-like units, prior width 0.01).
    cut : float -- the selection threshold the nominal analysis cuts on.
    softness : float -- sigmoid width; smaller tracks the hard cut more
        closely at the cost of a steeper (but still finite) gradient.
    """
    cvn_score = jnp.asarray(cvn_score, dtype=jnp.float64)
    shift = jnp.asarray(shift, dtype=jnp.float64)
    return jax.nn.sigmoid((cvn_score + shift - cut) / softness)
