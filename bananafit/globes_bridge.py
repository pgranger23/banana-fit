"""Bridge from ``mango.globes.GlobesExperiment`` to bananafit ``SampleModel``\\ s.

``mango.globes`` computes, per channel ``c``, a reconstructed-energy spectrum

    N_c[i] = sum_j K_c[i, j] * P_c[j] + const_c[i]

where the sum runs over the GLoBES *sampling* (true-energy) grid nodes ``j``
(``exp.samp_c``, weighted by the sampling bin width ``exp.samp_w`` -- a
midpoint/rectangle quadrature rule already folded into ``K_c`` at parse time,
see ``GlobesExperiment._precompute``), ``K_c[i, j]`` is the *oscillation-
independent* kernel

    K_c[i, j] = post_eff_c[i] * R_c[i, j] * flux_c[j] * xsec_c[j]
                * samp_w[j] * norm * pre_eff_c[j]

(``R_c`` the smearing/migration matrix true->reco, ``norm`` the POT x
nucleons x unit-conversion x ``scale`` factor from ``GlobesExperiment._norm``),
``P_c[j]`` is the oscillation probability at sampling node ``j`` (or exactly 1
for a ``NOSC_`` channel -- GLoBES' notation for "count this rate without
applying oscillation", used here for NC channels), and ``const_c[i]`` is an
oscillation-independent additive background (smeared ``pre_bkg`` plus direct
``post_bkg``; for the DUNE TDR config both are absent so ``const_c`` is
identically zero, but the general path below still exists). A ``rule``
(``exp.rules[...]``) sums ``coef * N_c`` over its signal and background
channel lists, unweighted by ``coef`` != 1 in this config (all coefficients
are ``1.0``), giving ``exp.spectra()[rule]``.

This module turns that *exact* sum-of-products into a set of bananafit
"events": one event per nonzero ``(i, j)`` kernel cell (``true_energy =
exp.samp_c[j]``, ``reco_bin = i``, ``nominal_weight = coef * K_c[i, j]``,
``flavor_from/to`` = the channel's initial/final flavor, or the bananafit
``-1`` "unoscillated" convention for a ``NOSC_`` channel), plus one event per
nonzero ``const_c[i]`` entry (same ``-1``/``-1`` unoscillated convention, so
its bananafit oscillation weight is forced to exactly ``1.0`` regardless of
the physics point -- reproducing the additive, oscillation-independent
``+ const_c[i]`` term). Concatenating every channel referenced by a rule
(signal and background alike, each cell scaled by its GLoBES ``coef``) into
one ``EventSample``, then letting bananafit's ``segment_sum``-by-``reco_bin``
regroup by ``i``, reproduces ``K_c @ P_c + const_c`` summed over channels --
i.e. ``exp.spectra()[rule]`` -- bin for bin, up to floating-point summation
order (see the exactness test in ``tests/test_globes_bridge.py``).

The oscillation probabilities themselves come from
``bananafit.oscillation.ConstantMatterOsc``, matching
``GlobesExperiment._osc``'s use of ``mango.probability_constant`` at the
*same* baseline/density/ye and the *same* (default, "nufast") backend --
this only works for a single-layer matter profile (``len(exp.lengths) ==
1``, true for the DUNE TDR ``$profiletype = 3`` config with one
``$densitytab``/``$lengthtab`` entry each); a layered profile
(``probability_profile``) is not implemented here (see the ``NotImplementedError``
below) since bananafit ships no profile-based ``OscConfig`` to target.

No splines, norms, or ``extra_weight`` hooks are attached here (DESIGN.md
§7's "systematics attached later" hook is left to the caller) -- every
returned ``SampleModel`` has ``splines=None, norms=None``.
"""

from __future__ import annotations

import numpy as np

from .events import make_sample_prebinned
from .model import SampleModel
from .oscillation import ConstantMatterOsc

__all__ = ["samples_from_globes"]


def _rebin_map(fine_edges: np.ndarray, coarse_edges: np.ndarray, *, atol: float = 1e-9):
    """Map each native (fine) reco bin index to a coarse bin index.

    ``coarse_edges`` must be a strictly increasing subset of ``fine_edges``
    (within ``atol``) -- i.e. every coarse bin is an exact union of whole
    fine bins, so regrouping by this map is an *exact* rebinning (no partial
    fine-bin splitting), equivalent to ``np.add.reduceat`` at matching
    indices. Fine bins outside ``[coarse_edges[0], coarse_edges[-1])`` map to
    the overflow slot ``n_coarse`` (bananafit's convention: ``reco_bin ==
    n_bins`` is dropped from the likelihood).
    """
    fine_edges = np.asarray(fine_edges, dtype=np.float64)
    coarse_edges = np.asarray(coarse_edges, dtype=np.float64)
    if coarse_edges.ndim != 1 or coarse_edges.shape[0] < 2:
        raise ValueError("rebin_edges must be a 1-D array of length >= 2")
    if not np.all(np.diff(coarse_edges) > 0):
        raise ValueError("rebin_edges must be strictly increasing")

    idx = []
    for e in coarse_edges:
        m = np.nonzero(np.isclose(fine_edges, e, atol=atol, rtol=0.0))[0]
        if m.size == 0:
            raise ValueError(
                f"rebin edge {e!r} does not align with any native GLoBES "
                "reco bin edge; rebin_edges must be a subset of the "
                "experiment's fine reco_edges for an exact rebinning"
            )
        idx.append(int(m[0]))
    idx = np.asarray(idx, dtype=np.int64)

    n_coarse = coarse_edges.shape[0] - 1
    n_fine = fine_edges.shape[0] - 1
    bin_map = np.full(n_fine, n_coarse, dtype=np.int32)  # default: overflow
    for k in range(n_coarse):
        bin_map[idx[k]:idx[k + 1]] = k
    return bin_map, n_coarse


def samples_from_globes(exp, *, rebin_edges=None, min_weight: float = 0.0):
    """Turn a ``mango.globes.GlobesExperiment`` into bananafit ``SampleModel``\\ s.

    One ``SampleModel`` (and one ``EventSample``) per GLoBES rule
    (``exp.rules``), replicating ``exp.spectra()`` exactly (see module
    docstring). No energy window is applied (``exp.spectra()`` does not
    apply ``@energy_window`` either -- that only happens inside
    ``exp.chi2``); apply a window downstream (e.g. via ``rebin_edges``, or a
    likelihood-level mask) if desired.

    Parameters
    ----------
    exp : mango.globes.GlobesExperiment
        A loaded GLoBES experiment (``mango.globes.load(...)``).
    rebin_edges : dict[str, array_like] or None
        Optional coarser reco-bin edges, keyed by rule name. Each array must
        be strictly increasing and every edge must coincide (within
        ``1e-9``) with one of ``exp.reco_edges`` -- i.e. every coarse bin is
        an exact union of native GLoBES fine bins, so the rebinning is exact
        (see :func:`_rebin_map`). Rules not present in the dict (or when
        ``rebin_edges`` is ``None``) keep the native GLoBES binning. Native
        fine bins outside a rule's coarse range fall into bananafit's
        overflow slot (dropped from the likelihood).
    min_weight : float
        Only cells (kernel entries or additive-background entries) with
        weight strictly greater than ``min_weight`` become events. The
        default ``0.0`` drops only exact zeros (mostly the off-band entries
        of the banded smearing matrix) -- an exact, lossless prune. A
        positive value trades a small (quantifiable) approximation for fewer
        events.

    Returns
    -------
    (list[SampleModel], dict)
        One ``SampleModel`` per rule, in ``exp.rules`` iteration order, plus
        a metadata dict (rule names -> edges/channel decomposition/event
        counts, and the matter-profile parameters used) for debugging and
        plotting.
    """
    if len(exp.lengths) != 1:
        raise NotImplementedError(
            "samples_from_globes only supports a single-layer matter "
            "profile (GlobesExperiment.lengths of length 1, matching "
            "bananafit.oscillation.ConstantMatterOsc); this experiment has "
            f"{len(exp.lengths)} layers ($profiletype with multiple "
            "densitytab/lengthtab entries or a PREM chord) -- a layered-"
            "profile OscConfig is not implemented in bananafit, so bridging "
            "this experiment would require adding one rather than "
            "approximating it here."
        )

    baseline_km = float(exp.baseline)
    density_gcc = float(exp.density[0])
    ye = float(exp.ye)

    samp_c = np.asarray(exp.samp_c, dtype=np.float64)
    reco_c = np.asarray(exp.reco_c, dtype=np.float64)
    fine_edges = np.asarray(exp.reco_edges, dtype=np.float64)
    nreco = exp.nreco

    samples = []
    meta = {
        "baseline_km": baseline_km,
        "density_gcc": density_gcc,
        "ye": ye,
        "native_reco_edges": fine_edges,
        "rules": {},
    }

    for rule_name, rule in exp.rules.items():
        parts = [(coef, cn, True) for coef, cn in rule.signal]
        parts += [(coef, cn, False) for coef, cn in rule.background]

        te_parts, rb_parts, ff_parts, ft_parts, anti_parts, w_parts = [], [], [], [], [], []
        chan_meta = []

        for coef, cn, is_signal in parts:
            ch = exp.channels[cn]
            K = np.asarray(exp.kernels[cn], dtype=np.float64) * coef
            const = np.asarray(exp.const[cn], dtype=np.float64) * coef

            if ch.noosc:
                flav_from, flav_to = -1, -1
            else:
                flav_from, flav_to = int(ch.src), int(ch.dst)

            i_idx, j_idx = np.nonzero(K > min_weight)
            n_kernel = int(i_idx.size)
            kernel_weight_sum = float(K[i_idx, j_idx].sum()) if n_kernel else 0.0
            if n_kernel:
                te_parts.append(samp_c[j_idx])
                rb_parts.append(i_idx.astype(np.int32))
                ff_parts.append(np.full(n_kernel, flav_from, dtype=np.int32))
                ft_parts.append(np.full(n_kernel, flav_to, dtype=np.int32))
                anti_parts.append(np.full(n_kernel, bool(ch.anti), dtype=bool))
                w_parts.append(K[i_idx, j_idx])

            const_mask = const > min_weight
            n_const = int(np.count_nonzero(const_mask))
            const_weight_sum = float(const[const_mask].sum()) if n_const else 0.0
            if n_const:
                ib = np.nonzero(const_mask)[0]
                # Purely additive, oscillation-independent term: -1/-1 forces
                # a bananafit oscillation weight of exactly 1.0 (see
                # oscillation._gather_and_guard), so the placeholder
                # true_energy (bin center) never affects the weight -- it
                # only needs to be a finite, physically sensible value fed
                # into the (discarded) mango.probability_constant call for
                # this event's nu/antinu slice.
                te_parts.append(reco_c[ib])
                rb_parts.append(ib.astype(np.int32))
                ff_parts.append(np.full(n_const, -1, dtype=np.int32))
                ft_parts.append(np.full(n_const, -1, dtype=np.int32))
                anti_parts.append(np.full(n_const, bool(ch.anti), dtype=bool))
                w_parts.append(const[const_mask])

            chan_meta.append(dict(
                channel=cn, coef=coef, is_signal=is_signal,
                src=int(ch.src), dst=int(ch.dst), anti=bool(ch.anti), noosc=bool(ch.noosc),
                n_kernel_cells=n_kernel, n_const_cells=n_const,
                kernel_weight_sum=kernel_weight_sum, const_weight_sum=const_weight_sum,
            ))

        if te_parts:
            true_energy = np.concatenate(te_parts)
            reco_bin = np.concatenate(rb_parts)
            flavor_from = np.concatenate(ff_parts)
            flavor_to = np.concatenate(ft_parts)
            anti = np.concatenate(anti_parts)
            nominal_weight = np.concatenate(w_parts)
        else:
            true_energy = np.zeros(0, dtype=np.float64)
            reco_bin = np.zeros(0, dtype=np.int32)
            flavor_from = np.zeros(0, dtype=np.int32)
            flavor_to = np.zeros(0, dtype=np.int32)
            anti = np.zeros(0, dtype=bool)
            nominal_weight = np.zeros(0, dtype=np.float64)

        edges_used = fine_edges
        n_bins = nreco
        if rebin_edges is not None and rule_name in rebin_edges:
            coarse_edges = np.asarray(rebin_edges[rule_name], dtype=np.float64)
            bin_map, n_bins = _rebin_map(fine_edges, coarse_edges)
            reco_bin = bin_map[reco_bin]
            edges_used = coarse_edges

        sample, _perm = make_sample_prebinned(
            rule_name,
            true_energy=true_energy,
            reco_bin=reco_bin,
            n_bins=n_bins,
            flavor_from=flavor_from,
            flavor_to=flavor_to,
            anti=anti,
            nominal_weight=nominal_weight,
            bin_edges=edges_used,
        )
        osc = ConstantMatterOsc(baseline_km=baseline_km, density_gcc=density_gcc, ye=ye)
        samples.append(SampleModel(events=sample, splines=None, norms=None, osc=osc))

        meta["rules"][rule_name] = dict(
            edges=edges_used,
            n_bins=n_bins,
            n_events=int(true_energy.shape[0]),
            channels=chan_meta,
        )

    return samples, meta
