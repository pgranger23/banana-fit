"""Tests for :mod:`bananafit.globes_bridge` -- validates that
:func:`samples_from_globes` reproduces ``mango.globes.GlobesExperiment.spectra``
*exactly* (module docstring of ``globes_bridge.py`` explains the pipeline
mapping), that exact rebinning to the DUNE FD analysis binning agrees with a
``np.add.reduceat`` of the native spectrum, that gradients through the
bridge match finite differences, that the assembled forward model jits
cleanly, and reports event counts / min_weight-pruning sanity.

Uses DUNE's public TDR GLoBES configuration, shipped in the MANGO repository
(https://github.com/pgranger23/mango-osc, ``analyses/dune/globes/``): set
``MANGO_OSC_DIR`` to a checkout, or keep one next to this repository (as
``mango-osc`` or ``jaxnu-osc``). Skipped if none is found.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import os

import numpy as np
import pytest

import mango
import mango.globes as globes
import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.globes_bridge import samples_from_globes
from bananafit.likelihood import poisson_loglike
from bananafit.model import build_forward_model
from bananafit.parameters import ParameterSet, osc_param_specs


def _find_dune_glb() -> Path:
    rel = Path("analyses") / "dune" / "globes" / "DUNE_GLoBES.glb"
    roots = [Path(os.environ["MANGO_OSC_DIR"])] if os.environ.get("MANGO_OSC_DIR") else []
    sib = Path(__file__).resolve().parents[2]
    roots += [sib / "mango-osc", sib / "jaxnu-osc"]
    for r in roots:
        if (r / rel).is_file():
            return r / rel
    return roots[-1] / rel  # absent: the tests skip


DUNE_GLB = _find_dune_glb()

# DUNE FD analysis binning (GeV).
FD_ANALYSIS_EDGES = np.array(
    [0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
     3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 10.0],
    dtype=np.float64,
)

_OSC_ORDER = ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def exp():
    if not DUNE_GLB.exists():
        pytest.skip(f"DUNE GLoBES config not found at {DUNE_GLB}")
    return globes.load(str(DUNE_GLB))


@pytest.fixture(scope="module")
def parameter_set():
    return ParameterSet(osc_param_specs("NO"))


def _rule_names(exp):
    return list(exp.rules.keys())


def _osc_point(**overrides):
    """A (physical_vector, mango.OscParams) pair at NuFIT-NO +/- overrides."""
    nu = mango.nufit_no()
    base = dict(
        theta12=float(nu.theta12), theta13=float(nu.theta13),
        theta23=float(nu.theta23), deltacp=float(nu.deltacp),
        dm21=float(nu.dm21), dm31=float(nu.dm31),
    )
    base.update(overrides)
    vec = np.array([base[k] for k in _OSC_ORDER], dtype=np.float64)
    params = mango.OscParams(**{k: jnp.asarray(base[k]) for k in _OSC_ORDER})
    return vec, params


# 3+ distinct physics points: nominal NuFIT-NO, a different deltacp, and a
# sign-flipped dm31 (inverted-ordering-like) -- per the task's validation spec.
TEST_POINTS = [
    dict(),
    dict(deltacp=-1.6),
    dict(dm31=-2.5e-3 + 7.42e-5),
    dict(deltacp=0.9, theta23=0.72),
]


# --- 1. exactness vs exp.spectra() -------------------------------------------


def test_exact_reproduction_of_globes_spectra(exp, parameter_set):
    """build_forward_model(samples).expected_by_sample reproduces
    exp.spectra() rule-by-rule, bin-by-bin, at several physics points."""
    samples, meta = samples_from_globes(exp)
    fm = build_forward_model(samples, parameter_set)

    for kw in TEST_POINTS:
        vec, params = _osc_point(**kw)
        ref = exp.spectra(params)
        got = fm.expected_by_sample(vec)
        for rule in _rule_names(exp):
            mu, _w2 = got[rule]
            mu = np.asarray(mu)
            ref_rule = np.asarray(ref[rule])
            assert mu.shape == ref_rule.shape == (exp.nreco,)
            np.testing.assert_allclose(
                mu, ref_rule, rtol=1e-10, atol=1e-9,
                err_msg=f"rule {rule!r} mismatch at point {kw}",
            )


def test_metadata_channel_decomposition_sums_to_spectrum(exp):
    """The per-channel kernel/const weight sums recorded in the metadata,
    summed over a rule's channels, equal exp.spectra()[rule] at P_osc == 1
    everywhere -- an independent cross-check of the bookkeeping (not just
    the oscillated total)."""
    samples, meta = samples_from_globes(exp)
    for rule_name, rule in exp.rules.items():
        chans = meta["rules"][rule_name]["channels"]
        total_unosc = sum(c["kernel_weight_sum"] + c["const_weight_sum"] for c in chans)
        # reference: sum over channels of coef * sum_ij K_ij (i.e. with P == 1)
        ref = 0.0
        for coef, cn in list(rule.signal) + list(rule.background):
            ref += coef * float(np.asarray(exp.kernels[cn]).sum())
            ref += coef * float(np.asarray(exp.const[cn]).sum())
        np.testing.assert_allclose(total_unosc, ref, rtol=1e-12)


# --- 2. rebinning to the FD analysis edges ---------------------------------


def test_rebinning_matches_reduceat_of_native(exp, parameter_set):
    fine_edges = np.asarray(exp.reco_edges, dtype=np.float64)

    samples_fine, _ = samples_from_globes(exp)
    fm_fine = build_forward_model(samples_fine, parameter_set)

    rebin_edges = {rule: FD_ANALYSIS_EDGES for rule in _rule_names(exp)}
    samples_coarse, meta_coarse = samples_from_globes(exp, rebin_edges=rebin_edges)
    fm_coarse = build_forward_model(samples_coarse, parameter_set)

    vec, _params = _osc_point(deltacp=0.9)
    fine = fm_fine.expected_by_sample(vec)
    coarse = fm_coarse.expected_by_sample(vec)

    # position of each analysis edge within the native fine edges
    idx = np.array([
        np.nonzero(np.isclose(fine_edges, e, atol=1e-9))[0][0] for e in FD_ANALYSIS_EDGES
    ])
    n_coarse = len(FD_ANALYSIS_EDGES) - 1

    for rule in _rule_names(exp):
        assert meta_coarse["rules"][rule]["n_bins"] == n_coarse
        native_mu = np.asarray(fine[rule][0])
        coarse_mu = np.asarray(coarse[rule][0])
        assert coarse_mu.shape == (n_coarse,)
        # np.add.reduceat's last segment runs to the end of the array by
        # default, so truncate to exactly the analysis range before
        # reducing -- everything past edges[-1] is bananafit's overflow slot,
        # correctly dropped from `coarse_mu`, and must be excluded here too.
        trunc = native_mu[: idx[-1]]
        expected_coarse = np.add.reduceat(trunc, idx[:-1])
        np.testing.assert_allclose(coarse_mu, expected_coarse, rtol=1e-10, atol=1e-9)


def test_rebin_edges_must_align_with_native_bins(exp):
    with pytest.raises(ValueError):
        samples_from_globes(exp, rebin_edges={"nue_app": np.array([0.0, 0.6, 2.0])})


# --- 3. gradient vs finite differences ---------------------------------------


def test_gradient_matches_finite_differences(exp, parameter_set):
    samples, _ = samples_from_globes(exp)
    fm = build_forward_model(samples, parameter_set)

    def total(vec):
        mu, _w2 = fm.expected(vec)
        return jnp.sum(mu)

    vec0, _params = _osc_point()
    grad = np.asarray(jax.grad(total)(jnp.asarray(vec0)))

    # per-parameter step sized to its natural scale (angles O(1) rad,
    # dm21/dm31 O(1e-3..1e-4) eV^2).
    eps = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8])
    fd = np.empty(6)
    for i in range(6):
        vp, vm = vec0.copy(), vec0.copy()
        vp[i] += eps[i]
        vm[i] -= eps[i]
        fd[i] = (float(total(jnp.asarray(vp))) - float(total(jnp.asarray(vm)))) / (2 * eps[i])

    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-3)


# --- 4. event-count sanity + min_weight pruning ------------------------------


def test_event_counts_and_min_weight_pruning(exp, parameter_set, capsys):
    samples, meta = samples_from_globes(exp)
    for rule in _rule_names(exp):
        n_events = meta["rules"][rule]["n_events"]
        print(f"[globes_bridge] sample {rule!r}: {n_events} events "
              f"({exp.nreco} reco bins x up to {exp.nsamp} true-energy nodes)")
        assert n_events > 0

    fm_default = build_forward_model(samples, parameter_set)
    vec, _params = _osc_point()
    mu_default, _ = fm_default.expected(vec)
    total_default = float(jnp.sum(mu_default))

    # min_weight = -1 disables pruning entirely (keeps exact-zero cells,
    # which contribute exactly zero to any sum): this is the "no pruning at
    # all" reference the default (min_weight=0.0, dropping only exact
    # zeros) must reproduce losslessly.
    samples_nopar, _ = samples_from_globes(exp, min_weight=-1.0)
    fm_nopar = build_forward_model(samples_nopar, parameter_set)
    mu_nopar, _ = fm_nopar.expected(vec)
    total_nopar = float(jnp.sum(mu_nopar))

    rel = abs(total_default - total_nopar) / abs(total_nopar)
    assert rel < 1e-6, f"default min_weight pruning changed the total by {rel:.3e} (>1e-6)"


# --- 5. jit -------------------------------------------------------------------


def test_expected_under_jit_matches_eager(exp, parameter_set):
    samples, _ = samples_from_globes(exp)
    fm = build_forward_model(samples, parameter_set)
    vec, _params = _osc_point(deltacp=1.1)
    vec = jnp.asarray(vec)

    eager_mu, eager_w2 = fm.expected(vec)

    jit_expected = jax.jit(fm.expected)
    jit_mu, jit_w2 = jit_expected(vec)

    np.testing.assert_allclose(np.asarray(jit_mu), np.asarray(eager_mu), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jit_w2), np.asarray(eager_w2), rtol=1e-12, atol=1e-10)


def test_logdensity_style_call_under_jit_matches_eager(exp, parameter_set):
    """A minimal Asimov Poisson logdensity built on top of `expected` --
    the shape of call bananafit's posterior.py assembles -- compiles under
    jit and matches its eager evaluation."""
    samples, _ = samples_from_globes(exp)
    fm = build_forward_model(samples, parameter_set)
    vec0, _params = _osc_point()
    asimov, _ = fm.expected(jnp.asarray(vec0))  # Asimov "data" at the truth

    def logdensity(vec):
        mu, _w2 = fm.expected(vec)
        return jnp.sum(poisson_loglike(mu, asimov))

    vec1, _ = _osc_point(deltacp=0.4)
    vec1 = jnp.asarray(vec1)
    eager = logdensity(vec1)
    jitted = jax.jit(logdensity)(vec1)
    np.testing.assert_allclose(float(jitted), float(eager), rtol=1e-12, atol=1e-10)
