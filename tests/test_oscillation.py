"""Tests for bananafit.oscillation (DESIGN.md Section 8).

``bananafit.events`` (the real ``EventSample``, DESIGN.md Section 3) is being
written in parallel by another agent and may not exist yet. This file does
NOT import it; instead it builds a tiny local ``_Sample`` stand-in exposing
exactly the attribute names ``bananafit.oscillation`` duck-types against
(``true_energy``, ``flavor_from``, ``flavor_to``, ``anti_split``,
``n_events``). Full integration against the real ``EventSample`` is covered
once ``events.py`` lands (e.g. in ``tests/test_model.py`` / an integration
suite) -- not here.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mango
import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.oscillation import ConstantMatterOsc, EarthOsc, OscConfig, VacuumOsc

BASELINE_KM = 1300.0
DENSITY_GCC = 2.8
YE = 0.5


# --- local EventSample stand-in (DESIGN.md Section 3 attribute names only) --


@dataclasses.dataclass(frozen=True)
class _Sample:
    true_energy: jax.Array
    flavor_from: jax.Array
    flavor_to: jax.Array
    anti_split: int

    @property
    def n_events(self) -> int:
        return self.true_energy.shape[0]


def _mixed_sample() -> _Sample:
    """9 events, anti_split=5: a mix of appearance/disappearance channels,
    nu and antinu, plus an unoscillated (NC-like) event in each block."""
    true_energy = jnp.array(
        [0.6, 1.2, 2.0, 3.0, 4.0, 0.7, 1.5, 2.5, 3.5], dtype=jnp.float64
    )
    #                        numu->nue  numu->numu  nue->nue    NC     nue->numu | anti versions...
    flavor_from = jnp.array([1, 1, 0, -1, 0, 1, 1, 0, -1], dtype=jnp.int32)
    flavor_to = jnp.array([0, 1, 0, -1, 1, 0, 1, 0, -1], dtype=jnp.int32)
    return _Sample(true_energy=true_energy, flavor_from=flavor_from,
                   flavor_to=flavor_to, anti_split=5)


def _all_nu_sample() -> _Sample:
    true_energy = jnp.array([0.8, 1.6, 3.2], dtype=jnp.float64)
    flavor_from = jnp.array([1, 0, -1], dtype=jnp.int32)
    flavor_to = jnp.array([0, 0, -1], dtype=jnp.int32)
    n = true_energy.shape[0]
    return _Sample(true_energy=true_energy, flavor_from=flavor_from,
                   flavor_to=flavor_to, anti_split=n)


def _all_antinu_sample() -> _Sample:
    true_energy = jnp.array([0.9, 2.1, 4.4], dtype=jnp.float64)
    flavor_from = jnp.array([1, 1, 0], dtype=jnp.int32)
    flavor_to = jnp.array([1, 0, 0], dtype=jnp.int32)
    return _Sample(true_energy=true_energy, flavor_from=flavor_from,
                   flavor_to=flavor_to, anti_split=0)


def _empty_sample() -> _Sample:
    z = jnp.zeros((0,), dtype=jnp.float64)
    zi = jnp.zeros((0,), dtype=jnp.int32)
    return _Sample(true_energy=z, flavor_from=zi, flavor_to=zi, anti_split=0)


# --- reference implementation: direct elementwise mango calls in a loop -----


def _reference_weights(params, sample, baseline, density, ye, *, vacuum=False):
    E = np.asarray(sample.true_energy)
    ff = np.asarray(sample.flavor_from)
    ft = np.asarray(sample.flavor_to)
    n = sample.n_events
    split = sample.anti_split
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if ff[i] == -1:
            out[i] = 1.0
            continue
        anti = i >= split
        if vacuum:
            p = mango.probability_vacuum(params, jnp.asarray(E[i]), baseline, anti=anti)
        else:
            p = mango.probability_constant(
                params, jnp.asarray(E[i]), baseline, density=density, ye=ye, anti=anti
            )
        out[i] = float(p[int(ft[i]), int(ff[i])])
    return out


# --- finite-difference helpers ----------------------------------------------


def _fd_step(v: float) -> float:
    return 1e-6 * max(abs(v), 1e-4)


def _central_diff(f, x0: float) -> float:
    h = _fd_step(x0)
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


OSC_FIELDS = ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")


# --- basic correctness -------------------------------------------------------


def test_weights_match_direct_loop_mixed_sample():
    params = mango.nufit_no()
    sample = _mixed_sample()
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    w = np.asarray(osc.weights(params, sample))
    ref = _reference_weights(params, sample, BASELINE_KM, DENSITY_GCC, YE)

    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-12)
    # NC / unoscillated events (index 3 = nu block, 8 = antinu block) get
    # exactly weight 1.0, not merely "close to" 1.0.
    assert w[3] == 1.0
    assert w[8] == 1.0


@pytest.mark.parametrize(
    "sample_fn", [_all_nu_sample, _all_antinu_sample], ids=["all_nu", "all_antinu"]
)
def test_weights_edge_case_pure_splits(sample_fn):
    params = mango.nufit_no()
    sample = sample_fn()
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    w = np.asarray(osc.weights(params, sample))
    ref = _reference_weights(params, sample, BASELINE_KM, DENSITY_GCC, YE)

    assert w.shape == (sample.n_events,)
    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-12)


def test_weights_empty_sample():
    params = mango.nufit_no()
    sample = _empty_sample()
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)
    w = osc.weights(params, sample)
    assert w.shape == (0,)
    assert w.dtype == jnp.float64


def test_vacuum_osc_matches_direct_loop():
    params = mango.nufit_no()
    sample = _mixed_sample()
    osc = VacuumOsc(BASELINE_KM)

    w = np.asarray(osc.weights(params, sample))
    ref = _reference_weights(params, sample, BASELINE_KM, None, None, vacuum=True)

    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-12)
    assert w[3] == 1.0
    assert w[8] == 1.0


def test_vacuum_osc_equals_constant_matter_osc_zero_density():
    params = mango.nufit_no()
    sample = _mixed_sample()
    w_vacuum = np.asarray(VacuumOsc(BASELINE_KM).weights(params, sample))
    w_zero_density = np.asarray(
        ConstantMatterOsc(BASELINE_KM, density_gcc=0.0).weights(params, sample)
    )
    np.testing.assert_allclose(w_vacuum, w_zero_density, rtol=1e-12, atol=1e-14)


# --- jit ---------------------------------------------------------------------


def test_jit_matches_eager():
    params = mango.nufit_no()
    sample = _mixed_sample()
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    w_eager = osc.weights(params, sample)

    @jax.jit
    def jitted(params, osc):
        return osc.weights(params, sample)

    w_jit = jitted(params, osc)
    # jit vs eager can differ at the ULP level (different XLA fusion / op
    # ordering), not bit-for-bit -- tight but non-zero tolerance.
    np.testing.assert_allclose(
        np.asarray(w_jit), np.asarray(w_eager), rtol=1e-12, atol=1e-14
    )


# --- gradients vs finite differences -----------------------------------------


def _loss_wrt_params(params, osc, sample):
    return jnp.sum(osc.weights(params, sample))


def test_grad_wrt_osc_params_matches_finite_differences():
    params = mango.nufit_no()
    sample = _mixed_sample()
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    grad = jax.grad(lambda p: _loss_wrt_params(p, osc, sample))(params)

    for field in OSC_FIELDS:
        v0 = float(getattr(params, field))

        def f(x, field=field):
            p2 = dataclasses.replace(params, **{field: jnp.asarray(x)})
            return float(_loss_wrt_params(p2, osc, sample))

        fd = _central_diff(f, v0)
        analytic = float(getattr(grad, field))
        np.testing.assert_allclose(
            analytic, fd, rtol=1e-5, atol=1e-6,
            err_msg=f"grad mismatch for OscParams.{field}",
        )


def test_grad_wrt_baseline_and_density_matches_finite_differences():
    params = mango.nufit_no()
    sample = _mixed_sample()
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    def loss_from_osc(osc):
        return _loss_wrt_params(params, osc, sample)

    grad_osc = jax.grad(loss_from_osc)(osc)

    def f_baseline(x):
        return float(loss_from_osc(ConstantMatterOsc(x, DENSITY_GCC, YE)))

    def f_density(x):
        return float(loss_from_osc(ConstantMatterOsc(BASELINE_KM, x, YE)))

    fd_baseline = _central_diff(f_baseline, BASELINE_KM)
    fd_density = _central_diff(f_density, DENSITY_GCC)

    np.testing.assert_allclose(
        float(grad_osc.baseline_km), fd_baseline, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        float(grad_osc.density_gcc), fd_density, rtol=1e-5, atol=1e-6
    )


def test_grad_finite_with_nc_events_present():
    """The where-guard test: NC/unoscillated events (flavor_from == -1) must
    not leak NaN into the gradient through the discarded gather branch."""
    params = mango.nufit_no()
    sample = _mixed_sample()
    assert bool(jnp.any(sample.flavor_from == -1))  # sanity: NC events present
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    grad_params = jax.grad(lambda p: _loss_wrt_params(p, osc, sample))(params)
    for field in OSC_FIELDS:
        assert np.isfinite(float(getattr(grad_params, field))), field

    grad_osc = jax.grad(lambda o: _loss_wrt_params(params, o, sample))(osc)
    assert np.isfinite(float(grad_osc.baseline_km))
    assert np.isfinite(float(grad_osc.density_gcc))
    assert np.isfinite(float(grad_osc.ye))

    # Also exercise the per-event weight vector itself (not just the summed
    # loss): jacobian of every weight w.r.t. every param must be finite.
    jac = jax.jacrev(lambda p: osc.weights(p, sample))(params)
    for field in OSC_FIELDS:
        assert np.all(np.isfinite(np.asarray(getattr(jac, field)))), field


# --- sterile smoke test -------------------------------------------------------


def test_sterile_smoke():
    st = mango.Sterile3plus1(
        theta12=jnp.asarray(0.5836),
        theta13=jnp.asarray(0.1495),
        theta23=jnp.asarray(0.8587),
        theta14=jnp.asarray(0.05),
        theta24=jnp.asarray(0.03),
        theta34=jnp.asarray(0.02),
        delta13=jnp.asarray(3.40),
        delta24=jnp.asarray(0.0),
        dm21=jnp.asarray(7.42e-5),
        dm31=jnp.asarray(2.515e-3),
        dm41=jnp.asarray(1.0),
    )
    sample = _mixed_sample()  # flavors are all in {-1, 0, 1, 2}
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)

    w = osc.weights(st, sample)
    assert w.shape == (sample.n_events,)
    w_np = np.asarray(w)
    assert np.all(np.isfinite(w_np))
    assert w_np[3] == 1.0
    assert w_np[8] == 1.0

    ref = _reference_weights(st, sample, BASELINE_KM, DENSITY_GCC, YE)
    np.testing.assert_allclose(w_np, ref, rtol=1e-10, atol=1e-12)

    # Differentiable w.r.t. sterile params too (n_flavors == 4 path).
    grad = jax.grad(lambda p: jnp.sum(osc.weights(p, sample)))(st)
    assert np.isfinite(float(grad.theta14))
    assert np.isfinite(float(grad.dm41))


# --- protocol / reserved interface -------------------------------------------


def test_constant_matter_osc_satisfies_protocol():
    osc = ConstantMatterOsc(BASELINE_KM, DENSITY_GCC, YE)
    assert isinstance(osc, OscConfig)
    assert isinstance(VacuumOsc(BASELINE_KM), OscConfig)


def test_earth_osc_reserved_not_implemented():
    with pytest.raises(NotImplementedError):
        EarthOsc()
