"""Tests for bananafit.parameters (DESIGN.md Section 6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

import jaxnu
import bananafit  # noqa: F401  (import enforces float64, per house style)
from bananafit.parameters import (
    Flat,
    Gaussian,
    MVGaussianBlock,
    ParameterSet,
    ParamSpec,
    VonMises,
    osc_param_specs,
)


# --- a representative mixed set: unbounded / positive / interval / fixed /
#     circular, with one per-param Gaussian and one correlated MV block ------


def _mixed_specs():
    return (
        ParamSpec("unb", -0.4, Flat(), (None, None)),
        ParamSpec("pos", 3.0, Gaussian(3.0, 0.7), (0.0, None)),
        ParamSpec("neg", -2.0, Flat(), (None, 0.0)),
        ParamSpec("inv", 0.3, Flat(), (0.0, 1.0)),
        ParamSpec("fix", 42.0, Flat(), (None, None), fixed=True),
        ParamSpec("circ", 1.0, Flat(), (None, None), circular=True),
        ParamSpec("blk_a", 1.0, Flat(), (None, None)),
        ParamSpec("blk_b", -1.0, Flat(), (None, None)),
    )


def _mixed_block():
    mean = jnp.array([1.0, -1.0])
    cov = jnp.array([[0.4, 0.1], [0.1, 0.9]])
    return MVGaussianBlock(("blk_a", "blk_b"), mean, cov)


def _mixed_set():
    return ParameterSet(_mixed_specs(), (_mixed_block(),))


# --- constrain / unconstrain round trip --------------------------------------


def test_round_trip_constrain_unconstrain():
    ps = _mixed_set()
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = jnp.asarray(rng.normal(size=ps.n_free))
        physical = ps.constrain(z)
        z2 = ps.unconstrain(physical)
        np.testing.assert_allclose(np.asarray(z2), np.asarray(z), atol=1e-12)


def test_round_trip_from_physical_side():
    ps = _mixed_set()
    physical0 = ps.constrain(ps.init_z)
    z = ps.unconstrain(physical0)
    physical1 = ps.constrain(z)
    # free coordinates round-trip exactly; fixed coordinates are untouched by
    # construction (never even read from physical0 on this path).
    free_idx = [i for i, s in enumerate(ps.specs) if not s.fixed]
    np.testing.assert_allclose(
        np.asarray(physical1)[free_idx], np.asarray(physical0)[free_idx], atol=1e-10
    )


def test_unconstrain_accepts_dict():
    ps = _mixed_set()
    physical = ps.constrain(ps.init_z)
    as_dict = {n: physical[i] for i, n in enumerate(ps.names)}
    z_from_dict = ps.unconstrain(as_dict)
    z_from_array = ps.unconstrain(physical)
    np.testing.assert_allclose(
        np.asarray(z_from_dict), np.asarray(z_from_array), atol=1e-12
    )


# --- fixed params never move ---------------------------------------------------


def test_fixed_params_never_move():
    ps = _mixed_set()
    fix_idx = ps.names.index("fix")
    rng = np.random.default_rng(1)
    for _ in range(10):
        z = jnp.asarray(rng.normal(scale=5.0, size=ps.n_free))
        physical = ps.constrain(z)
        assert float(physical[fix_idx]) == 42.0


def test_fixed_params_excluded_from_free_names():
    ps = _mixed_set()
    assert "fix" not in ps.free_names
    assert ps.n_free == ps.n_params - 1


# --- log_det_jacobian vs jax.jacobian -----------------------------------------


def test_log_det_jacobian_matches_full_jacobian_slogdet():
    ps = _mixed_set()
    rng = np.random.default_rng(2)
    free_rows = [i for i, s in enumerate(ps.specs) if not s.fixed]
    for _ in range(10):
        z = jnp.asarray(rng.normal(size=ps.n_free))
        J = jax.jacobian(ps.constrain)(z)  # (n_params, n_free)
        J_free = J[free_rows, :]
        _, logdet_ref = jnp.linalg.slogdet(J_free)
        logdet_ours = ps.log_det_jacobian(z)
        np.testing.assert_allclose(
            float(logdet_ours), float(logdet_ref), rtol=1e-8, atol=1e-10
        )


def test_jacobian_restricted_to_fixed_rows_is_zero():
    ps = _mixed_set()
    fix_idx = ps.names.index("fix")
    z = ps.init_z
    J = jax.jacobian(ps.constrain)(z)
    np.testing.assert_allclose(np.asarray(J[fix_idx]), 0.0, atol=0.0)


# --- log_prior vs hand-computed Normal + MVN ----------------------------------


def test_log_prior_matches_hand_computed():
    ps = _mixed_set()
    physical = ps.constrain(ps.init_z)
    physical_np = np.asarray(physical)

    pos_idx = ps.names.index("pos")
    expected = norm(3.0, 0.7).logpdf(physical_np[pos_idx])

    a_idx, b_idx = ps.names.index("blk_a"), ps.names.index("blk_b")
    mean = np.array([1.0, -1.0])
    cov = np.array([[0.4, 0.1], [0.1, 0.9]])
    expected += multivariate_normal(mean=mean, cov=cov).logpdf(
        physical_np[[a_idx, b_idx]]
    )
    # every other param is Flat: contributes 0.

    np.testing.assert_allclose(
        float(ps.log_prior(physical)), expected, rtol=1e-10, atol=1e-10
    )


def test_log_prior_varies_correctly_with_pos_param():
    ps = _mixed_set()
    physical = np.asarray(ps.constrain(ps.init_z))
    pos_idx = ps.names.index("pos")

    physical_shift = physical.copy()
    physical_shift[pos_idx] = 3.7
    diff_ours = float(ps.log_prior(jnp.asarray(physical_shift))) - float(
        ps.log_prior(jnp.asarray(physical))
    )
    diff_ref = norm(3.0, 0.7).logpdf(3.7) - norm(3.0, 0.7).logpdf(physical[pos_idx])
    np.testing.assert_allclose(diff_ours, diff_ref, rtol=1e-10)


# --- VonMises: periodic prior for circular parameters -------------------------


def test_von_mises_log_prob_is_2pi_periodic():
    """The one property that matters for a regularizing prior on a
    ``circular=True`` parameter: log_prob(x) must equal log_prob(x + 2*pi*k)
    for *every* integer k, exactly (not just approximately near the support
    window) -- otherwise it silently breaks the physical 2*pi symmetry the
    sampler relies on for a circular coordinate (unlike Gaussian, see
    parameters.py's VonMises docstring)."""
    prior = VonMises(0.7, 2.0)
    x0 = jnp.array(0.9)
    base = float(prior.log_prob(x0))
    for k in range(-4, 5):
        shifted = x0 + 2.0 * jnp.pi * k
        np.testing.assert_allclose(float(prior.log_prob(shifted)), base, atol=1e-9)


def test_von_mises_log_prob_periodic_near_wrap_boundary():
    """Same periodicity check, but starting near +-pi (the support
    boundary) and with mu itself near the boundary -- the case that a naive
    (non-wrapping) use of numpyro's VonMises could get wrong given its fixed
    (-pi, pi] support constraint."""
    prior = VonMises(3.05, 0.44)  # mu close to +pi
    for x0 in (jnp.array(3.1), jnp.array(-3.1), jnp.array(3.13)):
        base = float(prior.log_prob(x0))
        for k in (-2, -1, 1, 2):
            shifted = x0 + 2.0 * jnp.pi * k
            np.testing.assert_allclose(float(prior.log_prob(shifted)), base, atol=1e-9)


def test_von_mises_matches_scipy_von_mises_pdf():
    from scipy.stats import vonmises

    mu, kappa = -0.6, 1.3
    prior = VonMises(mu, kappa)
    xs = np.array([-3.0, -0.6, 0.0, 1.5, 3.0])
    expected = vonmises(kappa, loc=mu).logpdf(xs)
    got = np.asarray(prior.log_prob(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_parameter_set_log_prior_periodic_for_von_mises_circular_param():
    """End-to-end (not just the bare Prior class): a ParameterSet whose
    circular parameter carries a VonMises prior gives an exactly 2*pi
    periodic log_prior in the physical (= z, identity transform) value --
    this is what actually protects NUTS/RWM branch-hopping from being
    penalized relative to the "canonical" replica."""
    specs = (
        ParamSpec("deltacp", 0.4, VonMises(0.4, 1.0 / 1.5**2), (None, None), circular=True),
        ParamSpec("other", 1.0, Flat(), (None, None)),
    )
    ps = ParameterSet(specs)
    assert ps.circular_free_names == ("deltacp",)

    base_physical = np.array([0.9, 1.0])
    base = float(ps.log_prior(jnp.asarray(base_physical)))
    for k in (-2, -1, 1, 2):
        shifted = base_physical.copy()
        shifted[0] += 2.0 * np.pi * k
        got = float(ps.log_prior(jnp.asarray(shifted)))
        np.testing.assert_allclose(got, base, atol=1e-8)


def test_von_mises_prior_breaks_symmetry_that_gaussian_would_have_broken():
    """Contrast check: the old Gaussian(mu, sigma) regularizer is *not*
    periodic (this is exactly the bug being fixed) -- confirms the test
    above is actually exercising real behavior, not something that would
    have passed for Gaussian too."""
    gaussian = Gaussian(0.4, 1.5)
    x0 = jnp.array(0.9)
    shifted = x0 + 2.0 * jnp.pi
    assert float(gaussian.log_prob(x0)) != pytest.approx(float(gaussian.log_prob(shifted)))


# --- double-counted prior raises ----------------------------------------------


def test_double_counted_prior_raises():
    specs = (
        ParamSpec("a", 1.0, Gaussian(1.0, 0.5), (None, None)),
        ParamSpec("b", -1.0, Flat(), (None, None)),
    )
    block = MVGaussianBlock(("a", "b"), jnp.array([1.0, -1.0]), jnp.eye(2))
    with pytest.raises(ValueError):
        ParameterSet(specs, (block,))


def test_block_referencing_unknown_name_raises():
    specs = (ParamSpec("a", 1.0, Flat(), (None, None)),)
    block = MVGaussianBlock(("a", "missing"), jnp.zeros(2), jnp.eye(2))
    with pytest.raises(ValueError):
        ParameterSet(specs, (block,))


def test_duplicate_names_raise():
    specs = (
        ParamSpec("a", 1.0, Flat(), (None, None)),
        ParamSpec("a", 2.0, Flat(), (None, None)),
    )
    with pytest.raises(ValueError):
        ParameterSet(specs)


def test_bad_bounds_raise():
    with pytest.raises(ValueError):
        ParameterSet((ParamSpec("a", 1.0, Flat(), (2.0, 1.0)),))


def test_init_outside_bounds_raises():
    with pytest.raises(ValueError):
        ParameterSet((ParamSpec("a", 5.0, Flat(), (0.0, 1.0)),))


# --- osc_param_specs -----------------------------------------------------------


def test_osc_param_specs_names_and_order():
    specs = osc_param_specs()
    assert tuple(s.name for s in specs) == (
        "theta12", "theta13", "theta23", "deltacp", "dm21", "dm31",
    )


def test_osc_param_specs_inits_match_nufit_no():
    specs = osc_param_specs(ordering="NO")
    nu = jaxnu.nufit_no()
    by_name = {s.name: s for s in specs}
    for field in ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31"):
        assert by_name[field].init == pytest.approx(float(getattr(nu, field)))


def test_osc_param_specs_deltacp_is_circular_and_unbounded():
    specs = osc_param_specs()
    by_name = {s.name: s for s in specs}
    assert by_name["deltacp"].circular is True
    assert by_name["deltacp"].bounds == (None, None)


def test_osc_param_specs_dm31_sign_by_ordering():
    no_specs = {s.name: s for s in osc_param_specs(ordering="NO")}
    io_specs = {s.name: s for s in osc_param_specs(ordering="IO")}
    assert no_specs["dm31"].init > 0
    assert no_specs["dm31"].bounds[0] == 0.0
    assert io_specs["dm31"].init < 0
    assert io_specs["dm31"].bounds[1] == 0.0


def test_osc_param_specs_builds_valid_parameter_set():
    ps = ParameterSet(osc_param_specs())
    assert ps.n_params == 6
    assert ps.n_free == 6
    physical = ps.constrain(ps.init_z)
    nu = jaxnu.nufit_no()
    for i, field in enumerate(
        ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")
    ):
        np.testing.assert_allclose(
            float(physical[i]), float(getattr(nu, field)), rtol=1e-10
        )


# --- circular_free_names --------------------------------------------------------


def test_circular_free_names_lists_only_free_circular_params():
    ps = _mixed_set()
    assert ps.circular_free_names == ("circ",)


def test_circular_free_names_excludes_fixed_circular_param():
    specs = (
        ParamSpec("deltacp_fixed", 0.5, Flat(), (None, None), fixed=True, circular=True),
        ParamSpec("deltacp_free", 0.5, Flat(), (None, None), circular=True),
        ParamSpec("linear", 1.0, Flat(), (None, None)),
    )
    ps = ParameterSet(specs)
    assert ps.circular_free_names == ("deltacp_free",)
    assert "deltacp_fixed" not in ps.free_names


# --- wrap_circular / to_dict ---------------------------------------------------


def test_wrap_circular_only_touches_circular_params():
    ps = _mixed_set()
    physical = np.array(ps.constrain(ps.init_z))  # np.array copies (writeable)
    circ_idx = ps.names.index("circ")
    physical[circ_idx] = 4.0 * np.pi + 0.3  # far outside (-pi, pi]
    wrapped = np.asarray(ps.wrap_circular(jnp.asarray(physical)))
    assert -np.pi < wrapped[circ_idx] <= np.pi
    np.testing.assert_allclose(
        np.sin(wrapped[circ_idx]), np.sin(physical[circ_idx]), atol=1e-10
    )
    np.testing.assert_allclose(
        np.cos(wrapped[circ_idx]), np.cos(physical[circ_idx]), atol=1e-10
    )
    other_idx = [i for i in range(ps.n_params) if i != circ_idx]
    np.testing.assert_allclose(wrapped[other_idx], physical[other_idx])


def test_to_dict_maps_names_to_values():
    ps = _mixed_set()
    physical = ps.constrain(ps.init_z)
    d = ps.to_dict(physical)
    assert set(d.keys()) == set(ps.names)
    for i, n in enumerate(ps.names):
        if not ps.specs[i].circular:
            assert float(d[n]) == pytest.approx(float(physical[i]))


# --- jit / grad ----------------------------------------------------------------


def test_constrain_under_jit_matches_eager():
    ps = _mixed_set()
    z = ps.init_z
    eager = ps.constrain(z)
    jitted = jax.jit(ps.constrain)(z)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-12)


def test_log_det_jacobian_under_jit_matches_eager():
    ps = _mixed_set()
    z = ps.init_z
    eager = ps.log_det_jacobian(z)
    jitted = jax.jit(ps.log_det_jacobian)(z)
    np.testing.assert_allclose(float(jitted), float(eager), rtol=1e-12)


def test_grad_of_log_prior_plus_log_det_jacobian_is_finite():
    ps = _mixed_set()

    def logdensity(z):
        physical = ps.constrain(z)
        return ps.log_prior(physical) + ps.log_det_jacobian(z)

    z = ps.init_z
    val = logdensity(z)
    grad = jax.grad(logdensity)(z)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad)))

    jit_grad = jax.jit(jax.grad(logdensity))(z)
    np.testing.assert_allclose(np.asarray(jit_grad), np.asarray(grad), rtol=1e-8)


def test_grad_matches_finite_differences_for_free_coord():
    ps = _mixed_set()

    def logdensity(z):
        physical = ps.constrain(z)
        return ps.log_prior(physical) + ps.log_det_jacobian(z)

    z = np.asarray(ps.init_z, dtype=np.float64)
    grad = np.asarray(jax.grad(logdensity)(jnp.asarray(z)))

    h = 1e-6
    for i in range(len(z)):
        zp = z.copy()
        zp[i] += h
        zm = z.copy()
        zm[i] -= h
        fd = (float(logdensity(jnp.asarray(zp))) - float(logdensity(jnp.asarray(zm)))) / (2 * h)
        np.testing.assert_allclose(grad[i], fd, rtol=1e-3, atol=1e-4)
