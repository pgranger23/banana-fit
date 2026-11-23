"""Parameters, bijective transforms, and priors — the single source of truth
for "what is being fit" (DESIGN.md §6).

All of it is one ordered sequence of small, static :class:`ParamSpec` records plus an
optional list of correlated Gaussian blocks (:class:`MVGaussianBlock`, the
covariance-penalty equivalent), assembled into a :class:`ParameterSet`.

The sampler never sees bounded/fixed/circular parameters directly: it walks on
an unconstrained vector ``z in R^n_free``. ``ParameterSet.constrain`` maps that
to the full physical vector (fixed entries reinserted as compile-time
constants), and ``log_det_jacobian`` supplies the change-of-variables term so
that sampling in ``z``-space with a flat proposal is equivalent to sampling the
physical parameters under their bounds. Bounded transforms are
``numpyro.distributions.transforms.biject_to`` bijectors chosen once, in plain
Python, from each spec's ``bounds`` at :class:`ParameterSet` construction time
(build-time == static; only the *values* threaded through them are traced), so
``constrain``/``unconstrain``/``log_det_jacobian`` are ordinary jit- and
grad-safe functions with no data-dependent Python control flow.

``circular`` parameters (deltacp) are not reflected at a boundary, as a
random-walk sampler may do: for gradient-based samplers a reflection
introduces a non-differentiable kink. Instead they are sampled unbounded on
R with the identity transform -- the physical posterior is exactly 2*pi
periodic, so NUTS explores it as happily as any other unbounded coordinate --
and wrapped into ``(-pi, pi]`` only when reporting (:meth:`wrap_circular`).
Correspondingly, a *regularizing prior* on a circular parameter should be
periodic too: plain :class:`Gaussian` breaks the 2*pi symmetry (its density
strictly prefers ``mu`` over the physically-identical ``mu + 2*pi``
replica); use :class:`VonMises` instead, which is exact under wrap-around by
construction.
"""

from __future__ import annotations

import dataclasses
from typing import Union

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.distributions.transforms import biject_to

import mango

__all__ = [
    "Prior",
    "Gaussian",
    "VonMises",
    "Flat",
    "ParamSpec",
    "MVGaussianBlock",
    "ParameterSet",
    "osc_param_specs",
]


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def _wrap_pi(x):
    """Wrap ``x`` (radians, any shape) into ``(-pi, pi]``. Shared by
    :meth:`ParameterSet.wrap_circular` and :class:`VonMises`'s ``log_prob``
    -- the one place both need "canonical representative of this 2*pi
    equivalence class"."""
    wrapped = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    return jnp.where(wrapped <= -jnp.pi, wrapped + 2.0 * jnp.pi, wrapped)


@dataclasses.dataclass(frozen=True)
class Gaussian:
    """Independent Gaussian prior, ``N(mu, sigma^2)``.

    Not periodic: on a ``circular=True`` parameter (deltacp) this silently
    breaks the 2*pi symmetry the sampler otherwise relies on -- the density
    strictly decreases away from ``mu``, so the physically-identical replica
    ``mu + 2*pi`` is penalized relative to ``mu`` even though the likelihood
    cannot tell them apart. Fine for a genuinely linear/bounded parameter
    (theta13's reactor constraint); use :class:`VonMises` instead for a
    regularizing prior on deltacp or any other circular parameter.
    """

    mu: float
    sigma: float

    def log_prob(self, x):
        return dist.Normal(self.mu, self.sigma).log_prob(x)


@dataclasses.dataclass(frozen=True)
class VonMises:
    """Periodic prior for a ``circular=True`` parameter -- the circular
    analogue of :class:`Gaussian`, ``VonMises(mu, kappa)`` via
    ``numpyro.distributions.VonMises``.

    ``circular`` parameters (see this module's docstring) are sampled
    unbounded on all of R with the identity transform, so their physical
    value ``x`` and every ``x + 2*pi*k`` are the same point on the circle.
    ``numpyro``'s ``VonMises`` has *support* restricted to ``(-pi, pi]``
    (``dist.VonMises(...).support`` is a fixed ``Circular(-pi, pi)``
    constraint, independent of ``mu``) -- evaluating it directly on an
    unbounded ``x`` risks numpyro's optional sample validation flagging (or,
    worse, silently returning ``-inf`` for) values outside that window even
    though its density is mathematically ``2*pi``-periodic in ``x`` (its
    ``log_prob`` already reduces ``(x - mu) mod 2*pi`` internally). This
    class removes that footgun by explicitly wrapping ``x`` into
    ``(-pi, pi]`` with :func:`_wrap_pi` -- the same recipe as
    :meth:`ParameterSet.wrap_circular` -- before calling ``log_prob``, so the
    result is *exactly* ``2*pi``-periodic in the raw unwrapped ``x`` (unlike
    :class:`Gaussian`, whose ``±2*pi`` replicas are not equivalent).

    ``kappa`` (concentration) plays the role of ``1/sigma**2``: for
    ``kappa >~ 1`` the small-angle limit of ``VonMises(mu, kappa)`` matches
    ``Gaussian(mu, sigma=1/sqrt(kappa))``. To reproduce an existing
    ``Gaussian(mu, sigma)`` regularizer's strength, use
    ``kappa = 1 / sigma**2`` -- e.g. the deltacp regularizer historically
    written ``Gaussian(mu, 1.5)`` becomes ``VonMises(mu, kappa=1/1.5**2)``
    (``kappa`` ~= 0.44).
    """

    mu: float
    kappa: float

    def log_prob(self, x):
        return dist.VonMises(self.mu, self.kappa, validate_args=False).log_prob(_wrap_pi(x))


@dataclasses.dataclass(frozen=True)
class Flat:
    """Improper flat prior: contributes exactly 0 to the log-posterior."""

    def log_prob(self, x):
        # Independent of x by construction (improper prior); returning a
        # bare python float (rather than jnp.zeros_like(x)) keeps this a true
        # no-op under jit/grad -- it never touches x at all.
        return 0.0


Prior = Union[Gaussian, VonMises, Flat]


# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """One parameter's static metadata. Plain dataclass, NOT a pytree: name,
    bounds, fixed/circular flags and the prior object are all compile-time
    configuration, never traced.
    """

    name: str
    init: float
    prior: Prior = dataclasses.field(default_factory=Flat)
    bounds: tuple = (None, None)
    fixed: bool = False
    circular: bool = False


@dataclasses.dataclass(frozen=True)
class MVGaussianBlock:
    """Correlated Gaussian prior over a named subset of parameters -- the
    MaCh3 covariance-matrix penalty equivalent (e.g. a flux+xsec covariance
    imported from a near-detector fit).

    The Cholesky factor of ``cov`` is precomputed at construction (build
    time) so every ``log_prob`` call at run time is a triangular solve, not a
    fresh decomposition.
    """

    names: tuple
    mean: jax.Array
    cov: jax.Array
    scale_tril: jax.Array = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        mean = jnp.asarray(self.mean, dtype=jnp.float64)
        cov = jnp.asarray(self.cov, dtype=jnp.float64)
        names = tuple(self.names)
        if len(names) != mean.shape[-1] or cov.shape != (len(names), len(names)):
            raise ValueError(
                f"MVGaussianBlock shape mismatch: {len(names)} names, "
                f"mean {mean.shape}, cov {cov.shape}"
            )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "scale_tril", jnp.linalg.cholesky(cov))

    def log_prob(self, x):
        """``x``: physical values for ``self.names``, in that order."""
        return dist.MultivariateNormal(loc=self.mean, scale_tril=self.scale_tril).log_prob(x)


def _build_transform(spec: ParamSpec):
    """Pick the bijector R -> support(spec) from bounds/circular. Called once
    per spec at :class:`ParameterSet` construction (static, not traced).
    """
    if spec.circular:
        # Sampled unbounded; bounds (if any) are ignored by design.
        return biject_to(constraints.real)
    lo, hi = spec.bounds
    if lo is None and hi is None:
        con = constraints.real
    elif lo is not None and hi is None:
        con = constraints.greater_than(float(lo))
    elif lo is None and hi is not None:
        con = constraints.less_than(float(hi))
    else:
        con = constraints.interval(float(lo), float(hi))
    return biject_to(con)


@dataclasses.dataclass(frozen=True)
class ParameterSet:
    """Ordered collection of :class:`ParamSpec`, plus optional correlated
    :class:`MVGaussianBlock` priors. Owns the constrain/unconstrain bijection
    and the prior. Not a JAX pytree: everything here (names, bounds, fixed
    masks, the per-parameter ``Transform`` objects) is compile-time static;
    only the arrays passed *into* its methods (``z``, ``physical``) are
    traced.
    """

    specs: tuple
    blocks: tuple = ()

    # Derived, computed in __post_init__ (object.__setattr__ since frozen).
    names: tuple = dataclasses.field(init=False, repr=False)
    free_names: tuple = dataclasses.field(init=False, repr=False)
    circular_free_names: tuple = dataclasses.field(init=False, repr=False)
    n_params: int = dataclasses.field(init=False, repr=False)
    n_free: int = dataclasses.field(init=False, repr=False)
    init_z: jax.Array = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        specs = tuple(self.specs)
        blocks = tuple(self.blocks)
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "blocks", blocks)

        names = tuple(s.name for s in specs)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate parameter name(s): {dupes}")
        name_to_idx = {n: i for i, n in enumerate(names)}

        for spec in specs:
            lo, hi = spec.bounds
            if lo is not None and hi is not None and not (lo < hi):
                raise ValueError(
                    f"parameter {spec.name!r}: bounds {spec.bounds} not lo < hi"
                )
            if not spec.circular:
                if lo is not None and spec.init < lo:
                    raise ValueError(
                        f"parameter {spec.name!r}: init {spec.init} < lower bound {lo}"
                    )
                if hi is not None and spec.init > hi:
                    raise ValueError(
                        f"parameter {spec.name!r}: init {spec.init} > upper bound {hi}"
                    )

        covered = {}
        for block in blocks:
            for n in block.names:
                if n not in name_to_idx:
                    raise ValueError(
                        f"MVGaussianBlock references unknown parameter {n!r}"
                    )
                if n in covered:
                    raise ValueError(
                        f"parameter {n!r} is covered by more than one MVGaussianBlock"
                    )
                covered[n] = block
                spec = specs[name_to_idx[n]]
                if isinstance(spec.prior, Gaussian):
                    raise ValueError(
                        f"parameter {n!r} has both a per-parameter Gaussian prior "
                        "and an MVGaussianBlock prior -- double-counted prior"
                    )

        fixed_mask = tuple(s.fixed for s in specs)
        free_names = tuple(n for n, f in zip(names, fixed_mask) if not f)
        free_index = {}
        j = 0
        for n, f in zip(names, fixed_mask):
            if not f:
                free_index[n] = j
                j += 1

        transforms = tuple(_build_transform(s) for s in specs)

        # names of the free (non-fixed) circular parameters, in free_names
        # order -- diagnostics.py's rhat/ess/summary/to_arviz duck-type
        # contract reads this to know which z-space coordinates need
        # sin/cos-based (rather than linear) treatment. Since circular specs
        # use the identity transform (_build_transform), these free_names
        # entries are exactly the raw physical deltacp-like value: no
        # constrain() needed to interpret them circularly.
        object.__setattr__(self, "circular_free_names", tuple(
            s.name for s in specs if s.circular and not s.fixed
        ))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "free_names", free_names)
        object.__setattr__(self, "n_params", len(specs))
        object.__setattr__(self, "n_free", len(free_names))
        object.__setattr__(self, "_name_to_idx", name_to_idx)
        object.__setattr__(self, "_fixed_mask", fixed_mask)
        object.__setattr__(self, "_free_index", tuple(
            free_index.get(n) for n in names
        ))
        object.__setattr__(self, "_transforms", transforms)
        object.__setattr__(self, "_circular_mask", tuple(s.circular for s in specs))

        inits = jnp.asarray([float(s.init) for s in specs], dtype=jnp.float64)
        object.__setattr__(self, "init_z", self.unconstrain(inits))

    # -- transform ----------------------------------------------------

    def constrain(self, z):
        """``z`` (n_free,) unconstrained -> (n_params,) physical, fixed
        params reinserted as constants."""
        z = jnp.asarray(z)
        parts = []
        for i, spec in enumerate(self.specs):
            if self._fixed_mask[i]:
                parts.append(jnp.asarray(spec.init, dtype=z.dtype))
            else:
                j = self._free_index[i]
                parts.append(self._transforms[i](z[j]))
        return jnp.stack(parts)

    def unconstrain(self, values):
        """(n_params,) physical (array or ``dict[str, float]``) -> (n_free,)
        unconstrained. Fixed parameters are dropped."""
        if isinstance(values, dict):
            arr = jnp.asarray(
                [values[n] for n in self.names], dtype=jnp.float64
            )
        else:
            arr = jnp.asarray(values)
        parts = []
        for i in range(len(self.specs)):
            if self._fixed_mask[i]:
                continue
            parts.append(self._transforms[i].inv(arr[i]))
        if not parts:
            return jnp.zeros((0,), dtype=arr.dtype)
        return jnp.stack(parts)

    def log_det_jacobian(self, z):
        """Log |det d(constrain)/dz| at ``z``, summed over free parameters
        (each transform acts on its own scalar coordinate, so the Jacobian of
        ``constrain`` restricted to the free-parameter output rows is
        diagonal and this sum equals its slogdet)."""
        z = jnp.asarray(z)
        total = jnp.zeros((), dtype=z.dtype)
        for i in range(len(self.specs)):
            if self._fixed_mask[i]:
                continue
            j = self._free_index[i]
            zi = z[j]
            xi = self._transforms[i](zi)
            total = total + self._transforms[i].log_abs_det_jacobian(zi, xi)
        return total

    # -- priors ---------------------------------------------------------

    def log_prior(self, physical):
        """Sum of per-parameter priors (Flat contributes 0) plus correlated
        MVGaussianBlock terms, evaluated at the full physical vector."""
        physical = jnp.asarray(physical)
        total = jnp.zeros((), dtype=physical.dtype)
        for i, spec in enumerate(self.specs):
            total = total + spec.prior.log_prob(physical[i])
        for block in self.blocks:
            idx = jnp.asarray([self._name_to_idx[n] for n in block.names])
            total = total + block.log_prob(physical[idx])
        return total

    # -- reporting helpers ------------------------------------------------

    def wrap_circular(self, physical):
        """Wrap ``circular`` parameters into ``(-pi, pi]``; all other entries
        pass through unchanged. Reporting only -- never used in the hot
        (logdensity) path."""
        physical = jnp.asarray(physical)
        wrapped = _wrap_pi(physical)
        mask = jnp.asarray(self._circular_mask, dtype=bool)
        return jnp.where(mask, wrapped, physical)

    def to_dict(self, physical):
        """``(n_params,)`` physical vector -> ``{name: value}``, circular
        parameters wrapped for display."""
        wrapped = self.wrap_circular(physical)
        return {n: wrapped[i] for i, n in enumerate(self.names)}


# ---------------------------------------------------------------------------
# Standard oscillation parameters
# ---------------------------------------------------------------------------


def osc_param_specs(ordering: str = "NO") -> tuple:
    """The standard six oscillation :class:`ParamSpec`, named exactly
    ``theta12, theta13, theta23, deltacp, dm21, dm31`` (radians, eV^2),
    initialised from :func:`mango.nufit_no` and given flat priors with
    sensible physical bounds.

    ``ordering``: ``"NO"`` (normal, ``dm31 > 0``, the default -- matches
    ``mango.nufit_no()`` directly) or ``"IO"`` (inverted, ``dm31 < 0``).
    mango ships no dedicated inverted-ordering fit point, so for ``"IO"`` the
    magnitude of ``dm31`` is kept and only its sign is flipped -- an
    approximation (the true NuFIT IO best fit differs slightly in theta23 and
    deltacp too), adequate as a chain-starting point, not as a physics
    result.
    """
    nu = mango.nufit_no()
    ordering = ordering.upper()
    if ordering not in ("NO", "IO"):
        raise ValueError(f"ordering must be 'NO' or 'IO', got {ordering!r}")

    half_pi = float(jnp.pi / 2.0)
    dm31_init = float(nu.dm31)
    if ordering == "IO":
        dm31_init = -abs(dm31_init)
        dm31_bounds = (None, 0.0)
    else:
        dm31_bounds = (0.0, None)

    return (
        ParamSpec("theta12", float(nu.theta12), Flat(), (0.0, half_pi)),
        ParamSpec("theta13", float(nu.theta13), Flat(), (0.0, half_pi)),
        ParamSpec("theta23", float(nu.theta23), Flat(), (0.0, half_pi)),
        ParamSpec("deltacp", float(nu.deltacp), Flat(), (None, None), False, True),
        ParamSpec("dm21", float(nu.dm21), Flat(), (0.0, None)),
        ParamSpec("dm31", dm31_init, Flat(), dm31_bounds),
    )
