"""Total posterior assembly (DESIGN.md §9).

Wires a built :class:`~bananafit.model.ForwardModel`, an observed binned data
vector, and a :class:`~bananafit.parameters.ParameterSet` into the one
function the whole package exists to differentiate:

    logdensity(z) = loglike(mu(constrain(z)), data) + log_prior(constrain(z))
                    + log_det_jacobian(z)

``BinnedPosterior`` is a thin, *stateless* container -- not a JAX pytree
(same convention as ``ParameterSet``: everything on it is either already a
pytree in its own right (``model``), plain static configuration (``stat``,
``parameter_set``), or a fixed data array (``data``) that never changes
across a fit). Its methods are ordinary Python functions closing over
``self``'s attributes; wrapping ``posterior.logdensity`` in ``jax.jit``
traces exactly the array math (segment sums, spline Horner evaluation,
oscillation probabilities, priors) with everything else baked in as
compile-time constants -- "one jitted function, closed over static data"
(DESIGN.md §1).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from .likelihood import barlow_beeston_lite_loglike, poisson_loglike
from .model import ForwardModel
from .parameters import ParameterSet

__all__ = ["BinnedPosterior", "make_posterior", "asimov_data"]

_STATS = ("poisson", "barlow-beeston")


@dataclasses.dataclass(frozen=True)
class BinnedPosterior:
    """Binned-likelihood posterior over a :class:`ForwardModel`.

    Prefer the :func:`make_posterior` builder (validates ``stat`` and
    ``data`` shape, coerces dtypes) over constructing this directly.
    """

    model: ForwardModel
    data: jax.Array
    parameter_set: ParameterSet
    stat: str = "poisson"

    # -- convenience passthroughs -----------------------------------------

    @property
    def z0(self) -> jax.Array:
        """Chain/optimizer start in unconstrained (z) space -- ``parameter_set.init_z``."""
        return self.parameter_set.init_z

    def constrain(self, z: jax.Array) -> jax.Array:
        return self.parameter_set.constrain(z)

    def to_dict(self, physical: jax.Array) -> dict:
        return self.parameter_set.to_dict(physical)

    # -- likelihood / posterior ---------------------------------------------

    def loglike(self, physical: jax.Array) -> jax.Array:
        """Binned log-likelihood at a physical parameter point (no prior,
        no jacobian)."""
        physical = jnp.asarray(physical, dtype=jnp.float64)
        mu, w2 = self.model.expected(physical)
        if self.stat == "poisson":
            return poisson_loglike(mu, self.data)
        return barlow_beeston_lite_loglike(mu, w2, self.data)

    def logdensity_physical(self, physical: jax.Array) -> jax.Array:
        """``loglike(physical) + log_prior(physical)`` -- no change-of-variables
        jacobian (use :meth:`logdensity` for the z-space posterior the
        sampler actually walks on)."""
        physical = jnp.asarray(physical, dtype=jnp.float64)
        return self.loglike(physical) + self.parameter_set.log_prior(physical)

    def logdensity(self, z: jax.Array) -> jax.Array:
        """The hot path: unconstrained-space log-posterior, scalar.

        ``loglike(mu(constrain(z)), data) + log_prior(constrain(z)) +
        log_det_jacobian(z)`` -- a pure function of ``z`` (everything else is
        closed over), safe to pass straight to ``jax.jit`` / ``jax.grad`` /
        blackjax kernels.
        """
        z = jnp.asarray(z, dtype=jnp.float64)
        physical = self.constrain(z)
        return self.logdensity_physical(physical) + self.parameter_set.log_det_jacobian(z)


def make_posterior(
    model: ForwardModel,
    data: jax.Array,
    parameter_set: ParameterSet,
    *,
    stat: str = "poisson",
) -> BinnedPosterior:
    """Build a :class:`BinnedPosterior`, validating ``stat`` and ``data``'s
    shape against ``model.total_bins``."""
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}, got {stat!r}")
    data = jnp.asarray(data, dtype=jnp.float64)
    if data.shape != (model.total_bins,):
        raise ValueError(
            f"data has shape {data.shape}, expected ({model.total_bins},) "
            "(= model.total_bins)"
        )
    return BinnedPosterior(model=model, data=data, parameter_set=parameter_set, stat=stat)


def asimov_data(
    model: ForwardModel,
    parameter_set: ParameterSet,
    overrides: dict[str, float] = {},  # noqa: B006 -- never mutated, see below
) -> jax.Array:
    """The Asimov dataset: ``model.expected(physical)[0]`` (``mu``) at the
    ``parameter_set`` init point, with any names in ``overrides`` replaced by
    the given physical value.

    ``overrides`` is read-only here (only ``dict.update`` on a freshly-built
    local dict, never on the argument itself), so the mutable default is
    safe despite the usual Python gotcha.
    """
    physical_by_name = {spec.name: spec.init for spec in parameter_set.specs}
    physical_by_name.update(overrides)
    physical = jnp.asarray(
        [physical_by_name[n] for n in parameter_set.names], dtype=jnp.float64
    )
    mu, _ = model.expected(physical)
    return mu
