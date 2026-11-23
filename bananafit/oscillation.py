"""Per-event oscillation weights via mango (DESIGN.md Section 8).

An ``OscConfig`` maps an oscillation-parameter point (``OscParams``, or any
mango params pytree -- ``Sterile3plus1``, ``NFlavorParams``, ...) and an
``EventSample`` (DESIGN.md Section 3) to a per-event weight ``(N,)`` array:
``P(flavor_from[i] -> flavor_to[i])`` at ``true_energy[i]``, with events
flagged unoscillated (``flavor_from == flavor_to == -1``, e.g. NC background)
getting weight exactly ``1.0``.

This module only *reads* four attributes off ``sample`` --
``true_energy``, ``flavor_from``, ``flavor_to``, ``anti_split`` -- plus
``n_events`` (total event count). It never imports ``bananafit.events``: any
object exposing those five attributes (duck-typed) works, which keeps this
module buildable/testable independently of the parallel work on
``events.py`` (see DESIGN.md Section 3 for the canonical ``EventSample``).

``anti_split`` is a *static* Python int (see DESIGN.md Section 3 rationale):
events are pre-sorted so ``[0:anti_split)`` are neutrinos and
``[anti_split:N)`` are antineutrinos. That lets the adapter make exactly two
mango calls on contiguous, statically-shaped slices via plain Python slicing
-- jit-safe (no dynamic shapes, no per-event branching under trace) -- and to
skip a call entirely (at trace/build time, a static condition) when one slice
is empty.

mango probability-matrix convention (see ``mango.oscillator``):
``P[..., beta, alpha] = P(nu_alpha -> nu_beta)``, i.e. gather
``P[i, flavor_to[i], flavor_from[i]]``.
"""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

import mango


@typing.runtime_checkable
class OscConfig(typing.Protocol):
    """Structural interface for a per-event oscillation-weight provider.

    ``SampleModel.osc`` (DESIGN.md Section 7) is typed against this Protocol.
    Any object with a matching ``weights`` method satisfies it structurally
    -- no base-class inheritance required (house style: no inheritance
    trees).
    """

    def weights(self, params: typing.Any, sample: typing.Any) -> jax.Array:
        """Per-event oscillation weight, shape ``(sample.n_events,)``, f64.

        ``params`` is any mango params pytree (``OscParams``,
        ``Sterile3plus1``, ``NFlavorParams``, ...). ``sample`` duck-types the
        ``EventSample`` interface (DESIGN.md Section 3): ``true_energy``,
        ``flavor_from``, ``flavor_to``, ``anti_split``, ``n_events``.
        """
        ...


def _gather_and_guard(P: jax.Array, flavor_from: jax.Array,
                      flavor_to: jax.Array) -> jax.Array:
    """Per-event gather ``P[i, flavor_to[i], flavor_from[i]]`` with the
    unoscillated-event where-guard.

    ``P`` is ``(n, F, F)`` (``F`` = 3 normally, but can exceed 3 for sterile
    params -- handled generically since we only ever index it with
    ``flavor_from``/``flavor_to`` values, never assume ``F == 3``).

    Events with ``flavor_from == -1`` (paired with ``flavor_to == -1`` by the
    ``EventSample`` convention -- e.g. NC background treated as unoscillated)
    get weight exactly ``1.0``. Double-``where`` (mango convention, see
    DESIGN.md Section 1 "Gradient safety"): the gather itself always uses a
    *safe* clipped index (``-1 -> 0``), so it is always a valid, finite,
    ordinary array element -- never NaN/inf and never an out-of-bounds
    access -- even for the events whose result is about to be discarded.
    Only *then* does the outer ``where`` pick ``1.0`` instead. This ordering
    matters: computing the discarded branch some NaN-producing way first
    (e.g. gathering at index -1 and relying on the outer ``where`` to hide
    it) would still poison the reverse-mode gradient, because
    ``grad(where(c, a, b))`` includes contributions through *both* branches.
    """
    unosc = flavor_from == -1
    safe_from = jnp.where(unosc, 0, flavor_from)
    safe_to = jnp.where(unosc, 0, flavor_to)
    idx = jnp.arange(flavor_from.shape[0])
    w = P[idx, safe_to, safe_from]
    return jnp.where(unosc, jnp.asarray(1.0, dtype=w.dtype), w)


def _osc_weights(sample: typing.Any, prob_fn) -> jax.Array:
    """Split ``sample`` at the static ``anti_split``, call ``prob_fn`` on each
    non-empty (nu, antinu) slice, gather+guard, and concatenate back to
    ``(N,)`` in the original (sorted) event order.

    ``prob_fn(E_slice, anti) -> (n, F, F)`` probability matrix; the ``params``
    it closes over are exactly what makes the result differentiable w.r.t.
    the oscillation parameters (mango autodiff-first).
    """
    n = sample.n_events
    split = sample.anti_split
    pieces = []
    if split > 0:
        sl = slice(0, split)
        P = prob_fn(sample.true_energy[sl], anti=False)
        pieces.append(_gather_and_guard(P, sample.flavor_from[sl], sample.flavor_to[sl]))
    if split < n:
        sl = slice(split, n)
        P = prob_fn(sample.true_energy[sl], anti=True)
        pieces.append(_gather_and_guard(P, sample.flavor_from[sl], sample.flavor_to[sl]))
    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(pieces, axis=0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConstantMatterOsc:
    """Long-baseline oscillation through constant-density matter (DUNE/T2K-
    like), via ``mango.probability_constant``.

    ``baseline_km``, ``density_gcc`` and ``ye`` are **array leaves**, not
    static metadata -- deliberately, per DESIGN.md Section 8 ("static
    leaves? -> leaves (differentiable is free)"): they are ordinary pytree
    data, so ``jax.grad``/``jax.jacrev`` flows through them exactly like the
    oscillation parameters, at zero extra cost. This means, for instance,
    the matter density can itself be a free (fitted or MCMC-sampled)
    parameter of the analysis -- a "matter-density systematic" -- with no
    special-casing anywhere else in the stack: whatever assembles the
    parameter vector into this dataclass just writes a traced value into
    ``density_gcc`` instead of a constant.

    ``weights(params, sample)`` makes exactly two ``mango.probability_constant``
    calls (nu slice with ``anti=False``, antinu slice with ``anti=True``) at
    the default backend (``"nufast"`` -- fast analytic 3-flavor, with
    automatic fallback to a matrix-exponential backend for NSI/sterile
    ``params``, so this adapter needs no backend selection of its own).
    """

    baseline_km: jax.Array
    density_gcc: jax.Array
    ye: jax.Array = 0.5

    def weights(self, params: typing.Any, sample: typing.Any) -> jax.Array:
        def prob_fn(E, anti):
            return mango.probability_constant(
                params, E, self.baseline_km, density=self.density_gcc,
                ye=self.ye, anti=anti,
            )
        return _osc_weights(sample, prob_fn)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class VacuumOsc:
    """Vacuum oscillation (e.g. a short-baseline or reactor sample where
    matter effects are negligible), via ``mango.probability_vacuum``.

    Equivalent to ``ConstantMatterOsc(baseline_km, density_gcc=0.0)`` --
    ``mango.probability_vacuum`` is itself a ``density=0`` convenience
    wrapper around ``probability_constant`` (see ``mango.oscillator``). Kept
    as a separate, explicit class (rather than asking callers to remember
    ``density_gcc=0.0``) both for readability at call sites (a vacuum sample
    should say so) and because it carries one fewer differentiable leaf
    (no meaningless "density" to fit against for a sample that has none).
    """

    baseline_km: jax.Array

    def weights(self, params: typing.Any, sample: typing.Any) -> jax.Array:
        def prob_fn(E, anti):
            return mango.probability_vacuum(params, E, self.baseline_km, anti=anti)
        return _osc_weights(sample, prob_fn)


class EarthOsc:
    """Atmospheric oscillation adapter -- INTERFACE RESERVED, v2 (DESIGN.md
    Section 13: "Deliberate non-goals (v1)").

    Not implemented. Reserved shape, for when atmospheric samples land:

    - Would wrap ``mango.probability_earth(params, E, cos_zenith, ...)``
      (the PREM-Earth entrypoint; see ``mango`` README "Quick start") instead
      of ``probability_constant``, evaluated batched over ``(E, cos_zenith)``
      pairs -- one call per event, or a shared grid + gather if events share
      a common ``(E, cz)`` binning, mirroring the nu/antinu split used here.
    - Needs a per-event ``cos_zenith`` array, which is **not** part of the
      current ``EventSample`` schema (DESIGN.md Section 3) -- to be added as
      an additional field (an "extras" leaf) when this lands, without
      disturbing ``ConstantMatterOsc``/``VacuumOsc`` above.
    - Same gather + unoscillated-event where-guard as
      ``ConstantMatterOsc.weights`` (``_gather_and_guard`` above is written
      generically enough to be reused as-is).
    - Would satisfy the same ``OscConfig`` protocol, so ``SampleModel.osc``
      (DESIGN.md Section 7) would not need to change to accept it.
    """

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        raise NotImplementedError(
            "EarthOsc is a reserved interface (DESIGN.md Section 13, v1 "
            "non-goal) -- atmospheric oscillation weighting is not "
            "implemented yet. Use ConstantMatterOsc or VacuumOsc."
        )
