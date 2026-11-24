"""R-hat / ESS diagnostics and arviz conversion for :class:`sampling.MCMCRun`
(DESIGN.md §10).

Dependency discipline (own-files rule): this module imports only
``jax``/``blackjax``/``numpy`` at module scope. ``arviz`` is imported lazily
inside :func:`to_arviz` only, so importing ``bananafit.diagnostics`` never
requires arviz to be installed (DESIGN.md lists it as an optional "report"
extra). Nothing is imported from other ``bananafit`` modules -- functions here
take a plain :class:`sampling.MCMCRun`-shaped object (duck-typed: anything
with ``.positions (n_chains, n_samples, n_free)``, ``.log_prob (n_chains,
n_samples)`` and ``.info (dict)`` attributes works) plus an optional
duck-typed ``parameter_set``.

``parameter_set`` duck-type contract (matches DESIGN §6's ``ParameterSet``,
but accepted structurally so this module never imports ``parameters.py``):

- ``.names``: sequence of ``n_params`` physical parameter names, in the
  order ``.constrain`` emits them (fixed parameters included).
- ``.free_names``: sequence of ``n_free`` names for the *sampled* z-space
  coordinates (not currently used by ``to_arviz`` itself, since z-space
  fallback naming only kicks in when ``parameter_set is None``, but part of
  the documented contract per DESIGN §6/§10).
- ``.circular_free_names``: sequence of the ``.free_names`` entries that are
  circular (e.g. ``"deltacp"``) -- read by :func:`rhat`/:func:`ess`/
  :func:`summary` (via ``parameter_set=``) to switch those coordinates from
  linear to sin/cos-based statistics, and by :func:`to_arviz` to wrap them
  around their circular mean instead of blindly into ``(-pi, pi]``. Since
  circular parameters use the identity transform (``parameters.py``), the
  raw z-space coordinate *is* the physical angle -- no ``constrain`` needed
  to interpret it circularly.
- ``.constrain(z) -> (n_params,)``: unconstrained z-space vector to physical
  values (fixed params reinserted as constants), vectorizable with
  ``jax.vmap``.
- ``.wrap_circular(physical) -> (n_params,)``: wraps circular parameters
  (e.g. deltacp) into their canonical range; identity elsewhere. Also
  vmap-able. (``to_arviz`` no longer calls this directly -- see its
  docstring -- but it remains part of the contract for callers that want the
  plain ``(-pi, pi]`` convention.)

See ``tests/test_sampling.py`` for a minimal stub implementing this contract
(identity ``constrain``/``wrap_circular``, empty ``circular_free_names``).

Circular-aware diagnostics (rhat/ess/summary/to_arviz)
-------------------------------------------------------
A branch-hopping chain on an unbounded circular coordinate (e.g. deltacp
sampled on all of R, truth near a wrap boundary) visits both a value ``x``
and its physically-identical replica ``x + 2*pi`` -- measured directly on
the large-scale RWM run, 69% of draws landed outside ``(-pi, pi]``. Treated
as an ordinary linear coordinate this makes R-hat/ESS report severe
"non-mixing" between chains that are actually in perfect agreement (they
just represent the same angle with a different integer winding), and a
naive mean/std or a blind ``mod (-pi, pi]`` wrap can split one physical mode
into two, or truncate a mode that straddles the wrap boundary.

The fix used throughout this module: for a name in ``circular_names``
(explicit, or read off ``parameter_set.circular_free_names``), map the
z-space draws through ``sin``/``cos`` before handing them to blackjax's
diagnostics (both are exactly ``2*pi``-periodic, so winding number is
invisible to them) and report the *worse* of the two component diagnostics
(higher R-hat / lower ESS -- the conservative choice, since sin and cos
alias different features of the circular distribution and either could be
the one that exposes a real mixing problem). For point estimates,
:func:`summary` reports the circular mean (``atan2(mean(sin), mean(cos))``)
and circular standard deviation (``sqrt(-2*ln(R))``, ``R`` = mean resultant
length) in place of the linear mean/std. :func:`to_arviz` wraps circular
draws into the window centered on their own circular mean rather than a
fixed ``(-pi, pi]`` -- the wrap boundary moves to sit opposite the mode
instead of through it, so a mode near ``+-pi`` is never artificially split.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import blackjax.diagnostics as _bj_diag

__all__ = ["rhat", "ess", "summary", "to_arviz", "octant_indicator", "lockstep_efficiency"]


def _param_names(n_free: int, names: Sequence[str] | None) -> list:
    if names is None:
        return [f"z{i}" for i in range(n_free)]
    names = list(names)
    if len(names) != n_free:
        raise ValueError(
            f"names has {len(names)} entries but run has {n_free} parameters"
        )
    return names


def _resolve_circular_names(
    parameter_set: Any | None, circular_names: Sequence[str] | None
) -> tuple:
    """``circular_names`` (explicit) wins; otherwise read
    ``parameter_set.circular_free_names`` if present; otherwise none."""
    if circular_names is not None:
        return tuple(circular_names)
    if parameter_set is not None and hasattr(parameter_set, "circular_free_names"):
        return tuple(parameter_set.circular_free_names)
    return ()


def _circular_indices(names: Sequence[str], circular_names: Sequence[str]) -> list:
    name_set = set(circular_names)
    return [i for i, n in enumerate(names) if n in name_set]


def _circular_columns(positions, circ_idx):
    """Only the circular coordinates, ``(n_chains, n_samples, len(circ_idx))``.
    Both blackjax diagnostics treat every trailing coordinate independently,
    so the sin/cos statistics of a circular coordinate do not need sin/cos
    of the whole (possibly multi-GB) chain array. Indexed in place: a numpy
    (memmap) history is sliced on the host, a jax array on its own device."""
    return positions[..., np.asarray(circ_idx)]


def rhat(
    run,
    names: Sequence[str] | None = None,
    parameter_set: Any | None = None,
    circular_names: Sequence[str] | None = None,
) -> dict:
    """Gelman-Rubin R-hat per parameter, from ``blackjax.diagnostics.
    potential_scale_reduction``. ``run.positions`` is ``(n_chains,
    n_samples, n_free)``; blackjax's ``chain_axis=0, sample_axis=1`` default
    matches that layout exactly, and the function broadcasts over any extra
    trailing axes -- so a single call handles every parameter at once,
    returning ``(n_free,)`` (squeezed down to a 0-d scalar if ``n_free==1``,
    hence the ``jnp.atleast_1d`` guard below).

    ``parameter_set``/``circular_names`` (see module docstring): for any
    named circular parameter, R-hat is instead ``max(rhat(sin(x)),
    rhat(cos(x)))`` -- plain linear R-hat on an unbounded, branch-hopping
    circular coordinate is not meaningful (two chains exploring the same
    physical mode via different windings look like they have not mixed).

    Returns ``{name: float(rhat)}``.
    """
    positions = run.positions
    values = np.array(jnp.atleast_1d(  # np.array (not asarray): force a writable copy
        _bj_diag.potential_scale_reduction(positions, chain_axis=0, sample_axis=1)
    ))
    names = _param_names(values.shape[0], names)
    circ = _resolve_circular_names(parameter_set, circular_names)
    circ_idx = _circular_indices(names, circ)
    if circ_idx:
        circ_pos = _circular_columns(positions, circ_idx)
        r_sin = np.asarray(jnp.atleast_1d(
            _bj_diag.potential_scale_reduction(jnp.sin(circ_pos), chain_axis=0, sample_axis=1)
        ))
        r_cos = np.asarray(jnp.atleast_1d(
            _bj_diag.potential_scale_reduction(jnp.cos(circ_pos), chain_axis=0, sample_axis=1)
        ))
        for j, i in enumerate(circ_idx):
            values[i] = max(r_sin[j], r_cos[j])
    return {n: float(v) for n, v in zip(names, values)}


def ess(
    run,
    names: Sequence[str] | None = None,
    parameter_set: Any | None = None,
    circular_names: Sequence[str] | None = None,
) -> dict:
    """Effective sample size per parameter, from ``blackjax.diagnostics.
    effective_sample_size`` (same ``chain_axis=0, sample_axis=1`` convention
    as :func:`rhat`, same all-parameters-at-once call). Returns ``{name:
    float(ess)}``.

    ``parameter_set``/``circular_names`` (see module docstring and
    :func:`rhat`): for any named circular parameter, ESS is instead
    ``min(ess(sin(x)), ess(cos(x)))`` -- the conservative (worse) of the two
    component ESS values.
    """
    positions = run.positions
    values = np.array(jnp.atleast_1d(  # np.array (not asarray): force a writable copy
        _bj_diag.effective_sample_size(positions, chain_axis=0, sample_axis=1)
    ))
    names = _param_names(values.shape[0], names)
    circ = _resolve_circular_names(parameter_set, circular_names)
    circ_idx = _circular_indices(names, circ)
    if circ_idx:
        circ_pos = _circular_columns(positions, circ_idx)
        e_sin = np.asarray(jnp.atleast_1d(
            _bj_diag.effective_sample_size(jnp.sin(circ_pos), chain_axis=0, sample_axis=1)
        ))
        e_cos = np.asarray(jnp.atleast_1d(
            _bj_diag.effective_sample_size(jnp.cos(circ_pos), chain_axis=0, sample_axis=1)
        ))
        for j, i in enumerate(circ_idx):
            values[i] = min(e_sin[j], e_cos[j])
    return {n: float(v) for n, v in zip(names, values)}


def octant_indicator(run, parameter_set, name: str = "theta23") -> dict:
    """Mixing of the binary octant indicator ``1[theta23 > pi/4]``.

    The per-parameter R-hat/ESS of :func:`rhat`/:func:`ess` can look healthy
    while the chains rarely change octant: within one octant the draws mix
    well, and only the octant fraction converges slowly. This function
    measures that fraction directly. ``parameter_set`` must provide
    ``.constrain`` and ``.names`` (module docstring); ``name`` is the mixing
    angle in radians.

    Returns ``{"p_upper", "ess", "mcse", "rhat", "switches_per_chain",
    "p_upper_per_chain"}``. ``mcse = sqrt(p (1 - p) / ess)``. ``ess`` and
    ``rhat`` are ``nan`` when no chain ever changes octant (the within-chain
    variance is zero, so they are undefined); ``switches_per_chain`` then
    shows it.
    """
    names = list(parameter_set.names)
    idx = names.index(name)
    pos = jnp.asarray(run.positions)
    n_chains, n_samples, n_free = pos.shape
    theta = jax.vmap(lambda z: parameter_set.constrain(z)[idx])(pos.reshape(-1, n_free))
    ind = np.asarray(theta > jnp.pi / 4, dtype=float).reshape(n_chains, n_samples)
    p = float(ind.mean())
    switches = (np.diff(ind, axis=1) != 0).sum(axis=1)
    if switches.sum() == 0:
        e = r = float("nan")
    else:
        e = float(jnp.ravel(_bj_diag.effective_sample_size(jnp.asarray(ind), chain_axis=0, sample_axis=1))[0])
        r = float(jnp.ravel(_bj_diag.potential_scale_reduction(jnp.asarray(ind), chain_axis=0, sample_axis=1))[0])
    mcse = float(np.sqrt(p * (1.0 - p) / e)) if np.isfinite(e) and e > 0 else float("nan")
    return {"p_upper": p, "ess": e, "mcse": mcse, "rhat": r,
            "switches_per_chain": switches.tolist(),
            "p_upper_per_chain": ind.mean(axis=1).tolist()}


def lockstep_efficiency(run) -> dict:
    """How much of a vmapped NUTS run's leapfrog work was useful.

    ``run_mcmc`` advances all chains together under ``jax.vmap``. blackjax's
    NUTS tree is a while-loop, so at every step the batch keeps integrating
    until the chain with the longest trajectory stops: the cost of a step is
    the *maximum* leapfrog count over chains, while each chain only needed its
    own. This returns ``{"mean_steps", "mean_max_steps", "efficiency"}`` with
    ``efficiency = mean_steps / mean_max_steps`` (1 = no waste). On a device
    that runs the chains in parallel (GPU) the waste is idle lanes; on CPU,
    where vmapping does not run chains in parallel, it is extra wall time.

    Exact for ``thinning=1``. With thinning, ``run.info`` holds per-block
    means, so the per-step maximum is underestimated and ``efficiency`` is an
    upper bound.
    """
    if "num_integration_steps" not in run.info:
        raise ValueError("lockstep_efficiency needs a NUTS run (info['num_integration_steps']).")
    n = np.asarray(run.info["num_integration_steps"], dtype=float)  # (n_chains, n_samples)
    mean_steps = float(n.mean())
    mean_max = float(n.max(axis=0).mean())
    return {"mean_steps": mean_steps, "mean_max_steps": mean_max,
            "efficiency": mean_steps / mean_max if mean_max > 0 else float("nan")}


def summary(
    run,
    names: Sequence[str] | None = None,
    parameter_set: Any | None = None,
    circular_names: Sequence[str] | None = None,
) -> dict:
    """Print and return a per-parameter table of mean/std/rhat/ess (z-space
    -- this is a sampler diagnostic, not a physical-space report; use
    :func:`to_arviz` + arviz's own summary for physical-space reporting).

    ``parameter_set``/``circular_names`` (see module docstring): for any
    named circular parameter, ``"mean"``/``"std"`` report the *circular*
    mean/std (``atan2(mean(sin), mean(cos))`` and
    ``sqrt(-2*ln(mean resultant length))``) instead of the linear moments --
    a branch-hopping chain's linear mean/std over raw unbounded draws is not
    a meaningful summary of a periodic posterior -- and ``rhat``/``ess`` are
    the circular-aware values from :func:`rhat`/:func:`ess`. The returned
    row schema is unchanged (``{"mean", "std", "rhat", "ess"}``) for full
    backward compatibility; circular rows are marked with a trailing
    ``"(circ)"`` in the printed table only.
    """
    n_free = run.positions.shape[-1]
    names = _param_names(n_free, names)
    circ = _resolve_circular_names(parameter_set, circular_names)
    circ_idx = set(_circular_indices(names, circ))

    positions = np.asarray(run.positions)
    mean = positions.mean(axis=(0, 1))
    std = positions.std(axis=(0, 1), ddof=1)

    if circ_idx:
        sin_mean = np.sin(positions).mean(axis=(0, 1))
        cos_mean = np.cos(positions).mean(axis=(0, 1))
        resultant = np.clip(np.sqrt(sin_mean**2 + cos_mean**2), 1e-12, 1.0)
        circ_mean = np.arctan2(sin_mean, cos_mean)
        circ_std = np.sqrt(-2.0 * np.log(resultant))
        for i in circ_idx:
            mean[i] = circ_mean[i]
            std[i] = circ_std[i]

    r = rhat(run, names, parameter_set=parameter_set, circular_names=circular_names)
    e = ess(run, names, parameter_set=parameter_set, circular_names=circular_names)

    table = {
        n: {"mean": float(mean[i]), "std": float(std[i]), "rhat": r[n], "ess": e[n]}
        for i, n in enumerate(names)
    }

    header = f"{'name':>15s} {'mean':>12s} {'std':>12s} {'rhat':>8s} {'ess':>10s}"
    print(header)
    print("-" * len(header))
    for i, n in enumerate(names):
        row = table[n]
        tag = " (circ)" if i in circ_idx else ""
        print(
            f"{n:>15s} {row['mean']:12.4g} {row['std']:12.4g} "
            f"{row['rhat']:8.4f} {row['ess']:10.1f}{tag}"
        )
    return table


def to_arviz(run, parameter_set: Any | None = None):
    """Convert an :class:`sampling.MCMCRun` to an arviz ``InferenceData``
    (physical space if ``parameter_set`` is given, z-space with fallback
    names ``"z0".."z{n_free-1}"`` otherwise).

    arviz 1.2.0 API note (checked against the installed package): the
    ``InferenceData`` *class* is deprecated in favor of ``xarray``'s
    ``DataTree`` (``arviz.InferenceData`` still exists but importing/using it
    raises a ``MigrationWarning``); the supported entry point is
    ``arviz.from_dict(data, ...)`` where ``data`` is ``{group_name: {var_name:
    array}}``. Called with no ``sample_dims``, ``from_dict`` infers ``(chain,
    draw)`` from each variable's two leading axes -- exactly ``run.positions``
    /``run.info``'s layout -- so no explicit ``sample_dims``/``dims`` plumbing
    is needed here. Verified: ``isinstance(arviz.from_dict(...), arviz.
    InferenceData)`` is ``True``, so the returned object is a drop-in
    ``InferenceData`` for downstream arviz plotting/IO despite the
    DataTree-backed implementation.

    Circular parameters (see module docstring): rather than
    ``parameter_set.wrap_circular``'s blind wrap into a fixed ``(-pi, pi]``
    window (which can split a mode that straddles the boundary, or -- as
    measured on the large-scale RWM run, 69% of raw deltacp draws outside
    ``(-pi, pi]`` -- leave the bulk of a branch-hopped chain's mass on the
    "wrong" side of it), each ``parameter_set.circular_free_names`` entry is
    wrapped into the window centered on its own pooled circular mean
    (``atan2(mean(sin(x)), mean(cos(x)))`` over all chains/draws), i.e.
    ``(circular_mean - pi, circular_mean + pi]``. The circular mean is also
    stored as ``idata.posterior.attrs[f"{name}_circular_mean"]`` so
    downstream plotting/reporting code can recover the window without
    recomputing it.
    """
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "to_arviz requires the optional 'arviz' dependency "
            "(DESIGN.md's 'report' extra); install with `pip install arviz`."
        ) from exc

    positions = jnp.asarray(run.positions)  # (n_chains, n_samples, n_free)
    n_chains, n_samples, n_free = positions.shape

    circular_means = {}
    if parameter_set is None:
        names = [f"z{i}" for i in range(n_free)]
        physical = np.asarray(positions)
    else:
        names = list(parameter_set.names)
        constrain = jax.vmap(jax.vmap(parameter_set.constrain))
        physical = np.array(constrain(positions))  # np.array (not asarray): force a writable copy
        if physical.shape[-1] != len(names):
            raise ValueError(
                f"parameter_set.constrain(...) produced {physical.shape[-1]} "
                f"values but parameter_set.names has {len(names)} entries"
            )

        circular_names = set(getattr(parameter_set, "circular_free_names", ()))
        for i, name in enumerate(names):
            if name not in circular_names:
                continue
            x = physical[..., i]
            mu = float(np.arctan2(np.sin(x).mean(), np.cos(x).mean()))
            circular_means[name] = mu
            physical[..., i] = np.mod(x - mu + np.pi, 2.0 * np.pi) - np.pi + mu

    posterior = {name: physical[..., i] for i, name in enumerate(names)}
    sample_stats = {key: np.asarray(value) for key, value in run.info.items()}

    attrs = {"posterior": {f"{n}_circular_mean": mu for n, mu in circular_means.items()}} \
        if circular_means else None
    return az.from_dict({"posterior": posterior, "sample_stats": sample_stats}, attrs=attrs)
