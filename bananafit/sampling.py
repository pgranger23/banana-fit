"""blackjax-based MCMC drivers (DESIGN.md §10).

One uniform ``run_mcmc`` contract wraps blackjax's gradient-based kernels
(NUTS, HMC, MALA) and the gradient-free adaptive random walk (RWM, MaCh3
parity), all multi-chain via ``vmap`` and all built from a single jitted
per-chain warmup+scan function.

Dependency discipline (own-files rule): this module imports only
``jax``/``blackjax``/``optax`` (plus stdlib). It imports **nothing** from the
rest of ``bananafit`` -- ``parameters.py`` etc. are being written in parallel
by other agents. Interfaces that DESIGN.md ties to ``ParameterSet`` (e.g. the
``parameter_set`` argument of ``diagnostics.to_arviz``) are accepted here as
duck-typed objects; see ``diagnostics.py`` for the documented contract.

Float64: DESIGN §1 says float64 is enforced once, in ``bananafit/__init__.py``
(``jax.config.update("jax_enable_x64", True)``), and no other module may
mutate that global config (config mutation is not something a pure-function
module should do as a side effect of being imported). Concretely: **you must
``import bananafit`` (or otherwise enable x64) before calling anything in this
module** with float64 targets -- ``sampling.py`` does not do it for you. The
test module enforces this by importing ``bananafit`` first.
"""

from __future__ import annotations

import functools
import time
import warnings
from dataclasses import dataclass, field
from collections.abc import Callable

from typing import NamedTuple

import jax
import jax.numpy as jnp
import blackjax
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.mcmc.random_walk import RWState, normal as _rw_normal_step

Array = jax.Array

# Roberts-Rosenthal (2001) optimal RWM scaling constant, 2.38^2.
_ROBERTS_ROSENTHAL_C = 2.38**2

# MALA's analogous optimal-acceptance target (Roberts & Rosenthal 1998).
_MALA_TARGET_ACCEPT = 0.574

# Stan's standard HMC-family target acceptance rate (Hoffman & Gelman 2014) --
# the same default ``blackjax.window_adaptation`` uses for its dual-averaging
# step-size phase; reused verbatim for the ``warmup="step-size"`` fast path so
# a Laplace-seeded run targets the same acceptance rate a window-adapted run
# would.
_HMC_TARGET_ACCEPT = 0.80

_WARMUP_MODES = ("window", "step-size", "none")

# blackjax.hmc requires num_integration_steps with no default; window_adaptation
# takes it as an "extra_parameter". This default is a reasonable trajectory
# length for a first pass -- pass step_scale/algorithm-specific tuning at a
# higher level if a model needs something else.
_DEFAULT_HMC_INTEGRATION_STEPS = 10

_ALGORITHMS = ("nuts", "hmc", "mala", "rwm")


# ---------------------------------------------------------------------------
# MCMCRun
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MCMCRun:
    """Output of :func:`run_mcmc`.

    Registered as a pytree (not just a plain dataclass): ``positions``,
    ``log_prob``, ``info`` and ``tuned`` are all array/dict-of-array data --
    treating ``MCMCRun`` as a pytree lets a caller ``jax.jit``/``vmap`` a
    post-processing function directly over a run (e.g. batch several fits'
    results) without manually unpacking it first. Only ``algorithm`` is
    static metadata (a Python string driving diagnostics/reporting code
    paths -- it must be hashable for jit caching, which rules out it being a
    data leaf).

    Attributes
    ----------
    positions:
        ``(n_chains, n_samples, n_free)``, z-space (unconstrained) draws.
    log_prob:
        ``(n_chains, n_samples)``, target log-density at each draw.
    info:
        dict of per-algorithm diagnostic arrays, each ``(n_chains,
        n_samples)``. Always includes ``"acceptance_rate"``. NUTS/HMC also
        carry ``"is_divergent"`` (bool: any divergent transition within the
        thinning block that produced this draw). MALA/RWM also carry
        ``"is_accepted"`` (float mean acceptance within the block).
    tuned:
        dict of per-chain tuned kernel parameters. NUTS/HMC:
        ``"step_size" (n_chains,)``, ``"inverse_mass_matrix"
        (n_chains, n_free)`` (diagonal metric). HMC additionally
        ``"num_integration_steps" (n_chains,)``. MALA: ``"step_size"
        (n_chains,)``. RWM: ``"proposal_cov" (n_chains, n_free, n_free)``.
    algorithm:
        one of ``"nuts" | "hmc" | "mala" | "rwm"``.
    """

    positions: Array
    log_prob: Array
    info: dict
    tuned: dict
    algorithm: str = field(metadata=dict(static=True))


# ---------------------------------------------------------------------------
# helpers shared by every algorithm branch
# ---------------------------------------------------------------------------


def _prepare_chain_init(init, n_chains, init_jitter, key):
    """Broadcast/jitter ``init`` to ``(n_chains, n_free)``.

    ``init`` may be ``(n_free,)`` (replicated across chains, optionally
    jittered by an iid Gaussian of std ``init_jitter``) or already
    ``(n_chains, n_free)`` (taken as-is; ``n_chains`` is then inferred from
    its shape, overriding the ``n_chains`` argument).
    """
    init = jnp.asarray(init)
    if init.ndim == 1:
        n_free = init.shape[0]
        chain_init = jnp.broadcast_to(init, (n_chains, n_free))
        if init_jitter:
            noise = init_jitter * jax.random.normal(key, (n_chains, n_free))
            chain_init = chain_init + noise
        return chain_init, n_chains, n_free
    elif init.ndim == 2:
        n_chains, n_free = init.shape
        return init, n_chains, n_free
    else:
        raise ValueError(
            f"init must be 1-D (n_free,) or 2-D (n_chains, n_free), got "
            f"shape {init.shape}"
        )


def _info_field(info, name):
    """Field access that works for both a blackjax-style NamedTuple info
    (attribute access) and a plain dict info (the blocked-mixture RWM
    kernel below returns a dict so its per-move-type field set can be built
    dynamically from a caller-supplied move list, which a NamedTuple's fixed
    schema cannot do)."""
    return info[name] if isinstance(info, dict) else getattr(info, name)


def _reduce_block_info(info, spec):
    """Reduce one thinning-block's worth of per-step ``info`` (a blackjax
    NamedTuple or plain dict, leading axis = block/thinning length) down to
    scalars per the ``spec`` dict {field_name: "mean" | "any"}. "mean"
    averages the field over the whole block (all steps, kept or discarded,
    contribute to diagnostics like acceptance rate); "any" flags a block as
    divergent if *any* step within it (kept or discarded) was divergent, so
    thinning never silently hides a divergence.
    """
    out = {}
    for name, mode in spec.items():
        x = _info_field(info, name)
        if mode == "mean":
            out[name] = jnp.mean(x, axis=0)
        elif mode == "any":
            out[name] = jnp.any(x, axis=0)
        else:  # pragma: no cover - internal misconfiguration
            raise ValueError(f"unknown reduction mode {mode!r}")
    return out


def _maybe_wrap_progress(body_fn, n_outer, progress):
    """Lazily wrap a scan body with jax_tqdm's ``scan_tqdm`` if requested and
    importable; otherwise return ``body_fn`` unchanged (silent skip, per
    spec). ``body_fn`` must have signature ``(carry, (index, xs)) -> ...``
    (jax_tqdm requires the leading element of the scanned ``xs`` tuple to be
    ``jnp.arange(n_outer)``, which the callers below already supply).
    """
    if not progress:
        return body_fn
    try:
        from jax_tqdm import scan_tqdm
    except ImportError:
        return body_fn
    return scan_tqdm(n_outer)(body_fn)


def _thinned_scan(step_fn, state, keys_flat, n_samples, thinning, info_spec, progress):
    """Run ``n_samples * thinning`` kernel steps, keeping only every
    ``thinning``-th draw. Implemented as an outer scan of length
    ``n_samples`` whose body runs an inner scan of length ``thinning`` and
    reduces its ``info`` down to scalars (see ``_reduce_block_info``) -- the
    intermediate (discarded) positions are never materialized outside the
    inner scan, so memory scales with ``n_samples`` not
    ``n_samples * thinning``.
    """
    keys = keys_flat.reshape(n_samples, thinning, keys_flat.shape[-1])

    def outer(state, xs):
        _, keys_block = xs

        def inner(state, k):
            state, info = step_fn(k, state)
            return state, info

        state, infos = jax.lax.scan(inner, state, keys_block)
        info_out = _reduce_block_info(infos, info_spec)
        return state, (state.position, state.logdensity, info_out)

    outer = _maybe_wrap_progress(outer, n_samples, progress)

    _, (positions, log_prob, infos) = jax.lax.scan(
        outer, state, (jnp.arange(n_samples), keys)
    )
    return positions, log_prob, infos


# ---------------------------------------------------------------------------
# gradient-based algorithms: nuts / hmc / mala
# ---------------------------------------------------------------------------

_HMC_FAMILY_INFO_SPEC = {"acceptance_rate": "mean", "is_divergent": "any"}
# NUTS-only: blackjax's NUTSInfo (unlike plain HMC's HMCInfo) additionally
# carries tree-depth diagnostics -- num_trajectory_expansions (doublings
# actually used, capped by max_num_doublings) and num_integration_steps
# (leapfrog count, up to 2**max_num_doublings) -- verified fields against the
# installed blackjax 1.6.2 NUTSInfo. Surfacing their per-block mean here is
# what makes the "mean tree depth"/leapfrog-count pathology (see
# benchmarks/NUTS_TUNING.md and examples/LARGE_FIT_REPORT.md's "scaling
# cliff") directly visible in run.info instead of only inferable from wall
# time; purely additive relative to _HMC_FAMILY_INFO_SPEC, so this cannot
# affect algorithm="hmc" (HMCInfo has no such fields, hence the separate spec).
_NUTS_INFO_SPEC = {
    **_HMC_FAMILY_INFO_SPEC,
    "num_trajectory_expansions": "mean",
    "num_integration_steps": "mean",
}
_MALA_INFO_SPEC = {"acceptance_rate": "mean", "is_accepted": "mean"}


def _mala_warmup(logdensity, key, x0, n_warmup, target=_MALA_TARGET_ACCEPT,
                  initial_step_size=1.0):
    """Step-size-only warmup for MALA via dual averaging.

    Deviation from the literal DESIGN.md text ("nuts/hmc/mala:
    blackjax.window_adaptation per chain"): verified empirically against the
    installed blackjax 1.6.2 that ``blackjax.window_adaptation(blackjax.mala,
    ...)`` raises ``TypeError: build_kernel.<locals>.kernel() takes 4
    positional arguments but 5 were given``. ``staged_adaptation`` (the engine
    behind ``window_adaptation``) unconditionally calls the mcmc kernel as
    ``kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix,
    **extra)`` -- correct for the HMC family, but MALA's kernel signature is
    ``(rng_key, state, logdensity_fn, step_size)`` (no mass matrix; MALA
    preconditions with the identity by construction). So MALA gets its own,
    much simpler, warmup: plain step-size dual averaging (the same
    Nesterov/Hoffman-Gelman scheme ``window_adaptation`` itself uses for step
    size) targeting the Roberts-Rosenthal-optimal 0.574 acceptance rate for
    Langevin proposals, with no mass-matrix adaptation at all.
    """
    kernel = blackjax.mala.build_kernel()
    state = blackjax.mala.init(x0, logdensity)
    da_init, da_update, da_final = dual_averaging_adaptation(target)
    da_state = da_init(initial_step_size)

    def body(carry, k):
        state, da_state = carry
        step_size = da_final(da_state)
        state, info = kernel(k, state, logdensity, step_size)
        da_state = da_update(da_state, info.acceptance_rate)
        return (state, da_state), None

    keys = jax.random.split(key, n_warmup)
    (state, da_state), _ = jax.lax.scan(body, (state, da_state), keys)
    return state, {"step_size": da_final(da_state)}


def _hmc_family_step_size_warmup(logdensity, key, x0, n_warmup, kernel, extra_kwargs,
                                  inverse_mass_matrix, target=_HMC_TARGET_ACCEPT,
                                  initial_step_size=1.0, axis_name=None):
    """Step-size-only warmup for the HMC family (nuts/hmc) via dual
    averaging, with the inverse mass matrix held *fixed* at a caller-supplied
    value (typically a MAP+Laplace covariance estimate, DESIGN.md's
    ``laplace_covariance``) instead of being estimated online.

    This is the gradient-based analogue of ``_mala_warmup`` -- same Nesterov
    dual-averaging scheme, just targeting the HMC-family's Stan-standard 0.80
    acceptance rate instead of MALA's 0.574 -- and it is ``run_mcmc``'s
    ``warmup="step-size"`` mode: unlike ``blackjax.window_adaptation`` (whose
    cost is dominated by the successive-doubling *mass-matrix* estimation
    windows, not the step-size dual averaging itself), this skips mass-matrix
    estimation entirely, so a full ``n_warmup``-step call here is far cheaper
    than the equivalent window-adapted call.

    ``axis_name`` (set when the chains are vmapped with that axis name): the
    dual averaging is driven by the acceptance rate averaged over all chains
    at each step, so every chain ends with the same step size (see
    ``run_mcmc``'s ``shared_step_size``).
    """
    state = blackjax.hmc.init(x0, logdensity)  # nuts.init is hmc.init (shared)
    da_init, da_update, da_final = dual_averaging_adaptation(target)
    da_state = da_init(initial_step_size)

    def body(carry, k):
        state, da_state = carry
        step_size = da_final(da_state)
        state, info = kernel(
            k, state, logdensity, step_size, inverse_mass_matrix, **extra_kwargs
        )
        acc = info.acceptance_rate
        if axis_name is not None:
            acc = jax.lax.pmean(acc, axis_name)
        da_state = da_update(da_state, acc)
        return (state, da_state), None

    keys = jax.random.split(key, n_warmup)
    (state, da_state), _ = jax.lax.scan(body, (state, da_state), keys)
    return state, da_final(da_state)


def _with_hop(step_fn, logdensity, hop_prob, hop_fn):
    """Follow each NUTS/HMC/MALA step, with probability ``hop_prob``, by a
    Metropolis hop ``z -> hop_fn(z)``.

    ``hop_fn`` must be a unit-Jacobian involution (same contract as the RWM
    hop, see the module comment above ``_build_mixture_rwm_kernel``), so the
    hop is a symmetric deterministic proposal and the plain Metropolis ratio
    is exact. The gradient step and the hop each leave the target invariant,
    so their composition does too. The proposed point costs one gradient;
    under ``vmap`` (all chains batched) both branches of the ``cond`` run, so
    in practice about one extra gradient per step, small next to a NUTS
    trajectory.
    Hops act in the sampling phase only; warmup is unchanged.
    """
    vg = jax.value_and_grad(logdensity)

    def step(key, state):
        k_step, k_choice, k_acc = jax.random.split(key, 3)
        state, info = step_fn(k_step, state)
        is_hop = jax.random.uniform(k_choice) < hop_prob
        proposal = hop_fn(state.position)
        # skipped when no hop is drawn, except under vmap (both branches run)
        ld_p, grad_p = jax.lax.cond(is_hop, lambda: vg(proposal),
                                    lambda: (state.logdensity, state.logdensity_grad))
        log_acc = jnp.where(jnp.isfinite(ld_p), jnp.minimum(0.0, ld_p - state.logdensity), -jnp.inf)
        accept = jnp.logical_and(is_hop, jnp.log(jax.random.uniform(k_acc)) < log_acc)
        hopped = state._replace(position=proposal, logdensity=ld_p, logdensity_grad=grad_p)
        state = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), hopped, state)
        info = {**info._asdict(), "is_hop": is_hop, "hop_accepted": accept}
        return state, info

    return step


def _run_gradient_based(logdensity, chain_init, *, algorithm, n_chains, n_warmup,
                         n_samples, key, thinning, progress,
                         inverse_mass_matrix=None, step_size=None, warmup="window",
                         max_num_doublings=None, target_acceptance_rate=None,
                         hop_prob=0.0, hop_fn=None, shared_step_size=None):
    if warmup not in _WARMUP_MODES:
        raise ValueError(f"warmup must be one of {_WARMUP_MODES}, got {warmup!r}")
    pooled_ok = warmup == "step-size" and algorithm in ("nuts", "hmc")
    if shared_step_size is None:
        shared_step_size = pooled_ok
    if shared_step_size and not pooled_ok:
        raise ValueError(
            "shared_step_size needs algorithm='nuts'/'hmc' and warmup='step-size' "
            "(the chains' acceptance rates are pooled across the vmapped chain "
            "axis during step-size warmup)."
        )
    if max_num_doublings is not None and algorithm != "nuts":
        raise ValueError(
            "max_num_doublings is a NUTS-specific tree-depth cap (hmc has a "
            f"fixed trajectory length, mala builds no tree); got algorithm="
            f"{algorithm!r}."
        )

    info_spec = _MALA_INFO_SPEC if algorithm == "mala" else (
        _NUTS_INFO_SPEC if algorithm == "nuts" else _HMC_FAMILY_INFO_SPEC
    )
    if hop_fn is not None:
        info_spec = {**info_spec, "is_hop": "mean", "hop_accepted": "mean"}
    warm_key, sample_key = jax.random.split(key)
    warm_keys = jax.random.split(warm_key, n_chains)
    sample_keys = jax.random.split(sample_key, n_chains)
    n_free = chain_init.shape[-1]

    if algorithm == "mala":
        if inverse_mass_matrix is not None:
            raise ValueError(
                "algorithm='mala' has no mass matrix (MALA preconditions "
                "with the identity by construction, see _mala_warmup's "
                "docstring); inverse_mass_matrix is not supported for mala."
            )

        if warmup == "none":
            if step_size is None:
                raise ValueError("warmup='none' requires an explicit step_size.")
            fixed_step_size = jnp.asarray(step_size)

            def warmup_one(k, x0):
                del k
                state = blackjax.mala.init(x0, logdensity)
                return state, {"step_size": fixed_step_size}

        else:
            # "window" and "step-size" coincide for mala: blackjax's
            # window_adaptation cannot drive a mala kernel at all (see
            # _mala_warmup's docstring for the verified TypeError), so
            # step-size-only dual averaging has *always* been mala's only
            # warmup here -- "window" is accepted only for a uniform
            # cross-algorithm ``warmup=`` argument, not because it does
            # anything different from "step-size" for this algorithm.
            init_ss = 1.0 if step_size is None else step_size
            mala_target = _MALA_TARGET_ACCEPT if target_acceptance_rate is None \
                else target_acceptance_rate

            def warmup_one(k, x0):
                return _mala_warmup(logdensity, k, x0, n_warmup,
                                     target=mala_target, initial_step_size=init_ss)

        def make_step(tuned_i):
            return blackjax.mala(logdensity, step_size=tuned_i["step_size"]).step

    else:
        algo = blackjax.nuts if algorithm == "nuts" else blackjax.hmc
        extra = (
            {"num_integration_steps": _DEFAULT_HMC_INTEGRATION_STEPS}
            if algorithm == "hmc"
            else {}
        )
        if algorithm == "nuts" and max_num_doublings is not None:
            # A plain Python int closed over here (never a value pulled out of
            # a per-chain vmapped/traced pytree) -- verified against installed
            # blackjax 1.6.2 that blackjax.nuts(..., max_num_doublings=X)
            # requires a *concrete* int (it statically sizes a trajectory-
            # storage array); passing a traced array raises "Shapes must be
            # 1D sequences of concrete values of integer type". Threading it
            # through this ``extra`` dict is safe precisely because ``extra``
            # is captured by closure into warmup_one/kernel calls below, not
            # part of any jax.vmap-mapped argument.
            extra["max_num_doublings"] = max_num_doublings
        hmc_target = _HMC_TARGET_ACCEPT if target_acceptance_rate is None \
            else target_acceptance_rate

        if warmup == "none":
            if step_size is None:
                raise ValueError("warmup='none' requires an explicit step_size.")
            imm = (
                jnp.ones(n_free) if inverse_mass_matrix is None
                else jnp.asarray(inverse_mass_matrix)
            )
            fixed_step_size = jnp.asarray(step_size)

            def warmup_one(k, x0):
                del k
                state = blackjax.hmc.init(x0, logdensity)
                tuned = {"step_size": fixed_step_size, "inverse_mass_matrix": imm}
                if algorithm == "hmc":
                    tuned["num_integration_steps"] = _DEFAULT_HMC_INTEGRATION_STEPS
                if algorithm == "nuts" and max_num_doublings is not None:
                    # informational only (mirrors hmc's num_integration_steps
                    # entry) -- make_step below always uses the closed-over
                    # max_num_doublings directly for nuts, never this array,
                    # per the staticness constraint noted above.
                    tuned["max_num_doublings"] = jnp.asarray(max_num_doublings)
                return state, tuned

        elif warmup == "step-size":
            # The cheap path (DESIGN motivation: measured ~1.1-1.4s/draw for
            # full window_adaptation on a correlated 8-param toy, 468s for
            # 80 warmup steps on the 20k-event model) -- a MAP+Laplace
            # covariance (verified PD/dense-or-diagonal, see
            # laplace_covariance) stands in directly for blackjax's
            # estimated inverse_mass_matrix (blackjax.mcmc.metrics.
            # default_metric/gaussian_euclidean accept either a 1-D diagonal
            # or 2-D dense array -- verified against the installed blackjax
            # 1.6.2 source), and only the step size is dual-averaged.
            if inverse_mass_matrix is None:
                raise ValueError(
                    "warmup='step-size' requires inverse_mass_matrix (e.g. "
                    "from laplace_covariance()); use warmup='window' to "
                    "adapt it online instead, or warmup='none' with an "
                    "explicit step_size to skip tuning entirely."
                )
            imm = jnp.asarray(inverse_mass_matrix)
            init_ss = 1.0 if step_size is None else step_size
            kernel = algo.build_kernel()

            def warmup_one(k, x0):
                state, tuned_step_size = _hmc_family_step_size_warmup(
                    logdensity, k, x0, n_warmup, kernel, extra, imm,
                    target=hmc_target, initial_step_size=init_ss,
                    axis_name="chain" if shared_step_size else None,
                )
                tuned = {"step_size": tuned_step_size, "inverse_mass_matrix": imm}
                if algorithm == "hmc":
                    tuned["num_integration_steps"] = _DEFAULT_HMC_INTEGRATION_STEPS
                if algorithm == "nuts" and max_num_doublings is not None:
                    tuned["max_num_doublings"] = jnp.asarray(max_num_doublings)
                return state, tuned

        else:  # "window" -- default, backward-compatible behavior
            window_kwargs = dict(extra)
            if inverse_mass_matrix is not None:
                imm_seed = jnp.asarray(inverse_mass_matrix)
                window_kwargs["is_mass_matrix_diagonal"] = imm_seed.ndim == 1
                window_kwargs["initial_inverse_mass_matrix"] = imm_seed
            if step_size is not None:
                window_kwargs["initial_step_size"] = step_size
            if target_acceptance_rate is not None:
                window_kwargs["target_acceptance_rate"] = target_acceptance_rate

            def warmup_one(k, x0):
                warm = blackjax.window_adaptation(algo, logdensity, **window_kwargs)
                (state, tuned), _ = warm.run(k, x0, num_steps=n_warmup)
                return state, tuned

        def make_step(tuned_i):
            if algorithm == "hmc":
                return blackjax.hmc(
                    logdensity,
                    step_size=tuned_i["step_size"],
                    inverse_mass_matrix=tuned_i["inverse_mass_matrix"],
                    num_integration_steps=tuned_i["num_integration_steps"],
                ).step
            nuts_kwargs = dict(
                step_size=tuned_i["step_size"],
                inverse_mass_matrix=tuned_i["inverse_mass_matrix"],
            )
            if max_num_doublings is not None:
                # Deliberately the closed-over run_mcmc-level Python int, NOT
                # tuned_i.get("max_num_doublings") -- see the staticness note
                # by the ``extra["max_num_doublings"] = ...`` assignment
                # above. Both are equal in value (same knob, every chain);
                # only the closed-over form is safe to hand to blackjax.nuts
                # from inside this vmapped (run_one) context.
                nuts_kwargs["max_num_doublings"] = max_num_doublings
            return blackjax.nuts(logdensity, **nuts_kwargs).step

    if shared_step_size:
        states, tuned = jax.vmap(warmup_one, axis_name="chain")(warm_keys, chain_init)
    else:
        states, tuned = jax.vmap(warmup_one)(warm_keys, chain_init)

    def run_one(state, tuned_i, k):
        step_fn = make_step(tuned_i)
        if hop_fn is not None:
            step_fn = _with_hop(step_fn, logdensity, hop_prob, hop_fn)
        keys_flat = jax.random.split(k, n_samples * thinning)
        return _thinned_scan(
            step_fn, state, keys_flat, n_samples, thinning, info_spec, progress
        )

    positions, log_prob, infos = jax.vmap(run_one)(states, tuned, sample_keys)

    if algorithm in ("nuts", "hmc"):
        n_divergent = int(jnp.sum(infos["is_divergent"]))
        if n_divergent > 0:
            warnings.warn(
                f"{algorithm}: {n_divergent} divergent transitions across all "
                f"chains/draws (out of {infos['is_divergent'].size} recorded "
                f"blocks) -- inspect run.info['is_divergent'] and consider "
                f"reparameterizing or raising target_acceptance_rate.",
                RuntimeWarning,
                stacklevel=2,
            )

    return MCMCRun(
        positions=positions,
        log_prob=log_prob,
        info=infos,
        tuned=tuned,
        algorithm=algorithm,
    )


# ---------------------------------------------------------------------------
# rwm (MaCh3 parity: no gradients, no window adaptation)
# ---------------------------------------------------------------------------

_RWM_INFO_SPEC = {"acceptance_rate": "mean", "is_accepted": "mean"}


# ---------------------------------------------------------------------------
# rwm mixture proposal: Gaussian RWM + deterministic-reflection "mode hop"
# ---------------------------------------------------------------------------
#
# Motivation (a DUNE-like stats-only FD fit): a local
# Gaussian RWM proposal cannot cross the deep likelihood valley separating
# the two theta23-octant modes (sin^2theta23 ~ x vs ~ 1-x) at any practical
# step scale -- the valley is many sigma wide in the tuned proposal's own
# units. MaCh3 handles this with dedicated symmetric "swap" moves. This is
# the same idea, generalized to reflect an arbitrary subset of z-space
# coordinates about the origin.
#
# Why z_i -> -z_i is the right reflection: every free ParamSpec's bijector is
# built once from its bounds (``parameters.py``'s ``biject_to``); for a
# parameter bounded to a symmetric-around-its-midpoint finite interval
# ``(lo, hi)`` (in particular ``(0, 1)``, used for a flat-prior
# ``sin^2theta`` parameter), that bijector's inverse is exactly a scaled
# sigmoid centered at ``z=0``, so ``t(-z) == lo + hi - t(z)`` -- verified
# numerically against the installed numpyro for the ``(0, 1)`` case, where
# this reduces to ``t(-z) == 1 - t(z)``, i.e. the physical octant reflection
# ``x -> 1-x``. Other free coordinates at their own ``hop_indices`` entries
# reflect the same way; coordinates *not* listed in ``hop_indices`` are left
# untouched.
#
# CAVEAT (found empirically on the DUNE-like stats-only FD model this was
# built for): a pure per-coordinate reflection only works well if the *other*
# free coordinates barely move between the two modes. When they don't --
# e.g. that model's wrong-theta23-octant mode also sits at a measurably
# different sin^2theta13 (the octant-theta13 degeneracy) -- reflecting theta23 alone while leaving every other
# coordinate at the *current* (correct-octant) chain value proposes a point
# nowhere near the actual wrong-octant mode, so it is essentially always
# rejected (measured: ~0/10000 hops accepted in production-scale tuning).
# ``hop_move_fn`` (below) generalizes the reflection to an arbitrary
# caller-supplied involution -- e.g. a translate-swap keyed off two
# gradient-free-optimizer-located mode anchors -- when the default
# origin-reflection is not enough.
#
# Why this preserves detailed balance without a Hastings correction: this is
# a *mixture of two transition kernels* chosen by a state-independent coin
# flip each step -- w.p. ``1-hop_prob`` the ordinary symmetric additive
# Gaussian RWM step (P(step|x) = P(-step|x+step)), w.p. ``hop_prob`` a
# deterministic involution R (R(R(z)) = z by construction: negating a fixed
# set of coordinates twice is the identity, and a pure translation-based swap
# built from two fixed anchor points -- see ``hop_move_fn`` -- is likewise an
# involution when the anchors sit deep in well-separated basins) with unit
# Jacobian (a signed permutation, or a pure translation -- both have Jacobian
# determinant exactly +-1). Each sub-kernel *individually* satisfies detailed
# balance w.r.t. the target (the Gaussian step is the standard RWM
# symmetric-proposal case; the reflection/translate step is a symmetric
# proposal too, since for a bijection R, "propose R(z) from z" and "propose
# R(z') from z'=R(z), i.e. recover z" are the same single transition
# considered from either end -- a deterministic symmetric proposal is the
# discrete analogue of a symmetric continuous one). A probabilistic mixture
# of kernels that individually leave a target distribution pi invariant also
# leaves pi invariant (Tierney 1994) -- so the plain Metropolis acceptance
# ``min(1, pi(z')/pi(z))`` (no ``proposal_logdensity_fn`` / Hastings ratio
# term) is exactly correct for both branches, with no additional correction
# needed for the mixing itself. This is validated against bimodal toys in
# ``tests/test_sampling.py`` (both a reflection-friendly toy and an
# asymmetric one demonstrating why ``hop_move_fn`` is sometimes needed).


class MixtureRWInfo(NamedTuple):
    """Per-step info for the Gaussian+reflection mixture RWM kernel.
    ``acceptance_rate``/``is_accepted`` match plain RWM's ``RWInfo`` (minus
    its unused ``proposal`` field); ``is_hop``/``hop_accepted`` are additional
    -- whether this step's proposal came from the reflection branch, and
    whether that particular proposal was accepted -- so a caller can recover
    the hop-specific acceptance rate (``mean(hop_accepted) / mean(is_hop)``)
    separately from the overall acceptance rate.
    """

    acceptance_rate: Array
    is_accepted: Array
    is_hop: Array
    hop_accepted: Array


_RWM_MIXTURE_INFO_SPEC = {
    "acceptance_rate": "mean",
    "is_accepted": "mean",
    "is_hop": "mean",
    "hop_accepted": "mean",
}


def _proposal_scale_matrix(cov):
    """Square-root factor of a proposal covariance, in the form blackjax's
    ``random_walk.normal(sigma)`` expects: it draws ``sigma @ xi`` (matrix) or
    ``sigma * xi`` (vector/scalar), i.e. ``sigma`` is a *scale*, not a
    covariance. Passing a covariance there realises ``cov @ cov.T`` -- the
    squared shape, which freezes every stiff direction."""
    cov = jnp.asarray(cov)
    if cov.ndim == 2:
        return jnp.linalg.cholesky(cov)
    return jnp.sqrt(cov)


def _build_mixture_rwm_kernel(logdensity, sigma, hop_prob, hop_indices, hop_move_fn=None,
                              *, scale_matrix=None):
    """``(init_fn, step_fn)`` for the Gaussian-RWM + hop-move mixture kernel
    described above. Hand-rolled (not routed through blackjax's
    ``rmh``/``additive_step_random_walk``) purely to surface
    ``is_hop``/``hop_accepted`` per-step diagnostics -- the actual
    Metropolis math is the same plain symmetric-proposal acceptance either
    kernel would use.

    ``hop_move_fn``, when given, is the hop proposal used *instead of* the
    default per-coordinate origin reflection (``z.at[hop_indices]
    .multiply(-1.0)``) -- any caller-supplied ``Array -> Array`` involution
    with unit-Jacobian is valid (see the module comment above for why); the
    caller is responsible for that guarantee, since it cannot be checked
    generically at trace time.

    ``sigma`` is the (fully scaled) proposal *covariance*. ``scale_matrix``,
    when given, is ``_proposal_scale_matrix(sigma)`` already computed by the
    caller (``sigma`` is then ignored) -- :func:`_run_rwm` computes it
    eagerly, exactly as this function would, and passes it in as a jit
    argument so the compiled chunk program can be reused across calls.
    """
    gauss_step = _rw_normal_step(
        _proposal_scale_matrix(sigma) if scale_matrix is None else scale_matrix)
    if hop_move_fn is None:
        hop_idx = jnp.asarray(hop_indices, dtype=jnp.int32)
        hop_move_fn = lambda z: z.at[hop_idx].multiply(-1.0)  # noqa: E731

    def init_fn(x0):
        return RWState(x0, logdensity(x0))

    def step_fn(rng_key, state):
        position, ld = state
        key_choice, key_gauss, key_accept = jax.random.split(rng_key, 3)

        is_hop = jax.random.uniform(key_choice) < hop_prob
        gauss_proposal = position + gauss_step(key_gauss, position)
        hop_proposal = hop_move_fn(position)
        proposed_position = jnp.where(is_hop, hop_proposal, gauss_proposal)

        proposed_ld = logdensity(proposed_position)
        log_p_accept = jnp.minimum(0.0, proposed_ld - ld)
        accept = jnp.log(jax.random.uniform(key_accept)) < log_p_accept

        new_position = jnp.where(accept, proposed_position, position)
        new_ld = jnp.where(accept, proposed_ld, ld)

        info = MixtureRWInfo(
            acceptance_rate=jnp.exp(log_p_accept),
            is_accepted=accept,
            is_hop=is_hop,
            hop_accepted=jnp.logical_and(accept, is_hop),
        )
        return RWState(new_position, new_ld), info

    return init_fn, step_fn


# ---------------------------------------------------------------------------
# rwm blocked mixture: Metropolis-within-Gibbs over an arbitrary set of
# NAMED, pi-invariant move types (generalizes the 2-way Gaussian+hop mixture
# above to N move types, each with its own static probability and its own
# coordinate subset/covariance -- motivated by a DUNE-like joint ND+FD fit: the full
# 292-dim proposal above mixes theta23 (bimodality handled by the hop move)
# but essentially never mixes deltacp/dm21/dm31/theta12 -- each full-vector
# move perturbs all 292 coordinates at once, so acceptance is dominated by
# the other ~286 coordinates and the slow-mixing osc directions barely move
# net of the metropolis accept/reject. In the stats-only fit these SAME
# directions mix fine in a
# LOW-dimensional osc-only proposal once systematics are held fixed -- i.e.
# this is a proposal-dimensionality problem, not a genuine posterior
# pathology. Blocking (Metropolis-within-Gibbs: propose a LOW-dimensional
# move restricted to a coordinate subset, leaving every other coordinate
# exactly as-is) is the standard fix -- this is exactly what MaCh3-style
# "parameter block" RWM samplers do, generalized here from this module's own
# existing "whole-vector Gaussian XOR whole-vector involution" 2-way mixture.
#
# Each block move is EITHER:
#   - a Gaussian step restricted to a coordinate subset ``indices``: draw
#     ``delta_block ~ N(0, sigma_block)`` (``sigma_block`` = ``step_scale *
#     cov_block``, exactly the same "target covariance x scalar step_scale"
#     convention as the whole-vector RWM case above), scatter it into an
#     all-zero ``(n_free,)`` vector at ``indices``, and add to the current
#     position -- every coordinate NOT in ``indices`` is therefore returned
#     UNCHANGED, exactly (not just "unlikely to move"); or
#   - the existing hop move (an involution, e.g. the octant translate-swap),
#     applied to the FULL vector as before.
#
# Why this still needs no Hastings correction: this is still a *mixture of
# individually pi-invariant transition kernels* chosen by a state-independent
# categorical draw each step (Tierney 1994, same argument as the 2-way
# mixture's module comment above) -- a block Gaussian RWM step restricted to
# a coordinate subset is the textbook Metropolis-within-Gibbs case (a
# symmetric proposal on the sub-vector, identity on the rest, is a symmetric
# proposal on the FULL vector: "propose z' = z + (0,...,delta,...,0)" and its
# reverse "propose z = z' - (0,...,delta,...,0)" have the same density by the
# block's own Gaussian symmetry) and the hop branch is unchanged from above.
# So every branch, and hence the mixture, is a valid Metropolis-Hastings
# kernel for pi under the plain ``min(1, pi(z')/pi(z))`` acceptance used
# below -- no per-branch or per-mixture correction term needed.
#
# The categorical move-type draw uses a single ``jax.random.uniform`` against
# static (Python-float, not traced) cumulative probability thresholds via
# ``jnp.searchsorted``, then ``jax.lax.switch`` to select which (jitted)
# proposal branch actually runs -- static shapes throughout (every branch's
# output is a full ``(n_free,)`` vector), so this is exactly as jit/vmap/scan
# -friendly as the 2-way ``jnp.where`` mixture above, just generalized past
# 2 branches. Info is a plain dict (not a fixed-schema NamedTuple) with
# per-move-type ``is_<name>``/``<name>_accepted`` fields built dynamically
# from the caller-supplied move names, so a caller can recover EVERY move
# type's own acceptance rate (``mean(<name>_accepted) / mean(is_<name>)``),
# not just an aggregate -- the design's explicit "per-move-type acceptance
# tracked in info for diagnostics" requirement.


def _build_block_mixture_rwm_kernel(logdensity, n_free, moves, *, cumprobs=None):
    """``(init_fn, step_fn)`` for the N-way blocked-mixture RWM kernel.

    ``moves``: a list of dicts, each one of:
      - Gaussian block move: ``{"name": str, "prob": float,
        "kind": "gaussian", "indices": sequence[int], "sigma": (k, k) array}``
        (``sigma`` is the FULLY SCALED proposal covariance for this block,
        i.e. already ``step_scale * cov_block`` -- this function does no
        further scaling); ``k == len(indices)``.
      - hop move: ``{"name": str, "prob": float, "kind": "hop",
        "hop_move_fn": Array -> Array}`` (a caller-supplied unit-Jacobian
        involution on the FULL ``(n_free,)`` vector, exactly the existing
        ``hop_move_fn``/default-reflection contract above).

    ``sum(m["prob"] for m in moves)`` must equal 1 (validated by the caller,
    :func:`_run_rwm`, which also builds the "full-vector" and "hop" entries
    from ``proposal_cov``/``step_scale``/``hop_prob``/``hop_indices``/
    ``hop_move_fn`` so this function itself stays agnostic to which move is
    "the original 2-way mixture's moves" vs. a caller-added block).

    ``cumprobs`` / a Gaussian move's ``"chol"`` entry, when given, are the
    already-computed ``jnp.cumsum`` of the move probabilities /
    ``jnp.linalg.cholesky(sigma)`` (``"prob"``/``"sigma"`` are then not
    read): :func:`_run_rwm` computes them eagerly, exactly as below, and
    passes them in as jit arguments so one compiled chunk program serves
    every call with the same static configuration.
    """
    names = [m["name"] for m in moves]
    if cumprobs is None:
        probs = [float(m["prob"]) for m in moves]
        cumprobs = jnp.cumsum(jnp.asarray(probs))

    def _make_branch(m):
        if m["kind"] == "gaussian":
            idx = jnp.asarray(m["indices"], dtype=jnp.int32)
            chol = m["chol"] if "chol" in m else jnp.linalg.cholesky(jnp.asarray(m["sigma"]))
            k = idx.shape[0]

            def branch(key, position):
                z = jax.random.normal(key, (k,))
                delta_block = chol @ z
                delta = jnp.zeros((n_free,), dtype=position.dtype).at[idx].set(delta_block)
                return position + delta

            return branch
        elif m["kind"] == "hop":
            move_fn = m["hop_move_fn"]

            def branch(key, position):
                del key
                return move_fn(position)

            return branch
        else:  # pragma: no cover - internal misconfiguration
            raise ValueError(f"unknown move kind {m['kind']!r}")

    branches = [_make_branch(m) for m in moves]
    n_moves = len(branches)

    def init_fn(x0):
        return RWState(x0, logdensity(x0))

    def step_fn(rng_key, state):
        position, ld = state
        key_choice, key_prop, key_accept = jax.random.split(rng_key, 3)

        u = jax.random.uniform(key_choice)
        move_idx = jnp.clip(jnp.searchsorted(cumprobs, u, side="right", method="compare_all"), 0, n_moves - 1)

        proposed_position = jax.lax.switch(move_idx, branches, key_prop, position)

        proposed_ld = logdensity(proposed_position)
        log_p_accept = jnp.minimum(0.0, proposed_ld - ld)
        accept = jnp.log(jax.random.uniform(key_accept)) < log_p_accept

        new_position = jnp.where(accept, proposed_position, position)
        new_ld = jnp.where(accept, proposed_ld, ld)

        info = {"acceptance_rate": jnp.exp(log_p_accept), "is_accepted": accept}
        for i, name in enumerate(names):
            is_this = move_idx == i
            info[f"is_{name}"] = is_this
            info[f"{name}_accepted"] = jnp.logical_and(accept, is_this)

        return RWState(new_position, new_ld), info

    return init_fn, step_fn


def _default_hop_move(hop_prob, hop_indices, hop_move_fn, n_free):
    """Shared hop-move-vs-hop-indices validation/defaulting used by both the
    2-way mixture (:func:`_run_rwm`'s ``hop_prob``-only path) and the blocked
    mixture (``block_moves``-non-empty path) -- factored out so both stay in
    sync rather than duplicating the same checks."""
    if not (0.0 < hop_prob < 1.0):
        raise ValueError(f"hop_prob must be in (0, 1), got {hop_prob}")
    if hop_move_fn is None:
        if not hop_indices:
            raise ValueError(
                "hop_prob > 0 requires either hop_move_fn (a caller-"
                "supplied involution) or a non-empty hop_indices "
                "(z-space coordinate indices to reflect about 0, e.g. "
                "the free index of a sin^2theta-parameterized angle for "
                "an octant hop)."
            )
        bad = [i for i in hop_indices if not (0 <= i < n_free)]
        if bad:
            raise ValueError(f"hop_indices out of range [0, {n_free}): {bad}")
        return _ReflectHop(tuple(int(i) for i in hop_indices))
    return hop_move_fn


@dataclass(frozen=True)
class _ReflectHop:
    """The default origin-reflection hop ``z.at[indices].multiply(-1.0)`` as
    a hashable, value-compared callable (a fresh lambda per ``run_mcmc``
    call would be a new static jit argument every call, i.e. a recompile per
    chunk). Same arithmetic as the lambda it replaces."""

    indices: tuple

    def __call__(self, z):
        return z.at[jnp.asarray(self.indices, dtype=jnp.int32)].multiply(-1.0)


class _ByIdentity:
    """Hashable-by-identity wrapper for a callable that is not hashable
    itself (e.g. an instance of a non-frozen dataclass with ``__call__``):
    such a callable can still be a static jit argument, cached per object."""

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def __hash__(self):
        return id(self.fn)

    def __eq__(self, other):
        return isinstance(other, _ByIdentity) and other.fn is self.fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _static_callable(fn):
    """``fn`` itself if hashable (functions, jitted functions, bound methods
    -- all compare by identity), else wrapped in :class:`_ByIdentity`."""
    if fn is None:
        return None
    try:
        hash(fn)
    except TypeError:
        return _ByIdentity(fn)
    return fn


class _RWMRunConfig(NamedTuple):
    """Static (hashable) part of one RWM ``run_mcmc`` call: everything that
    fixes the traced program. Arrays that may change between chunks (the
    proposal scale/Cholesky factors, move probabilities, chain state, keys)
    are jit arguments instead, see :func:`_rwm_run_compiled`.

    ``kernel`` is ``("plain",)``, ``("mixture", hop_prob, hop_indices,
    hop_move_fn)`` or ``("block", n_free, ((name, "gaussian", indices) |
    (name, "hop", hop_move_fn), ...))``. Callables compare by identity (the
    runners build ``logdensity``/``hop_move_fn`` once per process), the
    default reflection hop by value (:class:`_ReflectHop`)."""

    logdensity: Callable
    kernel: tuple
    n_warmup: int
    n_samples: int
    thinning: int
    progress: bool
    info_spec: tuple


def _rwm_kernel_from_config(logdensity, kernel, dyn):
    """Rebuild ``(init_fn, step_fn)`` inside the jit trace from the static
    ``kernel`` description and the per-call arrays ``dyn`` -- the same
    builders, with the same eagerly-precomputed factors, as before."""
    kind = kernel[0]
    if kind == "block":
        _, n_free, moves_static = kernel
        moves = []
        for (name, mkind, payload), chol in zip(moves_static, dyn["chols"]):
            if mkind == "gaussian":
                moves.append({"name": name, "kind": "gaussian", "indices": payload, "chol": chol})
            else:
                moves.append({"name": name, "kind": "hop", "hop_move_fn": payload})
        return _build_block_mixture_rwm_kernel(logdensity, n_free, moves, cumprobs=dyn["cumprobs"])
    if kind == "mixture":
        _, hop_prob, hop_indices, hop_move_fn = kernel
        return _build_mixture_rwm_kernel(logdensity, None, hop_prob, hop_indices,
                                         hop_move_fn=hop_move_fn, scale_matrix=dyn["scale"])
    rw = blackjax.additive_step_random_walk.normal_random_walk(logdensity, sigma=dyn["scale"])
    return rw.init, rw.step


@functools.partial(jax.jit, static_argnames=("cfg",))
def _rwm_run_compiled(dyn, chain_init, warm_keys, sample_keys, *, cfg):
    """Burn-in + thinned sampling for all chains as ONE jitted program,
    cached by jax on ``cfg`` (static) and the argument shapes. Before this,
    ``_run_rwm`` built fresh closures and ran the vmapped scans eagerly, so
    every ``run_mcmc`` call (every chunk of a chunked production run)
    re-traced and re-compiled the whole log-density (~50-140 s per call on
    a V100 for a large model). Same operations, same
    keys, same order as the eager version."""
    init_fn, step_fn = _rwm_kernel_from_config(cfg.logdensity, cfg.kernel, dyn)
    info_spec = dict(cfg.info_spec)
    n_warmup, n_samples, thinning = cfg.n_warmup, cfg.n_samples, cfg.thinning

    def burn_in(k, x0):
        state = init_fn(x0)
        keys = jax.random.split(k, n_warmup)

        def body(state, kk):
            state, _ = step_fn(kk, state)
            return state, None

        state, _ = jax.lax.scan(body, state, keys)
        return state

    states = jax.vmap(burn_in)(warm_keys, chain_init)

    def run_one(state, k):
        keys_flat = jax.random.split(k, n_samples * thinning)
        return _thinned_scan(
            step_fn, state, keys_flat, n_samples, thinning, info_spec, cfg.progress
        )

    return jax.vmap(run_one)(states, sample_keys)


def _run_rwm(logdensity, chain_init, *, n_chains, n_warmup, n_samples, key,
             thinning, progress, proposal_cov, step_scale, n_free,
             hop_prob=0.0, hop_indices=None, hop_move_fn=None, block_moves=None):
    if proposal_cov is None:
        raise ValueError(
            "algorithm='rwm' requires proposal_cov (a user-supplied matrix, a "
            "MAP+Laplace estimate from laplace_covariance(), or similar)."
        )
    proposal_cov = jnp.asarray(proposal_cov)
    scale = _ROBERTS_ROSENTHAL_C / n_free if step_scale is None else step_scale
    sigma = scale * proposal_cov

    if block_moves:
        # Blocked Metropolis-within-Gibbs mixture (module comment above
        # _build_block_mixture_rwm_kernel): the "full-vector" move gets
        # whatever probability isn't claimed by the caller's blocks + hop.
        block_prob_sum = sum(float(bm["prob"]) for bm in block_moves)
        full_prob = 1.0 - hop_prob - block_prob_sum
        if full_prob < -1.0e-9:
            raise ValueError(
                f"block_moves probabilities ({block_prob_sum}) + hop_prob "
                f"({hop_prob}) exceed 1.0 -- the full-vector move's implied "
                f"probability would be negative."
            )
        full_prob = max(full_prob, 0.0)
        moves = [{
            "name": "full", "prob": full_prob, "kind": "gaussian",
            "indices": tuple(range(n_free)), "sigma": sigma,
        }]
        seen_names = {"full"}
        for bm in block_moves:
            name = bm["name"]
            if name in seen_names:
                raise ValueError(
                    f"block_moves has a duplicate/reserved name {name!r} "
                    f"(names so far: {sorted(seen_names)})."
                )
            seen_names.add(name)
            if bm.get("kind", "gaussian") == "hop":
                # an extra deterministic hop (a second mode pair, e.g. a delta_CP mirror next to the
                # octant hop): same unit-Jacobian involution contract as hop_move_fn
                if not callable(bm.get("hop_move_fn")):
                    raise ValueError(f"block_moves[{name!r}] of kind 'hop' needs a callable 'hop_move_fn'.")
                moves.append({"name": name, "prob": float(bm["prob"]), "kind": "hop",
                              "hop_move_fn": bm["hop_move_fn"]})
                continue
            idx = tuple(int(i) for i in bm["indices"])
            bad = [i for i in idx if not (0 <= i < n_free)]
            if bad:
                raise ValueError(
                    f"block_moves[{name!r}]['indices'] out of range "
                    f"[0, {n_free}): {bad}"
                )
            bm_step_scale = bm.get("step_scale")
            if bm_step_scale is None:
                bm_step_scale = _ROBERTS_ROSENTHAL_C / len(idx)
            bm_sigma = bm_step_scale * jnp.asarray(bm["cov"])
            moves.append({
                "name": name, "prob": float(bm["prob"]), "kind": "gaussian",
                "indices": idx, "sigma": bm_sigma,
            })
        if hop_prob > 0.0:
            if "hop" in seen_names:
                raise ValueError("block_moves may not use the reserved name 'hop' when hop_prob > 0.")
            hop_fn = _default_hop_move(hop_prob, hop_indices, hop_move_fn, n_free)
            moves.append({"name": "hop", "prob": hop_prob, "kind": "hop", "hop_move_fn": hop_fn})
        total_prob = sum(m["prob"] for m in moves)
        if abs(total_prob - 1.0) > 1.0e-6:
            raise ValueError(
                f"block_moves + hop_prob (+ the implied full-vector move) "
                f"must sum to 1.0, got {total_prob}."
            )
        # Static move description + the arrays _build_block_mixture_rwm_kernel
        # would compute from it (cumsum of probs, Cholesky of each block
        # sigma), computed here eagerly exactly as it did and passed to the
        # cached program as arguments (they change between chunks).
        kernel_static = ("block", n_free, tuple(
            (m["name"], "gaussian", tuple(int(i) for i in m["indices"]))
            if m["kind"] == "gaussian" else (m["name"], "hop", _static_callable(m["hop_move_fn"]))
            for m in moves
        ))
        dyn = {
            "cumprobs": jnp.cumsum(jnp.asarray([float(m["prob"]) for m in moves])),
            "chols": tuple(
                jnp.linalg.cholesky(jnp.asarray(m["sigma"])) if m["kind"] == "gaussian" else None
                for m in moves
            ),
        }
        info_spec = {"acceptance_rate": "mean", "is_accepted": "mean"}
        for m in moves:
            info_spec[f"is_{m['name']}"] = "mean"
            info_spec[f"{m['name']}_accepted"] = "mean"
    elif hop_prob > 0.0:
        if not (0.0 < hop_prob < 1.0):
            raise ValueError(f"hop_prob must be in (0, 1), got {hop_prob}")
        if hop_move_fn is None:
            if not hop_indices:
                raise ValueError(
                    "hop_prob > 0 requires either hop_move_fn (a caller-"
                    "supplied involution) or a non-empty hop_indices "
                    "(z-space coordinate indices to reflect about 0, e.g. "
                    "the free index of a sin^2theta-parameterized angle for "
                    "an octant hop)."
                )
            bad = [i for i in hop_indices if not (0 <= i < n_free)]
            if bad:
                raise ValueError(f"hop_indices out of range [0, {n_free}): {bad}")
        kernel_static = ("mixture", hop_prob,
                         tuple(hop_indices) if hop_indices else None,
                         _static_callable(hop_move_fn))
        dyn = {"scale": _proposal_scale_matrix(sigma)}
        info_spec = _RWM_MIXTURE_INFO_SPEC
    else:
        kernel_static = ("plain",)
        dyn = {"scale": _proposal_scale_matrix(sigma)}
        info_spec = _RWM_INFO_SPEC

    warm_key, sample_key = jax.random.split(key)
    warm_keys = jax.random.split(warm_key, n_chains)
    sample_keys = jax.random.split(sample_key, n_chains)

    cfg = _RWMRunConfig(
        logdensity=_static_callable(logdensity), kernel=kernel_static, n_warmup=int(n_warmup),
        n_samples=int(n_samples), thinning=int(thinning), progress=bool(progress),
        info_spec=tuple(info_spec.items()),
    )
    positions, log_prob, infos = _rwm_run_compiled(
        dyn, chain_init, warm_keys, sample_keys, cfg=cfg)

    mean_acc = float(jnp.mean(infos["acceptance_rate"]))
    if mean_acc < 0.01 or mean_acc > 0.5:
        warnings.warn(
            f"rwm: mean acceptance rate {mean_acc:.3f} is outside the sane "
            f"band [1%, 50%] (MaCh3 sanity check) -- proposal_cov/step_scale "
            f"likely need retuning.",
            RuntimeWarning,
            stacklevel=2,
        )

    tuned = {
        "proposal_cov": jnp.broadcast_to(sigma, (n_chains,) + sigma.shape)
    }
    # (block-move names/probs are static Python metadata, not array data --
    # deliberately NOT added to `tuned`, since MCMCRun.tuned is a pytree data
    # field (jax.tree_util.register_dataclass) expected to hold array leaves;
    # per-move-type diagnostics live in `info` instead, see
    # _build_block_mixture_rwm_kernel's docstring.)
    if hop_prob > 0.0:
        tuned["hop_prob"] = jnp.asarray(hop_prob)
        if hop_indices:
            tuned["hop_indices"] = jnp.asarray(hop_indices)
        mean_hop_rate = float(jnp.mean(infos["is_hop"]))
        if mean_hop_rate > 0.0 and (
            mean_hop_rate < 0.5 * hop_prob or mean_hop_rate > 1.5 * hop_prob
        ):
            warnings.warn(
                f"rwm mixture: observed hop-proposal rate {mean_hop_rate:.4f} "
                f"deviates >50% from configured hop_prob={hop_prob:.4f} -- "
                f"unexpected for a state-independent coin flip, check "
                f"n_samples*thinning is large enough to average out.",
                RuntimeWarning,
                stacklevel=2,
            )

    return MCMCRun(
        positions=positions,
        log_prob=log_prob,
        info=infos,
        tuned=tuned,
        algorithm="rwm",
    )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_mcmc(
    logdensity: Callable[[Array], Array],
    init: Array,
    *,
    algorithm: str = "nuts",
    n_chains: int = 4,
    n_warmup: int = 1000,
    n_samples: int = 2000,
    seed: int = 0,
    init_jitter: float = 0.0,
    thinning: int = 1,
    progress: bool = False,
    proposal_cov: Array | None = None,
    step_scale: float | None = None,
    hop_prob: float = 0.0,
    hop_indices: tuple | None = None,
    hop_move_fn: Callable[[Array], Array] | None = None,
    block_moves: list | None = None,
    inverse_mass_matrix: Array | None = None,
    step_size: float | None = None,
    warmup: str = "window",
    max_num_doublings: int | None = None,
    target_acceptance_rate: float | None = None,
    shared_step_size: bool | None = None,
) -> MCMCRun:
    """Run multi-chain MCMC on ``logdensity`` (a scalar function of a
    ``(n_free,)`` z-space vector) and return an :class:`MCMCRun`.

    Parameters
    ----------
    logdensity:
        ``z -> scalar``. Built once, closed over static data (DESIGN §1's
        "hot path is one jitted function" rule) -- ``run_mcmc`` jits/vmaps
        around it but never traces into or modifies it.
    init:
        ``(n_free,)`` (replicated to ``n_chains``, optionally jittered by
        ``init_jitter``) or ``(n_chains, n_free)`` (taken as-is; ``n_chains``
        is then inferred from its shape).
    algorithm:
        ``"nuts" | "hmc" | "mala"`` (gradient-based, tuned by warmup) or
        ``"rwm"`` (gradient-free additive-step random walk with a correlated
        Gaussian proposal, MaCh3-style; requires ``proposal_cov``).
    n_chains, n_warmup, n_samples, seed:
        standard MCMC knobs. All randomness is derived deterministically from
        ``jax.random.PRNGKey(seed)`` (see the determinism test in
        ``tests/test_sampling.py``) -- same seed reproduces bit-identical
        output.
    init_jitter:
        std of iid Gaussian noise added per-chain to a 1-D ``init`` (0 =
        every chain starts at the exact same point).
    thinning:
        keep every ``thinning``-th post-warmup draw; implemented as a nested
        scan (see ``_thinned_scan``) so the full unthinned chain is never
        materialized.
    progress:
        wrap the sampling scan with a jax_tqdm progress bar if the
        ``jax_tqdm`` package is importable; silently does nothing otherwise.
    proposal_cov, step_scale:
        ``rwm`` only. ``proposal_cov`` (required) is the target covariance
        shape of the correlated Gaussian proposal (e.g. from
        ``laplace_covariance``); the actual proposal covariance is
        ``step_scale * proposal_cov``, with ``step_scale`` defaulting to the
        Roberts-Rosenthal ``2.38**2 / n_free`` optimal-scaling constant.
    hop_prob, hop_indices, hop_move_fn:
        Deterministic hop moves between modes (e.g. theta23 octants).
        ``nuts``/``hmc``/``mala``: each sampling step is followed, with
        probability ``hop_prob``, by a Metropolis hop (see ``_with_hop``;
        not applied during warmup). ``rwm``: enables the Gaussian+hop mixture kernel (see the
        module comment above ``_build_mixture_rwm_kernel``): with
        probability ``hop_prob`` (default 0, i.e. off -- plain Gaussian RWM,
        unchanged behavior), propose a hop move instead of the usual
        Gaussian step; both branches use the ordinary symmetric-proposal
        Metropolis acceptance (no Hastings correction needed -- see the
        module comment for why). Two ways to specify the hop move
        (``hop_prob > 0`` requires exactly one):
        - ``hop_indices`` (the default mechanism): a deterministic
          reflection ``z_i -> -z_i`` for every ``i in hop_indices`` (other
          coordinates untouched). For a parameter with a flat prior on a
          symmetric interval like ``(0, 1)`` (e.g. this repo's
          sin^2theta23 convention), this is exactly the physical involution
          ``x -> lo + hi - x`` (e.g. octant swap ``sin^2theta23 -> 1 -
          sin^2theta23``). Only correct if the *other* free coordinates
          barely differ between the two modes being connected -- see
          ``hop_move_fn`` below if they don't.
        - ``hop_move_fn``: a caller-supplied ``Array -> Array`` involution
          (``hop_move_fn(hop_move_fn(z)) == z`` and unit-Jacobian, e.g. a
          translation-based swap between two mode anchors located
          off-line), used verbatim in
          place of the ``hop_indices`` reflection. The caller is
          responsible for the involution/Jacobian guarantee -- it is not
          (cannot be, in general) checked here.
        ``run.info`` gains ``"is_hop"``/``"hop_accepted"`` (per-block means)
        when enabled, from which the hop-specific acceptance rate is
        ``mean(hop_accepted) / mean(is_hop)``.
    block_moves:
        ``rwm`` only. Generalizes the 2-way Gaussian+hop mixture above to a
        Metropolis-within-Gibbs mixture over an arbitrary number of NAMED,
        pi-invariant move types (see the module comment above
        ``_build_block_mixture_rwm_kernel`` -- built for the DUNE Sec-6
        joint-fit rescue: a 292-dim full-vector RWM proposal essentially
        never nets a move along a handful of slow-mixing directions, because
        each step's acceptance is dominated by the other ~286 coordinates;
        blocking -- proposing a move restricted to just those directions,
        leaving everything else EXACTLY unchanged -- fixes this without
        touching detailed balance). ``None``/``[]`` (default) is the
        existing 2-way (or plain, if ``hop_prob==0`` too) behavior,
        UNCHANGED -- this is strictly additive/opt-in.

        A list of dicts, each ``{"name": str, "prob": float,
        "indices": Sequence[int], "cov": Array, "step_scale": float |
        None}``: propose ``delta ~ N(0, step_scale * cov)`` restricted to
        the coordinates in ``indices`` (every other coordinate is returned
        UNCHANGED, not merely perturbed less), with probability ``prob``.
        ``step_scale`` defaults to the Roberts-Rosenthal
        ``2.38**2 / len(indices)`` optimal-scaling constant FOR THAT BLOCK's
        OWN dimensionality (not the whole-vector ``2.38**2 / n_free`` --
        this per-block dimension factor is the entire point of blocking:
        a 6-dim move should be scaled for 6 dims, not 292). ``"full"`` and
        ``"hop"`` are reserved move names (the whole-vector Gaussian move
        built from ``proposal_cov``/``step_scale`` above, and the hop move
        built from ``hop_prob``/``hop_indices``/``hop_move_fn`` if
        ``hop_prob > 0``, respectively) -- do not reuse them.
        ``sum(bm["prob"] for bm in block_moves) + hop_prob`` must be
        ``<= 1``; the remainder is the full-vector move's own probability
        (so e.g. ``block_moves=[osc(0.3), solar(0.1)], hop_prob=0.1`` gives
        the full move probability ``0.5`` automatically -- it is not passed
        explicitly). An entry ``{"name": str, "prob": float, "kind": "hop",
        "hop_move_fn": Array -> Array}`` is instead a further deterministic
        hop (same unit-Jacobian involution contract as ``hop_move_fn``), for
        a second pair of modes next to the main hop. Each block move, the full move, and the hop move are
        all individually pi-invariant symmetric-proposal Metropolis kernels
        (see the module comment for the detailed-balance argument), so the
        SAME ``min(1, pi(z')/pi(z))`` acceptance used everywhere else in
        this module is exactly correct for the mixture -- no extra
        correction term. ``run.info`` gains ``"is_<name>"``/
        ``"<name>_accepted"`` for EVERY move (including ``"is_full"``/
        ``"full_accepted"`` and, if ``hop_prob > 0``, ``"is_hop"``/
        ``"hop_accepted"``), so every move type's own acceptance rate is
        directly recoverable, not just an aggregate.
    inverse_mass_matrix:
        ``"nuts" | "hmc"`` only (``"mala"`` has no mass matrix; ``"rwm"``
        uses ``proposal_cov`` instead). A caller-supplied inverse mass
        matrix -- dense ``(n_free, n_free)`` or diagonal ``(n_free,)``, both
        accepted directly by blackjax's ``metrics.default_metric`` (verified
        against the installed blackjax 1.6.2) -- typically a MAP+Laplace
        estimate from :func:`laplace_covariance`. ``None`` (default) leaves
        mass-matrix estimation to whatever ``warmup`` mode requires it.
        When given with ``warmup="window"``, it seeds (rather than replaces)
        the usual online mass-matrix estimation via
        ``window_adaptation``'s ``initial_inverse_mass_matrix``.
    step_size:
        ``"nuts" | "hmc" | "mala"`` only. Initial (``warmup="window"`` /
        ``"step-size"``) or fixed, final (``warmup="none"``) step size.
        ``None`` uses blackjax's/``_mala_warmup``'s own default initial
        value.
    warmup:
        ``"window"`` (default, unchanged behavior): full
        ``blackjax.window_adaptation`` (mass matrix + step size) for
        nuts/hmc, ``_mala_warmup`` (step size only, mala has no mass matrix)
        for mala. ``"step-size"``: skip mass-matrix estimation -- for
        nuts/hmc this *requires* ``inverse_mass_matrix`` and runs only
        Nesterov dual-averaging step-size adaptation against it (the cheap
        path motivated by a measured ~1.1-1.4s/draw window-adaptation cost
        on an 8-param correlated toy; see :func:`laplace_seeded_nuts`); for
        mala it is identical to ``"window"`` (mala never estimates a mass
        matrix regardless). ``"none"``: no warmup loop at all -- requires an
        explicit ``step_size`` (``inverse_mass_matrix`` defaults to the
        identity if not given); sampling starts immediately from ``init``.
    max_num_doublings:
        ``"nuts"`` only (``None`` default = blackjax's own default of 10,
        i.e. up to ``2**10 = 1024`` leapfrog steps per transition).
        Caps the number of trajectory-doubling rounds NUTS's tree-building
        can perform before returning, bounding worst-case per-transition
        cost -- the mitigation for the "scaling cliff" documented in
        ``examples/LARGE_FIT_REPORT.md``'s "Execution notes": step-size-only
        dual averaging can drift the step size down while chasing a stiff
        direction, and once small enough, satisfying the U-turn criterion in
        flatter directions can require the full ``2**max_num_doublings``
        leapfrog steps -- each one a full gradient evaluation -- every
        transition. Applied both during warmup (threaded into
        ``window_adaptation``'s/the step-size kernel's own NUTS calls, so a
        long ``warmup="window"`` run cannot runaway either) and sampling.
        Verified against installed blackjax 1.6.2:
        ``blackjax.nuts(..., max_num_doublings=k)`` requires ``k`` to be a
        concrete (untraced) Python int -- it statically sizes a
        trajectory-storage array -- so this value must stay, and does stay,
        a plain closed-over Python int through every vmapped chain rather
        than flowing through the same per-chain ``tuned`` dict pipeline as
        ``step_size``/``inverse_mass_matrix`` (confirmed empirically: handing
        blackjax.nuts a per-chain *array* value pulled back out of
        ``window_adaptation``'s own returned ``tuned`` dict raises "Shapes
        must be 1D sequences of concrete values of integer type"). Raises
        ``ValueError`` if given for ``algorithm != "nuts"``. Diagnostics:
        ``run.info["num_trajectory_expansions"]``/``["num_integration_steps"]``
        (NUTS-only fields, per-block means) report the tree depth/leapfrog
        count actually used, so a capped run's effectiveness is directly
        checkable.
    target_acceptance_rate:
        ``"nuts" | "hmc" | "mala"`` only. Overrides the dual-averaging
        step-size target acceptance probability -- ``None`` (default) keeps
        each algorithm's own default (0.80 for nuts/hmc, matching both
        ``blackjax.window_adaptation``'s and this module's
        ``warmup="step-size"`` fast path; 0.574, the Roberts-Rosenthal-
        optimal Langevin rate, for mala). Lowering it (e.g. 0.6) is the
        other documented mitigation for the same scaling-cliff pathology as
        ``max_num_doublings``: a lower target accepts a larger step size,
        directly counteracting dual averaging's tendency to chase the
        stiffest direction down to a pathologically small step. Applied
        during whichever warmup does step-size adaptation (``"window"`` via
        ``window_adaptation``'s own ``target_acceptance_rate``; ``"step-size"``
        via the same Nesterov dual averaging used internally); has no effect
        under ``warmup="none"`` (no adaptation runs at all).

    shared_step_size:
        ``nuts``/``hmc`` with ``warmup="step-size"``:
        adapt one step size for all chains, from the acceptance rate pooled
        across chains at each warmup step, instead of one per chain. Under
        ``vmap`` every NUTS step costs the longest trajectory among the
        chains, so a chain whose short warmup settled on a small step size
        sets the cost for all of them (measured on the 8-parameter
        validation toy: per-chain step sizes 0.15-0.66, lockstep
        efficiency 0.46; pooling gave 1.8-2.3x the minimum ESS per second
        and removed the divergences). Default ``None``: on for
        ``nuts``/``hmc`` with ``warmup="step-size"`` (the Laplace-seeded
        recipe), off otherwise. ``False`` restores one step size per chain
        (the behaviour before v0.3.0); ``True`` with any other algorithm or
        warmup raises.

    Compiled-program cache (``algorithm="rwm"``): burn-in + sampling run as one
    jitted program, cached on the log-density callable and the static settings
    (kernel structure, chain/step counts, thinning, hop settings); proposal
    covariance, step scales, move probabilities, start point and seed are
    arguments, so repeated calls (chunks, tune sweeps) reuse it without a
    re-trace. The cache keeps the log-density -- and every array it closes
    over -- alive until ``jax.clear_caches()``: call that when a process
    switches to another model, or before a call with different static
    settings if device memory is tight.
    """
    algorithm = algorithm.lower()
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"algorithm must be one of {_ALGORITHMS}, got {algorithm!r}")

    key = jax.random.PRNGKey(seed)
    key, jitter_key = jax.random.split(key)
    chain_init, n_chains, n_free = _prepare_chain_init(
        init, n_chains, init_jitter, jitter_key
    )

    if algorithm == "rwm":
        if (inverse_mass_matrix is not None or step_size is not None or warmup != "window"
                or max_num_doublings is not None or target_acceptance_rate is not None
                or shared_step_size):  # None is falsy: only an explicit True is rejected
            raise ValueError(
                "inverse_mass_matrix/step_size/warmup/max_num_doublings/"
                "target_acceptance_rate/shared_step_size are gradient-based-algorithm options "
                "(nuts/hmc/mala); algorithm='rwm' uses proposal_cov/step_scale "
                "instead."
            )
        return _run_rwm(
            logdensity,
            chain_init,
            n_chains=n_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            key=key,
            thinning=thinning,
            progress=progress,
            proposal_cov=proposal_cov,
            step_scale=step_scale,
            n_free=n_free,
            hop_prob=hop_prob,
            hop_indices=hop_indices,
            hop_move_fn=hop_move_fn,
            block_moves=block_moves,
        )
    elif block_moves:
        raise ValueError(
            "block_moves is an algorithm='rwm'-only option (the blocked "
            f"Metropolis-within-Gibbs kernel); got algorithm={algorithm!r}."
        )

    hop_fn = None
    if hop_prob != 0.0:
        hop_fn = _default_hop_move(hop_prob, hop_indices, hop_move_fn, n_free)
    elif hop_indices is not None or hop_move_fn is not None:
        raise ValueError(
            "hop_indices/hop_move_fn have no effect without hop_prob > 0 "
            f"(got hop_prob={hop_prob}, algorithm={algorithm!r})."
        )

    return _run_gradient_based(
        logdensity,
        chain_init,
        algorithm=algorithm,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        key=key,
        thinning=thinning,
        progress=progress,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=step_size,
        warmup=warmup,
        max_num_doublings=max_num_doublings,
        target_acceptance_rate=target_acceptance_rate,
        hop_prob=hop_prob,
        hop_fn=hop_fn,
        shared_step_size=shared_step_size,
    )


# ---------------------------------------------------------------------------
# MAP pre-fit + Laplace covariance
# ---------------------------------------------------------------------------


def map_fit(
    logdensity: Callable[[Array], Array],
    init: Array,
    *,
    n_steps: int = 500,
    learning_rate: float | None = None,
    method: str = "lbfgs",
) -> tuple[Array, Array]:
    """Maximize ``logdensity`` from ``init`` via optax; returns ``(z_map,
    logdensity(z_map))``.

    Used to seed MCMC chains and, via :func:`laplace_covariance`, to build a
    Laplace proposal covariance for ``rwm``. ``method="lbfgs"`` uses
    ``optax.lbfgs`` (installed optax 0.2.8 has it) with the canonical
    ``value_and_grad_from_state`` + zoom-linesearch pattern from its own
    docstring; ``method="adam"`` is a plain fixed-step fallback for optax
    versions without L-BFGS. Both loops run for exactly ``n_steps`` (a
    ``jax.lax.scan``, no early stopping) -- ``n_steps=500`` comfortably
    converges the 3-d Gaussian smoke test to 1e-3 in the mean.
    """
    import optax

    init = jnp.asarray(init)
    neg_logdensity = lambda z: -logdensity(z)  # noqa: E731

    if method == "lbfgs" and hasattr(optax, "lbfgs"):
        solver = optax.lbfgs() if learning_rate is None else optax.lbfgs(
            learning_rate=learning_rate
        )
        value_and_grad = optax.value_and_grad_from_state(neg_logdensity)

        def body(carry, _):
            params, opt_state = carry
            value, grad = value_and_grad(params, state=opt_state)
            updates, opt_state = solver.update(
                grad, opt_state, params, value=value, grad=grad,
                value_fn=neg_logdensity,
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

    else:
        lr = 1e-2 if learning_rate is None else learning_rate
        solver = optax.adam(lr)

        def body(carry, _):
            params, opt_state = carry
            grad = jax.grad(neg_logdensity)(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

    opt_state = solver.init(init)
    (z_map, _), _ = jax.lax.scan(body, (init, opt_state), None, length=n_steps)
    return z_map, logdensity(z_map)


def conditional_modes(logdensity: Callable[[Array], Array], z_start: Array, index: int,
                      values, *, n_steps: int = 300) -> tuple[Array, Array]:
    """Maximise ``logdensity`` over every coordinate except ``index``, with
    ``z[index]`` held at each entry of ``values`` (a profile scan).

    Each fit starts from the finished fit at the nearest value already done
    (the first from ``z_start``), so the scan follows one ridge. Returns
    ``(modes, logdensities)``: ``modes`` is ``(len(values), n_free)`` (full
    vectors, ``modes[k, index] == values[k]``), in the order of ``values``.
    Meant for :func:`octant_mirror_hop`; L-BFGS via optax as in
    :func:`map_fit`, one compiled solver reused for every value.
    """
    import numpy as np
    import optax

    z_start = jnp.asarray(z_start)
    values = np.asarray(values, dtype=float)
    n_free = z_start.shape[0]
    others = jnp.asarray([i for i in range(n_free) if i != index])

    def full(zo, v):
        return jnp.zeros(n_free, z_start.dtype).at[others].set(zo).at[index].set(v)

    def neg(zo, v):
        return -logdensity(full(zo, v))

    solver = optax.lbfgs()

    @jax.jit
    def solve(zo0, v):
        f = lambda zo: neg(zo, v)  # noqa: E731
        value_and_grad = optax.value_and_grad_from_state(f)

        def body(carry, _):
            zo, st = carry
            val, g = value_and_grad(zo, state=st)
            upd, st = solver.update(g, st, zo, value=val, grad=g, value_fn=f)
            return (optax.apply_updates(zo, upd), st), None

        (zo, _), _ = jax.lax.scan(body, (zo0, solver.init(zo0)), None, length=n_steps)
        return zo, -f(zo)

    order = np.argsort(np.abs(values - float(z_start[index])))
    modes = [None] * len(values)
    lds = [None] * len(values)
    done = []
    for k in order:
        j = min(done, key=lambda j: abs(values[j] - values[k])) if done else None
        zo0 = z_start[others] if j is None else modes[j][others]
        zo, ld = solve(zo0, values[k])
        modes[k], lds[k] = full(zo, values[k]), ld
        done.append(k)
    return jnp.stack(modes), jnp.stack(lds)


def octant_mirror_hop(index: int, grid, modes) -> Callable[[Array], Array]:
    """Octant hop that follows the posterior ridge: an involution
    ``z -> (-z_i, z_o - g(z_i) + g(-z_i))`` for the mixing-angle coordinate
    ``i = index`` and all other coordinates ``z_o``.

    ``g`` interpolates (piecewise-linearly, clamped outside ``grid``) the
    conditional modes of the other coordinates as a function of ``z_i``,
    e.g. ``modes`` from :func:`conditional_modes` on a ``grid`` that covers
    both octants. With a bounded angle on a symmetric interval (the
    ``(0, pi/2)`` or ``(0, 1)`` bijector), ``z_i -> -z_i`` is the exact
    octant mirror, and the shift moves the other coordinates from the ridge
    point of this ``z_i`` to the ridge point of the mirrored one, so a point
    on the ridge lands on the ridge. The default reflection
    (``hop_indices=(index,)``) is the special case ``g = const``.

    Valid for any ``g`` (applying it twice gives back ``z``; the Jacobian is
    block-triangular with determinant -1), so a poor ``grid`` only lowers
    the hop acceptance and cannot bias the posterior. Pass the result as
    ``run_mcmc(..., hop_prob=p, hop_move_fn=octant_mirror_hop(...))``.
    """
    grid = jnp.asarray(grid, dtype=float)
    modes = jnp.asarray(modes, dtype=float)
    order = jnp.argsort(grid)
    grid, modes = grid[order], modes[order]
    modes = modes.at[:, index].set(0.0)

    def g(t):
        return jax.vmap(lambda col: jnp.interp(t, grid, col), in_axes=1)(modes)

    def hop(z):
        zi = z[index]
        shifted = z - g(zi) + g(-zi)
        return shifted.at[index].set(-zi)

    return hop


def laplace_covariance(logdensity: Callable[[Array], Array], z_map: Array) -> Array:
    """Laplace approximation to the posterior covariance at ``z_map``:
    ``inv(-hessian(logdensity)(z_map))``, symmetrized.

    Falls back to the Moore-Penrose pseudo-inverse (with a ``RuntimeWarning``)
    if ``-hessian`` is not positive definite at ``z_map`` -- can happen at a
    saddle point or a MAP that hasn't fully converged. This function runs
    eagerly (not jitted): the PD check and the warning are ordinary Python
    control flow on concrete values, which is exactly the kind of
    "conditional on traced values" DESIGN §1 forbids *inside* the hot
    ``logdensity`` path but is fine for a once-per-fit diagnostic utility
    like this one.

    Hessian recipe: reverse-over-reverse (``jax.jacrev(jax.grad(logdensity))``),
    **not** ``jax.hessian`` (which is forward-over-reverse,
    ``jax.jacfwd(jax.jacrev(f))``). Any ``logdensity`` built from
    ``bananafit/splines.py``'s ``SplineTable`` (both the dense and factorized
    representations) routes through that module's ``_curve_gather``/
    ``_cell_event_product`` ``jax.custom_vjp`` primitives -- deliberately
    reverse-mode-only (their custom backward pass is
    the fix for a GPU scatter-add pathology, and ``jax.custom_vjp`` never
    defines a JVP rule). ``jax.hessian``'s outer ``jacfwd`` needs a JVP
    through the whole graph, so it raises ``TypeError: can't apply
    forward-mode autodiff (jvp) to a custom_vjp function`` the moment
    ``logdensity`` touches a spline table -- confirmed directly (this was
    the root cause of 3 ``tests/test_endtoend.py`` failures).
    ``jax.jacrev(jax.grad(f))`` instead vmaps a VJP pullback (reverse-mode)
    over cotangent basis vectors for *both* differentiation layers -- no
    ``jvp`` anywhere in the graph, so it composes cleanly with a
    reverse-mode-only ``custom_vjp`` (verified directly here, on tiny
    dense- and factorized-``SplineTable``-backed posteriors, in
    ``tests/test_sampling.py::test_laplace_covariance_matches_finite_diff_hessian_through_splines``,
    and via the 3 previously-failing ``tests/test_endtoend.py`` cases).
    Mathematically identical to ``jax.hessian`` wherever both are
    defined -- this is purely an AD-strategy substitution, not an
    approximation -- and no more expensive asymptotically: both are
    ``O(n_free)`` VJP/JVP evaluations of a already-scalar-output function.
    """
    z_map = jnp.asarray(z_map)
    hess = jax.jacrev(jax.grad(logdensity))(z_map)
    hess = 0.5 * (hess + hess.T)
    neg_hess = -hess

    eigvals = jnp.linalg.eigvalsh(neg_hess)
    min_eig = float(jnp.min(eigvals))
    if min_eig <= 0.0:
        warnings.warn(
            f"laplace_covariance: -hessian(logdensity) is not positive "
            f"definite at z_map (min eigenvalue {min_eig:.3e}); falling back "
            f"to the pseudo-inverse. The Laplace approximation may be a poor "
            f"description of the posterior here.",
            RuntimeWarning,
            stacklevel=2,
        )
        cov = jnp.linalg.pinv(neg_hess)
    else:
        cov = jnp.linalg.inv(neg_hess)
    return 0.5 * (cov + cov.T)


# ---------------------------------------------------------------------------
# convenience: MAP + Laplace-seeded MCMC
# ---------------------------------------------------------------------------


def laplace_seeded_nuts(
    logdensity: Callable[[Array], Array],
    z0: Array | None = None,
    *,
    n_chains: int,
    n_samples: int,
    n_warmup: int = 200,
    seed: int = 0,
    map_kwargs: dict | None = None,
    algorithm: str = "nuts",
    init_jitter: float = 0.0,
    thinning: int = 1,
    progress: bool = False,
    max_num_doublings: int | None = None,
    target_acceptance_rate: float | None = None,
) -> tuple[MCMCRun, dict]:
    """MAP fit -> Laplace covariance -> Laplace-seeded MCMC, in one call.

    The cheap-warmup recipe this module's docstrings/DESIGN motivation call
    for: :func:`map_fit` (L-BFGS to the mode) then :func:`laplace_covariance`
    (``inv(-hessian)`` at the mode) give an inverse-mass-matrix/proposal-
    covariance estimate essentially for free relative to a full MCMC warmup;
    this function chains them straight into :func:`run_mcmc`.

    - ``algorithm in {"nuts", "hmc"}`` (default ``"nuts"``): runs
      ``run_mcmc(..., inverse_mass_matrix=cov, warmup="step-size")`` --
      the Laplace covariance stands in for blackjax's online mass-matrix
      estimate, so warmup only has to dual-average the step size (see
      ``warmup="step-size"``'s docstring in :func:`run_mcmc` for the
      measured cost motivation).
    - ``algorithm == "rwm"``: runs
      ``run_mcmc(..., algorithm="rwm", proposal_cov=cov)`` -- the same
      Laplace covariance, now used the way ``proposal_cov`` was already
      designed to be used (MaCh3-style correlated-Gaussian RWM proposal;
      DESIGN.md's own suggested source for it), started from ``z_map``
      instead of an arbitrary ``init``.

    Parameters
    ----------
    logdensity, z0:
        as in :func:`map_fit`/:func:`run_mcmc`; ``z0`` (required) is the
        MAP optimizer's starting point, *not* the MCMC init directly -- the
        found ``z_map`` is what chains actually start from.
    n_chains, n_samples, n_warmup, seed, init_jitter, thinning, progress:
        forwarded to :func:`run_mcmc` (``n_warmup`` here is the *cheap*
        step-size-only/RWM-burn-in warmup length, not a window-adaptation
        length -- it can typically be much shorter than a window-adapted
        run's ``n_warmup`` for the same quality, since there is no
        mass-matrix estimation left to do).
    map_kwargs:
        extra keyword arguments forwarded to :func:`map_fit` (e.g.
        ``n_steps``, ``method``).
    algorithm:
        ``"nuts"`` (default), ``"hmc"``, or ``"rwm"``.
    max_num_doublings, target_acceptance_rate:
        forwarded to :func:`run_mcmc` unchanged (``"nuts"``/``"hmc"``/mixed
        only -- see :func:`run_mcmc`'s docstring; both are the caller-facing
        knobs for the ``warmup="step-size"`` "scaling cliff" this recipe is
        otherwise most exposed to, since the fixed Laplace mass matrix alone
        does not bound worst-case tree-doubling cost -- see
        ``examples/LARGE_FIT_REPORT.md``'s "Execution notes"). Not valid with
        ``algorithm="rwm"`` (``run_mcmc`` raises ``ValueError`` in that case,
        same as passing them alongside ``algorithm="rwm"`` directly).

    Returns
    -------
    ``(run, info)`` where ``run`` is the :class:`MCMCRun` and ``info`` is
    ``{"z_map", "logdensity_at_map", "cov", "map_time_s",
    "laplace_time_s"}`` (the last two are wall-clock seconds for the MAP
    fit and the Hessian/inverse respectively -- the "warmup/setup time
    measured separately from sampling" this function's docstring promises).
    """
    if z0 is None:
        raise ValueError("laplace_seeded_nuts requires z0 (the MAP optimizer's start point).")
    if algorithm not in ("nuts", "hmc", "rwm"):
        raise ValueError(f"algorithm must be one of 'nuts', 'hmc', 'rwm', got {algorithm!r}")
    map_kwargs = {} if map_kwargs is None else map_kwargs

    t0 = time.perf_counter()
    z_map, ld_map = map_fit(logdensity, z0, **map_kwargs)
    z_map.block_until_ready()
    map_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cov = laplace_covariance(logdensity, z_map)
    cov.block_until_ready()
    laplace_time_s = time.perf_counter() - t0

    common = dict(
        n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup, seed=seed,
        init_jitter=init_jitter, thinning=thinning, progress=progress,
    )
    if algorithm == "rwm":
        if max_num_doublings is not None or target_acceptance_rate is not None:
            raise ValueError(
                "max_num_doublings/target_acceptance_rate are gradient-based-"
                "algorithm options (nuts/hmc); algorithm='rwm' does not use them."
            )
        run = run_mcmc(logdensity, z_map, algorithm="rwm", proposal_cov=cov, **common)
    else:
        run = run_mcmc(
            logdensity, z_map, algorithm=algorithm, inverse_mass_matrix=cov,
            warmup="step-size", max_num_doublings=max_num_doublings,
            target_acceptance_rate=target_acceptance_rate, **common,
        )

    info = {
        "z_map": z_map,
        "logdensity_at_map": ld_map,
        "cov": cov,
        "map_time_s": map_time_s,
        "laplace_time_s": laplace_time_s,
    }
    return run, info
