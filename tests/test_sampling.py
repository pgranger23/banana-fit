"""Tests for bananafit.sampling / bananafit.diagnostics (DESIGN.md §10).

Keep this file's own footprint minimal: import ``bananafit`` first (enables
float64 globally, per DESIGN §1 -- sampling.py itself never mutates jax
config), then only ``bananafit.sampling`` / ``bananafit.diagnostics``. Targets
are small analytic Gaussians so full-file runtime stays well under the ~3 min
budget.
"""

import warnings
from types import SimpleNamespace

import bananafit  # noqa: F401  -- enables jax_enable_x64 before anything else runs

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bananafit.sampling import (
    run_mcmc, map_fit, laplace_covariance, laplace_seeded_nuts, MCMCRun,
)
from bananafit import diagnostics


# ---------------------------------------------------------------------------
# shared target: correlated 3-d Gaussian, correlation 0.8
# ---------------------------------------------------------------------------

D = 3
TRUE_MEAN = jnp.array([1.0, -2.0, 0.5])
_RHO = 0.8
_SIGMA = jnp.array([1.0, 1.5, 0.8])
_CORR = jnp.full((D, D), _RHO) + jnp.eye(D) * (1.0 - _RHO)
TRUE_COV = _SIGMA[:, None] * _CORR * _SIGMA[None, :]
_PREC = jnp.linalg.inv(TRUE_COV)


def logdensity(z):
    dz = z - TRUE_MEAN
    return -0.5 * jnp.dot(dz, _PREC @ dz)


INIT = jnp.zeros(D)


# ---------------------------------------------------------------------------
# NUTS: statistical recovery
# ---------------------------------------------------------------------------


def test_nuts_recovers_gaussian():
    run = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=500,
        n_samples=1000, seed=0,
    )
    assert isinstance(run, MCMCRun)
    assert run.algorithm == "nuts"
    assert run.positions.shape == (2, 1000, D)
    assert run.log_prob.shape == (2, 1000)
    assert np.all(np.isfinite(np.asarray(run.positions)))
    assert np.all(np.isfinite(np.asarray(run.log_prob)))

    r = diagnostics.rhat(run)
    e = diagnostics.ess(run)
    assert all(v < 1.05 for v in r.values()), r
    assert all(v > 100 for v in e.values()), e

    pos = np.asarray(run.positions).reshape(-1, D)
    mean = pos.mean(axis=0)
    min_ess = min(e.values())
    tol = 3.0 * np.sqrt(np.diag(np.asarray(TRUE_COV))) / np.sqrt(min_ess)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < tol), (mean, tol)

    cov = np.cov(pos.T)
    rel_err = np.abs(cov - np.asarray(TRUE_COV)) / np.abs(np.asarray(TRUE_COV))
    assert np.all(rel_err < 0.15), rel_err


def test_nuts_deterministic():
    run1 = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=100,
        n_samples=100, seed=42,
    )
    run2 = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=100,
        n_samples=100, seed=42,
    )
    np.testing.assert_array_equal(
        np.asarray(run1.positions), np.asarray(run2.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(run1.log_prob), np.asarray(run2.log_prob)
    )


def test_nuts_thinning_and_2d_init():
    chain_init = jnp.stack([INIT, INIT + 0.1])
    run = run_mcmc(
        logdensity, chain_init, algorithm="nuts", n_warmup=50, n_samples=40,
        thinning=3, seed=1,
    )
    assert run.positions.shape == (2, 40, D)


# ---------------------------------------------------------------------------
# RWM: MaCh3-style adaptive-covariance random walk
# ---------------------------------------------------------------------------


def test_rwm_recovers_gaussian_with_true_covariance():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        run = run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=2, n_warmup=1000,
            n_samples=3000, seed=0, proposal_cov=TRUE_COV,
        )
    assert run.algorithm == "rwm"
    acc = float(np.mean(np.asarray(run.info["acceptance_rate"])))
    assert 0.15 < acc < 0.6, acc

    pos = np.asarray(run.positions).reshape(-1, D)
    mean = pos.mean(axis=0)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < 0.3), mean


@pytest.mark.parametrize("path", ["plain", "mixture", "block_moves", "diag"])
def test_rwm_realised_proposal_covariance_is_scale_times_cov(path):
    """Regression: blackjax's random-walk ``sigma`` is a scale matrix, so the
    plain and 2-way kernels once realised ``(s*C)(s*C)^T = s^2 C^2`` instead of
    the documented ``s*C``. On a flat target every
    proposal is accepted, so successive differences are the proposal steps."""
    cov = jnp.array([[4.0, 0.6], [0.6, 0.25]])
    kwargs = dict(algorithm="rwm", n_chains=1, n_warmup=0, n_samples=40000,
                  seed=0, step_scale=1.0, proposal_cov=cov)
    if path == "mixture":
        kwargs.update(hop_prob=0.01, hop_move_fn=lambda z: z)
    elif path == "block_moves":
        kwargs.update(block_moves=[{"name": "b", "prob": 0.01, "indices": (0,),
                                     "cov": jnp.eye(1), "step_scale": 1e-12}])
    elif path == "diag":
        kwargs.update(proposal_cov=jnp.diag(cov))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = run_mcmc(lambda z: 0.0 * jnp.sum(z), jnp.zeros(2), **kwargs)
    steps = np.diff(np.asarray(run.positions)[0], axis=0)
    steps = steps[np.abs(steps).max(axis=1) > 1e-5]  # drop identity hops / tiny block moves
    expected = np.asarray(cov) if path != "diag" else np.diag(np.diag(np.asarray(cov)))
    np.testing.assert_allclose(np.cov(steps.T), expected, rtol=0.05, atol=0.03)


def test_rwm_requires_proposal_cov():
    with pytest.raises(ValueError):
        run_mcmc(logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10)


def test_rwm_acceptance_sanity_warning():
    # A wildly oversized proposal drives acceptance far below 1%.
    with pytest.warns(RuntimeWarning, match="acceptance rate"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=1, n_warmup=50,
            n_samples=200, seed=0, proposal_cov=TRUE_COV, step_scale=1e5,
        )


# ---------------------------------------------------------------------------
# RWM mixture kernel: octant-hop / mode-jumping proposal
# ---------------------------------------------------------------------------
#
# Validates the ``hop_prob``/``hop_indices`` mixture proposal (see the
# module comment above ``bananafit.sampling._build_mixture_rwm_kernel``) on a
# cheap 1-D analytic bimodal target, symmetric about ``z=0`` with UNEQUAL
# mode weights -- scaled-down stand-in for the DUNE stats-only FD
# theta23-octant degeneracy this kernel was built for (a flat prior on
# sin^2theta23 in ``(0,1)`` makes ``z -> -z`` exactly the physical octant
# reflection ``x -> 1-x``, see that module's docstring for the numerical
# verification). Unequal weights (0.3 / 0.7) test that the mixture kernel
# recovers the *correct relative mass*, not merely "visits both modes".

_HOP_MU = 5.0
_HOP_SIGMA = 0.5
_HOP_W1 = 0.3  # mass of the z<0 mode; z>0 mode carries 1-_HOP_W1 = 0.7


def _bimodal_logdensity(mu=_HOP_MU, sigma=_HOP_SIGMA, w1=_HOP_W1):
    """``w1*N(-mu,sigma) + (1-w1)*N(+mu,sigma)`` as a scalar-z logdensity.
    ``mu/sigma = 10`` here -- modes are separated by 10 within-mode standard
    deviations, far enough that a locally-tuned Gaussian RWM proposal never
    crosses in any practically reachable chain length (see
    ``test_rwm_plain_cannot_cross_bimodal_valley`` below), yet cheap enough
    for a unit test (a single scalar coordinate, no mango/model machinery).
    """
    log_w1 = jnp.log(w1)
    log_w2 = jnp.log1p(-w1)

    def logdensity(z):
        z = jnp.asarray(z).reshape(())
        lp1 = log_w1 - 0.5 * ((z + mu) / sigma) ** 2
        lp2 = log_w2 - 0.5 * ((z - mu) / sigma) ** 2
        return jax.scipy.special.logsumexp(jnp.stack([lp1, lp2]))

    return logdensity


def test_rwm_plain_cannot_cross_bimodal_valley():
    """Sanity check on the toy itself: plain RWM (hop_prob=0), tuned to the
    within-mode scale and started in the low-mass mode, stays there --
    reproducing this task's premise that a local RWM proposal cannot
    populate the second mode without help."""
    target = _bimodal_logdensity()
    tight_cov = jnp.array([[_HOP_SIGMA**2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = run_mcmc(
            target, jnp.array([-_HOP_MU]), algorithm="rwm", n_chains=4,
            n_warmup=2000, n_samples=20000, seed=0, proposal_cov=tight_cov,
        )
    frac_upper = float(jnp.mean(run.positions[..., 0] > 0.0))
    assert frac_upper < 0.01, frac_upper


def test_rwm_mixture_hop_recovers_bimodal_toy():
    """The octant-hop mixture kernel populates both modes with the correct
    analytic mass split (0.3 / 0.7), regardless of which mode the chain
    starts in, with sane overall and hop-specific acceptance rates."""
    target = _bimodal_logdensity()
    tight_cov = jnp.array([[_HOP_SIGMA**2]])
    true_frac_upper = 1.0 - _HOP_W1

    for start in (-_HOP_MU, _HOP_MU):
        run = run_mcmc(
            target, jnp.array([start]), algorithm="rwm", n_chains=4,
            n_warmup=2000, n_samples=20000, seed=1, proposal_cov=tight_cov,
            hop_prob=0.1, hop_indices=(0,),
        )
        assert run.algorithm == "rwm"
        z = np.asarray(run.positions[..., 0])
        frac_upper = float(np.mean(z > 0.0))
        assert abs(frac_upper - true_frac_upper) < 0.05, (start, frac_upper)

        acc = float(jnp.mean(run.info["acceptance_rate"]))
        assert 0.01 < acc < 0.6, acc

        hop_rate = float(jnp.mean(run.info["is_hop"]))
        assert 0.05 < hop_rate < 0.2, hop_rate  # ~ configured hop_prob=0.1
        hop_acc = float(jnp.mean(run.info["hop_accepted"])) / hop_rate
        assert 0.05 < hop_acc <= 1.0, hop_acc


def test_rwm_mixture_requires_hop_indices():
    with pytest.raises(ValueError, match="hop_indices"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=TRUE_COV, hop_prob=0.1,
        )


def test_rwm_mixture_hop_prob_out_of_range_rejected():
    with pytest.raises(ValueError, match="hop_prob"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=TRUE_COV, hop_prob=1.5, hop_indices=(0,),
        )


def test_rwm_mixture_hop_prob_rejected_for_gradient_algorithms():
    with pytest.raises(ValueError, match="hop_prob"):
        run_mcmc(
            logdensity, INIT, algorithm="nuts", n_warmup=10, n_samples=10,
            hop_prob=0.1, hop_indices=(0,),
        )


def test_rwm_mixture_default_hop_prob_matches_plain_rwm():
    """hop_prob=0.0 (the default) is bit-identical to not passing it at all
    -- the mixture kernel's addition is strictly opt-in."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_a = run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=2, n_warmup=20,
            n_samples=50, seed=0, proposal_cov=TRUE_COV,
        )
        run_b = run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=2, n_warmup=20,
            n_samples=50, seed=0, proposal_cov=TRUE_COV, hop_prob=0.0,
        )
    np.testing.assert_array_equal(
        np.asarray(run_a.positions), np.asarray(run_b.positions)
    )


# ---------------------------------------------------------------------------
# RWM mixture kernel: hop_move_fn (arbitrary involution) -- translate-swap
# ---------------------------------------------------------------------------
#
# Found on a DUNE-like stats-only FD model: a plain per-
# coordinate reflection (hop_indices) only works when every *other*
# coordinate is (nearly) unchanged between the two modes. This 2-D toy has
# mode B offset from mode A in a SECOND coordinate too (not just mirrored in
# the first) -- exactly the shape of the real failure (sin2theta13 also
# shifts between theta23 octants) -- to validate the ``hop_move_fn`` escape
# hatch this repo added in response: a translation-based swap keyed off two
# mode anchors, still a valid unit-Jacobian involution (see the module
# comment above ``bananafit.sampling._build_mixture_rwm_kernel``).

_SWAP_MU_A = jnp.array([5.0, 3.0])
_SWAP_MU_B = jnp.array([-5.0, -1.0])  # offset in BOTH coordinates from -mu_A
_SWAP_SIGMA = 0.5
_SWAP_W1 = 0.3  # mass near mode A


def _offset_bimodal_logdensity(mu_a=_SWAP_MU_A, mu_b=_SWAP_MU_B,
                                sigma=_SWAP_SIGMA, w1=_SWAP_W1):
    log_w1 = jnp.log(w1)
    log_w2 = jnp.log1p(-w1)

    def logdensity(z):
        z = jnp.asarray(z)
        lp1 = log_w1 - 0.5 * jnp.sum((z - mu_a) ** 2) / sigma**2
        lp2 = log_w2 - 0.5 * jnp.sum((z - mu_b) ** 2) / sigma**2
        return jax.scipy.special.logsumexp(jnp.stack([lp1, lp2]))

    return logdensity


def _translate_swap(z, mu_a=_SWAP_MU_A, mu_b=_SWAP_MU_B, side_index=0):
    """``z -> mu_b + (z - mu_a)`` if ``z`` is on mode a's side (coordinate
    ``side_index`` positive), else the reverse -- a pure translation, hence
    unit Jacobian, and an involution as long as the two modes are separated
    enough (relative to their own width) along ``side_index`` that the
    post-swap point reliably re-classifies to the other side (true here:
    mode separation is 20 sigma along coordinate 0)."""
    z = jnp.asarray(z)
    is_a = z[side_index] > 0.0
    return jnp.where(is_a, mu_b + (z - mu_a), mu_a + (z - mu_b))


def test_rwm_reflect_hop_fails_on_offset_bimodal_toy():
    """Sanity check reproducing the real-model failure: a plain reflection
    hop, started in the low-mass mode, cannot find the second mode when it
    is offset in a coordinate the reflection does not touch -- the observed
    mode-A occupancy stays far from the true 0.3, and the hop-specific
    acceptance rate is near zero (mirrors the ~0/10000 measured on the real
    FD model at production settings)."""
    target = _offset_bimodal_logdensity()
    cov = jnp.eye(2) * _SWAP_SIGMA**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = run_mcmc(
            target, _SWAP_MU_A, algorithm="rwm", n_chains=4, n_warmup=2000,
            n_samples=20000, seed=1, proposal_cov=cov,
            hop_prob=0.1, hop_indices=(0,),
        )
    frac_a = float(np.mean(np.asarray(run.positions[..., 0]) > 0.0))
    assert abs(frac_a - _SWAP_W1) > 0.2, frac_a  # far from the true 0.3
    hop_rate = float(jnp.mean(run.info["is_hop"]))
    hop_acc = float(jnp.mean(run.info["hop_accepted"])) / hop_rate
    assert hop_acc < 0.01, hop_acc


def test_rwm_hop_move_fn_translate_swap_recovers_offset_bimodal_toy():
    """The fix: hop_move_fn's translate-swap correctly recovers the analytic
    mass split (0.3/0.7) with sane, nonzero hop acceptance, from either
    starting mode."""
    target = _offset_bimodal_logdensity()
    cov = jnp.eye(2) * _SWAP_SIGMA**2
    true_frac_a = _SWAP_W1

    for start in (_SWAP_MU_A, _SWAP_MU_B):
        run = run_mcmc(
            target, start, algorithm="rwm", n_chains=4, n_warmup=2000,
            n_samples=20000, seed=1, proposal_cov=cov,
            hop_prob=0.1, hop_move_fn=_translate_swap,
        )
        assert run.algorithm == "rwm"
        frac_a = float(np.mean(np.asarray(run.positions[..., 0]) > 0.0))
        assert abs(frac_a - true_frac_a) < 0.05, (start, frac_a)

        acc = float(jnp.mean(run.info["acceptance_rate"]))
        assert 0.01 < acc < 0.9, acc
        hop_rate = float(jnp.mean(run.info["is_hop"]))
        assert 0.05 < hop_rate < 0.2, hop_rate
        hop_acc = float(jnp.mean(run.info["hop_accepted"])) / hop_rate
        assert 0.1 < hop_acc <= 1.0, hop_acc


def test_rwm_hop_move_fn_requires_positive_hop_prob_or_indices():
    with pytest.raises(ValueError, match="hop_move_fn|hop_indices"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=TRUE_COV, hop_prob=0.1,
        )


def test_rwm_hop_move_fn_rejected_for_gradient_algorithms():
    with pytest.raises(ValueError, match="hop_move_fn"):
        run_mcmc(
            logdensity, INIT, algorithm="nuts", n_warmup=10, n_samples=10,
            hop_move_fn=lambda z: -z,
        )


# ---------------------------------------------------------------------------
# RWM blocked mixture: Metropolis-within-Gibbs over named move types
# ---------------------------------------------------------------------------
#
# Motivated by a DUNE-like joint ND+FD fit: the 292-dim full-vector RWM
# proposal above mixes theta23 (helped by the hop move) but essentially never
# nets a move for deltacp/dm21/dm31/theta12, while
# these SAME directions mix fine in a low-dimensional
# osc-only proposal once the other ~286 coordinates are held fixed. The
# ``block_moves`` argument (see the module comment above
# ``bananafit.sampling._build_block_mixture_rwm_kernel``) generalizes the
# 2-way Gaussian+hop mixture to N named move types, each restricted to its
# own coordinate subset, each individually pi-invariant. Three properties
# are tested here per the task's own list:
#   (a) each move type ALONE (prob=1 for that move) preserves the correct
#       analytic marginal for its own block, on a target where the blocks
#       are independent (so the OTHER block's fixed value doesn't bias the
#       moving block's conditional == marginal distribution);
#   (c) (tested together with (a), same run) a block move's mask touches
#       ONLY its own coordinates -- the other block's positions are EXACTLY
#       (bit-identically) unchanged for the whole chain, not merely
#       "usually" unchanged -- PLUS a direct, accept/reject-independent
#       check on the raw proposal step (logdensity held constant so every
#       proposal is accepted, isolating the proposal mechanism itself from
#       Metropolis noise);
#   (b) the FULL mixture (full-vector + two Gaussian blocks + a hop move,
#       mirroring the real Sec-6 design's four move types) recovers the
#       correct analytic marginals -- mean/cov of a correlated block AND the
#       mode-weight split of a bimodal ("octant") coordinate -- confirming
#       the mixture as a whole still targets the true joint density
#       regardless of how the target's blocks correlate with each other.

_BLK_MEAN_A = jnp.array([1.0, -1.0])
_BLK_COV_A = jnp.array([[1.0, 0.5], [0.5, 1.0]])
_BLK_MEAN_B = jnp.array([2.0, 0.5])
_BLK_COV_B = jnp.array([[0.7, -0.2], [-0.2, 0.9]])
_BLK_PREC_A = jnp.linalg.inv(_BLK_COV_A)
_BLK_PREC_B = jnp.linalg.inv(_BLK_COV_B)


def _independent_two_block_logdensity(z):
    """4-D target, blocks A=(0,1) and B=(2,3), INDEPENDENT of each other
    (block-diagonal covariance) -- so holding one block fixed leaves the
    other's CONDITIONAL distribution identical to its own MARGINAL,
    exactly what a single block-move-alone run samples."""
    za, zb = z[0:2] - _BLK_MEAN_A, z[2:4] - _BLK_MEAN_B
    return -0.5 * jnp.dot(za, _BLK_PREC_A @ za) - 0.5 * jnp.dot(zb, _BLK_PREC_B @ zb)


@pytest.mark.parametrize(
    "name,indices,mean,cov,other_slice",
    [
        ("blockA", (0, 1), _BLK_MEAN_A, _BLK_COV_A, slice(2, 4)),
        ("blockB", (2, 3), _BLK_MEAN_B, _BLK_COV_B, slice(0, 2)),
    ],
)
def test_block_move_alone_recovers_own_marginal_and_freezes_other_block(
    name, indices, mean, cov, other_slice
):
    init = jnp.array([0.5, -0.5, 1.5, 1.0])  # off-mean start, both blocks
    other_init = np.asarray(init)[other_slice]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = run_mcmc(
            _independent_two_block_logdensity, init, algorithm="rwm",
            n_chains=3, n_warmup=1000, n_samples=6000, seed=2,
            proposal_cov=jnp.eye(4),  # unused: full move gets prob 0 below
            block_moves=[{"name": name, "prob": 1.0, "indices": indices, "cov": cov}],
        )
    pos = np.asarray(run.positions)

    # (c) mask: the OTHER block is bit-identical to its initial value at
    # every single kept draw, every chain -- not a statistical claim.
    other = pos[..., other_slice]
    np.testing.assert_array_equal(other, np.broadcast_to(other_init, other.shape))

    # (a) the MOVING block recovers its own known analytic marginal.
    moving = pos[..., indices].reshape(-1, 2)
    emp_mean = moving.mean(axis=0)
    emp_cov = np.cov(moving.T)
    assert np.all(np.abs(emp_mean - np.asarray(mean)) < 0.15), (name, emp_mean)
    assert np.all(np.abs(emp_cov - np.asarray(cov)) < 0.25), (name, emp_cov)

    acc = float(jnp.mean(run.info[f"is_{name}"]))
    assert acc > 0.99, acc  # the only move with nonzero prob


def test_block_mixture_proposal_masks_touch_only_own_coordinates():
    """Direct, accept/reject-independent check on the raw proposal step: a
    CONSTANT logdensity makes every Metropolis step accept with probability
    1 (log_p_accept == 0 always), isolating the proposal mechanism itself.
    For every step, whichever move fired, every coordinate NOT in that
    move's ``indices`` must be bit-identical before/after."""
    from bananafit.sampling import _build_block_mixture_rwm_kernel

    n_free = 6
    moves = [
        {"name": "full", "prob": 0.4, "kind": "gaussian",
         "indices": tuple(range(n_free)), "sigma": 0.01 * jnp.eye(n_free)},
        {"name": "osc", "prob": 0.3, "kind": "gaussian",
         "indices": (0, 1, 2, 3, 4), "sigma": 0.02 * jnp.eye(5)},
        {"name": "solar", "prob": 0.2, "kind": "gaussian",
         "indices": (0, 1), "sigma": 0.03 * jnp.eye(2)},
        {"name": "hop", "prob": 0.1, "kind": "hop",
         "hop_move_fn": lambda z: z.at[5].multiply(-1.0)},
    ]
    const_logdensity = lambda z: jnp.asarray(0.0)  # noqa: E731 -- always-accept
    init_fn, step_fn = _build_block_mixture_rwm_kernel(const_logdensity, n_free, moves)
    state = init_fn(jnp.arange(1.0, n_free + 1.0))

    def body(state, key):
        before = state.position
        state, info = step_fn(key, state)
        return state, (before, state.position, info)

    keys = jax.random.split(jax.random.PRNGKey(0), 4000)
    _, (before, after, infos) = jax.lax.scan(body, state, keys)
    before, after = np.asarray(before), np.asarray(after)

    # every step was accepted (const logdensity => log_p_accept == 0)
    assert np.all(np.asarray(infos["is_accepted"]))

    idx_by_move = {
        "full": (0, 1, 2, 3, 4, 5), "osc": (0, 1, 2, 3, 4),
        "solar": (0, 1), "hop": (5,),
    }
    for m in moves:
        fired = np.asarray(infos[f"is_{m['name']}"])
        assert fired.sum() > 0, m["name"]  # sanity: this move actually fired
        touched = set(idx_by_move[m["name"]])
        untouched = [i for i in range(n_free) if i not in touched]
        if untouched:
            np.testing.assert_array_equal(
                before[fired][:, untouched], after[fired][:, untouched],
                err_msg=f"move {m['name']!r} changed a coordinate outside {sorted(touched)}",
            )


_BOSC_MEAN = jnp.array([0.3, -0.6, 1.1, -1.4, 0.7])  # coords 0-4: correlated Gaussian
_BOSC_RHO = 0.5
_BOSC_SIGMA = jnp.array([0.4, 0.9, 0.6, 1.1, 0.3])
_BOSC_CORR = jnp.full((5, 5), _BOSC_RHO) + jnp.eye(5) * (1.0 - _BOSC_RHO)
_BOSC_COV = _BOSC_SIGMA[:, None] * _BOSC_CORR * _BOSC_SIGMA[None, :]
_BOSC_PREC = jnp.linalg.inv(_BOSC_COV)
_BOSC_HOP_MU = 4.0  # coord 5: bimodal, octant-like, symmetric hop z5 -> -z5
_BOSC_HOP_SIGMA = 0.3
_BOSC_HOP_W1 = 0.35


def _osc_like_logdensity(z):
    """6-D target mirroring the real Sec-6 shape: coords 0-4 a correlated
    Gaussian (stand-in for theta12/theta13/deltacp/dm21/dm31), coord 5 an
    independent SYMMETRIC bimodal (stand-in for the theta23 octant -- z5 ->
    -z5 is the exact hop involution, as in the real sin^2theta23 case)."""
    dz = z[:5] - _BOSC_MEAN
    lp_gauss = -0.5 * jnp.dot(dz, _BOSC_PREC @ dz)
    log_w1 = jnp.log(_BOSC_HOP_W1)
    log_w2 = jnp.log1p(-_BOSC_HOP_W1)
    lp1 = log_w1 - 0.5 * ((z[5] + _BOSC_HOP_MU) / _BOSC_HOP_SIGMA) ** 2
    lp2 = log_w2 - 0.5 * ((z[5] - _BOSC_HOP_MU) / _BOSC_HOP_SIGMA) ** 2
    lp_bimodal = jax.scipy.special.logsumexp(jnp.stack([lp1, lp2]))
    return lp_gauss + lp_bimodal


def test_block_mixture_recovers_analytic_marginals():
    """The full 4-way mixture (full + osc-block + solar-block + octant-hop,
    same shape as the real Sec-6 design) recovers BOTH the correlated
    Gaussian block's analytic mean/cov AND the bimodal coordinate's true
    mode-weight split -- the mixture as a whole still targets the exact
    joint density (Tierney 1994), independent of how the blocks are
    proposed."""
    full_cov = jnp.eye(6) * 0.05  # deliberately mismatched/generic shape --
    # correctness must not depend on the full move's covariance matching the
    # target; only efficiency does.
    init = jnp.concatenate([_BOSC_MEAN, jnp.array([_BOSC_HOP_MU])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = run_mcmc(
            _osc_like_logdensity, init, algorithm="rwm", n_chains=4,
            n_warmup=3000, n_samples=25000, seed=3, proposal_cov=full_cov,
            step_scale=1.0,
            block_moves=[
                {"name": "osc", "prob": 0.3, "indices": (0, 1, 2, 3, 4), "cov": _BOSC_COV},
                {"name": "solar", "prob": 0.1, "indices": (0, 1), "cov": _BOSC_COV[:2, :2]},
            ],
            hop_prob=0.1, hop_indices=(5,),
        )
    pos = np.asarray(run.positions)

    gauss_part = pos[..., :5].reshape(-1, 5)
    emp_mean = gauss_part.mean(axis=0)
    emp_cov = np.cov(gauss_part.T)
    assert np.all(np.abs(emp_mean - np.asarray(_BOSC_MEAN)) < 0.15), emp_mean
    assert np.all(np.abs(emp_cov - np.asarray(_BOSC_COV)) < 0.2), emp_cov

    frac_upper = float(np.mean(pos[..., 5] > 0.0))
    assert abs(frac_upper - (1.0 - _BOSC_HOP_W1)) < 0.07, frac_upper

    for name in ("full", "osc", "solar", "hop"):
        rate = float(jnp.mean(run.info[f"is_{name}"]))
        assert rate > 0.0, name
        acc = float(jnp.mean(run.info[f"{name}_accepted"])) / rate
        assert 0.0 < acc <= 1.0, (name, acc)


_BCP_MU = 3.0       # coord 0: second bimodal coordinate (delta_CP-mirror-like), asymmetric weights
_BCP_W1 = 0.2


def _two_bimodal_logdensity(z):
    """2-D product of two independent bimodals: coord 1 the symmetric octant-like one (main hop z1 -> -z1), coord 0
    a second pair of modes at -/+_BCP_MU with weights _BCP_W1 / 1 - _BCP_W1 (block-move hop z0 -> -z0)."""
    def bimodal(x, mu, w1):
        return jax.scipy.special.logsumexp(jnp.stack([jnp.log(w1) - 0.5 * ((x + mu) / 0.3) ** 2,
                                                      jnp.log1p(-w1) - 0.5 * ((x - mu) / 0.3) ** 2]))
    return bimodal(z[0], _BCP_MU, _BCP_W1) + bimodal(z[1], _BOSC_HOP_MU, _BOSC_HOP_W1)


def test_block_moves_extra_hop_recovers_both_mode_splits():
    """A ``kind="hop"`` block move next to the main hop: both independent bimodal coordinates reach their true weights;
    without the extra hop the second pair is never crossed (Gaussian steps of 0.3 cannot cross a 6-unit gap)."""
    init = jnp.array([-_BCP_MU, _BOSC_HOP_MU])
    kw = dict(algorithm="rwm", n_chains=4, n_warmup=2000, n_samples=20000, seed=5,
              proposal_cov=jnp.eye(2) * 0.09, step_scale=1.0, hop_prob=0.1, hop_indices=(1,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = run_mcmc(_two_bimodal_logdensity, init, block_moves=[
            {"name": "cphop", "prob": 0.1, "kind": "hop", "hop_move_fn": lambda z: z.at[0].multiply(-1.0)}], **kw)
        run0 = run_mcmc(_two_bimodal_logdensity, init, **kw)
    pos = np.asarray(run.positions)
    assert abs(float(np.mean(pos[..., 0] < 0)) - _BCP_W1) < 0.04
    assert abs(float(np.mean(pos[..., 1] < 0)) - _BOSC_HOP_W1) < 0.04
    for name in ("full", "hop", "cphop"):
        rate = float(jnp.mean(run.info[f"is_{name}"]))
        assert rate > 0.0 and float(jnp.mean(run.info[f"{name}_accepted"])) > 0.0, name
    assert float(np.mean(np.asarray(run0.positions)[..., 0] < 0)) == 1.0      # stuck in the start mode


def test_block_moves_hop_kind_needs_a_callable():
    with pytest.raises(ValueError, match="callable"):
        run_mcmc(logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10, proposal_cov=TRUE_COV,
                 block_moves=[{"name": "x", "prob": 0.1, "kind": "hop"}])


def test_block_moves_default_empty_matches_plain_rwm():
    """``block_moves=None``/``[]`` (default) is bit-identical to not passing
    it at all -- strictly additive/opt-in, same invariant as
    ``hop_prob=0.0``'s own back-compat test above."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_a = run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=2, n_warmup=20,
            n_samples=50, seed=0, proposal_cov=TRUE_COV,
        )
        run_b = run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=2, n_warmup=20,
            n_samples=50, seed=0, proposal_cov=TRUE_COV, block_moves=[],
        )
    np.testing.assert_array_equal(
        np.asarray(run_a.positions), np.asarray(run_b.positions)
    )


def test_block_moves_probabilities_exceeding_one_rejected():
    with pytest.raises(ValueError, match="exceed 1"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=TRUE_COV,
            block_moves=[{"name": "a", "prob": 0.6, "indices": (0,), "cov": jnp.eye(1)},
                         {"name": "b", "prob": 0.6, "indices": (1,), "cov": jnp.eye(1)}],
        )


def test_block_moves_reserved_name_rejected():
    with pytest.raises(ValueError, match="reserved|duplicate"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=TRUE_COV,
            block_moves=[{"name": "full", "prob": 0.1, "indices": (0,), "cov": jnp.eye(1)}],
        )


def test_block_moves_rejected_for_gradient_algorithms():
    with pytest.raises(ValueError, match="block_moves"):
        run_mcmc(
            logdensity, INIT, algorithm="nuts", n_warmup=10, n_samples=10,
            block_moves=[{"name": "a", "prob": 0.1, "indices": (0,), "cov": jnp.eye(1)}],
        )


def test_block_moves_out_of_range_indices_rejected():
    with pytest.raises(ValueError, match="out of range"):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=TRUE_COV,
            block_moves=[{"name": "a", "prob": 0.1, "indices": (D + 5,), "cov": jnp.eye(1)}],
        )


# ---------------------------------------------------------------------------
# MALA: smoke test
# ---------------------------------------------------------------------------


def test_mala_smoke():
    run = run_mcmc(
        logdensity, INIT, algorithm="mala", n_chains=2, n_warmup=500,
        n_samples=1000, seed=0,
    )
    assert run.algorithm == "mala"
    assert np.all(np.isfinite(np.asarray(run.positions)))
    assert np.all(np.isfinite(np.asarray(run.log_prob)))
    mean = np.asarray(run.positions).reshape(-1, D).mean(axis=0)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < 0.5), mean


# ---------------------------------------------------------------------------
# HMC: smoke test
# ---------------------------------------------------------------------------


def test_hmc_smoke():
    run = run_mcmc(
        logdensity, INIT, algorithm="hmc", n_chains=2, n_warmup=500,
        n_samples=500, seed=0,
    )
    assert run.algorithm == "hmc"
    assert np.all(np.isfinite(np.asarray(run.positions)))
    mean = np.asarray(run.positions).reshape(-1, D).mean(axis=0)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < 0.5), mean


# ---------------------------------------------------------------------------
# map_fit / laplace_covariance
# ---------------------------------------------------------------------------


def test_map_fit_finds_mean():
    z_map, ld_at_map = map_fit(logdensity, jnp.zeros(D), n_steps=500)
    assert np.allclose(np.asarray(z_map), np.asarray(TRUE_MEAN), atol=1e-3)
    assert np.isclose(float(ld_at_map), float(logdensity(TRUE_MEAN)), atol=1e-6)


def test_map_fit_adam_method():
    z_map, ld_at_map = map_fit(
        logdensity, jnp.zeros(D), n_steps=3000, method="adam", learning_rate=0.05
    )
    assert np.allclose(np.asarray(z_map), np.asarray(TRUE_MEAN), atol=1e-2)


def test_laplace_covariance_recovers_true_covariance():
    z_map, _ = map_fit(logdensity, jnp.zeros(D), n_steps=500)
    cov = laplace_covariance(logdensity, z_map)
    assert np.allclose(np.asarray(cov), np.asarray(TRUE_COV), atol=1e-6)


def test_laplace_covariance_non_pd_fallback_warns():
    def flat_logdensity(z):
        # zero curvature -> -hessian is exactly singular (PSD, not PD).
        return jnp.asarray(0.0) * jnp.sum(z)

    with pytest.warns(RuntimeWarning, match="positive definite"):
        cov = laplace_covariance(flat_logdensity, jnp.zeros(D))
    assert np.all(np.isfinite(np.asarray(cov)))


# ---------------------------------------------------------------------------
# laplace_covariance through splines.py's custom-VJP gathers (regression for
# the jax.hessian/jvp-of-custom_vjp TypeError, see laplace_covariance's
# docstring) -- ``jax.hessian``
# (forward-over-reverse) cannot differentiate through SplineTable at all;
# laplace_covariance now uses jax.jacrev(jax.grad(...)) (reverse-over-
# reverse), which does. Cross-checked here against an independent,
# AD-free central-finite-difference Hessian of jax.grad(logdensity) (grad
# itself is plain reverse-mode -- no jvp involved -- so this reference does
# not share any code path with laplace_covariance's own AD strategy).
# ---------------------------------------------------------------------------

_SPLINE_KNOT_GRID = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def _build_dense_spline_logdensity(seed=0, n_syst=3, n_curves=4, n_events=25):
    from bananafit import splines

    rng = np.random.default_rng(seed)
    knot_x = np.tile(_SPLINE_KNOT_GRID, (n_syst, 1))
    knot_y = 1.0 + 0.3 * rng.standard_normal(size=(n_syst, n_curves, _SPLINE_KNOT_GRID.size))
    event_index = rng.integers(0, n_curves + 1, size=(n_syst, n_events)).astype(np.int32)
    table = splines.build_spline_table(
        knot_x, knot_y, event_index, boundary="natural", dtype=jnp.float64,
    )
    data = jnp.asarray(1.0 + 0.1 * rng.standard_normal(n_events))

    def logdensity_spline(z):
        w = table(z)
        return -0.5 * jnp.sum((w - data) ** 2) - 0.5 * jnp.sum(z**2)

    return logdensity_spline, n_syst


def _build_factorized_spline_logdensity(seed=0, n_syst=3, n_cells=6, n_events=25, n_curves=4):
    from bananafit import splines

    rng = np.random.default_rng(seed)
    knot_x = np.tile(_SPLINE_KNOT_GRID, (n_syst, 1))
    knot_y = 1.0 + 0.3 * rng.standard_normal(size=(n_syst, n_curves, _SPLINE_KNOT_GRID.size))
    cell_to_curve = rng.integers(0, n_curves + 1, size=(n_syst, n_cells)).astype(np.int32)
    event_cell = rng.integers(0, n_cells, size=n_events).astype(np.int32)
    table = splines.build_spline_table_factorized(
        knot_x, knot_y, cell_to_curve, event_cell, boundary="natural", dtype=jnp.float64,
    )
    data = jnp.asarray(1.0 + 0.1 * rng.standard_normal(n_events))

    def logdensity_spline(z):
        w = table(z)
        return -0.5 * jnp.sum((w - data) ** 2) - 0.5 * jnp.sum(z**2)

    return logdensity_spline, n_syst


def _finite_diff_hessian(f, z0, eps=1e-4):
    """Central-finite-difference Hessian of ``f`` via columns of
    ``jax.grad(f)`` (reverse-mode only -- no ``jvp`` anywhere, so this is an
    independent cross-check, not a rederivation of laplace_covariance's own
    AD strategy). Symmetrized like laplace_covariance itself.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    grad_fn = jax.grad(f)
    n = z0.shape[0]
    step = eps * np.maximum(np.abs(z0), 1.0)
    hess = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dz = np.zeros(n)
        dz[i] = step[i]
        g_plus = np.asarray(grad_fn(jnp.asarray(z0 + dz)))
        g_minus = np.asarray(grad_fn(jnp.asarray(z0 - dz)))
        hess[:, i] = (g_plus - g_minus) / (2.0 * step[i])
    return 0.5 * (hess + hess.T)


@pytest.mark.parametrize(
    "builder", [_build_dense_spline_logdensity, _build_factorized_spline_logdensity],
    ids=["dense", "factorized"],
)
def test_laplace_covariance_matches_finite_diff_hessian_through_splines(builder):
    logdensity_spline, n_syst = builder()

    # Sanity: jax.hessian (forward-over-reverse) is exactly the pre-existing
    # gap laplace_covariance's fix works around -- confirm it still fails,
    # so this test would catch a regression if splines.py's custom_vjp
    # primitives ever grew a JVP rule and laplace_covariance were reverted.
    with pytest.raises(TypeError, match="jvp"):
        jax.hessian(logdensity_spline)(jnp.zeros(n_syst))

    # Evaluate at the actual MAP (map_fit), not an arbitrary point: the
    # ``-0.5 * sum(z**2)`` regularizer in both builders guarantees a unique,
    # strongly concave interior maximum (genuinely positive-definite
    # curvature there), whereas an arbitrary z0 (e.g. 0) can land with
    # negative curvature in some spline-knot directions and trip
    # laplace_covariance's pinv fallback -- which is a different, already
    # separately-tested code path (test_laplace_covariance_non_pd_fallback_
    # warns), not what this test is checking.
    z_map, _ld = map_fit(logdensity_spline, jnp.zeros(n_syst), n_steps=500)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cov = laplace_covariance(logdensity_spline, z_map)

    fd_hess = _finite_diff_hessian(logdensity_spline, z_map, eps=1e-4)
    fd_cov = np.linalg.inv(-fd_hess)
    fd_cov = 0.5 * (fd_cov + fd_cov.T)

    np.testing.assert_allclose(np.asarray(cov), fd_cov, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# diagnostics.to_arviz
# ---------------------------------------------------------------------------


class _StubParameterSet:
    """Minimal duck-typed ParameterSet stand-in: identity constrain, no
    circular parameters. Matches the interface documented at the top of
    bananafit/diagnostics.py.
    """

    def __init__(self, names):
        self.names = list(names)
        self.free_names = list(names)

    def constrain(self, z):
        return z

    def wrap_circular(self, physical):
        return physical


def test_to_arviz_zspace_fallback_names():
    pytest.importorskip("arviz")
    run = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=50,
        n_samples=60, seed=0,
    )
    idata = diagnostics.to_arviz(run)
    post = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    for i in range(D):
        assert f"z{i}" in post.data_vars
    assert post["z0"].shape == (2, 60)


def test_to_arviz_with_parameter_set():
    pytest.importorskip("arviz")
    run = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=50,
        n_samples=60, seed=0,
    )
    names = ["a", "b", "c"]
    idata = diagnostics.to_arviz(run, parameter_set=_StubParameterSet(names))
    post = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    for n in names:
        assert n in post.data_vars
        assert post[n].shape == (2, 60)
    np.testing.assert_allclose(
        np.asarray(post["a"].values), np.asarray(run.positions[..., 0])
    )


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_returns_table(capsys):
    run = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=50,
        n_samples=60, seed=0,
    )
    table = diagnostics.summary(run, names=["a", "b", "c"])
    assert set(table.keys()) == {"a", "b", "c"}
    for row in table.values():
        assert set(row.keys()) == {"mean", "std", "rhat", "ess"}
    captured = capsys.readouterr()
    assert "rhat" in captured.out


# ---------------------------------------------------------------------------
# circular-aware rhat/ess/summary/to_arviz -- DESIGN.md-linked fix for
# deltacp-style branch-hopping (see bananafit/diagnostics.py's module
# docstring). Synthetic ``SimpleNamespace(positions=..., log_prob=...,
# info=...)`` runs -- same lightweight MCMCRun duck type
# ``examples/dune_large_fit.py``'s ``_run_like`` builds from checkpoints --
# so these are pure-numpy, no actual sampling needed.
# ---------------------------------------------------------------------------


class _StubCircularParameterSet:
    """Minimal duck-typed ParameterSet: one circular free param, one linear
    free param, identity constrain (as if both used the identity
    transform -- true for any circular param, and for this stub's purposes
    for the linear one too)."""

    def __init__(self, names, circular_names):
        self.names = list(names)
        self.free_names = list(names)
        self.circular_free_names = tuple(circular_names)

    def constrain(self, z):
        return z

    def wrap_circular(self, physical):
        wrapped = jnp.mod(physical + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return jnp.where(wrapped <= -jnp.pi, wrapped + 2.0 * jnp.pi, wrapped)


def _branch_hopping_positions(mu, sigma, n_chains=4, n_samples=2000, seed=0,
                               stuck_offsets=None):
    """Synthetic single-parameter chain: Gaussian around ``mu`` in angle
    space, optionally with each chain permanently offset by
    ``stuck_offsets[c] * 2*pi`` (the "some chains stuck on a different
    winding" pathology that makes linear R-hat blow up while the true
    posterior -- being 2*pi periodic -- has fully converged)."""
    rng = np.random.default_rng(seed)
    base = mu + sigma * rng.standard_normal((n_chains, n_samples))
    if stuck_offsets is not None:
        base = base + 2.0 * np.pi * np.asarray(stuck_offsets)[:, None]
    return base[..., None]  # (n_chains, n_samples, 1)


def test_rhat_ess_circular_names_recovers_convergence_linear_view_misses():
    """The scenario this whole fix targets: chains 2 and 3 are permanently
    on the x+2*pi replica of the exact same physical mode chains 0 and 1
    explore -- physically converged, but a linear view of the raw
    coordinate sees "two chains stuck apart from the other two" and
    reports a badly inflated R-hat / collapsed ESS."""
    positions = _branch_hopping_positions(
        mu=3.0, sigma=0.3, stuck_offsets=[0, 0, 1, 1], seed=1,
    )
    run = SimpleNamespace(positions=jnp.asarray(positions), log_prob=jnp.zeros(positions.shape[:2]), info={})

    r_linear = diagnostics.rhat(run, names=["deltacp"])
    e_linear = diagnostics.ess(run, names=["deltacp"])
    r_circ = diagnostics.rhat(run, names=["deltacp"], circular_names=["deltacp"])
    e_circ = diagnostics.ess(run, names=["deltacp"], circular_names=["deltacp"])

    assert r_linear["deltacp"] > 2.0, r_linear  # linear view: badly non-converged
    assert r_circ["deltacp"] < 1.01, r_circ  # circular view: actually converged
    assert e_circ["deltacp"] > 10.0 * e_linear["deltacp"]

    # parameter_set= gives the same answer as the equivalent circular_names=
    ps = _StubCircularParameterSet(["deltacp"], ["deltacp"])
    r_via_ps = diagnostics.rhat(run, names=["deltacp"], parameter_set=ps)
    np.testing.assert_allclose(r_via_ps["deltacp"], r_circ["deltacp"])


def test_rhat_ess_non_circular_param_unaffected_by_circular_names():
    """A non-circular parameter's rhat/ess must be untouched whether or not
    *other* parameters are marked circular."""
    lin = 1.0 + 0.4 * np.random.default_rng(2).standard_normal((4, 500))
    circ = _branch_hopping_positions(mu=0.0, sigma=0.5, n_samples=500, seed=3)[..., 0]
    positions = np.stack([lin, circ], axis=-1)
    run = SimpleNamespace(positions=jnp.asarray(positions), log_prob=jnp.zeros(positions.shape[:2]), info={})

    r_plain = diagnostics.rhat(run, names=["lin", "circ"])
    r_marked = diagnostics.rhat(run, names=["lin", "circ"], circular_names=["circ"])
    np.testing.assert_allclose(r_marked["lin"], r_plain["lin"])


def test_summary_circular_mean_std_match_analytic():
    # Half the chains permanently on the x+2*pi replica (the measured
    # large-scale-RWM pathology: 69% of raw draws outside (-pi, pi]) -- the
    # circular mean/std (invariant to sin/cos under any 2*pi shift) must
    # still recover the true mu/sigma; the linear mean is dragged toward
    # mu_true + pi by the offset half of the chains.
    mu_true, sigma_true = 3.0, 0.35
    positions = _branch_hopping_positions(
        mu=mu_true, sigma=sigma_true, n_samples=4000, seed=4, stuck_offsets=[0, 0, 1, 1],
    )
    run = SimpleNamespace(positions=jnp.asarray(positions), log_prob=jnp.zeros(positions.shape[:2]), info={})
    table = diagnostics.summary(run, names=["deltacp"], circular_names=["deltacp"])
    assert set(table["deltacp"].keys()) == {"mean", "std", "rhat", "ess"}
    assert table["deltacp"]["mean"] == pytest.approx(mu_true, abs=0.05)
    assert table["deltacp"]["std"] == pytest.approx(sigma_true, abs=0.05)

    # a *linear* summary on the same raw draws is not close to the truth --
    # confirms this is testing real behavior, not something linear stats
    # would have gotten right anyway.
    linear_table = diagnostics.summary(run, names=["deltacp"])
    assert linear_table["deltacp"]["mean"] != pytest.approx(mu_true, abs=0.05)


def test_to_arviz_wraps_circular_param_around_circular_mean():
    pytest.importorskip("arviz")
    mu_true = 3.0  # near +pi: a fixed (-pi, pi] wrap would split this mode
    positions = _branch_hopping_positions(mu=mu_true, sigma=0.4, n_samples=3000, seed=5)
    run = SimpleNamespace(positions=jnp.asarray(positions), log_prob=jnp.zeros(positions.shape[:2]), info={})
    ps = _StubCircularParameterSet(["deltacp"], ["deltacp"])

    idata = diagnostics.to_arviz(run, parameter_set=ps)
    post = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    x = np.asarray(post["deltacp"].values)

    # single contiguous window around mu_true, no artificial mode-splitting
    # (all draws land within one 2*pi-wide window centered near mu_true).
    assert x.max() - x.min() < 2.0 * np.pi
    assert x.mean() == pytest.approx(mu_true, abs=0.1)
    assert post.attrs["deltacp_circular_mean"] == pytest.approx(mu_true, abs=0.1)


def test_to_arviz_circular_wrap_does_not_touch_non_circular_params():
    pytest.importorskip("arviz")
    lin = 1.0 + 0.4 * np.random.default_rng(6).standard_normal((3, 400))
    circ = _branch_hopping_positions(mu=0.0, sigma=0.3, n_chains=3, n_samples=400, seed=7)[..., 0]
    positions = np.stack([lin, circ], axis=-1)
    run = SimpleNamespace(positions=jnp.asarray(positions), log_prob=jnp.zeros(positions.shape[:2]), info={})
    ps = _StubCircularParameterSet(["lin", "circ"], ["circ"])

    idata = diagnostics.to_arviz(run, parameter_set=ps)
    post = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    np.testing.assert_allclose(np.asarray(post["lin"].values), lin)


# ---------------------------------------------------------------------------
# Laplace-seeded NUTS: run_mcmc(inverse_mass_matrix=..., warmup=...) and the
# laplace_seeded_nuts convenience wrapper.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def map_and_cov():
    z_map, ld_map = map_fit(logdensity, jnp.zeros(D), n_steps=500)
    cov = laplace_covariance(logdensity, z_map)
    return z_map, ld_map, cov


def test_step_size_warmup_recovers_gaussian(map_and_cov):
    z_map, _ld_map, cov = map_and_cov
    run = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=2, n_warmup=200,
        n_samples=1000, seed=0, inverse_mass_matrix=cov, warmup="step-size",
    )
    assert run.algorithm == "nuts"
    r = diagnostics.rhat(run)
    e = diagnostics.ess(run)
    assert all(v < 1.05 for v in r.values()), r
    assert all(v > 100 for v in e.values()), e

    pos = np.asarray(run.positions).reshape(-1, D)
    mean = pos.mean(axis=0)
    min_ess = min(e.values())
    tol = 3.0 * np.sqrt(np.diag(np.asarray(TRUE_COV))) / np.sqrt(min_ess)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < tol), (mean, tol)


def test_step_size_warmup_leaves_mass_matrix_unadapted(map_and_cov):
    # warmup="step-size" is a *fixed*-mass-matrix warmup by design: unlike
    # window_adaptation, it never re-estimates the mass matrix from the
    # chain's own covariance, so tuned["inverse_mass_matrix"] must come back
    # bit-identical to what was passed in (this is precisely the mass-matrix-
    # estimation work it skips relative to warmup="window").
    z_map, _ld_map, cov = map_and_cov
    run = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=3, n_warmup=50,
        n_samples=10, seed=0, inverse_mass_matrix=cov, warmup="step-size",
    )
    for i in range(3):
        np.testing.assert_array_equal(
            np.asarray(run.tuned["inverse_mass_matrix"][i]), np.asarray(cov)
        )
    # diagonal form also accepted directly (verified against installed
    # blackjax 1.6.2's metrics.default_metric/gaussian_euclidean).
    diag_cov = jnp.diag(cov)
    run_diag = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=2, n_warmup=50,
        n_samples=10, seed=0, inverse_mass_matrix=diag_cov, warmup="step-size",
    )
    assert run_diag.tuned["inverse_mass_matrix"].shape == (2, D)


def test_step_size_warmup_needs_fewer_logdensity_calls():
    # The headline motivation for warmup="step-size": seeded with the
    # MAP+Laplace covariance, NUTS uses a good metric from the first warmup
    # step, while window_adaptation starts from the identity and spends its
    # early windows on long trajectories. On a real model the warmup cost is
    # the logdensity/gradient evaluations, so the test counts those instead
    # of timing the run: on a toy this small each run_mcmc call is ~3.5 s of
    # tracing and compilation whether n_warmup is 10 or 4000, so a wall-clock
    # comparison only compares compile times (and flipped on CI).
    # A target whose widths span 100x, with correlation, is the case the
    # fast path is for. Measured (seeds 0-3): window 14.7k-18.1k calls,
    # step-size 1.9k-3.1k, ratio <= 0.21; the 0.5 bound leaves room for
    # jax/blackjax RNG changes. n_chains=1 keeps the count exact (a vmapped
    # while_loop also runs the finished chains' masked iterations).
    sigma = jnp.array([1.0, 10.0, 0.1])
    corr = jnp.full((D, D), 0.8) + jnp.eye(D) * 0.2
    prec = jnp.linalg.inv(sigma[:, None] * corr * sigma[None, :])

    def scaled_logdensity(z):
        dz = z - TRUE_MEAN
        return -0.5 * jnp.dot(dz, prec @ dz)

    n_calls = [0]

    def _count(_x):
        n_calls[0] += 1

    def counted_logdensity(z):
        jax.debug.callback(_count, z[0])
        return scaled_logdensity(z)

    z_map, _ = map_fit(scaled_logdensity, jnp.zeros(D), n_steps=500)
    cov = laplace_covariance(scaled_logdensity, z_map)
    calls = {}
    for mode, kw in (
        ("window", {}),
        ("step-size", dict(inverse_mass_matrix=cov, warmup="step-size")),
    ):
        n_calls[0] = 0
        run_mcmc(
            counted_logdensity, z_map, algorithm="nuts", n_chains=1,
            n_warmup=400, n_samples=5, seed=0, **kw,
        )
        calls[mode] = n_calls[0]

    print(f"\nlogdensity calls, window warmup: {calls['window']}, "
          f"step-size warmup: {calls['step-size']}")
    assert calls["window"] > 0, calls
    assert calls["step-size"] < 0.5 * calls["window"], calls


def test_step_size_warmup_requires_inverse_mass_matrix():
    with pytest.raises(ValueError, match="step-size"):
        run_mcmc(
            logdensity, INIT, algorithm="nuts", n_warmup=10, n_samples=10,
            warmup="step-size",
        )


def test_warmup_none_requires_step_size():
    with pytest.raises(ValueError, match="step_size"):
        run_mcmc(
            logdensity, INIT, algorithm="nuts", n_warmup=10, n_samples=10,
            warmup="none",
        )


def test_warmup_none_starts_sampling_immediately(map_and_cov):
    z_map, _ld_map, cov = map_and_cov
    run = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=2, n_warmup=12345,
        n_samples=500, seed=0, inverse_mass_matrix=cov, step_size=0.5,
        warmup="none",
    )
    assert run.algorithm == "nuts"
    # tuned kernel parameters are exactly what was supplied -- no adaptation
    # of any kind happened.
    np.testing.assert_array_equal(
        np.asarray(run.tuned["step_size"]), np.full((2,), 0.5)
    )
    for i in range(2):
        np.testing.assert_array_equal(
            np.asarray(run.tuned["inverse_mass_matrix"][i]), np.asarray(cov)
        )
    assert np.all(np.isfinite(np.asarray(run.positions)))

    pos = np.asarray(run.positions).reshape(-1, D)
    mean = pos.mean(axis=0)
    e = diagnostics.ess(run)
    min_ess = max(min(e.values()), 1.0)
    tol = 5.0 * np.sqrt(np.diag(np.asarray(TRUE_COV))) / np.sqrt(min_ess)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < tol), (mean, tol)


def test_warmup_none_defaults_identity_mass_matrix():
    # inverse_mass_matrix is optional even for warmup="none" -- defaults to
    # the identity (only step_size is mandatory to skip warmup entirely).
    run = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=1, n_warmup=999,
        n_samples=5, seed=0, step_size=0.2, warmup="none",
    )
    np.testing.assert_array_equal(
        np.asarray(run.tuned["inverse_mass_matrix"]), np.ones((1, D))
    )


def test_warmup_invalid_value_raises():
    with pytest.raises(ValueError):
        run_mcmc(
            logdensity, INIT, algorithm="nuts", n_warmup=10, n_samples=10,
            step_size=0.1, warmup="bogus",
        )


def test_gradient_options_rejected_for_rwm(map_and_cov):
    _z_map, _ld_map, cov = map_and_cov
    with pytest.raises(ValueError):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=cov, warmup="step-size",
        )


def test_inverse_mass_matrix_rejected_for_mala(map_and_cov):
    _z_map, _ld_map, cov = map_and_cov
    with pytest.raises(ValueError):
        run_mcmc(
            logdensity, INIT, algorithm="mala", n_warmup=10, n_samples=10,
            inverse_mass_matrix=cov,
        )


def test_mala_warmup_none():
    run = run_mcmc(
        logdensity, INIT, algorithm="mala", n_chains=2, n_warmup=99999,
        n_samples=200, seed=0, step_size=0.3, warmup="none",
    )
    np.testing.assert_array_equal(
        np.asarray(run.tuned["step_size"]), np.full((2,), 0.3)
    )
    assert np.all(np.isfinite(np.asarray(run.positions)))


def test_max_num_doublings_caps_tree_depth():
    # A pathologically tiny fixed step size forces NUTS's tree-building to
    # want many doublings to satisfy the U-turn criterion -- exactly the
    # "scaling cliff" mechanism documented in examples/LARGE_FIT_REPORT.md.
    # With max_num_doublings capped, run.info's NUTS-only tree-depth fields
    # (num_trajectory_expansions/num_integration_steps, see _NUTS_INFO_SPEC)
    # must never exceed the cap; uncapped (blackjax's own default of 10)
    # should push noticeably deeper on the same posterior/step size.
    run_capped = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=5,
        n_samples=20, seed=0, step_size=1e-5, warmup="none",
        max_num_doublings=4,
    )
    assert "num_trajectory_expansions" in run_capped.info
    assert "num_integration_steps" in run_capped.info
    max_expansions = float(np.max(np.asarray(run_capped.info["num_trajectory_expansions"])))
    assert max_expansions <= 4.0, max_expansions
    max_leapfrog = float(np.max(np.asarray(run_capped.info["num_integration_steps"])))
    assert max_leapfrog <= 2 ** 4, max_leapfrog

    run_uncapped = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=5,
        n_samples=20, seed=0, step_size=1e-5, warmup="none",
    )
    max_expansions_uncapped = float(
        np.max(np.asarray(run_uncapped.info["num_trajectory_expansions"]))
    )
    assert max_expansions_uncapped > max_expansions, (
        max_expansions_uncapped, max_expansions
    )


def test_max_num_doublings_rejected_for_hmc():
    with pytest.raises(ValueError, match="max_num_doublings"):
        run_mcmc(
            logdensity, INIT, algorithm="hmc", n_chains=1, n_warmup=5,
            n_samples=5, step_size=0.1, warmup="none", max_num_doublings=4,
        )


def test_max_num_doublings_rejected_for_mala():
    with pytest.raises(ValueError, match="max_num_doublings"):
        run_mcmc(
            logdensity, INIT, algorithm="mala", n_chains=1, n_warmup=5,
            n_samples=5, step_size=0.1, warmup="none", max_num_doublings=4,
        )


def test_max_num_doublings_rejected_for_rwm(map_and_cov):
    _z_map, _ld_map, cov = map_and_cov
    with pytest.raises(ValueError):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_chains=1, n_warmup=5,
            n_samples=5, proposal_cov=cov, max_num_doublings=4,
        )


def test_max_num_doublings_capped_with_step_size_warmup(map_and_cov):
    # The intended combination for the large-fit mitigation: Laplace-seeded
    # step-size-only warmup, with a tree-depth cap active during warmup too
    # (not just sampling) so a long step-size warmup cannot itself run away.
    z_map, _ld_map, cov = map_and_cov
    run = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=2, n_warmup=100,
        n_samples=100, seed=0, inverse_mass_matrix=cov, warmup="step-size",
        max_num_doublings=6,
    )
    max_expansions = float(np.max(np.asarray(run.info["num_trajectory_expansions"])))
    assert max_expansions <= 6.0, max_expansions
    r = diagnostics.rhat(run)
    assert all(v < 1.1 for v in r.values()), r


# ---------------------------------------------------------------------------
# target_acceptance_rate: passed through to whichever dual-averaging warmup
# runs (window_adaptation's own target_acceptance_rate, or the internal
# Nesterov dual averaging used by warmup="step-size"/mala).
# ---------------------------------------------------------------------------


def test_target_acceptance_rate_passed_through_window():
    run = run_mcmc(
        logdensity, INIT, algorithm="nuts", n_chains=2, n_warmup=200,
        n_samples=200, seed=0, target_acceptance_rate=0.6,
    )
    assert run.algorithm == "nuts"
    assert np.all(np.isfinite(np.asarray(run.positions)))


def test_target_acceptance_rate_passed_through_step_size(map_and_cov):
    z_map, _ld_map, cov = map_and_cov
    run = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=2, n_warmup=200,
        n_samples=500, seed=0, inverse_mass_matrix=cov, warmup="step-size",
        target_acceptance_rate=0.6,
    )
    r = diagnostics.rhat(run)
    assert all(v < 1.1 for v in r.values()), r


def test_target_acceptance_rate_affects_mala_warmup():
    run = run_mcmc(
        logdensity, INIT, algorithm="mala", n_chains=2, n_warmup=500,
        n_samples=500, seed=0, target_acceptance_rate=0.4,
    )
    acc = float(np.mean(np.asarray(run.info["acceptance_rate"])))
    # A much lower target than MALA's usual 0.574 optimum should settle on a
    # correspondingly lower steady-state acceptance rate.
    assert acc < 0.55, acc


def test_target_acceptance_rate_rejected_for_rwm(map_and_cov):
    _z_map, _ld_map, cov = map_and_cov
    with pytest.raises(ValueError):
        run_mcmc(
            logdensity, INIT, algorithm="rwm", n_warmup=10, n_samples=10,
            proposal_cov=cov, target_acceptance_rate=0.6,
        )


def test_laplace_seeded_nuts_forwards_max_num_doublings_and_target_acceptance():
    run, info = laplace_seeded_nuts(
        logdensity, jnp.zeros(D), n_chains=2, n_samples=100, n_warmup=100,
        seed=0, map_kwargs={"n_steps": 500}, max_num_doublings=6,
        target_acceptance_rate=0.6,
    )
    assert run.algorithm == "nuts"
    max_expansions = float(np.max(np.asarray(run.info["num_trajectory_expansions"])))
    assert max_expansions <= 6.0, max_expansions


def test_laplace_seeded_nuts_rejects_tuning_knobs_for_rwm():
    with pytest.raises(ValueError):
        laplace_seeded_nuts(
            logdensity, jnp.zeros(D), n_chains=2, n_samples=10, n_warmup=10,
            seed=0, algorithm="rwm", map_kwargs={"n_steps": 100},
            max_num_doublings=4,
        )


def test_window_warmup_accepts_inverse_mass_matrix_seed(map_and_cov):
    # warmup="window" (the default) still runs full online mass-matrix
    # estimation, but an explicit inverse_mass_matrix now seeds it (via
    # window_adaptation's initial_inverse_mass_matrix) instead of being
    # ignored -- purely additive, does not change the None-default path.
    z_map, _ld_map, cov = map_and_cov
    run = run_mcmc(
        logdensity, z_map, algorithm="nuts", n_chains=2, n_warmup=100,
        n_samples=200, seed=0, inverse_mass_matrix=cov,
    )
    assert run.algorithm == "nuts"
    assert np.all(np.isfinite(np.asarray(run.positions)))


# ---------------------------------------------------------------------------
# laplace_seeded_nuts convenience wrapper
# ---------------------------------------------------------------------------


def test_laplace_seeded_nuts_recovers_gaussian_and_reports_timings():
    run, info = laplace_seeded_nuts(
        logdensity, jnp.zeros(D), n_chains=2, n_samples=1000, n_warmup=200,
        seed=0, map_kwargs={"n_steps": 500},
    )
    assert run.algorithm == "nuts"
    assert set(info.keys()) == {
        "z_map", "logdensity_at_map", "cov", "map_time_s", "laplace_time_s",
    }
    assert info["map_time_s"] > 0.0
    assert info["laplace_time_s"] > 0.0
    np.testing.assert_allclose(
        np.asarray(info["z_map"]), np.asarray(TRUE_MEAN), atol=1e-3
    )

    r = diagnostics.rhat(run)
    assert all(v < 1.05 for v in r.values()), r
    pos = np.asarray(run.positions).reshape(-1, D)
    mean = pos.mean(axis=0)
    e = diagnostics.ess(run)
    min_ess = min(e.values())
    tol = 3.0 * np.sqrt(np.diag(np.asarray(TRUE_COV))) / np.sqrt(min_ess)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < tol), (mean, tol)


def test_laplace_seeded_nuts_algorithm_rwm():
    run, info = laplace_seeded_nuts(
        logdensity, jnp.zeros(D), n_chains=2, n_samples=500, n_warmup=300,
        seed=0, algorithm="rwm", map_kwargs={"n_steps": 500},
    )
    assert run.algorithm == "rwm"
    pos = np.asarray(run.positions).reshape(-1, D)
    mean = pos.mean(axis=0)
    assert np.all(np.abs(mean - np.asarray(TRUE_MEAN)) < 0.3), mean
    # proposal_cov used by rwm is the reported Laplace covariance (scaled by
    # the Roberts-Rosenthal constant internally, see run_mcmc's docstring).
    assert "cov" in info


def test_laplace_seeded_nuts_requires_z0():
    with pytest.raises(ValueError):
        laplace_seeded_nuts(logdensity, n_chains=2, n_samples=10, n_warmup=10)


def test_laplace_seeded_nuts_invalid_algorithm():
    with pytest.raises(ValueError):
        laplace_seeded_nuts(
            logdensity, jnp.zeros(D), n_chains=2, n_samples=10, n_warmup=10,
            algorithm="bogus",
        )
