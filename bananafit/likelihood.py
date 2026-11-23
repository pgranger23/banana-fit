"""Binned test statistics: Poisson, Barlow-Beeston-lite, and covariance-
Gaussian log-likelihoods (DESIGN.md §9).

Pure functions on ``(mu, w2, n)`` (or ``(pred, data, cov_inv)``) arrays -- no
state, no pytree wrapping needed. They return **log-likelihood** (up to an
additive n-only constant, same convention as ``mango.stat.poisson_chi2``
dropping ``log(n!)``), not chi2: MCMC wants a smooth scalar to differentiate,
not a test statistic.

The Poisson/BB-lite kernels follow the mango "double-where" gradient-safety
idiom: any ``sqrt``/``log``/division that can land on its domain edge is
evaluated on a substituted safe value inside a ``jnp.where`` so reverse-mode
AD never back-propagates through the singular branch.

``gaussian_cov_loglike``/``build_nd_covariance`` add a correlated-bin
Gaussian near-detector (ND) likelihood,
``-log L_ND = 0.5 (lambda - n)^T Z^-1 (lambda - n) + prior term``, with ``Z``
the ND bin covariance built from "many-universes" throws of the ND detector
systematics. The *nominal* prediction's statistical uncertainty is added to
``Z``'s diagonal and ``Z`` is inverted once, because re-inverting it at every
MCMC step would be prohibitively expensive.
``build_nd_covariance`` implements the fractional-covariance recipe (ND
detector covariances are commonly stored as fractional ``TMatrixD`` objects,
e.g. named ``*_frac_cov``): absolute covariance ``C_ij = frac_cov_ij *
nominal_i * nominal_j``, diagonal statistical variance (Poisson, ``=
nominal_i``) always added, optional MC-statistical variance of the nominal MC
template (e.g. the same per-bin ``w2`` that feeds
``barlow_beeston_lite_loglike``) added on top when supplied -- a
diagonal-only approximation of the statistical part.
``cov_inv`` is precomputed once, in float64 numpy, at *build* time (never
inside the traced ``logdensity`` hot path), exactly like
``MVGaussianBlock.scale_tril`` in ``parameters.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "poisson_loglike",
    "barlow_beeston_lite_loglike",
    "gaussian_cov_loglike",
    "build_nd_covariance",
]


def poisson_loglike(mu, n, *, mc_floor: float = 1e-5):
    """Binned Poisson log-likelihood, ``sum_b n_b log(mu_b) - mu_b``
    (``log(n_b!)`` dropped -- an n-only constant, irrelevant for MCMC and for
    likelihood-ratio use).

    ``mu`` is floored at ``mc_floor`` *only inside the log* (MaCh3's
    low-MC-statistics floor): an empty-MC bin would otherwise put ``log(0) =
    -inf`` on the gradient path. The linear ``-mu`` term is smooth all the
    way to ``mu = 0`` and is left unfloored. Because the floored argument to
    `log` is always > 0, ``log(mu_floored)`` is always finite, so ``n=0``
    bins reduce to ``0 * finite = 0`` exactly -- no extra ``where`` needed
    for that case.

    Parameters
    ----------
    mu, n : (n_bins,) f64 arrays -- expected and observed counts per bin.
    mc_floor : float -- floor applied to ``mu`` inside the log only.
    """
    mu = jnp.asarray(mu)
    n = jnp.asarray(n)
    mu_floored = jnp.maximum(mu, mc_floor)
    return jnp.sum(n * jnp.log(mu_floored) - mu)


def barlow_beeston_lite_loglike(mu, w2, n, *, mc_floor: float = 1e-5):
    """Barlow-Beeston-lite (Conway, arXiv:1103.0354) binned log-likelihood --
    one nuisance scale factor ``beta_b`` per bin absorbing the MC's own
    (finite-statistics) uncertainty, profiled out in closed form. Equivalent
    to pyhf's ``staterror`` modifier and MaCh3's
    ``SampleHandlerBase::GetTestStatLLH`` "Barlow-Beeston" mode.

    Derivation
    ----------
    Per bin, let ``mu`` be the nominal MC-weighted expectation and ``w2`` the
    sum of squared event weights in that bin (so ``w2 / mu^2`` is the
    relative variance of the MC's estimate of the true rate, exact for
    unweighted MC where it reduces to ``1/N_eff``). The joint likelihood of
    data ``n`` and the (unknown, Gaussian-distributed) true/nominal ratio
    ``beta`` is

        L(beta) = Pois(n | beta * mu) * Gaus(beta | 1, sigma2),   sigma2 = w2 / mu^2

    i.e. ``log L(beta) = n log(beta*mu) - beta*mu - (beta-1)^2 / (2 sigma2)``.
    "Lite" means one nuisance parameter per bin rather than per (bin, sample);
    it is profiled analytically instead of fit: setting ``d log L/d beta = 0``
    and multiplying through by ``beta`` gives the quadratic

        beta^2 + (mu*sigma2 - 1) beta - n*sigma2 = 0

    whose discriminant ``D = (mu*sigma2-1)^2 + 4 n sigma2`` is a sum of two
    non-negative terms (n, sigma2 >= 0), hence always >= 0. Of its two roots
    only ``beta_+ = (-(mu*sigma2-1) + sqrt(D)) / 2`` is guaranteed
    non-negative (``sqrt(D) >= |mu*sigma2-1|``), so that is the one used.
    Plugging ``beta_+`` back into ``log L(beta)`` gives the profiled
    log-likelihood returned here.

    ``w2 -> 0`` limit: as ``sigma2 -> 0``, the quadratic's positive root
    tends continuously to ``beta = 1`` (a short Taylor expansion gives
    ``beta - 1 ~ (n - mu) sigma2 -> 0``), so the Poisson term reduces to the
    plain ``poisson_loglike``. The penalty term ``(beta-1)^2 / (2 sigma2)``
    is a genuine ``0/0`` there, but its analytic limit is also 0
    (``(beta-1)^2 ~ sigma2^2``, so the ratio ``~ sigma2 -> 0``): both branches
    are simply forced to exactly the (poisson_loglike-equal) ``sigma2 = 0``
    values below a small relative-variance threshold, which is exact at
    ``sigma2 = 0`` and only ever discards an ``O(sigma2)`` correction for
    ``0 < sigma2 <= threshold`` (also where naive evaluation of ``beta_+``
    would lose precision to cancellation in ``-b + sqrt(D)``).

    Parameters
    ----------
    mu, w2, n : (n_bins,) f64 arrays -- MC expectation, MC sum-of-squared-
        weights, and observed data per bin.
    mc_floor : float -- floor applied to ``mu`` (as in `poisson_loglike`) and
        to the ``beta*mu`` argument of the Poisson log, both inside logs
        only.
    """
    mu = jnp.asarray(mu)
    w2 = jnp.asarray(w2)
    n = jnp.asarray(n)

    mu_floored = jnp.maximum(mu, mc_floor)
    sigma2 = w2 / mu_floored**2

    # Relative-MC-variance threshold below which sigma2 is treated as exactly
    # zero: pure-Poisson regime (see derivation above), and the regime where
    # -b + sqrt(D) would otherwise lose precision to cancellation.
    threshold = 1e-8
    small = sigma2 <= threshold

    # Discriminant is analytically >= 0; double-where guards sqrt's gradient
    # at D = 0 (the mango `_safe_sqrt` idiom) and floating round-off.
    sigma2_safe = jnp.where(small, 1.0, sigma2)  # avoid using ~0 in the quadratic
    b = mu_floored * sigma2_safe - 1.0
    disc = b * b + 4.0 * n * sigma2_safe
    disc_big = disc > 0.0
    sqrt_disc = jnp.where(disc_big, jnp.sqrt(jnp.where(disc_big, disc, 1.0)), 0.0)
    beta_raw = 0.5 * (-b + sqrt_disc)

    beta = jnp.where(small, 1.0, beta_raw)

    # Poisson term: floor beta*mu_floored inside the log only, same reasoning
    # as poisson_loglike (also covers beta -> 0 at n = 0, large sigma2).
    beta_mu_log = jnp.maximum(beta * mu_floored, mc_floor)
    poisson_term = n * jnp.log(beta_mu_log) - beta * mu

    # Gaussian penalty on beta; guarded to 0 in the sigma2 -> 0 limit.
    sigma2_div = jnp.where(small, 1.0, sigma2)
    penalty = jnp.where(small, 0.0, (beta - 1.0) ** 2 / (2.0 * sigma2_div))

    return jnp.sum(poisson_term - penalty)


def gaussian_cov_loglike(pred, data, cov_inv):
    """Correlated-bin Gaussian log-likelihood (the ND chi2 term):
    ``-0.5 * (pred - data)^T cov_inv (pred - data)``.

    Used for an ND sample in place of a per-bin Poisson term, the chi2 being
    computed with the ND bin covariance matrix: ``pred`` is the binned MC
    expectation ``lambda(theta)`` (``mu`` in this package's convention),
    ``data`` the observed/Asimov binned counts, ``cov_inv`` the precomputed
    inverse of the *absolute* bin covariance (see :func:`build_nd_covariance`
    -- built once, outside the hot loop, since re-inverting per MCMC step is
    the cost this approximation avoids).

    Pure array algebra (matmul, no sqrt/log/division), so no where-guarding
    is needed for gradient safety -- unlike the Poisson kernels above, this
    function's gradient is already smooth and finite everywhere ``cov_inv``
    is a finite matrix.

    Parameters
    ----------
    pred, data : (n_bins,) f64 arrays.
    cov_inv : (n_bins, n_bins) f64 array -- the precomputed covariance inverse.
    """
    pred = jnp.asarray(pred, dtype=jnp.float64)
    data = jnp.asarray(data, dtype=jnp.float64)
    cov_inv = jnp.asarray(cov_inv, dtype=jnp.float64)
    resid = pred - data
    return -0.5 * resid @ cov_inv @ resid


def build_nd_covariance(frac_cov, nominal_pred, *, add_mc_stat_var=None):
    """Build the absolute ND bin covariance (and its inverse) from a
    fractional covariance matrix and the nominal prediction: the fractional
    covariance ``frac_cov`` (from "many universes" throws of the ND detector
    systematics, dimensionless) is converted to an absolute covariance by

        C_ij = frac_cov_ij * nominal_i * nominal_j

    then the statistical uncertainty of the nominal prediction is added to
    the diagonal (an approximation to re-inverting the covariance at every
    MCMC step) -- Poisson variance
    ``nominal_i`` per bin. ``add_mc_stat_var`` optionally adds a further
    per-bin variance to the diagonal (e.g. MC-statistical variance of the
    nominal template itself, sum-of-squared-weights ``w2``, for a
    Barlow-Beeston-lite-style extra term) -- ``None`` skips this.

    This is a *build-time* function: runs once in plain numpy at float64
    (never traced/jitted -- inverting the fixed covariance is the costly
    step, done once rather than per MCMC step), returning JAX arrays for use in the hot
    :func:`gaussian_cov_loglike` path.

    Parameters
    ----------
    frac_cov : (n_bins, n_bins) array_like -- fractional covariance.
    nominal_pred : (n_bins,) array_like -- nominal (Asimov) binned prediction.
    add_mc_stat_var : (n_bins,) array_like or None -- extra diagonal variance.

    Returns
    -------
    (cov, cov_inv) : both ``(n_bins, n_bins)`` f64 JAX arrays.

    Raises
    ------
    ValueError if the resulting absolute covariance is not positive-definite
    (a Cholesky decomposition is attempted as the PD check, discarded, and
    ``np.linalg.inv`` used for the actual inverse -- symmetrized first since
    floating-point ``frac_cov_ij * nominal_i * nominal_j`` construction can
    introduce a tiny asymmetry from rounding).
    """
    frac_cov = np.asarray(frac_cov, dtype=np.float64)
    nominal_pred = np.asarray(nominal_pred, dtype=np.float64)
    n_bins = nominal_pred.shape[0]
    if frac_cov.shape != (n_bins, n_bins):
        raise ValueError(
            f"frac_cov has shape {frac_cov.shape}, expected ({n_bins}, {n_bins}) "
            "(= nominal_pred.shape[0] squared)"
        )

    cov = frac_cov * np.outer(nominal_pred, nominal_pred)
    cov = 0.5 * (cov + cov.T)  # symmetrize (guards float rounding asymmetry)

    diag_add = nominal_pred.copy()  # statistical uncertainty of the nominal prediction
    if add_mc_stat_var is not None:
        add_mc_stat_var = np.asarray(add_mc_stat_var, dtype=np.float64)
        if add_mc_stat_var.shape != (n_bins,):
            raise ValueError(
                f"add_mc_stat_var has shape {add_mc_stat_var.shape}, expected ({n_bins},)"
            )
        diag_add = diag_add + add_mc_stat_var
    cov[np.diag_indices(n_bins)] += diag_add

    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "ND covariance is not positive-definite after adding the "
            "statistical diagonal term -- check frac_cov and nominal_pred"
        ) from exc

    cov_inv = np.linalg.inv(cov)
    return jnp.asarray(cov, dtype=jnp.float64), jnp.asarray(cov_inv, dtype=jnp.float64)
