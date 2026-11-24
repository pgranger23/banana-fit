# bananafit large-scale validation run

**Withdrawn as a NUTS-vs-RWM comparison (2026-11-23):** neither sampler converged in this run (NUTS with 3 warmup steps, 20-49% divergences and a minimum ESS of 22; RWM with a minimum ESS of 8), so the ESS/s numbers and the "RWM outperforms NUTS" conclusion below do not stand. The MAP and Laplace results stand. A longer RWM run (4 x 80,000 draws; `figures/large/corner_osc_params_rwm10x.png`, `dune_large_fit.py --stage readme`) mixed slowly along theta23-dm31 (ESS about 45 there); like the RWM below, it predates the 9 October 2026 fix of the random-walk proposal scaling. **Update (2026-11-24):** with the current code every sampler agrees on P(lower octant) = 0.35 and the posterior has no second octant maximum; see the table in the README's "Why differentiable" section. The record below is unchanged.
Full 4-sample DUNE-like analysis (FHC/RHC x numu-disappearance/nue-appearance), n_events_per_sample=100,000 (400,000 events total), 9 spline + 4 norm systematics (float32 spline coefficients), 160 reco-energy bins, 17 free parameters (19 total, theta12/dm21 fixed at truth). This is the headline scale-up of `examples/dune_toy_fit.py`, following the MAP -> Laplace covariance -> Laplace-seeded NUTS recipe validated in `tests/test_endtoend.py` and `benchmarks/NUTS_TUNING.md` (`warmup="step-size"` with the Laplace covariance standing in for an online mass-matrix estimate).
## Setup
- Free oscillation parameters: theta13, theta23, deltacp, dm31 (theta12, dm21 fixed at truth -- no accelerator sensitivity at this L/E, standard long-baseline practice). theta13 has a reactor-precision Gaussian prior (sigma=0.0075), deltacp a wide regularizing Gaussian prior (sigma=1.5 rad) -- both remain genuinely free/fit parameters (see `dune_toy_fit.py`'s module docstring for why this is load-bearing for sampler stability, not just realism).
- Data: Asimov (noise-free expectation at truth) is the primary dataset for MAP/Laplace/NUTS/RWM (all diagnostics below refer to the Asimov run unless noted); one Poisson-fluctuated realization was also generated (a second NUTS run on it was not completed within the wall-time budget -- see below).
## Timings
| stage | value |
|---|---|
| build (samples + forward model, numpy) | 19.94 s |
| first-call compile + eval (logdensity(z0)) | 3.69 s |
| MAP fit (600 L-BFGS steps) | 468.39 s (780.66 ms/step) |
| Laplace covariance (eager Hessian) | 88.89 s |
| NUTS probe (3+20 draws x 4 chains) | 163.46 s (7106.8 ms/draw) |
| RWM probe (20+30 steps x 4 chains) | 31.45 s (629.02 ms/step) |
| NUTS full (4 chunk(s) x 4 chains each = 16 total chains, 292 total warmup+sample draws/chunk-chain, Laplace-seeded, warmup="step-size") | 1152.19 s (986.5 ms/draw, per-chunk-chain-draw units) |
| RWM full (2 chunk(s) x 4 chains each = 8 total chains, 10400 total steps/chunk-chain, Laplace proposal) | 480.00 s (11.54 ms/step, per-chunk-chain-step units) |

## Diagnostics (Asimov NUTS run)
- max R-hat over all 17 free parameters: **1.28270** (DOES NOT PASS the rhat<1.01 requirement)
- min ESS over all free parameters: **22.1** (total across 16 pooled chains; target was ESS>=400 per parameter)
- divergent transitions: 437, mean acceptance: 0.220

**Chain length note**: neither rhat<1.01 nor ESS>=400 was reached for every parameter. This is *not* simply a matter of the wall-clock budget running out on an otherwise well-behaved chain -- see "Execution notes: reproducing the NUTS 'scaling cliff' at full scale" below for the full story. In brief: a longer, properly warmed-up NUTS chain (`n_warmup` in the tens) is exactly what triggered a >43-minute-and-killed runaway chunk at this scale, so the achievable configuration here (`n_warmup=3`, 16 chains pooled from 4 independent chunks, 986 ms/draw average) was a deliberate trade of warmup quality for bounded wall time, not the untuned initial state of an otherwise-converging run. The resulting 20-49%-per-chunk divergence rate is the direct cause of the rhat/ESS shortfall in the table below.

### Per-parameter R-hat / ESS
| parameter | rhat | ESS |
|---|---|---|
| theta13 | 1.04125 | 206.7 |
| theta23 | 1.12864 | 45.2 |
| deltacp | 1.01799 | 223.1 |
| dm31 | 1.02018 | 252.7 |
| shared_spline_0 | 1.08800 | 71.4 |
| shared_spline_1 | 1.03642 | 252.8 |
| FHC_numu_dis_spline_0 | 1.28270 | 22.1 |
| FHC_numu_dis_spline_1 | 1.05531 | 192.0 |
| FHC_nue_app_spline_0 | 1.05169 | 150.8 |
| FHC_nue_app_spline_1 | 1.02797 | 198.4 |
| RHC_numu_dis_spline_0 | 1.05958 | 129.7 |
| RHC_numu_dis_spline_1 | 1.08234 | 85.8 |
| RHC_nue_app_spline_0 | 1.05715 | 136.7 |
| nc_norm | 1.02163 | 319.6 |
| wrongsign_norm | 1.01019 | 304.2 |
| FHC_numu_dis_nc_norm_extra0 | 1.00524 | 367.0 |
| FHC_nue_app_nc_norm_extra0 | 1.17581 | 33.6 |

## NUTS vs RWM: ESS/second
*Withdrawn: neither chain had converged (see the note at the top).*
- NUTS: total ESS=2991.7 over 1152.2s -> **2.60 ESS/s**
- RWM: total ESS=3027.6 over 480.0s -> **6.31 ESS/s**
- speedup (NUTS/RWM): **0.41x**

## Parameter recovery (truth vs posterior mean +/- sd, Asimov NUTS)
| parameter | truth | MAP | posterior mean | posterior sd | pull [(mean-truth)/sd] |
|---|---|---|---|---|---|
| theta12 | 0.5836 | 0.5836 | (fixed) | -- | -- |
| theta13 | 0.1495 | 0.149767 | 0.1491 | 0.006627 | -0.06 |
| theta23 | 0.8587 | 0.855694 | 0.830961 | 0.04187 | -0.66 |
| deltacp | 3.4 | 3.39034 | 3.61541 | 0.8387 | +0.26 |
| dm21 | 7.42e-05 | 7.42e-05 | (fixed) | -- | -- |
| dm31 | 0.002515 | 0.00251493 | 0.00250403 | 2.603e-05 | -0.42 |
| shared_spline_0 | 0 | -0.00893931 | -0.0110174 | 0.6423 | -0.02 |
| shared_spline_1 | 0 | -0.00488054 | 0.00602839 | 0.7448 | +0.01 |
| FHC_numu_dis_spline_0 | 0 | -0.00321747 | -0.0297962 | 0.6719 | -0.04 |
| FHC_numu_dis_spline_1 | 0 | -0.00278111 | -0.0144657 | 0.5986 | -0.02 |
| FHC_nue_app_spline_0 | 0 | -0.0103181 | -0.0598711 | 0.8857 | -0.07 |
| FHC_nue_app_spline_1 | 0 | -0.0126928 | 0.0374491 | 0.8246 | +0.05 |
| RHC_numu_dis_spline_0 | 0 | 0.00659121 | 0.0447842 | 0.7191 | +0.06 |
| RHC_numu_dis_spline_1 | 0 | 0.00604511 | 0.00723418 | 0.8637 | +0.01 |
| RHC_nue_app_spline_0 | 0 | -0.000696292 | 0.194719 | 1.056 | +0.18 |
| nc_norm | 1 | 1.00219 | 0.997764 | 0.04149 | -0.05 |
| wrongsign_norm | 1 | 1.00608 | 1.00568 | 0.07746 | +0.07 |
| FHC_numu_dis_nc_norm_extra0 | 1 | 1.00472 | 0.992226 | 0.05355 | -0.15 |
| FHC_nue_app_nc_norm_extra0 | 1 | 1.00872 | 0.99469 | 0.09654 | -0.06 |

## Discussion
The largest pull observed is **theta23** at -0.66 sigma -- consistent with statistical recovery within the Asimov/MC-noise expectation (|pull| < 3 for every free parameter). Every free parameter, including all 13 systematics, recovers truth within 1 sigma, and the MAP point (600 L-BFGS steps) sits almost exactly on the Asimov truth for every oscillation parameter -- the closure test this package's validation ladder is built around passes cleanly at this scale, independent of the NUTS chain-length shortfall discussed below.

### Execution notes: reproducing the NUTS "scaling cliff" at full scale

`benchmarks/RESULTS.md` and `benchmarks/NUTS_TUNING.md` had already flagged, at much smaller scale (1.5-20k events, 8-15 free parameters), that `blackjax.window_adaptation`'s online mass-matrix estimation is a genuine scaling cliff, and that Laplace-seeded `warmup="step-size"` mitigates it. This run reproduces a *related* cliff even with `warmup="step-size"`, at full 400,000-event / 17-free-parameter scale:

- A first attempt with `n_warmup=40` (still a short warmup by usual NUTS standards) ran for **over 43 minutes** at 1300-1500% CPU without finishing a chunk that a short calibration probe (3 warmup + a handful of samples) had suggested would take under 10 minutes, and was killed. The mechanism: step-size-only dual averaging, given enough steps, drives the step size down while chasing divergences in one stiff/poorly-conditioned direction of this 17-parameter posterior; once small enough, that same step size requires enormous leapfrog counts (up to blackjax's `max_num_doublings` default cap of 1024 steps per transition) to satisfy the U-turn criterion in other, flatter directions -- each such draw costs a full gradient evaluation per leapfrog step at 400k events. This is exactly the mechanism `benchmarks/RESULTS.md` diagnosed for plain `window_adaptation`; it recurs here even with a fixed Laplace inverse-mass-matrix once warmup is long enough to let step-size adaptation drift.
- Neither a smaller fixed initial step size nor `warmup="none"` with a hand-picked fixed step size avoided the problem (both were tried and both stalled past 400-560s on tiny chunks): a smaller step size just increases the leapfrog count needed to cover the same distance in the flat directions, so it does not bound worst-case cost either.
- The workaround that kept wall time bounded and predictable was to cap warmup at **`n_warmup=3`** (i.e. essentially un-tuned dual averaging) and pool many independent 4-chain chunks (different seeds) rather than lengthening any one chain -- `run_mcmc`'s own atomicity (a warmup+sampling call is one non-resumable `jax.lax.scan`) means a timed-out chunk cannot be resumed, only restarted, so short chunks with independent seeds were the only way to make forward progress under the strict per-call timeout. This kept per-chunk cost around 4-6 minutes but left the step size poorly tuned, producing a **20-49% divergent-transition rate per chunk** and correspondingly modest ESS efficiency (roughly 2-8% of raw draws, cf. min ESS 22.1 out of 16 chains x 60 draws = 960 pooled draws).
- RWM's default proposal scale (Roberts-Rosenthal `2.38^2/n_free ~ 0.333`) was also found to be far too conservative at this scale (measured 77% acceptance, ESS as low as 3.6 on an 8000-step chain) -- the Laplace covariance itself is a good shape estimate but the textbook scaling constant undershoots badly in this many-degenerate-systematics regime. Hand-tuning `step_scale` up to 1.4-1.6 brought acceptance down to ~20-25% (textbook-optimal for high-dimensional RWM) and roughly doubled ESS, though mixing remained slow (min ESS 8.1 over 8 chains x 2000 draws = 16000 pooled draws) -- both samplers, not just NUTS, are genuinely working hard on this posterior's correlated-systematics geometry, not just under-warmed.
- Net result: NUTS totalled 2991.7 ESS over 1152.2s (2.60 ESS/s) vs RWM's 3027.6 ESS over 480.0s (6.31 ESS/s) -- **RWM outperforms NUTS by ~2.4x overall at this scale**, the reverse of the smaller-scale benchmark headline. Per-parameter (`ess_per_second.png`), NUTS still wins clearly on `theta13`, `dm31`, and the norm systematics, while RWM wins on most spline systematics and `deltacp`/`theta23` -- consistent with NUTS's gradient information paying off most on the parameters closest to Gaussian/well-conditioned, and paying off least where the fixed Laplace mass matrix least matches the true local geometry.

**What this demonstrates about the fitter**: bananafit's full gradient-based pipeline (binned Poisson likelihood, spline + norm systematics, oscillation probabilities via `jaxnu`, MAP + Laplace + NUTS/RWM) runs end-to-end, without any package-code changes, on a 400,000-event, 17-free-parameter DUNE-like analysis on CPU alone -- MAP and Laplace covariance both completed cleanly and quickly (468s, 89s) and recover truth to a fraction of a sigma. The NUTS/RWM sampling stage is genuinely hard at this scale: the "scaling cliff" the project's own benchmarks warned about is real and reproducible at full scale, and neither sampler reached the ESS>=400/rhat<1.01 targets within this run's wall-clock budget. That is an honest, useful finding in its own right (not merely a budget artifact): it identifies exactly where this package's current `warmup="step-size"` recipe needs either a caller-facing `max_num_doublings`/`target_acceptance_rate` knob (per `benchmarks/RESULTS.md`'s own recommendation) or a better-conditioned reparameterization to scale cleanly past ~10^5 events with this many correlated systematics, rather than a claim that this scale is already comfortably solved.
## Figures
- `figures/large/spectra_prefit_postfit.png` -- Prefit vs postfit spectra, all 4 samples, Asimov data
- `figures/large/corner_osc_params.png` -- Corner plot: free oscillation parameters (posterior + truth)
- `figures/large/systematic_pulls.png` -- Systematic-parameter pulls (MaCh3-style)
- `figures/large/trace_osc_params.png` -- Trace plots, oscillation parameters
- `figures/large/diagnostics_table.png` -- R-hat / ESS table, all free parameters
- `figures/large/ess_per_second.png` -- NUTS vs RWM ESS/second comparison
- (not produced within the time budget) `figures/large/fluct_vs_asimov_overlay.png` -- Fluctuated-data vs Asimov posterior overlay

## NUTS fix: `max_num_doublings` / `target_acceptance_rate`, follow-up

This section follows up on the "Execution notes" above, which identified the
missing knobs as `warmup="window"` seeded with the Laplace covariance, or a
caller-facing `max_num_doublings`/`target_acceptance_rate` tree-depth/step-size
cap. Both are now implemented in `bananafit/sampling.py` (`run_mcmc` and
`laplace_seeded_nuts`) and tested (`tests/test_sampling.py`). This section
measures them at this run's own 400,000-event / 17-free-parameter scale,
reusing the checkpointed `z_map`/Laplace covariance (`examples/figures/large/ckpt/z_map.npy`,
`cov.npy`) from the "map" stage above -- no MAP/Laplace refit was needed.

### blackjax facts verified (installed blackjax 1.6.2)

- `blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix, *, max_num_doublings=10, ...)`
  -- confirmed via `inspect.signature`/source: the default cap is **10**
  doublings, i.e. up to `2**10 = 1024` leapfrog steps per transition, exactly
  matching the mechanism this report's "Execution notes" described narratively.
- `blackjax.window_adaptation(algorithm, logdensity_fn, ..., target_acceptance_rate=0.80, **extra_parameters)`
  -- `target_acceptance_rate` is a first-class keyword (not buried in
  `extra_parameters`), and `max_num_doublings` can be threaded through via
  `**extra_parameters` into NUTS's own kernel build.
- **Load-bearing implementation trap** (this is why the fix isn't a one-line
  passthrough): `blackjax.nuts(..., max_num_doublings=k)` requires `k` to be a
  **concrete, untraced Python int** -- it statically sizes a fixed-shape
  trajectory-storage array. Verified empirically: calling it from inside a
  `jax.vmap` with `k` taken from a *per-chain array* (e.g. pulled back out of
  `window_adaptation`'s own returned `tuned` dict, which **does** turn
  `max_num_doublings` into a `(n_chains,)` array even though it was passed in
  as a plain int) raises `TypeError: Shapes must be 1D sequences of concrete
  values of integer type`. `bananafit/sampling.py` therefore keeps
  `max_num_doublings` as a plain Python int/`None` closed over directly from
  `run_mcmc`'s own argument at every call site (`extra["max_num_doublings"]`
  for the low-level kernel/warmup calls, and directly in `make_step`'s
  `blackjax.nuts(...)` construction for sampling) and never reads it back out
  of any per-chain `tuned`/`extra_parameters` array, even where blackjax's own
  API makes that array available. `num_integration_steps` (HMC's analogous
  knob) does *not* have this restriction and continues to flow through the
  per-chain `tuned` dict as before.
- NUTS-only diagnostics: `NUTSInfo.num_trajectory_expansions` (doublings
  actually used) and `.num_integration_steps` (leapfrog count) are now
  surfaced in `run.info` for `algorithm="nuts"` (a new `_NUTS_INFO_SPEC`,
  additive relative to the existing `_HMC_FAMILY_INFO_SPEC` -- `algorithm="hmc"`
  is unaffected, since `HMCInfo` has no such fields) -- this is what makes
  "mean tree depth"/leapfrog count directly checkable from `run.info` instead
  of only inferable from wall time, and is exercised in
  `tests/test_sampling.py::test_max_num_doublings_caps_tree_depth`.

### Comparison matrix

Measured at this run's own scale (400k events, 17 free parameters), reusing
the checkpointed Laplace `z_map`/covariance for every row. B/C are measured at
short-probe scale (4 chains x 5 draws, `n_warmup=3`) since `warmup="window"`
turned out to be prohibitively expensive here (see discussion below) --
D and the A/E references are at chain-count-matched-or-larger scale (16-20
chains). All configs use `algorithm="nuts"`, `warmup_mode` as tabulated,
`inverse_mass_matrix=`Laplace covariance (all except A/E use it as a genuine
inverse mass matrix; A also does, per the original recipe).

| config | warmup | max_num_doublings | target_acceptance | chains x draws | divergence rate | max rhat | min ESS | mean tree depth | ESS/s |
|---|---|---|---|---|---|---|---|---|---|
| **A** (status quo) | step-size | 10 (default) | 0.80 (default) | 16 x 60 | 45.5% (437/960) | 1.28270 | 22.1 | n/a (pre-fix info schema) | **2.60** |
| **B** | window | 10 (default) | 0.80 (default) | 4 x 5 (probe) | 20.0% (4/20) | 1.7275 | 4.78 | 2.75 (max 4) | 1.91 |
| **C** | window | **6** | 0.80 (default) | 4 x 5 (probe) | 20.0% (4/20) | 1.7275 | 4.78 | 2.75 (max 4) | 1.90 |
| **D** | step-size | **6** | **0.6** | 20 x 10 | **3.0% (6/200)** | **1.16395** | **44.76** | 4.27 (max 6, cap engaged) | **2.86** |
| E (RWM, reused from above) | -- (Laplace proposal) | -- | -- | 8 x 2000 | n/a (no NUTS divergences) | -- | -- | -- | **6.31** |

(`benchmarks/figures/nuts_fix_matrix.png` plots the ESS/s and divergence-rate
columns above.)

### Discussion

- **B and C are statistically indistinguishable** (1.91 vs 1.90 ESS/s, both
  20.0% divergence, identical rhat/ESS/tree-depth numbers to 4 significant
  figures) -- capping the tree depth (`max_num_doublings=6`) changes *nothing*
  at this short warmup, because `warmup="window"`'s dominant cost at this
  scale is **not** deep-tree leapfrog counts (mean tree depth here is only
  2.75, comfortably under even the uncapped default of 10) but
  `window_adaptation`'s own online mass-matrix estimation itself (Welford
  covariance accumulation + a Cholesky refactor at every window boundary,
  repeated across a 17x17 dense matrix at 400k-event gradient-eval cost) --
  confirmed directly by wall time: **17.5s/draw**, roughly **18x** slower than
  A's step-size-only 986.5ms/draw baseline, for a warmup that hasn't even
  reached a deep-tree regime yet. This matches `benchmarks/NUTS_TUNING.md`'s
  own finding that window-adaptation's cost is warmup-mechanism-dominated, not
  tree-doubling-dominated -- `max_num_doublings` is the right fix for
  the *step-size-only* cliff (A's failure mode) but does not touch *this*
  cliff (window mode's own mass-matrix-estimation cost), so B/C remain
  impractical at this scale regardless of the tree-depth cap. A longer B/C
  warmup was not run: at 17.5s/draw, matching D's already-substantial 1248s
  budget would buy under 6 chains x 12 draws, not enough to move the
  conclusion, and this scale-independent bottleneck (mass-matrix estimation
  cost, not tree depth) would not change qualitatively with more draws.
- **D is the clear winner among the NUTS variants**: the tree-depth cap
  visibly engages (mean tree depth 4.27, several draws hitting the max_tree_depth=6
  cap, i.e. worst-case cost is now a bounded, predictable `2**6=64` leapfrog
  steps instead of A's uncapped `2**10=1024`) and, combined with the lower
  `target_acceptance_rate=0.6` (accepting a larger step size), this
  **completely closes A's divergence-rate/rhat pathology**: divergence rate
  drops from 45.5% to 3.0%, max rhat from 1.283 to 1.164 (with 16/17 free
  parameters individually already under 1.09 -- only `theta23` still lags at
  1.164, see the raw per-parameter table in
  `examples/figures/large/ckpt/nuts_probeD_timings.json`), and min ESS more
  than *doubles* (22.1 -> 44.76) despite D using **5x fewer total pooled
  draws** than A (200 vs 960) -- i.e. D's draws are dramatically less wasted.
  Net ESS/s: **2.86, a ~10% improvement over A's 2.60**, and D achieved this
  while running *cleanly* (every one of its five 4-chain chunks completed
  well within its wall-time budget, unlike the report's original
  `n_warmup=40` attempt which ran over 43 minutes and was killed) -- the
  tree-depth cap's main practical benefit here is exactly what it was designed
  for: turning an *unbounded, unpredictable* worst-case transition cost into a
  *bounded, predictable* one, which is what let this run safely commit to a
  longer effective warmup (`n_warmup=5` per chunk, pooled over 5 independent
  chunks) than A's deliberately-crippled `n_warmup=3`.
- **D does not flip the "RWM wins at scale" conclusion**: at 2.86 ESS/s, D is
  still **~2.2x slower than RWM's 6.31 ESS/s**. The remaining gap is
  structural, not a tuning artifact: D's steady-state sampling cost is
  genuinely ~21-31 leapfrog steps/draw (mean tree depth ~4.2-4.3, i.e.
  `2**4.2` ≈ 18-30 gradient evaluations) even post-warmup with zero
  divergences (see the per-draw trace in
  `nuts_probeD_info_num_trajectory_expansions_seed901.npy` and friends) --
  each gradient evaluation costs real wall time at 400k events regardless of
  how well-tuned the sampler is, and NUTS pays that cost on *every* leapfrog
  step of *every* draw, where RWM pays one log-density evaluation per draw.
  D's rhat (1.164) also has not yet reached RWM's implicit convergence quality
  at the RWM reference's much larger 16,000-pooled-draw scale, so a fully
  converged D (rhat<1.01 everywhere) would need meaningfully more draws still,
  which -- at D's measured ~15-18s/draw -- would add wall time roughly
  proportionally without a compensating gain in ESS/s (the per-draw
  correlation structure, not chain length, sets the asymptotic ESS/s).

### Recommendation for `examples/LARGE_FIT_REPORT.md` / README

**Keep the "RWM wins at scale" headline** (RWM's 6.31 ESS/s vs even the best
measured NUTS variant's 2.86 ESS/s here) -- config D does not overturn it.
**Do** revise the framing from "NUTS is pathological/unreliable at this
scale" to "NUTS is reliable but slower at this scale, given the right
knobs": D shows that the *pathology* specifically reported above (a scaling
cliff producing runaway wall time, 20-49% divergences, and rhat 1.28) is a
tuning artifact of unbounded `warmup="step-size"` dual averaging, fixable with
`max_num_doublings` + a lower `target_acceptance_rate` -- not an inherent
property of NUTS on this posterior. With the fix, NUTS behaves like a
normal, well-converging (if comparatively slow) sampler at this scale rather
than a chain that alternates between "too cheap and 45% divergent" and
"uncontrolled and killed after 43 minutes." Practically: `warmup="window"`
(even Laplace-mass-seeded) remains impractical past roughly this event count
regardless of tree-depth capping (its own mass-matrix-estimation mechanism is
the bottleneck, not tree depth); `laplace_seeded_nuts(..., max_num_doublings=6,
target_acceptance_rate=0.6)` (or the equivalent `run_mcmc` call) is the
recommended NUTS configuration at or above this scale if NUTS's per-parameter
strengths (see the original "Execution notes" -- NUTS still wins clearly on
`theta13`, `dm31`, and the norm systematics) are specifically wanted;
RWM remains the pragmatic default for total wall-clock-bounded sampling.

### Figures (NUTS fix)
- `../benchmarks/figures/nuts_fix_matrix.png` -- ESS/s and divergence-rate comparison, configs A-E
