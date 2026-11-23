# BANANA (`bananafit`) — design

**B**ayesian **A**utodiff **N**eutrino **A**nalysis & **N**uisance **A**ssessment:
a differentiable MCMC oscillation fitter in JAX.

Built on [MANGO](https://github.com/pgranger23/mango-osc) (`mango-osc` on PyPI, imported as `mango`). Takes a
**reconstructed MC event sample** + **spline-based systematic responses**
(event weight vs σ-variation of each systematic) and runs **efficient Bayesian
fits** of oscillation + nuisance parameters — gradient-based (NUTS/HMC/MALA via
blackjax, exploiting MANGO's exact gradients) and classic adaptive random-walk
Metropolis (MaCh3-style) with the *same* driver code.

The conceptual coverage follows MaCh3 (samplePDF → spline monolith → covariance
penalties → adaptive MCMC); the structure is functional: everything is a frozen
dataclass pytree + pure functions, composed at the top by ~10 lines of user code.

## 1. Design rules

- **Pure functions over frozen pytrees.** Every dataclass is
  `@jax.tree_util.register_dataclass` + `@dataclass(frozen=True)` (same pattern
  as `mango.OscParams`). Static metadata (names, shapes, bin counts) are
  `static` fields; arrays are leaves. No class owns a fit loop and a file handle
  and a covariance at once.
- **float64 mandatory** (set in `__init__.py`, same as MANGO). Exception: spline
  *coefficient tables* may be stored float32 (weights are O(1) smooth factors;
  memory is the binding constraint at 10⁶ events × 50 systematics), with the
  accumulation (histogramming, likelihood) always in float64.
- **The hot path is one jitted function**: `logdensity(z) -> scalar`, built once,
  closed over static data. Everything inside is dense fused array math — no
  Python branching on traced values, no per-event objects. Exception at
  multi-GB scale: XLA embeds closed-over arrays as constants in every
  executable, so the atmospheric per-event dial table must be a `jit`
  argument; closed over, it put 2.8 GB of constants in each executable and ran
  the V100 out of memory.
- **Precompute at build time, evaluate flat at run time** (as in MaCh3's
  spline monolith): spline coefficients, bin indices,
  channel slices, norm masks are all baked into arrays when the model is
  constructed; the per-step work is gather → Horner → product → segment_sum.
- **Gradient safety**: `where`-guard any `sqrt`/`log`/division that can hit its
  domain edge (MANGO's double-`where` pattern) so reverse-mode AD never sees NaN.
- **Reuse, don't rewrite**: oscillation physics = MANGO (`mango`); samplers + warmup +
  R-hat/ESS = `blackjax`; priors + bounded-parameter bijections =
  `numpyro.distributions`; posterior reporting = `arviz`; MAP pre-fit = `optax`.
  Hand-written (ecosystem gap, each ≲100 lines): batched Horner spline
  evaluator, Barlow-Beeston-lite likelihood, HDF5 schema.
- **Validation-first**: every module ships pytest tests; numerical kernels are
  validated against scipy / analytic references; the end-to-end fit against an
  Asimov closure test and the `mango.stat` Fisher matrix.

## 2. Module layout

```
bananafit/
  __init__.py      x64 enforcement + public re-exports
  events.py        EventSample struct-of-arrays pytree
  splines.py       spline coefficient tables + Horner eval
  norms.py         normalization systematics (masked multiplicative weights)
  shifts.py        FD reconstructed-energy shift systematics + differentiable binning
  parameters.py    ParamSpec/ParameterSet, transforms, priors
  likelihood.py    binned test statistics (Poisson, BB-lite)
  oscillation.py   per-event oscillation weights via MANGO
  model.py         ForwardModel: weights -> binned prediction
  posterior.py     logdensity assembly
  sampling.py      blackjax drivers, multi-chain, adaptation
  diagnostics.py   R-hat/ESS, arviz conversion, summaries
  io.py            HDF5 read/write of samples + splines
  root_import.py   ROOT TSpline3 import: MaCh3/T2K/DUNE spline-monolith compatibility
  globes_bridge.py bridge from mango.globes.GlobesExperiment to bananafit SampleModels
  toy.py           toy MC generator (DUNE-like)
tests/             one test file per module
examples/          end-to-end toy fit, docs
```

Dependencies: `jax`, `numpy`, `mango-osc` (MANGO), `blackjax>=1.6`, `numpyro>=0.21` (its
`distributions` only), `scipy` (spline building only, never at runtime),
`h5py`. Optional extras, as `pyproject.toml` lists them: `test` (`pytest`),
`report` (`arviz`, `matplotlib`), `extras` (`jax-tqdm`, `optax` for the MAP
pre-fit), `root` (`uproot`, ROOT TSpline3 import).

## 3. Data model (`events.py`)

One `EventSample` = one detection channel/selection (e.g. FHC νe-appearance).
Struct-of-arrays over N MC events:

```python
@register_dataclass @dataclass(frozen=True)
class EventSample:
    name:        str                   # static
    n_bins:      int                   # static: analysis bins for THIS sample
    true_energy: Array  # (N,) f64, GeV
    reco_bin:    Array  # (N,) int32 in [0, n_bins); out-of-window events get n_bins (overflow slot, dropped from likelihood)
    flavor_from: Array  # (N,) int32, MANGO's Flavor convention (0=e,1=mu,2=tau)
    flavor_to:   Array  # (N,) int32
    nominal_weight: Array  # (N,) f64: flux x xsec x eff x POT, all osc-independent factors
    anti_split:  int                   # static: events sorted so [0:anti_split) are neutrinos, [anti_split:N) antineutrinos
```

Constructor helper `make_sample(...)` does the sorting by `anti`, the reco-bin
digitization from bin edges (kept alongside as plain numpy metadata, not a
leaf), and validation (concrete inputs only — build time is not traced).
Multi-sample analyses are a `tuple[EventSample, ...]`; global bin offsets are
computed statically in `model.py`.

Rationale: `anti` as a static sorted split (not a per-event flag) lets the
oscillation adapter make exactly two MANGO calls (ν, ν̄) on contiguous slices
with static shapes — no doubled compute, no dynamic shapes.

## 4. Systematic response splines (`splines.py`)

MaCh3-equivalent of the spline monolith, JAX-native. A systematic's response is
a cubic spline through knots at σ-variations (typically −3…+3) giving a
multiplicative event weight. Key structural facts exploited:

1. The evaluation point x (the systematic's current value, in σ units) is **the
   same scalar for all events** → the active spline segment per systematic is a
   *scalar* `searchsorted`, then a single coefficient gather.
2. Many events share a response (splines produced per mode × kinematic bin) →
   dedup through an index map, exactly like MaCh3's monolith:

```python
@register_dataclass @dataclass(frozen=True)
class SplineTable:
    names:  tuple[str, ...]   # static, one per spline systematic (n_syst)
    knots:  Array  # (n_syst, n_knots) f64, sigma positions (NaN-padded if ragged)
    coeffs: Array  # (n_syst, n_curves, n_knots-1, 4) f32|f64, Horner coeffs per segment: w = ((a*t + b)*t + c)*t + d, t = x - knot[k]
    event_index: Array  # (n_syst, N) int32 -> curve index per event; curve 0 is ALWAYS the identity response (w=1) so "systematic doesn't affect event" is index 0
    def __call__(self, x: Array) -> Array:  # x: (n_syst,) -> weights (N,) = prod over systs
```

- Builder `build_spline_table(knot_x, knot_y, event_index, *, boundary="natural", dtype=float32)`
  precomputes coefficients with **numpy/scipy at build time** (`scipy.interpolate.CubicSpline`
  natural BC = ROOT `TSpline3` with explicit `"b2e2"` — NOT ROOT's default
  constructor boundary, see §11.1; also `"not-a-knot"`, `"monotone"` via PCHIP).
- Outside the knot range: **linear continuation** of the boundary segment
  (the library default; NOT what TSpline3/MaCh3 do -- they evaluate the end-segment cubic, `extrapolation="cubic"`, with
  `boundary="not-a-knot"` = ROOT's default coefficients), plus optional weight clamp
  `w in [0, w_cap]` applied at eval (MaCh3's knot-cap convention).
- Runtime eval per systematic: scalar segment lookup → gather `(n_curves, 4)`
  slab → Horner at scalar t → gather per event via `event_index` → running
  product in f64. Weights of all systs multiply into `(N,)`.
- Per-event splines are the degenerate case `event_index[s] = 1..N` (curve per
  event); toy-scale (10⁵ events × 20 systs) fits comfortably in memory. At
  analysis scale it does not fit as a closed-over constant: a per-event dial
  table of ~8×10⁷ (event, dial) pairs (2.3 GB of float32 knots) has to be
  passed to `jit` as an argument.

`io.py` stores the table verbatim (HDF5) — the HDF5 schema is the native format;
the optional ROOT `TSpline3` importer (`root_import.py`, via uproot) is
described in §11.1.

## 5. Normalization & functional systematics

Norm parameters (flat multiplicative weight on a selection of events) are just
1-knot responses; but they deserve the cheap path:

```python
@register_dataclass @dataclass(frozen=True)
class NormSet:
    names: tuple[str, ...]        # static (n_norm)
    mask:  Array   # (n_norm, N) f32 in {0,1} (or float response fractions)
    def __call__(self, x: Array) -> Array:  # (n_norm,) -> (N,): prod_n (1 + (x_n - 1) * mask_n)
```

with x_n having prior mean 1. Functional systematics (arbitrary user function
`(x, sample) -> (N,) weights`) are supported at the `model.py` level as a plain
callable hook — no framework needed.

## 6. Parameters, transforms, priors (`parameters.py`)

The single source of truth for "what is being fit" — the role MaCh3's
covariance objects play:

```python
@dataclass(frozen=True)            # plain dataclass, NOT a pytree (all static)
class ParamSpec:
    name: str
    init: float                    # nominal / chain start
    prior: Prior                   # Gaussian(mu, sigma) | Flat() | none: see below
    bounds: tuple[float|None, float|None] = (None, None)
    fixed: bool = False
    circular: bool = False         # deltacp

class ParameterSet:                # ordered collection of ParamSpec
    # construction: ParameterSet(specs) or ParameterSet.from_yaml(...) later
    free_names, n_free, ...
    def unconstrain(self, values: dict|Array) -> Array   # physical -> z (free params only)
    def constrain(self, z: Array) -> Array               # z -> physical vector (fixed params reinserted as constants)
    def log_prior(self, physical: Array) -> Array        # scalar, includes correlated blocks
    def log_det_jacobian(self, z: Array) -> Array        # of constrain, for sampling in z
```

- Bounded params use `numpyro.distributions.transforms.biject_to(interval/positive)`;
  unbounded are identity; `circular` params are sampled on ℝ and wrapped mod 2π
  only at reporting time (the posterior is periodic — valid and NUTS-friendly).
- Priors: `Gaussian(mu, sigma)`, `Flat()` (improper, contributes 0),
  and a **correlated block** `MVGaussian(names, mean, cov)` attached at the
  `ParameterSet` level (the MaCh3 covariance-matrix penalty, e.g. a flux+xsec
  covariance from a near-detector fit). Implemented with
  `numpyro.distributions` logpdfs.
- Fixed params are dropped from z entirely (never proposed), reinserted as
  constants in `constrain` — no sign-encoded state.
- Convention: oscillation parameters appear by their `OscParams` field names
  (`theta12, theta13, theta23, deltacp, dm21, dm31`, radians / eV²); a helper
  `osc_param_specs(nufit=..., ordering=...)` returns the standard six with
  NuFIT-like priors/bounds. `posterior.py` splits the physical vector into
  `OscParams` + systematic vector by name.

## 7. Forward model (`model.py`)

```python
@register_dataclass @dataclass(frozen=True)
class SampleModel:
    events:  EventSample
    splines: SplineTable | None
    norms:   NormSet | None
    osc:     OscConfig            # see oscillation.py
    extra_weight: Callable | None # static: functional-systematics hook (x_dict, events) -> (N,)

@register_dataclass @dataclass(frozen=True)
class ForwardModel:
    samples: tuple[SampleModel, ...]
    # static: global bin offsets; total_bins
    def expected(self, osc: OscParams, sys: Array) -> tuple[Array, Array]:
        # per sample: w = nominal * P_osc(osc) * splines(x_spl) * norms(x_norm) [* extra]
        # mu  = segment_sum(w,  reco_bin, num_segments=n_bins+1)[:-1]   (overflow slot dropped)
        # w2  = segment_sum(w*w, ...)[:-1]                              (for BB-lite)
        # returns concatenated (total_bins,) mu and w2
```

`sys` is the full systematic sub-vector; each `SampleModel` picks its named
entries via static index arrays computed at build time (systematics are shared
across samples by name — one θ_MAQE moves all samples coherently).

## 8. Oscillation weights (`oscillation.py`)

```python
@register_dataclass @dataclass(frozen=True)
class ConstantMatterOsc:          # long-baseline: DUNE/T2K-like
    baseline_km: float; density_gcc: float; ye: float = 0.5   # static leaves? -> leaves (differentiable is free)
    def weights(self, params: OscParams, sample: EventSample) -> Array  # (N,)

class EarthOsc: ...               # atmospheric: probability_earth over (E, cz); needs cos_zenith in EventSample extras — v2, interface reserved
```

Implementation: two calls to `mango.probability_constant(params, E_slice, ...)`
(ν slice with `anti=False`, ν̄ slice with `anti=True`; `backend="nufast"`,
auto-fallback handles NSI/steriles) returning `(n, 3, 3)`, then one
`P[i, flavor_to[i], flavor_from[i]]` gather. Events with
`flavor_from == flavor_to == -1` (e.g. intrinsic/NC backgrounds treated as
unoscillated) get weight 1 via a `where`. Fully differentiable w.r.t. `OscParams`
*and* baseline/density (free matter-density systematics come for free).

`EarthOsc` still raises `NotImplementedError`. Zenith-binned (atmospheric)
weighting is done outside the package, with MANGO (PREM layers,
production-height averaging, sub-sampling within bins), and passed in as
per-event weights.

## 9. Likelihood (`likelihood.py`)

Pure functions on `(mu, w2, data)` arrays, all returning **log-likelihood**
(not χ²; MCMC wants logs — `mango.stat.poisson_chi2` stays for frequentist use):

- `poisson_loglike(mu, n)` — `n·log μ − μ` (constants dropped), `where`-guarded
  at μ→0 with the MaCh3-style low-MC floor.
- `barlow_beeston_lite_loglike(mu, w2, n)` — Conway/BB-lite: per-bin scale β
  from the closed-form quadratic `β² + (μσ²−1)β − nσ² = 0` with `σ² = w2/μ²`,
  Poisson in βμ + Gaussian penalty in β. Reference formulas: Conway
  (arXiv:1103.0354), pyhf `staterror`, MaCh3 `GetTestStatLLH`. Validated
  against an independent numerical profile over β.
- (reserved) `icecube_loglike` — SAY effective-count likelihood, later.

Total posterior (assembled in `posterior.py`):
`logpost(z) = loglike(mu(constrain(z)), data) + log_prior(constrain(z)) + log_det_jacobian(z)`.

## 10. Sampling (`sampling.py`)

Thin, uniform drivers over blackjax — one contract for gradient and
non-gradient kernels (MaCh3 parity = `"rwm"`; API default = `"nuts"`). Which
one wins depends on the posterior: a gradient sampler is not automatically
faster, and a random walk with a Hessian (Laplace) proposal and an octant hop
is a strong baseline. Gradients are used for MAP/profile fits, Laplace
covariances and NUTS. The toy comparison in `benchmarks/NUTS_TUNING.md` that
ranked seeded NUTS above RWM is withdrawn (note at its top).

```python
run_mcmc(logdensity, init: Array | (chains, dim), *, algorithm="nuts",  # "nuts"|"hmc"|"mala"|"rwm"
         n_chains=4, n_warmup=1000, n_samples=2000, seed=0,
         init_jitter=..., progress=True, thinning=1) -> MCMCRun

@dataclass class MCMCRun:
    positions: (n_chains, n_samples, n_free)   # z-space
    log_prob:  (n_chains, n_samples)
    info: ...                                   # acceptance, divergences
    tuned: ...                                  # step size / mass matrix / rwm covariance
```

- **nuts/hmc/mala**: `blackjax.window_adaptation` per chain (vmapped), then
  vmapped `lax.scan` over the tuned kernel. Divergence count surfaced loudly.
- **rwm**: `blackjax.additive_step_random_walk` with correlated Gaussian
  proposal; proposal covariance from (in order of preference) a user-supplied
  matrix, a MAP+Fisher estimate, or a Haario-style adaptation phase; scaled
  `2.38²/d` (Roberts-Rosenthal), acceptance-rate sanity warnings (MaCh3's
  <1%/>50% checks).
- `map_fit(logdensity, init)` via optax L-BFGS/Adam — chain initialization and
  the Fisher/Laplace proposal seed.
- Multi-chain is `vmap`, multi-device later via sharding (structure ready).
- `diagnostics.py`: `blackjax.diagnostics` R-hat/ESS on the fly;
  `to_arviz(run, parameter_set) -> InferenceData` (physical space, circular
  wrap applied, correct names) for corner/trace plots and NetCDF persistence.

## 11. I/O (`io.py`) — HDF5 schema, version-tagged

```
/format_version = 1
/samples/<name>/
    true_energy, reco_bin, flavor_from, flavor_to, nominal_weight  (N,)
    attrs: n_bins, anti_split, bin_edges, baseline_km, density_gcc, ye
    splines/  names, knots (n_syst,k), coeffs (n_syst,c,k-1,4), event_index (n_syst,N)
    norms/    names, mask (n_norm, N)
/parameters/  (optional) specs serialized as YAML string attr
```

`save_analysis(path, samples&models)` / `load_analysis(path)` round-trip
everything needed for a fit. Splines may alternatively be stored as raw knot
y-values (`knot_y (n_syst, c, k)`) and rebuilt at load — both supported, raw
knots preferred for portability (coefficients are an implementation detail).

### 11.1 ROOT `TSpline3` import (`root_import.py`)

MaCh3/T2K/DUNE-convention spline files (one `TSpline3` per systematic, or per
(mode, kinematic-bin) curve, stored as `TKey`s scattered across a ROOT file's
directory tree) import via `uproot` — no ROOT/PyROOT dependency at runtime.
`uproot` cannot *write* `TSpline3` (not in its writable-class allowlist), so
this pathway is one-directional: real MaCh3 files in, `SplineTable` out; test
fixtures are real ROOT files written with PyROOT (available for tests, not a
runtime dependency).

- `read_tspline3_knots(path, *, name_parser=None) -> {syst: {curve_key: (knot_x, knot_y)}}`
  walks every `TDirectory`, finds `TSpline3` objects, and extracts knots from
  each one's `fPoly` array (`TSplinePoly3` sub-objects with scalar `fX`/`fY`
  members — `TSpline3` has no flat `fX`/`fY` arrays of its own). The object's
  *file key name* (not `TObject::GetName()`, which is often uniformly
  `"Spline3"`) and directory path are parsed via `name_parser(objname,
  dirpath) -> (syst_name, curve_key)`; default
  `default_name_parser` splits on `"_"` (first token = systematic, remaining
  int-or-str tokens = curve key, e.g. `"MAQE_0_3" -> ("MAQE", (0, 3))`).
  Only the file's own fitted `TSpline3` coefficients (`fB`/`fC`/`fD`) are
  never trusted or read — only knot positions/values.
- `spline_table_from_root(path, event_curve_map, *, boundary="natural", dtype=float32, name_parser=None) -> SplineTable`
  glues the above into `build_spline_table`: `event_curve_map` gives, per
  systematic, one curve key (or `None` for "unaffected" → identity, index 0)
  per event; coefficients are always refit locally at `boundary` (never
  read from the file), so the source file's own boundary condition is
  irrelevant to the result. All curves of one systematic must share a knot
  grid (`build_spline_table` stores one knot row per systematic); ragged
  knot counts across *different* systematics use the usual NaN-padding.

**Important boundary-condition finding** (from cross-checking against real
PyROOT `TSpline3.Eval`, ROOT 6.40): ROOT's *default* `TSpline3` construction
(no explicit `opt`) is **not** a natural spline — it disagreed with
`scipy.interpolate.CubicSpline(..., bc_type="natural")` by O(0.01–0.1) near
the knot-grid ends on a 7-knot test case. Explicit natural boundaries at both
ends (`TSpline3(..., "b2e2", 0, 0)`) reproduced `scipy`'s natural spline to
~1e-15 in-range. Since `spline_table_from_root` never trusts the source
file's fitted coefficients (only its knots), this is not a correctness bug
for the importer — but it means "natural" is a `bananafit`-side choice
applied uniformly on import, which may differ from whatever boundary
condition originally produced a given MaCh3 file's `TSpline3` objects near
the first/last knot. Separately, outside the knot range `SplineTable` linearly
continues the boundary segment (DESIGN.md Section 4, gradient-safety),
whereas `TSpline3::Eval` extrapolates with the full boundary cubic — these
diverge substantially a few knots out. The library default
(`extrapolation="linear"`) avoids the curving-away cubic; `extrapolation="cubic"`
with `boundary="not-a-knot"` reproduces `TSpline3` (§4). The beam analysis uses
the latter to match MaCh3.

## 12. Testing & validation ladder

1. Unit: splines vs `scipy.interpolate.CubicSpline` (values + derivatives,
   in/out of range); likelihoods vs analytic/pyhf-formula references; transforms
   round-trip + jacobian vs `jax.jacobian`; oscillation adapter vs direct MANGO calls.
2. Gradient checks: `jax.grad(logdensity)` vs central finite differences at
   several points (the MANGO convention).
3. Statistical: sampler drivers on analytic targets (correlated Gaussian,
   Rosenbrock) — R-hat < 1.01, ESS sane, moments right.
4. End-to-end: toy DUNE-like Asimov closure (posterior mode = truth), posterior
   width vs `mango.stat.fisher_matrix` Laplace approximation, NUTS vs RWM
   agreement on the same posterior.

## 13. Deliberate non-goals (v1)

Atmospheric zenith-binned oscillation weighting (interface reserved), PCA of
the systematic space, mass-ordering trans-dimensional jumps (run NO/IO as two
chains and Bayes-factor them), multi-device sharding, simulated annealing.
(ROOT `TSpline3` import was originally listed here; it is implemented, §11.1.)

Update (2026-11-05): two of these can be done outside the package.
Zenith-binned weighting uses MANGO directly; the package's `EarthOsc`
still raises (§8). Guarded mass-ordering and octant translation jumps are
built by the caller and passed to `run_mcmc` as `hop_move_fn` moves; `bananafit/sampling.py` has no
ordering move of its own.
