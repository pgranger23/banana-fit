"""Systematic-response spline monolith (DESIGN.md §4).

A systematic's event-weight response is a cubic spline through knots at
sigma-variations of that systematic (typically -3..+3): ``w(x) = S(x)``, one
curve per (mode x kinematic bin) or per event. The structural facts this
module exploits, mirroring MaCh3's spline monolith but flattened into dense
JAX arrays:

1. The evaluation point ``x`` (systematic's current MCMC value, in sigma
   units) is the *same scalar* for every event -> per systematic, finding the
   active segment is a *scalar* ``searchsorted``, not a per-event one.
2. Many events share one response curve (splines are produced per mode x
   kinematic bin, not per event) -> a ``(n_syst, N)`` integer index array
   dedups curves across events, exactly like MaCh3's monolith. Curve index 0
   is *always* the identity response (``w == 1``): "this systematic doesn't
   affect this event" is just ``event_index == 0``, with no branching.

Everything expensive (the linear solve for the spline coefficients) happens
once at *build* time with ``scipy.interpolate`` on plain numpy arrays. The
runtime hot path (``SplineTable.__call__``, evaluated on every MCMC step) is
pure gather + Horner + gather + product:

    per systematic s:  seg = clip(searchsorted(knots[s], x[s]) - 1)   # scalar
                        slab = coeffs[s, :, seg, :]                    # (n_curves, 4)
                        w[s, :] = Horner(slab, x[s] - knots[s, seg])   # (n_curves,)
    per event:          gather w[s, event_index[s, n]] for each s, then
                         multiply across s (accumulating in float64)

Coefficients are stored in Horner form ``w = ((a*t + b)*t + c)*t + d`` with
``t = x - knot[seg]``, matching ``scipy``'s ``PPoly`` convention exactly (its
``.c[k, i]`` is the coefficient of ``(x - x_i)**(3-k)`` on segment ``i``), so
building the table is a transpose, not a reformulation.

Memory, not compute, is the constraint at ~1e6 events x ~50 systematics: the
coefficient table is ``(n_syst, n_curves, n_knots-1, 4)`` and is stored in
``dtype`` (default ``float32``) for that reason -- weights are O(1) smooth
multiplicative factors, so float32 in the *table* costs nothing perceptible
in the final likelihood as long as the per-event product that consumes it
accumulates in float64 (which ``SplineTable.__call__`` does). Pass
``dtype=jnp.float64`` to ``build_spline_table`` for a full float64 table
(used by the gradient tests here, where float32 rounding would swamp a
central-difference comparison).

Outside the knot range there are two conventions, selected by ``extrapolation``
(``build_spline_table`` / ``build_spline_table_factorized``, stored on the table):

* ``"linear"`` (the library default, unchanged): continue linearly using the
  *value and slope at the boundary knot*. The linear form shares its value and
  first derivative with the in-range cubic exactly at the boundary (by
  construction -- see ``_eval_all_curves``), so the stitched function is C1 across
  the boundary and ``jax.grad`` is finite there.
* ``"cubic"``: keep evaluating the cubic of the end segment (``seg`` clamped to
  ``[0, n_real - 2]``, no ``where`` at all), so the function is a single
  polynomial across the boundary (C-infinity there, gradient finite).

**``"linear"`` is not the MaCh3/ROOT ``TSpline3`` convention.** MaCh3 evaluates ``y + dx (b + dx (c + d dx))`` with the coefficients of
the end segment for any ``x`` (``TSpline3_red::FindX``/``Eval``,
``MaCh3/Splines/SplineStructs.h:674-720``: ``if (segment >= nPoints-1) segment =
nPoints-2``; ``SplineBase::FindSplineSegment``, ``SplineBase.cpp:97``), i.e.
``extrapolation="cubic"``; together with ``boundary="not-a-knot"`` (ROOT's
``TSpline3`` coefficients) that reproduces the stored ``TSpline3`` objects to ~1e-15
inside and outside the knots. Cubic
extrapolation can go negative far outside the knots; MaCh3 does not floor an
individual spline weight in v >= 2.7.0 (it drops an event whose *total* weight is <= 0,
``SampleHandlerBase.cpp``), which is what ``weight_clip=(0, None)`` on the per-event
product reproduces here; v <= 2.6.1 additionally floors every spline weight at 0
(``BinnedSplineHandler::CalcSplineWeights``), reproduced by ``curve_floor=0.0``. The two differ only for
an event with two (or any even number of) simultaneously negative curves.

Factorized (grid-cell) index mode
------------------------------------------------------------------------
The dense ``event_index`` above is ``(n_syst, N)`` -- fine at FD scale
(~1e6 events) but prohibitive at ND scale (~1e7-1e8 events x ~55 systematics
x 4 bytes = tens of GB) when, as with the ND binned splines, curves are
*already* shared by every event landing in the same ``(mode, Etrue-bin,
Erec-bin, Yrec-bin)`` grid cell: the per-systematic dedup ``event_index``
does no *additional* work beyond re-deriving, per systematic, which curve
that shared cell maps to. :class:`SplineTable` supports a second,
mutually-exclusive representation that factors this out explicitly:

- ``event_cell``: ``(N,)`` int32 -- one small grid-cell id per event,
  shared across *all* systematics (this is the expensive-to-avoid,
  per-event-sized array, but only ``O(N)``, not ``O(n_syst * N)``).
- ``cell_to_curve``: ``(n_syst, n_cells)`` int32 -- the small per-systematic
  lookup from grid cell to curve index (``n_cells`` is the total number of
  occupied ``(mode, i, j, k)`` cells, typically hundreds-to-low-thousands --
  nowhere near ``N``).

At call time the effective per-(syst, event) curve index is reconstructed
with one cheap double gather, ``cell_to_curve[:, event_cell]`` (shape
``(n_syst, n_cells)`` gathered by a length-``N`` index -> ``(n_syst, N)``),
then fed into the exact same ``_curve_gather`` custom-VJP path as the dense
``event_index`` case -- the curve-0/identity-collision-dropping backward
pass (see ``_curve_gather_bwd``) is untouched and applies identically,
since by the time it runs the two representations have already been
unified into the same ``(n_syst, N)`` shape. Memory savings are in *storage*
(on disk, and as a persistent field of the built ``SplineTable``/pytree),
not in the transient compute buffer XLA allocates for that one gather at
call time -- exactly like the dense path already does for
``_eval_all_curves``'s output.

A :class:`SplineTable` has either ``event_index`` set (dense) or both
``event_cell`` and ``cell_to_curve`` set (factorized), never a mix -- built
via :func:`build_spline_table` (dense) or
:func:`build_spline_table_factorized` (factorized) respectively; the
unused representation's field(s) are left ``None`` (a valid, leaf-free JAX
pytree value, so ``jax.jit``/``vmap`` see a consistent static structure per
built table without any special-casing at the call site).

Factorized hot path: cell-level product, never an ``(n_syst, N)`` buffer
------------------------------------------------------------------------
The paragraph above (and the original factorized implementation) noted that
the memory savings of ``event_cell``/``cell_to_curve`` were in *storage*
only -- ``__call__`` used to reconstruct the dense ``(n_syst, N)`` index via
``cell_to_curve[:, event_cell]`` and feed it through the same
``_curve_gather`` path as the dense representation, so the transient compute
buffer was exactly as large as the dense case's (tens of GB at ND's
N~5e7-scale, which made the gradient pathologically slow and ran out of
memory at that scale). That is no
longer true: :func:`_factorized_event_weight` (used by ``__call__`` whenever
``event_cell is not None``) restructures the computation so that **neither
the forward nor the backward pass ever materializes an ``(n_syst, N)``-shaped
array**, exploiting the fact that the whole point of the factorized
representation is that ``event_cell`` (not the per-systematic curve index)
is what is actually shared across all events landing in the same grid cell:

1. Gather each systematic's per-*cell* value first, at the *small* scale --
   ``curve_val[s, c] = per_curve[s, cell_to_curve[s, c]]``, shape
   ``(n_syst, n_cells)`` -- via the existing ``_curve_gather`` custom-VJP
   (curve-0 dropping and all, unchanged; here it runs at ``n_cells`` scale,
   nowhere near the pathological regime it was built to avoid).
2. Take the product **across systematics** at cell scale:
   ``W[c] = prod_s curve_val[s, c]``, shape ``(n_cells,)``.
3. Gather out to event scale **once**, after the cross-systematic product
   is already done: ``w[n] = W[event_cell[n]]``, shape ``(N,)`` -- the
   *only* ``O(N)``-sized array this path ever touches (besides its own
   output), forward or backward.

The backward pass mirrors this exactly (see
:func:`_cell_event_product_bwd`'s docstring for the full derivation): the
cotangent flowing back through step 3 is summed by cell with **one**
``segment_sum(g, event_cell, num_segments=n_cells)`` -- a single ``N ->
n_cells`` scatter, shared across all 55 systematics because ``event_cell``
does not depend on which systematic is being differentiated -- and
everything after that (undoing the cross-systematic product, then
``_curve_gather``'s own already-cheap backward) operates at
``(n_syst, n_cells)`` scale, not ``(n_syst, N)``. This is the "1 moderate
scatter + tiny gathers" restructuring of what used to be 55 simultaneous
huge, skewed scatters -- mathematically exact (the cell-level product and
the per-event gather commute with the systematic product because every
event in a cell shares the *same* curve per systematic, by construction of
the factorized representation), not an approximation.

``weights_per_syst`` (the debugging per-systematic breakdown) still goes
through the dense reconstruction for the factorized path -- it is not the
hot path (never called by ``ForwardModel.expected``/``__call__``), and
returning a per-systematic ``(n_syst, N)`` breakdown is inherently
``O(n_syst * N)``-sized output regardless of representation, so there is no
memory-frugal way to serve that particular (diagnostic-only) contract.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

_BOUNDARIES = ("natural", "not-a-knot", "monotone")
_EXTRAPOLATIONS = ("linear", "cubic")


@jax.custom_vjp
def _curve_gather(per_curve: jax.Array, event_index: jax.Array) -> jax.Array:
    """``gathered[s, n] = per_curve[s, event_index[s, n]]`` -- the per-event
    curve lookup at the heart of :meth:`SplineTable.weights_per_syst`, split
    out as its own ``custom_vjp`` so the (expensive) backward pass can skip
    curve index 0 (see :func:`_curve_gather_bwd`).

    Forward is exactly the plain two-index gather this replaces; only the
    backward differs from JAX's own default gather VJP.
    """
    rows = jnp.arange(per_curve.shape[0])[:, None]
    return per_curve[rows, event_index]


def _curve_gather_fwd(per_curve: jax.Array, event_index: jax.Array):
    return _curve_gather(per_curve, event_index), (event_index, per_curve.shape[1])


def _curve_gather_bwd(res, g):
    """``d(per_curve)[s, c] = sum_{n: event_index[s, n] == c} g[s, n]``, a
    per-systematic ``segment_sum`` -- with curve index 0 (the identity
    response) *dropped* from the reduction, not merely zeroed.

    Why dropping (not just zeroing) curve 0 is exactly correct, not an
    approximation: :func:`build_spline_table` always gives curve 0 the exact
    coefficients ``a=b=c=0, d=1`` on *every* segment (see its docstring), so
    ``_eval_all_curves``'s Horner formula ``((a*t+b)*t+c)*t+d`` returns
    *exactly* 1.0 for curve 0 at every ``x`` -- both the in-range branch and
    the linear-continuation branches (built from the same all-zero/one
    coefficients). ``d(per_curve[s, 0])/dx`` is therefore identically zero,
    so whatever value ``d(per_curve)[s, 0]`` (this function's output) takes,
    the next step of the chain rule (back through ``_eval_all_curves``)
    multiplies it by that exact zero -- the final gradient w.r.t. ``x`` is
    unaffected by what is written into bucket 0.

    Why this matters for *performance*, not just as a harmless
    simplification: real ``event_index`` rows are heavily skewed toward
    curve 0 ("this systematic doesn't affect this event", ``splines.py``
    module docstring) -- empirically 6%-99.9% of events per systematic in
    a realistic far-detector analysis (median ~72%), i.e. curve 0 is
    frequently the target of hundreds of thousands of simultaneous
    scatter-add collisions out of ~440k events in a sample. Redirecting
    those entries to an out-of-bounds segment id (``n_curves``, one past the
    last real curve) makes ``jax.ops.segment_sum``'s default out-of-range
    handling *drop* them from the reduction outright -- removing them from
    the scatter-add's collision/atomic-update workload entirely, not just
    zeroing their contribution (a plain ``jnp.where``-based zero-out would
    still perform the same number of colliding atomic updates and gain
    nothing). This is the fix for a measured pathology:
    jax/XLA's scatter-add lowering for a reduction this skewed measured
    catastrophically slow under reverse-mode AD (~180-200s/call on Lyon's
    V100, jax/jaxlib 0.11.0) despite the *forward* gather and a uniformly-
    random-index synthetic benchmark of the same shape being fast in
    isolation.
    """
    event_index, n_curves = res
    dropped_index = jnp.where(event_index == 0, n_curves, event_index)
    d_per_curve = jax.vmap(
        lambda gi, idxi: jax.ops.segment_sum(gi, idxi, num_segments=n_curves)
    )(g, dropped_index)
    return d_per_curve, None


_curve_gather.defvjp(_curve_gather_fwd, _curve_gather_bwd)


@jax.custom_vjp
def _cell_event_product(curve_val: jax.Array, event_cell: jax.Array) -> jax.Array:
    """``w[n] = prod_s(curve_val[:, event_cell[n]])`` -- cross-systematic
    product at *cell* scale, then one gather out to *event* scale.

    ``curve_val``: ``(n_syst, n_cells)`` -- each systematic's response value
    for each grid cell (already gathered from that systematic's curve table,
    see :func:`_factorized_event_weight`). ``event_cell``: ``(N,)`` int, the
    per-event grid-cell id shared by every systematic.

    Forward touches exactly one ``(N,)``-sized array (the output); no
    ``(n_syst, N)`` intermediate is ever formed -- see
    :func:`_cell_event_product_bwd` for why the backward pass has the same
    property.
    """
    W = jnp.prod(curve_val, axis=0)  # (n_cells,)
    return W[event_cell]  # (N,)


def _cell_event_product_fwd(curve_val: jax.Array, event_cell: jax.Array):
    return _cell_event_product(curve_val, event_cell), (curve_val, event_cell)


def _cell_event_product_bwd(res, g):
    """Derivation (exact, not an approximation).

    Forward: ``W[c] = prod_s curve_val[s, c]``, ``w[n] = W[event_cell[n]]``.
    Given the incoming cotangent ``g[n] = dLoss/dw[n]`` (shape ``(N,)``):

    1. ``dLoss/dW[c] = sum_{n: event_cell[n] == c} g[n]`` -- because
       ``event_cell`` (not ``curve_val``) determines which events land on
       which ``W[c]``, this sum is **identical in form for every
       systematic** -- computed *once* as
       ``G = segment_sum(g, event_cell, num_segments=n_cells)``, the single
       "moderate" ``N -> n_cells`` scatter this function is named for
       (``n_cells`` is a few-to-low-thousands at ND scale, vs. ``N`` ~ 5e7 --
       see the module docstring's "Factorized hot path" section for the
       full before/after picture). This is the one place this whole
       composition touches an ``(N,)``-shaped array on the backward side.
    2. ``dW[c]/d(curve_val[s, c]) = prod_{s' != s} curve_val[s', c]`` (plain
       product rule) -- the "leave-one-out" product across the systematic
       axis. Computed via a prefix/suffix ``cumprod`` along axis 0
       (``O(n_syst * n_cells)``, trivial) rather than
       ``W[c] / curve_val[s, c]``: division would be unsafe whenever a
       curve's value is exactly (or near) zero, which the leave-one-out
       product handles for free (an empty/degenerate product is well-defined
       and exactly correct there, unlike ``0 / 0``).
    3. Chain rule: ``dLoss/d(curve_val[s, c]) = G[c] * leave_one_out[s, c]``,
       shape ``(n_syst, n_cells)`` -- small, elementwise, no further gather
       or scatter needed.

    Everything from step 2 onward is bounded by ``n_syst * n_cells``; only
    step 1's single segment_sum ever sees the full ``N``-scale cotangent.
    """
    curve_val, event_cell = res
    n_cells = curve_val.shape[1]
    G = jax.ops.segment_sum(g, event_cell, num_segments=n_cells)  # (n_cells,)

    prefix = jnp.cumprod(curve_val, axis=0)  # prefix[s] = prod_{0..s}
    suffix = jnp.cumprod(curve_val[::-1], axis=0)[::-1]  # suffix[s] = prod_{s..end}
    ones_row = jnp.ones((1, n_cells), dtype=curve_val.dtype)
    prefix_excl = jnp.concatenate([ones_row, prefix[:-1]], axis=0)  # prod_{0..s-1}
    suffix_excl = jnp.concatenate([suffix[1:], ones_row], axis=0)  # prod_{s+1..end}
    leave_one_out = prefix_excl * suffix_excl  # (n_syst, n_cells) == prod_{s'!=s} curve_val[s']

    d_curve_val = G[None, :] * leave_one_out
    return d_curve_val, None


_cell_event_product.defvjp(_cell_event_product_fwd, _cell_event_product_bwd)


def _factorized_event_weight(
    per_curve: jax.Array, cell_to_curve: jax.Array, event_cell: jax.Array
) -> jax.Array:
    """``(N,)`` total per-event systematic weight (product over all
    systematics) for the factorized representation -- the memory-safe
    replacement for ``jnp.prod(weights_per_syst(x), axis=0)`` that never
    materializes an ``(n_syst, N)`` array. See the module docstring's
    "Factorized hot path" section for the full picture.

    Not itself a fresh ``custom_vjp``: it is a plain composition of two
    ``custom_vjp`` primitives that both operate at the small ``(n_syst,
    n_cells)`` scale (:func:`_curve_gather`, reused unmodified here with
    ``cell_to_curve`` -- shape ``(n_syst, n_cells)`` -- standing in for the
    dense builder's ``(n_syst, N)`` ``event_index``; and
    :func:`_cell_event_product`) -- JAX's autodiff chains through both
    automatically, so this function needs no VJP of its own.
    """
    curve_val = _curve_gather(per_curve, cell_to_curve).astype(jnp.float64)  # (n_syst, n_cells)
    return _cell_event_product(curve_val, event_cell)  # (N,) f64


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SplineTable:
    """Flattened spline monolith: coefficients + per-event curve indices.

    See the module docstring for the evaluation algorithm. ``names``,
    ``knots`` and ``coeffs``/``event_index`` shapes follow DESIGN.md §4
    exactly; ``weight_clip`` is a small, static addition (a ``(lo, hi)`` pair,
    either bound optionally ``None``) needed to apply the "optional weight
    clamp" the design text calls for -- it is config, not data, so it is a
    static field like ``names``.
    """

    names: tuple[str, ...] = dataclasses.field(metadata=dict(static=True))
    knots: jax.Array  # (n_syst, n_knots) f64, NaN-padded if a syst has fewer knots
    coeffs: jax.Array  # (n_syst, n_curves, n_knots-1, 4) storage dtype, Horner [a,b,c,d]
    # Dense path (build_spline_table): event_index (n_syst, N) int32, curve
    # index per event, 0 == identity. Factorized path
    # (build_spline_table_factorized): event_index is None, event_cell (N,)
    # + cell_to_curve (n_syst, n_cells) are set instead -- see the module
    # docstring "Factorized (grid-cell) index mode" section. Exactly one of
    # the two representations is populated on any given instance.
    event_index: jax.Array | None = None
    weight_clip: tuple = dataclasses.field(
        default=(0.0, None), metadata=dict(static=True)
    )
    event_cell: jax.Array | None = None  # (N,) int32, factorized path only
    cell_to_curve: jax.Array | None = None  # (n_syst, n_cells) int32, factorized path only
    # Behaviour outside [x_min, x_max] of each systematic's real knots (static, config like
    # weight_clip): "linear" = boundary value + slope (library default), "cubic" = the end-segment
    # cubic (MaCh3's TSpline3_red::Eval) -- module docstring.
    extrapolation: str = dataclasses.field(default="linear", metadata=dict(static=True))
    # Optional floor applied to EACH curve's weight before the product over systematics (MaCh3 <= v2.6.1,
    # ``BinnedSplineHandler::CalcSplineWeights``: ``if (weight < 0) weight = 0``); None (default) = no per-curve floor, only the
    # product-level ``weight_clip`` (MaCh3 >= v2.7.0 drops an event whose TOTAL weight is <= 0). Static config.
    curve_floor: float | None = dataclasses.field(default=None, metadata=dict(static=True))

    def _eval_all_curves(self, x: jax.Array) -> jax.Array:
        """Per systematic, per curve, weight at ``x``: ``(n_syst, n_curves)``.

        Vectorized (``vmap``) over the systematic axis; the segment lookup
        and Horner evaluation for a single systematic are scalar-in,
        vector-out (one scalar ``x`` -> all curves of that systematic).
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        cubic = self.extrapolation == "cubic"  # static (pytree metadata)

        def eval_one(k_row, coeff_slab, xs):
            # NaN-padded (ragged) knot slots read as +inf for the search.
            # The *table's* n_knots (k_row.shape[0]) is a static upper bound
            # shared by every systematic; a systematic with fewer real knots
            # needs its segment clamped to *its own* last real segment
            # (n_real - 2), not the table's (n_knots - 2) -- otherwise, right
            # at/beyond that systematic's last real knot, searchsorted's
            # insertion point lands one segment past the last real one: a
            # padding slot whose coefficients were replicated from the last
            # real segment (build_spline_table) but are only valid relative
            # to *that* segment's start knot, not this one's. Using it with
            # this segment's (correct, real) k0 mismatches the polynomial's
            # reference point, corrupting the value (and, via w_high, the
            # boundary slope used for linear continuation beyond x_max).
            is_nan = jnp.isnan(k_row)
            n_real = jnp.sum(~is_nan)
            k_search = jnp.where(is_nan, jnp.inf, k_row)
            x_min = k_row[0]
            x_max = jnp.max(jnp.where(is_nan, -jnp.inf, k_row))

            # compare_all: n_knots compares, no XLA while loop (identical index, see shifts.hard_bin_index)
            seg_raw = jnp.searchsorted(k_search, xs, side="right", method="compare_all") - 1
            seg = jnp.clip(seg_raw, 0, n_real - 2)
            k0 = k_row[seg]
            coeff = coeff_slab[:, seg, :]  # (n_curves, 4)
            cdtype = coeff.dtype
            a, b, c, d = coeff[:, 0], coeff[:, 1], coeff[:, 2], coeff[:, 3]

            t = (xs - k0).astype(cdtype)
            w_in = ((a * t + b) * t + c) * t + d
            if cubic:
                # MaCh3: the end-segment cubic for any x (seg is clamped to [0, n_real - 2]
                # above); one polynomial, no seam, finite gradient everywhere.
                return w_in

            # Value & slope of the boundary segment evaluated exactly at the
            # boundary knot, then continued linearly. At x == x_min, seg is
            # clamped to 0 so k0 == x_min and t == 0, so w_low collapses to
            # exactly w_in there (d, with slope c) -- C1-matched by
            # construction, no special-casing at the seam.
            t_b = (x_max - k0).astype(cdtype)
            val_b = ((a * t_b + b) * t_b + c) * t_b + d
            slope_b = 3.0 * a * t_b * t_b + 2.0 * b * t_b + c
            w_high = val_b + slope_b * (xs - x_max).astype(cdtype)

            w_low = c * t + d  # t == xs - x_min here (seg 0, k0 == x_min)

            below = xs < x_min
            above = xs > x_max
            return jnp.where(below, w_low, jnp.where(above, w_high, w_in))

        out = jax.vmap(eval_one)(self.knots, self.coeffs, x)
        if self.curve_floor is not None:
            out = jnp.maximum(out, jnp.asarray(self.curve_floor, dtype=out.dtype))   # static branch; curve 0 (== 1) is unaffected
        return out

    def _effective_event_index(self) -> jax.Array:
        """``(n_syst, N)`` curve index per (systematic, event), regardless
        of which representation this table was built with (see module
        docstring "Factorized (grid-cell) index mode").

        The ``event_cell is not None`` check is a *static* branch: which
        representation a given :class:`SplineTable` instance uses is part
        of its pytree structure (a ``None`` field carries no leaves), fixed
        at build time -- so this resolves at trace time under ``jax.jit``,
        not per-call, exactly like the ``lo is not None`` checks in
        ``__call__`` below.
        """
        if self.event_cell is not None:
            if self.cell_to_curve is None:
                raise ValueError(
                    "SplineTable has event_cell set but cell_to_curve is None "
                    "-- the factorized representation requires both"
                )
            # cell_to_curve: (n_syst, n_cells), event_cell: (N,) ->
            # (n_syst, N), one gather per systematic row.
            return jnp.take(self.cell_to_curve, self.event_cell, axis=1)
        if self.event_index is None:
            raise ValueError(
                "SplineTable has neither event_index (dense) nor "
                "event_cell/cell_to_curve (factorized) set -- build it via "
                "build_spline_table or build_spline_table_factorized"
            )
        return self.event_index

    def weights_per_syst(self, x: jax.Array) -> jax.Array:
        """Per-event weight contributed by each systematic separately.

        ``(n_syst, N)`` float64, *not* multiplied together and *not*
        weight-clipped -- for debugging / per-systematic plots. Slower than
        ``__call__`` is not a goal here, but this does skip the final
        product + clip that the hot path needs.
        """
        per_curve = self._eval_all_curves(x)  # (n_syst, n_curves)
        event_index = self._effective_event_index()  # (n_syst, N), either path
        gathered = _curve_gather(per_curve, event_index)  # (n_syst, N)
        return gathered.astype(jnp.float64)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Per-event weight: product over systematics, accumulated in float64.

        ``x``: ``(n_syst,)`` f64, the current value of every spline
        systematic in sigma units. Returns ``(N,)`` f64.

        Factorized tables (``event_cell is not None``) take the memory-safe
        path (:func:`_factorized_event_weight`, module docstring "Factorized
        hot path" section) that never materializes an ``(n_syst, N)`` array,
        forward or backward -- this is a *static* branch (which
        representation a table uses is fixed at build time, part of its
        pytree structure), not a per-call cost. Dense tables are unchanged.
        """
        if self.event_cell is not None:
            x = jnp.asarray(x, dtype=jnp.float64)
            per_curve = self._eval_all_curves(x)  # (n_syst, n_curves)
            w = _factorized_event_weight(per_curve, self.cell_to_curve, self.event_cell)
        else:
            gathered = self.weights_per_syst(x)  # (n_syst, N) f64
            w = jnp.prod(gathered, axis=0)
        lo, hi = self.weight_clip
        if lo is not None or hi is not None:
            lo_v = -jnp.inf if lo is None else lo
            hi_v = jnp.inf if hi is None else hi
            w = jnp.clip(w, lo_v, hi_v)
        return w


def _validate_knot_shapes(knot_x: np.ndarray, knot_y: np.ndarray) -> tuple[int, int, int, int]:
    """Shared ``knot_x``/``knot_y`` shape checks for both builders.

    Returns ``(n_syst, n_knots, n_curves_in, n_curves)`` where ``n_curves =
    n_curves_in + 1`` (the ``+1`` is the prepended identity curve).
    """
    if knot_x.ndim != 2:
        raise ValueError(f"knot_x must be (n_syst, n_knots), got shape {knot_x.shape}")
    n_syst, n_knots = knot_x.shape

    if knot_y.ndim != 3 or knot_y.shape[0] != n_syst or knot_y.shape[2] != n_knots:
        raise ValueError(
            "knot_y must be (n_syst, n_curves, n_knots) = "
            f"({n_syst}, *, {n_knots}), got {knot_y.shape}"
        )
    n_curves_in = knot_y.shape[1]
    n_curves = n_curves_in + 1  # + prepended identity
    return n_syst, n_knots, n_curves_in, n_curves


def _resolve_names(names: Sequence[str] | None, n_syst: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"syst_{i}" for i in range(n_syst))
    names = tuple(names)
    if len(names) != n_syst:
        raise ValueError(f"names has {len(names)} entries, expected n_syst={n_syst}")
    return names


def _fit_coeffs(
    knot_x: np.ndarray, knot_y: np.ndarray, boundary: str, names: tuple[str, ...]
) -> np.ndarray:
    """Fit every curve of every systematic, prepending the identity curve.

    Returns ``(n_syst, n_curves, n_knots - 1, 4)`` float64 Horner
    coefficients -- the shared inner loop of both :func:`build_spline_table`
    and :func:`build_spline_table_factorized` (only how curves are *indexed
    per event* differs between the two; how they are *fit from knots* does
    not).
    """
    if boundary not in _BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; choose from {_BOUNDARIES}")

    n_syst, n_knots, n_curves_in, n_curves = _validate_knot_shapes(knot_x, knot_y)
    n_seg = n_knots - 1
    coeffs = np.zeros((n_syst, n_curves, n_seg, 4), dtype=np.float64)
    coeffs[:, 0, :, 3] = 1.0  # identity curve: a=b=c=0, d=1 on every segment

    for s in range(n_syst):
        valid = ~np.isnan(knot_x[s])
        n_valid = int(valid.sum())
        if n_valid < 2:
            raise ValueError(
                f"systematic {s} ({names[s]!r}) has fewer than 2 valid (non-NaN) knots"
            )
        x_valid = knot_x[s, valid]
        if np.any(np.diff(x_valid) <= 0):
            raise ValueError(
                f"knot_x for systematic {s} ({names[s]!r}) must be strictly increasing"
            )
        n_seg_valid = n_valid - 1

        for c in range(n_curves_in):
            y_valid = knot_y[s, c, valid]
            if boundary == "monotone":
                pp = PchipInterpolator(x_valid, y_valid)
            else:
                pp = CubicSpline(x_valid, y_valid, bc_type=boundary)
            # pp.c: (4, n_seg_valid), pp.c[k, i] is the coeff of (x-x_i)**(3-k)
            # on segment i -- exactly our Horner [a, b, c, d] convention.
            seg_coeffs = pp.c.T  # (n_seg_valid, 4)
            coeffs[s, c + 1, :n_seg_valid, :] = seg_coeffs
            if n_seg_valid < n_seg:
                # Padding beyond the last real knot is never selected at eval
                # time (the search treats NaN knots as +inf, clamping to the
                # last real segment); replicate it so the array is still
                # well-defined if ever inspected directly (e.g. via io.py).
                coeffs[s, c + 1, n_seg_valid:, :] = seg_coeffs[-1]

    return coeffs


def _check_extrapolation(extrapolation: str) -> str:
    if extrapolation not in _EXTRAPOLATIONS:
        raise ValueError(f"unknown extrapolation {extrapolation!r}; choose from {_EXTRAPOLATIONS}")
    return extrapolation


def _resolve_weight_clip(weight_clip: tuple) -> tuple:
    lo, hi = weight_clip
    lo = None if lo is None else float(lo)
    hi = None if hi is None else float(hi)
    return lo, hi


def build_spline_table(
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    event_index: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    boundary: str = "natural",
    dtype=jnp.float32,
    weight_clip: tuple = (0.0, None),
    extrapolation: str = "linear",
    curve_floor: float | None = None,
) -> SplineTable:
    """Build a :class:`SplineTable` from knot positions/values (build time only).

    Args:
      knot_x: ``(n_syst, n_knots)`` sigma positions. Rows may share a common
        grid; a systematic with fewer real knots than others pads the tail
        with NaN (ragged support) -- but all curves of *one* systematic must
        share that systematic's grid (DESIGN.md §4, "fine to require a
        common grid per systematic across its curves").
      knot_y: ``(n_syst, n_curves, n_knots)`` response values, curves 1..C
        (the identity response is *not* passed in here -- see below).
      event_index: ``(n_syst, N)`` int, curve index per event where ``0``
        means "unaffected" (identity) and ``k`` (``k >= 1``) means
        ``knot_y[:, k - 1, :]``. This numbering already matches the storage
        layout below, since the identity curve is *prepended*.
      boundary: ``"natural"`` (zero second derivative at the ends),
        ``"not-a-knot"`` (what ROOT's ``TSpline3`` does by default, i.e.
        what MaCh3 evaluates), or ``"monotone"``
        (``scipy.interpolate.PchipInterpolator``, shape-preserving).
        The default stays ``"natural"``; for the DUNE knot sets the two
        differ negligibly once throws are truncated to the parameter
        bounds.
      dtype: storage dtype of the coefficient table (default float32 --
        see module docstring).
      weight_clip: ``(lo, hi)``, either side optionally ``None``; applied to
        the final per-event product in ``SplineTable.__call__``.
      extrapolation: ``"linear"`` (default: boundary value + slope outside the
        knots) or ``"cubic"`` (the end-segment cubic, what MaCh3 evaluates; module
        docstring). ``boundary="not-a-knot"`` + ``extrapolation="cubic"`` on the REAL
        (NaN-padded, ragged) knots is MaCh3's ``TSpline3`` response.
      curve_floor: ``None`` (default) or a number; each curve's weight is floored there BEFORE the product over systematics
        (MaCh3 <= v2.6.1 floors every spline weight at 0, ``curve_floor=0.0``; >= v2.7.0 does not). See the module docstring.

    Returns:
      A :class:`SplineTable` with curve 0 of every systematic set to the
      identity response (``a=b=c=0, d=1`` exactly, for every segment --
      built directly rather than spline-fit through ``y==1``, so it is exact
      bit-for-bit in any storage dtype, not merely numerically close), using
      the dense ``(n_syst, N)`` ``event_index`` representation -- see
      :func:`build_spline_table_factorized` for the memory-frugal
      alternative when many events share few grid cells across all
      systematics at once.
    """
    knot_x = np.asarray(knot_x, dtype=np.float64)
    knot_y = np.asarray(knot_y, dtype=np.float64)
    event_index = np.asarray(event_index, dtype=np.int32)

    n_syst, n_knots, n_curves_in, n_curves = _validate_knot_shapes(knot_x, knot_y)

    if event_index.ndim != 2 or event_index.shape[0] != n_syst:
        raise ValueError(
            f"event_index must be (n_syst, N) = ({n_syst}, *), got {event_index.shape}"
        )
    if event_index.size and (event_index.min() < 0 or event_index.max() >= n_curves):
        raise ValueError(
            f"event_index values must be in [0, {n_curves - 1}] "
            f"(0=identity, 1..{n_curves_in}=knot_y rows); got range "
            f"[{event_index.min()}, {event_index.max()}]"
        )

    names = _resolve_names(names, n_syst)
    extrapolation = _check_extrapolation(extrapolation)
    coeffs = _fit_coeffs(knot_x, knot_y, boundary, names)
    lo, hi = _resolve_weight_clip(weight_clip)

    return SplineTable(
        names=names,
        knots=jnp.asarray(knot_x, dtype=jnp.float64),
        coeffs=jnp.asarray(coeffs, dtype=dtype),
        event_index=jnp.asarray(event_index, dtype=jnp.int32),
        weight_clip=(lo, hi),
        extrapolation=extrapolation,
        curve_floor=None if curve_floor is None else float(curve_floor),
    )


def build_spline_table_factorized(
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    cell_to_curve: np.ndarray,
    event_cell: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    boundary: str = "natural",
    dtype=jnp.float32,
    weight_clip: tuple = (0.0, None),
    extrapolation: str = "linear",
    curve_floor: float | None = None,
) -> SplineTable:
    """Build a :class:`SplineTable` using the factorized grid-cell index
    (module docstring "Factorized (grid-cell) index mode") instead of a
    dense ``(n_syst, N)`` ``event_index``.

    Use this when events are naturally grouped into a shared, per-event
    "cell" (e.g. a ``(mode, Etrue-bin, Erec-bin, Yrec-bin)`` grid cell, as
    for the DUNE ND binned splines) that already determines every
    systematic's curve at once -- storing one ``(N,)`` cell id plus one
    small ``(n_syst, n_cells)`` lookup is far cheaper than a dense
    ``(n_syst, N)`` array at ND-scale event counts (tens of millions of
    events x ~55 systematics).

    Args:
      knot_x, knot_y: same as :func:`build_spline_table`.
      cell_to_curve: ``(n_syst, n_cells)`` int, curve index per (systematic,
        grid cell); same ``0``-means-identity / ``k>=1``-means-``knot_y[:,
        k-1, :]`` convention as ``event_index`` in the dense builder.
      event_cell: ``(N,)`` int, grid-cell id per event, shared across every
        systematic (indexes ``cell_to_curve``'s second axis).
      boundary, dtype, weight_clip, extrapolation, curve_floor: same as :func:`build_spline_table`.

    Returns:
      A :class:`SplineTable` with ``event_index=None`` and
      ``event_cell``/``cell_to_curve`` populated instead; identical curve
      fitting (including the exact identity curve 0) to
      :func:`build_spline_table`.
    """
    knot_x = np.asarray(knot_x, dtype=np.float64)
    knot_y = np.asarray(knot_y, dtype=np.float64)
    cell_to_curve = np.asarray(cell_to_curve, dtype=np.int32)
    event_cell = np.asarray(event_cell, dtype=np.int32)

    n_syst, n_knots, n_curves_in, n_curves = _validate_knot_shapes(knot_x, knot_y)

    if cell_to_curve.ndim != 2 or cell_to_curve.shape[0] != n_syst:
        raise ValueError(
            f"cell_to_curve must be (n_syst, n_cells) = ({n_syst}, *), "
            f"got {cell_to_curve.shape}"
        )
    n_cells = cell_to_curve.shape[1]
    if cell_to_curve.size and (cell_to_curve.min() < 0 or cell_to_curve.max() >= n_curves):
        raise ValueError(
            f"cell_to_curve values must be in [0, {n_curves - 1}] "
            f"(0=identity, 1..{n_curves_in}=knot_y rows); got range "
            f"[{cell_to_curve.min()}, {cell_to_curve.max()}]"
        )

    if event_cell.ndim != 1:
        raise ValueError(f"event_cell must be (N,), got shape {event_cell.shape}")
    if event_cell.size and (event_cell.min() < 0 or event_cell.max() >= n_cells):
        raise ValueError(
            f"event_cell values must be in [0, {n_cells - 1}] (indexing "
            f"cell_to_curve's {n_cells}-cell axis); got range "
            f"[{event_cell.min()}, {event_cell.max()}]"
        )

    names = _resolve_names(names, n_syst)
    extrapolation = _check_extrapolation(extrapolation)
    coeffs = _fit_coeffs(knot_x, knot_y, boundary, names)
    lo, hi = _resolve_weight_clip(weight_clip)

    return SplineTable(
        names=names,
        knots=jnp.asarray(knot_x, dtype=jnp.float64),
        coeffs=jnp.asarray(coeffs, dtype=dtype),
        event_index=None,
        weight_clip=(lo, hi),
        event_cell=jnp.asarray(event_cell, dtype=jnp.int32),
        cell_to_curve=jnp.asarray(cell_to_curve, dtype=jnp.int32),
        extrapolation=extrapolation,
        curve_floor=None if curve_floor is None else float(curve_floor),
    )
