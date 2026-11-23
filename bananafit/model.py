"""Forward model: per-event weights -> binned prediction (DESIGN.md §7).

Two pieces:

- :class:`SampleModel` bundles one :class:`~bananafit.events.EventSample` with
  the systematic-response objects that act on it (a
  :class:`~bananafit.splines.SplineTable`, a :class:`~bananafit.norms.NormSet`,
  either optionally ``None``) and an oscillation-weight provider (``osc``,
  duck-typing :class:`bananafit.oscillation.OscConfig` -- e.g.
  :class:`~bananafit.oscillation.ConstantMatterOsc`). It carries no knowledge
  of *where* its systematic parameters live in the global fit vector -- that
  resolution happens once, at :func:`build_forward_model` time.

- :class:`ForwardModel` is the built object: a tuple of ``SampleModel`` plus,
  per sample, static index arrays (Python tuples of ``int``, baked in at
  build time -- never traced) that say which entries of the global physical
  parameter vector are this sample's six oscillation parameters, its spline
  systematics (in the sample's own ``SplineTable.names`` order), and its norm
  parameters (in the sample's own ``NormSet.names`` order). Systematics are
  named, so the same physical index feeds every sample that references that
  name -- "one parameter moves all samples coherently" (DESIGN.md §7).

Evaluating the model is the hot path: gather the relevant physical values,
multiply the weight factors, ``segment_sum`` into bins, drop the overflow
slot, concatenate across samples. No Python branching on traced values --
the only Python-level branching (``splines is None``, etc.) happens on the
static structure of ``SampleModel``, which is fixed at build/trace time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

import mango

from .events import EventSample
from .norms import NormSet
from .oscillation import OscConfig
from .parameters import ParameterSet
from .shifts import EnergyShift, hard_bin_index, soft_bin_weights
from .splines import SplineTable

__all__ = ["SampleModel", "ForwardModel", "build_forward_model"]


def _static_field(**kwargs):
    return dataclasses.field(metadata=dict(static=True), **kwargs)


# The six OscParams field names, in the exact order mango.OscParams expects
# (DESIGN.md §6, "Convention: oscillation parameters appear by their
# OscParams field names") -- verified to match mango.OscParams's own
# dataclass field order, though we pass them by keyword below so that fact
# is never load-bearing.
_OSC_NAMES = ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SampleModel:
    """One sample's events plus the systematic-response objects acting on it.

    ``events``, ``splines``, ``norms`` and ``osc`` are all pytree children
    (``splines``/``norms`` may be ``None`` -- JAX treats that as a valid,
    leaf-free pytree). ``extra_weight`` is a static (non-traced) optional
    hook for functional systematics not expressible as a spline or norm:
    ``extra_weight(sys_physical_by_name, events) -> (N,) weights``, where
    ``sys_physical_by_name`` is a ``dict[str, Array]`` mapping *every* name
    in the analysis's :class:`~bananafit.parameters.ParameterSet` (not just
    this sample's systematics) to its current physical value, and ``events``
    is this sample's ``EventSample`` -- so the hook can read whatever it
    needs by name.

    ``energy_shift`` is an optional :class:`~bananafit.shifts.EnergyShift`
    (DESIGN.md addendum: FD energy-shift
    systematics, ``bananafit.shifts``). When present, this sample's bin
    assignment is *not* the fixed ``events.reco_bin`` -- it is recomputed
    every evaluation from the shift-parameter-dependent shifted energy (soft
    tent-kernel or hard MaCh3-parity binning, per ``energy_shift.mode``).
    When ``None`` (the default -- every pre-existing ``SampleModel`` in the
    test suite and elsewhere), the fixed-bin fast path is used unchanged and
    bit-identical to before this field existed.
    """

    events: EventSample
    splines: SplineTable | None
    norms: NormSet | None
    osc: OscConfig
    extra_weight: Callable | None = _static_field(default=None)
    energy_shift: EnergyShift | None = None


@dataclasses.dataclass(frozen=True)
class _SampleIndex:
    """Static per-sample bookkeeping resolved once at build time: which
    entries of the global physical parameter vector feed this sample, and
    where this sample's bins land in the concatenated global bin vector.

    Plain (non-pytree) frozen dataclass of hashable ints/tuples -- this is
    exactly the kind of "metadata field" ``jax.tree_util.register_dataclass``
    wants: static, hashable, immutable, safe to use as part of a jit cache
    key.
    """

    osc_idx: tuple[int, ...]     # length 6, indices into physical for _OSC_NAMES
    spline_idx: tuple[int, ...]  # indices into physical, in splines.names order
    norm_idx: tuple[int, ...]    # indices into physical, in norms.names order
    shift_idx: tuple[int, ...]   # indices into physical, in energy_shift.param_names order (empty if no energy_shift)
    n_bins: int
    bin_offset: int


def _resolve_names(names: Sequence[str], name_to_idx: dict, *, context: str) -> tuple[int, ...]:
    missing = [n for n in names if n not in name_to_idx]
    if missing:
        raise ValueError(
            f"{context}: name(s) {missing} not found in ParameterSet "
            f"(have {sorted(name_to_idx)})"
        )
    return tuple(name_to_idx[n] for n in names)


def _event_weights(
    sample: SampleModel, meta: _SampleIndex, physical: jax.Array, param_names: tuple[str, ...]
) -> jax.Array:
    """Per-event weight ``(N,)`` f64 for one sample at one physical point."""
    events = sample.events
    osc_vals = physical[jnp.asarray(meta.osc_idx, dtype=jnp.int32)]
    osc_params = mango.OscParams(
        theta12=osc_vals[0],
        theta13=osc_vals[1],
        theta23=osc_vals[2],
        deltacp=osc_vals[3],
        dm21=osc_vals[4],
        dm31=osc_vals[5],
    )
    w = events.nominal_weight * sample.osc.weights(osc_params, events)

    if sample.splines is not None:
        x_spl = physical[jnp.asarray(meta.spline_idx, dtype=jnp.int32)]
        w = w * sample.splines(x_spl)

    if sample.norms is not None:
        x_norm = physical[jnp.asarray(meta.norm_idx, dtype=jnp.int32)]
        w = w * sample.norms(x_norm)

    if sample.extra_weight is not None:
        sys_dict = {n: physical[i] for i, n in enumerate(param_names)}
        w = w * sample.extra_weight(sys_dict, events)

    return w.astype(jnp.float64)


def _bin_sample(
    sample: SampleModel, meta: _SampleIndex, physical: jax.Array, param_names: tuple[str, ...]
) -> tuple[jax.Array, jax.Array]:
    w = _event_weights(sample, meta, physical, param_names)
    n_bins = meta.n_bins
    # per-row squared weight; a compressed sample (EventSample.w2_scale set) rescales it to the exact
    # sum of squares of the events the row stands for
    ww = w * w if sample.events.w2_scale is None else w * w * sample.events.w2_scale

    if sample.energy_shift is None:
        # Fast fixed-bin path -- bit-identical to before energy_shift existed.
        mu = jax.ops.segment_sum(w, sample.events.reco_bin, num_segments=n_bins + 1)[:n_bins]
        w2 = jax.ops.segment_sum(ww, sample.events.reco_bin, num_segments=n_bins + 1)[:n_bins]
        return mu.astype(jnp.float64), w2.astype(jnp.float64)

    shift = sample.energy_shift
    x_shift = physical[jnp.asarray(meta.shift_idx, dtype=jnp.int32)]
    e_shifted = shift.model.shift(shift.components, x_shift)
    edges = jnp.asarray(shift.bin_edges, dtype=jnp.float64)

    if shift.mode == "hard":
        bin_idx = hard_bin_index(e_shifted, edges)
        mu = jax.ops.segment_sum(w, bin_idx, num_segments=n_bins + 1)[:n_bins]
        w2 = jax.ops.segment_sum(ww, bin_idx, num_segments=n_bins + 1)[:n_bins]
        return mu.astype(jnp.float64), w2.astype(jnp.float64)

    # "soft": tent-kernel weight split between bin_lo and bin_lo + 1.
    bin_lo, frac_lo = soft_bin_weights(e_shifted, edges)
    frac_hi = 1.0 - frac_lo
    bin_hi = bin_lo + 1
    mu_lo = jax.ops.segment_sum(w * frac_lo, bin_lo, num_segments=n_bins + 1)[:n_bins]
    mu_hi = jax.ops.segment_sum(w * frac_hi, bin_hi, num_segments=n_bins + 2)[:n_bins]
    w2_lo = jax.ops.segment_sum(ww * frac_lo, bin_lo, num_segments=n_bins + 1)[:n_bins]
    w2_hi = jax.ops.segment_sum(ww * frac_hi, bin_hi, num_segments=n_bins + 2)[:n_bins]
    return (mu_lo + mu_hi).astype(jnp.float64), (w2_lo + w2_hi).astype(jnp.float64)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ForwardModel:
    """Weights -> binned prediction, over a tuple of :class:`SampleModel`.

    ``samples`` is a pytree child (a tuple of ``SampleModel``, each itself a
    pytree); everything else is static, computed once by
    :func:`build_forward_model`: per-sample parameter-index bookkeeping
    (``sample_meta``), the global bin layout (``total_bins`` = sum of each
    sample's ``n_bins``, overflow excluded), and the full ordered parameter
    name tuple (``param_names``, needed to build the ``sys_physical_by_name``
    dict passed to any ``extra_weight`` hooks).
    """

    samples: tuple[SampleModel, ...]
    sample_meta: tuple[_SampleIndex, ...] = _static_field()
    total_bins: int = _static_field()
    param_names: tuple[str, ...] = _static_field()

    def expected(self, physical: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``physical`` (n_params,) -> ``(mu, w2)``, each ``(total_bins,)`` f64."""
        physical = jnp.asarray(physical, dtype=jnp.float64)
        mu_parts = []
        w2_parts = []
        for sample, meta in zip(self.samples, self.sample_meta):
            mu_s, w2_s = _bin_sample(sample, meta, physical, self.param_names)
            mu_parts.append(mu_s)
            w2_parts.append(w2_s)
        return jnp.concatenate(mu_parts), jnp.concatenate(w2_parts)

    def expected_by_sample(self, physical: jax.Array) -> dict[str, tuple[jax.Array, jax.Array]]:
        """Same as :meth:`expected`, but keyed by sample name and not
        concatenated -- for reporting. ``jnp.concatenate`` of the values in
        sample order reproduces :meth:`expected`'s output exactly."""
        physical = jnp.asarray(physical, dtype=jnp.float64)
        out = {}
        for sample, meta in zip(self.samples, self.sample_meta):
            out[sample.events.name] = _bin_sample(sample, meta, physical, self.param_names)
        return out

    def event_weights(self, physical: jax.Array, sample_index: int | str) -> jax.Array:
        """Per-event weight ``(N,)`` for one sample -- debugging aid.

        ``sample_index`` may be a position in ``self.samples`` or a sample
        name (matched against ``sample.events.name``).
        """
        physical = jnp.asarray(physical, dtype=jnp.float64)
        if isinstance(sample_index, str):
            matches = [i for i, s in enumerate(self.samples) if s.events.name == sample_index]
            if not matches:
                names = [s.events.name for s in self.samples]
                raise ValueError(f"no sample named {sample_index!r} (have {names})")
            pos = matches[0]
        else:
            pos = sample_index
        return _event_weights(self.samples[pos], self.sample_meta[pos], physical, self.param_names)


def build_forward_model(samples: Sequence[SampleModel], parameter_set: ParameterSet) -> ForwardModel:
    """Resolve per-sample systematic/oscillation-parameter indices and bin
    offsets against ``parameter_set`` (DESIGN.md §7), at Python/build time.

    Every spline systematic name and every norm name referenced by any
    sample must exist in ``parameter_set.names`` -- systematics are shared
    across samples by name (one parameter moves all samples that reference
    it coherently); a name missing from the ``ParameterSet`` is a build-time
    ``ValueError``, not a silent no-op. Likewise all six oscillation
    parameter names (``theta12, theta13, theta23, deltacp, dm21, dm31``)
    must be present.
    """
    samples = tuple(samples)
    name_to_idx = {n: i for i, n in enumerate(parameter_set.names)}

    sample_meta = []
    offset = 0
    for sample in samples:
        sample_name = sample.events.name
        osc_idx = _resolve_names(
            _OSC_NAMES, name_to_idx, context=f"sample {sample_name!r} oscillation parameters"
        )
        if sample.splines is None:
            spline_idx: tuple[int, ...] = ()
        else:
            spline_idx = _resolve_names(
                sample.splines.names, name_to_idx,
                context=f"sample {sample_name!r} spline systematics",
            )
        if sample.norms is None:
            norm_idx: tuple[int, ...] = ()
        else:
            norm_idx = _resolve_names(
                sample.norms.names, name_to_idx,
                context=f"sample {sample_name!r} norm systematics",
            )

        if sample.energy_shift is None:
            shift_idx: tuple[int, ...] = ()
        else:
            shift_idx = _resolve_names(
                sample.energy_shift.param_names, name_to_idx,
                context=f"sample {sample_name!r} energy-shift parameters",
            )

        n_bins = sample.events.n_bins
        sample_meta.append(_SampleIndex(
            osc_idx=osc_idx, spline_idx=spline_idx, norm_idx=norm_idx, shift_idx=shift_idx,
            n_bins=n_bins, bin_offset=offset,
        ))
        offset += n_bins

    return ForwardModel(
        samples=samples,
        sample_meta=tuple(sample_meta),
        total_bins=offset,
        param_names=tuple(parameter_set.names),
    )
