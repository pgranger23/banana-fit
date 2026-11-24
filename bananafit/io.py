"""HDF5 persistence for event samples, splines, and norms (DESIGN.md Section 11).

Schema (version-tagged, ``/format_version``)::

    /format_version = 1
    /samples/<name>/
        true_energy, reco_bin, flavor_from, flavor_to, nominal_weight   (N,)
        attrs: name, n_bins, anti_split, bin_edges (optional), plus any
               caller-supplied osc_attrs (e.g. baseline_km, density_gcc, ye)
        splines/
            names, knots (n_syst,k), coeffs (n_syst,c,k-1,4), event_index (n_syst,N)
            knot_y (n_syst,c-1,k)   -- OPTIONAL, only if the caller passed
                                       spline_knot_y=... to save_sample
            attrs: boundary, weight_clip_lo, weight_clip_hi, coeff_dtype
        norms/
            names, mask (n_norm, N)
    /parameters/
        attrs: yaml   -- OPTIONAL, the parameter-spec YAML *string* stored
                          verbatim (no parsing here -- that is the caller's
                          business, see DESIGN.md Section 6)

This module only ever touches plain HDF5 primitives (datasets/attrs/groups)
and the real dataclasses' public constructors -- it never reaches into
private state of ``EventSample``/``SplineTable``/``NormSet``. Everything
numpy at the HDF5 boundary: arrays are ``np.asarray``'d on the way in
(accepting jax arrays), and reconstructed with explicit ``jnp.asarray(...,
dtype=...)`` on the way out so storage dtypes (notably the spline
coefficient table's float32) round-trip bit-exactly by default.

Two representations of a spline systematic's response may be stored
side-by-side, per DESIGN.md Section 11:

- The coefficient table (``coeffs`` + ``knots`` + ``event_index`` + names +
  attrs) -- always written when a :class:`~bananafit.splines.SplineTable` is
  given. This is what :func:`load_sample` reads by default
  (``rebuild="from-coeffs"``): a bit-exact round trip of exactly the
  ``SplineTable`` that was saved, no recomputation.
- The raw knot y-values (``knot_y``) -- written only if the caller passes
  ``spline_knot_y=`` (the array originally handed to
  ``build_spline_table``), since a ``SplineTable`` itself does not retain
  them (only the fitted coefficients). This is the portable representation:
  ``load_sample(..., rebuild="from-knots")`` re-runs
  ``build_spline_table`` from these raw knots instead of trusting the stored
  coefficients -- useful for e.g. swapping ``boundary``/``dtype`` on load,
  or verifying the coefficient table against an independent re-fit.
"""

from __future__ import annotations

import contextlib
import os

import h5py
import jax.numpy as jnp
import numpy as np

from bananafit.events import EventSample
from bananafit.norms import NormSet
from bananafit.splines import SplineTable, _with_auto_sort_cache, build_spline_table

FORMAT_VERSION = 1

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

# Attribute keys owned by this schema on a /samples/<name> group -- anything
# else found there on load is assumed to be caller-supplied osc_attrs.
_RESERVED_SAMPLE_ATTRS = frozenset({"name", "n_bins", "anti_split", "bin_edges"})


# --- small helpers -----------------------------------------------------------


def _is_group(obj) -> bool:
    # h5py.File subclasses h5py.Group, so this covers both "a group inside
    # an already-open file" and "an already-open file itself" (write/read at
    # its root).
    return isinstance(obj, h5py.Group)


@contextlib.contextmanager
def _resolve_group(group_or_path, mode: str):
    """Yield an ``h5py.Group`` to work in, opening/closing a file iff needed.

    ``group_or_path`` is either an already-open ``h5py.Group``/``h5py.File``
    (used as-is, caller owns its lifetime -- ``mode`` is ignored) or a
    path-like (a new file is opened in ``mode`` and closed on exit).
    """
    if _is_group(group_or_path):
        yield group_or_path
    else:
        f = h5py.File(os.fspath(group_or_path), mode)
        try:
            yield f
        finally:
            f.close()


def _write_dataset(group, name, arr, *, dtype=None):
    arr = np.asarray(arr) if dtype is None else np.asarray(arr, dtype=dtype)
    kwargs = {}
    if arr.size > 0:
        # gzip + shuffle: the byte-shuffle filter helps gzip's ratio a lot
        # on the (mostly-smooth, mostly-similar-magnitude) float arrays here.
        kwargs["compression"] = "gzip"
        kwargs["shuffle"] = True
    return group.create_dataset(name, data=arr, **kwargs)


def _write_names(group, key, names):
    # Variable-length UTF-8 strings; not compressed (tiny, and vlen string
    # datasets do not benefit from the byte-shuffle filter).
    group.create_dataset(key, data=list(names), dtype=_STR_DTYPE)


def _read_names(group, key) -> tuple[str, ...]:
    return tuple(group[key].asstr()[...])


def _attr_to_py(v):
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def _require(group, key, what: str):
    if key not in group:
        raise ValueError(
            f"bananafit.io: {what} {group.name!r} is missing required "
            f"dataset/group {key!r} (truncated or corrupted file?)"
        )


def _require_attr(group, key, what: str):
    if key not in group.attrs:
        raise ValueError(
            f"bananafit.io: {what} {group.name!r} is missing required "
            f"attribute {key!r} (truncated or corrupted file?)"
        )


# --- events --------------------------------------------------------------


def _write_events(group, events: EventSample) -> None:
    if getattr(events, "w2_scale", None) is not None:
        raise ValueError("bananafit.io: cannot save a compressed EventSample (w2_scale is set; "
                         "the schema stores one row per MC event) -- save the uncompressed sample")
    _write_dataset(group, "true_energy", events.true_energy, dtype=np.float64)
    _write_dataset(group, "reco_bin", events.reco_bin, dtype=np.int32)
    _write_dataset(group, "flavor_from", events.flavor_from, dtype=np.int32)
    _write_dataset(group, "flavor_to", events.flavor_to, dtype=np.int32)
    _write_dataset(group, "nominal_weight", events.nominal_weight, dtype=np.float64)

    group.attrs["name"] = events.name
    group.attrs["n_bins"] = int(events.n_bins)
    group.attrs["anti_split"] = int(events.anti_split)
    if events.bin_edges is not None:
        group.attrs["bin_edges"] = np.asarray(events.bin_edges, dtype=np.float64)


def _read_events(group) -> tuple[EventSample, int]:
    for key in ("true_energy", "reco_bin", "flavor_from", "flavor_to", "nominal_weight"):
        _require(group, key, "sample group")
    for key in ("name", "n_bins", "anti_split"):
        _require_attr(group, key, "sample group")

    n = group["true_energy"].shape[0]
    for key in ("reco_bin", "flavor_from", "flavor_to", "nominal_weight"):
        length = group[key].shape[0]
        if length != n:
            raise ValueError(
                f"bananafit.io: sample group {group.name!r}: dataset {key!r} "
                f"has length {length}, expected {n} (matching true_energy)"
            )

    bin_edges = None
    if "bin_edges" in group.attrs:
        bin_edges = tuple(float(e) for e in group.attrs["bin_edges"])

    events = EventSample(
        name=str(group.attrs["name"]),
        n_bins=int(group.attrs["n_bins"]),
        anti_split=int(group.attrs["anti_split"]),
        bin_edges=bin_edges,
        true_energy=jnp.asarray(group["true_energy"][...], dtype=jnp.float64),
        reco_bin=jnp.asarray(group["reco_bin"][...], dtype=jnp.int32),
        flavor_from=jnp.asarray(group["flavor_from"][...], dtype=jnp.int32),
        flavor_to=jnp.asarray(group["flavor_to"][...], dtype=jnp.int32),
        nominal_weight=jnp.asarray(group["nominal_weight"][...], dtype=jnp.float64),
    )
    return events, n


# --- splines -----------------------------------------------------------------


def _write_splines(group, table: SplineTable, *, knot_y=None, boundary: str) -> None:
    _write_dataset(group, "knots", table.knots, dtype=np.float64)

    coeffs_np = np.asarray(table.coeffs)  # preserve storage dtype (f32|f64)
    _write_dataset(group, "coeffs", coeffs_np, dtype=coeffs_np.dtype)

    _write_dataset(group, "event_index", table.event_index, dtype=np.int32)
    _write_names(group, "names", table.names)

    lo, hi = table.weight_clip
    group.attrs["weight_clip_lo"] = np.nan if lo is None else float(lo)
    group.attrs["weight_clip_hi"] = np.nan if hi is None else float(hi)
    group.attrs["boundary"] = boundary
    group.attrs["coeff_dtype"] = str(coeffs_np.dtype)
    if table.extrapolation != "linear":
        # only written when non-default, so existing files/round trips stay byte-identical
        group.attrs["extrapolation"] = table.extrapolation
    if table.curve_floor is not None:
        group.attrs["curve_floor"] = float(table.curve_floor)

    if knot_y is not None:
        knot_y_np = np.asarray(knot_y, dtype=np.float64)
        n_syst, n_knots = table.knots.shape
        n_curves = table.coeffs.shape[1]
        expected = (n_syst, n_curves - 1, n_knots)
        if knot_y_np.shape != expected:
            raise ValueError(
                f"bananafit.io: spline_knot_y has shape {knot_y_np.shape}, "
                f"expected {expected} = (n_syst, n_curves - 1, n_knots) to "
                "match the given SplineTable (curve 0 is the identity "
                "response and is not part of the raw knot_y)"
            )
        _write_dataset(group, "knot_y", knot_y_np, dtype=np.float64)


def _read_splines(group, n_events: int, *, rebuild: str) -> SplineTable:
    for key in ("knots", "coeffs", "event_index", "names"):
        _require(group, key, "splines group")

    names = _read_names(group, "names")
    knots = np.asarray(group["knots"][...], dtype=np.float64)
    coeffs = group["coeffs"][...]
    event_index = np.asarray(group["event_index"][...], dtype=np.int32)

    n_syst = knots.shape[0]
    if len(names) != n_syst:
        raise ValueError(
            f"bananafit.io: splines group {group.name!r}: names has "
            f"{len(names)} entries, expected n_syst={n_syst} (from knots)"
        )
    if coeffs.shape[0] != n_syst:
        raise ValueError(
            f"bananafit.io: splines group {group.name!r}: coeffs has "
            f"{coeffs.shape[0]} systematics, expected n_syst={n_syst} (from knots)"
        )
    if event_index.shape[0] != n_syst:
        raise ValueError(
            f"bananafit.io: splines group {group.name!r}: event_index has "
            f"{event_index.shape[0]} systematics, expected n_syst={n_syst} (from knots)"
        )
    if event_index.shape[1] != n_events:
        raise ValueError(
            f"bananafit.io: splines group {group.name!r}: event_index has "
            f"second dimension {event_index.shape[1]}, expected n_events="
            f"{n_events} (from the sample's event arrays)"
        )

    lo = float(group.attrs.get("weight_clip_lo", np.nan))
    hi = float(group.attrs.get("weight_clip_hi", np.nan))
    weight_clip = (None if np.isnan(lo) else lo, None if np.isnan(hi) else hi)
    extrapolation = str(group.attrs.get("extrapolation", "linear"))
    curve_floor = float(group.attrs["curve_floor"]) if "curve_floor" in group.attrs else None

    if rebuild == "from-coeffs":
        return _with_auto_sort_cache(SplineTable(
            names=names,
            knots=jnp.asarray(knots, dtype=jnp.float64),
            coeffs=jnp.asarray(coeffs),  # keep the exact stored dtype
            event_index=jnp.asarray(event_index, dtype=jnp.int32),
            weight_clip=weight_clip,
            extrapolation=extrapolation,
            curve_floor=curve_floor,
        ))
    if rebuild == "from-knots":
        if "knot_y" not in group:
            raise ValueError(
                "bananafit.io: rebuild='from-knots' requested but no knot_y "
                f"dataset is stored in {group.name!r} -- save_sample must "
                "be given spline_knot_y=... for this path to be available"
            )
        knot_y = np.asarray(group["knot_y"][...], dtype=np.float64)
        boundary = str(group.attrs.get("boundary", "natural"))
        dtype = np.dtype(group.attrs.get("coeff_dtype", str(coeffs.dtype)))
        return build_spline_table(
            knots,
            knot_y,
            event_index,
            names=names,
            boundary=boundary,
            dtype=dtype,
            weight_clip=weight_clip,
            extrapolation=extrapolation,
            curve_floor=curve_floor,
        )
    raise ValueError(
        f"bananafit.io: unknown rebuild={rebuild!r}; choose 'from-coeffs' or 'from-knots'"
    )


# --- norms ---------------------------------------------------------------


def _write_norms(group, norms: NormSet) -> None:
    mask_np = np.asarray(norms.mask)
    _write_dataset(group, "mask", mask_np, dtype=mask_np.dtype)
    _write_names(group, "names", norms.names)


def _read_norms(group, n_events: int) -> NormSet:
    for key in ("mask", "names"):
        _require(group, key, "norms group")
    names = _read_names(group, "names")
    mask = group["mask"][...]
    if mask.shape[0] != len(names):
        raise ValueError(
            f"bananafit.io: norms group {group.name!r}: mask has "
            f"{mask.shape[0]} rows, expected len(names)={len(names)}"
        )
    if mask.shape[1] != n_events:
        raise ValueError(
            f"bananafit.io: norms group {group.name!r}: mask has second "
            f"dimension {mask.shape[1]}, expected n_events={n_events}"
        )
    return NormSet(names=names, mask=jnp.asarray(mask))


# --- public: single sample ----------------------------------------------


def save_sample(
    group_or_path,
    events: EventSample,
    *,
    splines: SplineTable | None = None,
    norms: NormSet | None = None,
    osc_attrs: dict | None = None,
    spline_knot_y=None,
    spline_boundary: str = "natural",
) -> None:
    """Write one sample's events (+ optional splines/norms) to HDF5.

    ``group_or_path``: an already-open ``h5py.Group``/``h5py.File`` to write
    into (used by :func:`save_analysis`), or a path -- a fresh file is
    created (mode ``"w"``) and closed again.

    ``osc_attrs``: an optional flat dict of extra reporting metadata stored
    as HDF5 attributes on the sample group verbatim (e.g. ``baseline_km``,
    ``density_gcc``, ``ye`` for a ``ConstantMatterOsc`` -- DESIGN.md Section
    11 lists these, but this module has no dependency on ``oscillation.py``:
    any JSON-attr-able scalar is accepted). Keys must not collide with the
    schema's own attribute names (``name``, ``n_bins``, ``anti_split``,
    ``bin_edges``).

    ``spline_knot_y``: optional raw knot y-values (the array originally
    passed to ``build_spline_table``, shape ``(n_syst, n_curves-1,
    n_knots)`` -- curve 0/identity excluded), stored alongside the
    coefficient table so :func:`load_sample` can later rebuild from knots
    instead of trusting the stored coefficients. Ignored if ``splines`` is
    ``None``.

    ``spline_boundary``: the boundary condition ``build_spline_table`` was
    (or would be) called with; stored as an attr so a ``rebuild="from-knots"``
    load reproduces the same fit. Ignored if ``splines`` is ``None``.
    """
    with _resolve_group(group_or_path, "w") as group:
        _write_events(group, events)

        if osc_attrs:
            for key, value in osc_attrs.items():
                if key in _RESERVED_SAMPLE_ATTRS:
                    raise ValueError(
                        f"bananafit.io: osc_attrs key {key!r} collides with a "
                        "reserved sample attribute name "
                        f"({sorted(_RESERVED_SAMPLE_ATTRS)})"
                    )
                group.attrs[key] = value

        if splines is not None:
            _write_splines(
                group.create_group("splines"),
                splines,
                knot_y=spline_knot_y,
                boundary=spline_boundary,
            )

        if norms is not None:
            _write_norms(group.create_group("norms"), norms)


def load_sample(
    group_or_path,
    *,
    rebuild: str = "from-coeffs",
) -> tuple[EventSample, SplineTable | None, NormSet | None, dict]:
    """Read one sample's events (+ optional splines/norms) from HDF5.

    ``rebuild``: ``"from-coeffs"`` (default) trusts the stored coefficient
    table verbatim (bit-exact round trip of the ``SplineTable`` that was
    saved). ``"from-knots"`` instead re-runs ``build_spline_table`` from the
    raw ``knot_y`` dataset (requires the file to have one -- see
    ``save_sample``'s ``spline_knot_y``); the result matches the original
    ``SplineTable`` to float32/float64 rounding, not bit-exactly.

    Returns ``(events, splines_or_none, norms_or_none, osc_attrs)`` where
    ``osc_attrs`` is every extra sample-group attribute that is not part of
    this schema (see ``save_sample``'s ``osc_attrs``).
    """
    with _resolve_group(group_or_path, "r") as group:
        events, n = _read_events(group)
        osc_attrs = {
            key: _attr_to_py(value)
            for key, value in group.attrs.items()
            if key not in _RESERVED_SAMPLE_ATTRS
        }
        table = _read_splines(group["splines"], n, rebuild=rebuild) if "splines" in group else None
        norms = _read_norms(group["norms"], n) if "norms" in group else None
    return events, table, norms, osc_attrs


# --- public: whole analysis -----------------------------------------------


def save_analysis(
    path,
    samples: dict,
    *,
    parameter_yaml: str | None = None,
    spline_knot_y: dict | None = None,
    spline_boundary: dict | None = None,
) -> None:
    """Write a whole analysis (multiple named samples) to one HDF5 file.

    ``samples``: ``{name: (events, splines_or_none, norms_or_none,
    osc_attrs_or_none)}``.

    ``parameter_yaml``: optional parameter-spec YAML *string*, stored
    verbatim as ``/parameters`` attr ``yaml`` -- this module does not parse
    or validate it (``parameters.py``'s business, DESIGN.md Section 6).

    ``spline_knot_y`` / ``spline_boundary``: optional ``{name: ...}`` maps
    passed through to each sample's :func:`save_sample` call (see there);
    a missing entry means "no raw knots for this sample" / boundary
    defaults to ``"natural"``.
    """
    spline_knot_y = spline_knot_y or {}
    spline_boundary = spline_boundary or {}

    with h5py.File(os.fspath(path), "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        samples_group = f.create_group("samples")
        for name, entry in samples.items():
            events, splines, norms, osc_attrs = entry
            save_sample(
                samples_group.create_group(name),
                events,
                splines=splines,
                norms=norms,
                osc_attrs=osc_attrs,
                spline_knot_y=spline_knot_y.get(name),
                spline_boundary=spline_boundary.get(name, "natural"),
            )
        if parameter_yaml is not None:
            f.create_group("parameters").attrs["yaml"] = parameter_yaml


def load_analysis(
    path,
    *,
    rebuild: str = "from-coeffs",
) -> tuple[dict, str | None]:
    """Read a whole analysis file written by :func:`save_analysis`.

    Returns ``(samples, parameter_yaml)`` where ``samples`` is ``{name:
    (events, splines_or_none, norms_or_none, osc_attrs)}`` (same shape
    :func:`save_analysis` takes) and ``parameter_yaml`` is the verbatim YAML
    string, or ``None`` if the file has no ``/parameters`` group.

    Raises ``ValueError`` loudly on an unknown/missing ``/format_version``
    or a missing ``/samples`` group.
    """
    with h5py.File(os.fspath(path), "r") as f:
        if "format_version" not in f.attrs:
            raise ValueError(
                f"bananafit.io: {os.fspath(path)!r} has no /format_version "
                "attribute -- not a bananafit analysis file"
            )
        version = int(f.attrs["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"bananafit.io: unsupported format_version={version} in "
                f"{os.fspath(path)!r} (this bananafit reads version "
                f"{FORMAT_VERSION} only)"
            )
        if "samples" not in f:
            raise ValueError(
                f"bananafit.io: {os.fspath(path)!r} has no /samples group "
                "(truncated or corrupted file?)"
            )

        samples = {}
        for name in f["samples"]:
            samples[name] = load_sample(f["samples"][name], rebuild=rebuild)

        parameter_yaml = None
        if "parameters" in f and "yaml" in f["parameters"].attrs:
            parameter_yaml = str(f["parameters"].attrs["yaml"])

    return samples, parameter_yaml
