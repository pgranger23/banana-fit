"""ROOT ``TSpline3`` import: MaCh3/T2K/DUNE spline-monolith compatibility.

MaCh3-family fitters store each systematic's event-weight response as one (or
many, per mode/kinematic bin) ``TSpline3`` object per ROOT file, with knots at
sigma-variations of the systematic. This module bridges those files into
``bananafit.splines.SplineTable`` (DESIGN.md Section 4) in two steps:

1. :func:`read_tspline3_knots` walks a ROOT file (via ``uproot``, read-only,
   no ROOT/PyROOT dependency at runtime) and extracts the raw knot
   ``(x, y)`` arrays of every ``TSpline3`` object found, keyed by systematic
   name and "curve key" (typically a ``(mode, bin)`` tuple) parsed from each
   object's name.
2. :func:`spline_table_from_root` turns those raw knots plus a caller-supplied
   per-event curve assignment into a :class:`~bananafit.splines.SplineTable`
   via :func:`~bananafit.splines.build_spline_table` -- i.e. this module only
   ever produces the *portable* raw-knot representation (DESIGN.md Section
   11); coefficients are always refit locally with ``scipy`` at
   ``bananafit``'s chosen boundary condition, never trusted from the ROOT
   file's own (usually undocumented) spline construction.

Investigation notes (see also the appended section in DESIGN.md Section 11):

- ``uproot`` (5.x, no ROOT/PyROOT needed at runtime) reads ``TSpline3``
  generically via the file's embedded ``TStreamerInfo`` -- there is no
  hand-written ``uproot`` model for it, so the object surfaces as a
  dynamically generated class (``uproot.dynamic.Model_TSpline3_v*``) with a
  ``.all_members`` dict. Critically, ``TSpline3`` does **not** expose flat
  ``fX``/``fY`` arrays; the knots live one-per-element in ``fPoly``, an array
  of ``TSplinePoly3`` sub-objects each with scalar members ``fX``, ``fY``
  (plus ROOT's own fitted Hermite coefficients ``fB``, ``fC``, ``fD``, which
  this module deliberately ignores -- see above). This was verified against
  real files written by PyROOT (ROOT 6.40, available in this environment's
  LCG_110 view) and read back with ``uproot`` 5.7.1.
- ``uproot`` cannot *write* ``TSpline3`` (its writing support only covers a
  small allowlist of classes, and ``TSpline3``/``TSplinePoly3`` are not
  in it), so test fixtures are **real ROOT files written with PyROOT**
  (``ROOT.TSpline3(title, x, y, n)`` then ``.Write(key_name)``) -- not
  synthetic byte-level ``TSpline3`` construction, which would be fragile and
  wouldn't validate the ``fPoly`` extraction path against a real streamer.
  Since PyROOT is available here, the gold validation (this reader's
  ``build_spline_table`` output vs ``ROOT.TSpline3.Eval`` on the *same* file)
  is also exercised directly, not merely assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from bananafit.splines import SplineTable, build_spline_table

CurveKey = object  # typically int, str, or a tuple thereof (e.g. (mode, bin))
NameParser = Callable[[str, str], tuple[str, CurveKey]]


# --- name parsing --------------------------------------------------------


def default_name_parser(objname: str, dirpath: str = "") -> tuple[str, CurveKey]:
    """MaCh3-style token convention: split ``objname`` on ``"_"``.

    The first token is the systematic name; the remaining tokens (if any)
    form the curve key -- integer-looking tokens are parsed as ``int`` (e.g.
    mode/bin indices), everything else stays a string. A single remaining
    token collapses to a bare scalar key; two or more become a tuple.
    Objects with no remaining tokens (a bare systematic name, e.g. one
    ``TSpline3`` per systematic with no mode/bin split) get curve key ``0``.

    Examples::

        "MAQE"            -> ("MAQE", 0)
        "MAQE_2"          -> ("MAQE", 2)
        "MAQE_0_3"        -> ("MAQE", (0, 3))
        "CA5_numu_signal" -> ("CA5", ("numu", "signal"))

    ``dirpath`` (the ``TDirectory`` path the object was found under, ``""``
    at file root) is accepted for signature compatibility with custom
    parsers that want to fold directory structure into the systematic name
    or curve key (e.g. one subdirectory per sample) -- the default ignores
    it.
    """
    tokens = objname.split("_")
    syst = tokens[0]
    rest = tokens[1:]
    if not rest:
        return syst, 0

    def _tok(t: str):
        try:
            return int(t)
        except ValueError:
            return t

    parsed = tuple(_tok(t) for t in rest)
    return syst, (parsed[0] if len(parsed) == 1 else parsed)


# --- fPoly -> (knot_x, knot_y) extraction ---------------------------------


def _member(obj, key):
    members = getattr(obj, "all_members", None)
    if members is None or key not in members:
        available = sorted(members) if members is not None else []
        raise ValueError(
            f"bananafit.root_import: object has no member {key!r} "
            f"(available: {available}); this ROOT file's TSpline3 layout is "
            "not the one this reader expects -- see the module docstring"
        )
    return members[key]


def _knots_from_tspline3(obj, objname: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(knot_x, knot_y)`` from one uproot ``TSpline3`` model."""
    poly = _member(obj, "fPoly")
    n = int(_member(obj, "fNp"))
    if n < 1:
        raise ValueError(
            f"bananafit.root_import: TSpline3 {objname!r} has fNp={n} (<1) knots"
        )
    if len(poly) != n:
        raise ValueError(
            f"bananafit.root_import: TSpline3 {objname!r}: fPoly has "
            f"{len(poly)} entries, fNp says {n}"
        )
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = poly[i]
        pm = getattr(p, "all_members", None)
        if pm is None:
            # Fallback: some uproot versions may surface fPoly as a
            # structured/record array rather than an array of sub-objects.
            try:
                xs[i] = p["fX"]
                ys[i] = p["fY"]
                continue
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"bananafit.root_import: TSpline3 {objname!r}: could not "
                    f"read fX/fY from fPoly[{i}] (type {type(p)!r})"
                ) from exc
        xs[i] = pm["fX"]
        ys[i] = pm["fY"]
    if np.any(np.diff(xs) <= 0):
        raise ValueError(
            f"bananafit.root_import: TSpline3 {objname!r} has non-increasing "
            f"knot x-positions: {xs.tolist()}"
        )
    return xs, ys


# --- public: knot extraction ----------------------------------------------


def read_tspline3_knots(
    path,
    *,
    name_parser: NameParser | None = None,
) -> dict[str, dict[CurveKey, tuple[np.ndarray, np.ndarray]]]:
    """Walk a ROOT file and extract every ``TSpline3``'s raw knots.

    Recurses into ``TDirectory`` subdirectories. Each object's name (the
    ``TKey`` name, i.e. the name it was ``Write()``-ed under -- not
    necessarily ``TObject::GetName()``, since ``TSpline3`` objects are
    commonly all named ``"Spline3"`` internally and disambiguated only by
    the key they are written under) and directory path are passed through
    ``name_parser`` (default :func:`default_name_parser`) to get
    ``(syst_name, curve_key)``.

    Returns ``{syst_name: {curve_key: (knot_x, knot_y)}}`` with
    ``knot_x``/``knot_y`` plain float64 numpy arrays of matching length
    (``>= 1`` knot; :func:`build_spline_table` downstream requires ``>= 2``
    per curve, checked there).

    Raises ``ValueError`` if the file has no ``TSpline3`` objects at all, or
    if two objects parse to the same ``(syst_name, curve_key)``.
    """
    import uproot

    parser = name_parser or default_name_parser

    result: dict[str, dict[CurveKey, tuple[np.ndarray, np.ndarray]]] = {}
    found_any = False

    with uproot.open(path) as f:
        classnames = f.classnames(recursive=True, filter_classname="TSpline3")
        for keyname in classnames:
            found_any = True
            obj = f[keyname]
            path_no_cycle = keyname.rsplit(";", 1)[0]
            *dirparts, objname = path_no_cycle.split("/")
            dirpath = "/".join(dirparts)

            syst, curve_key = parser(objname, dirpath)
            knot_x, knot_y = _knots_from_tspline3(obj, objname)

            bucket = result.setdefault(syst, {})
            if curve_key in bucket:
                raise ValueError(
                    f"bananafit.root_import: {path!r}: duplicate curve key "
                    f"{curve_key!r} for systematic {syst!r} (object "
                    f"{keyname!r}); name_parser must map each TSpline3 to a "
                    "unique (syst, curve_key)"
                )
            bucket[curve_key] = (knot_x, knot_y)

    if not found_any:
        raise ValueError(f"bananafit.root_import: no TSpline3 objects found in {path!r}")

    return result


# --- public: SplineTable assembly -----------------------------------------


def spline_table_from_root(
    path,
    event_curve_map: Mapping[str, Sequence],
    *,
    boundary: str = "natural",
    dtype=np.float32,
    name_parser: NameParser | None = None,
    weight_clip: tuple = (0.0, None),
) -> SplineTable:
    """Build a :class:`~bananafit.splines.SplineTable` from a ROOT file.

    ``event_curve_map``: ``{syst_name: array_of_length_N}``, one entry per
    spline systematic the caller wants in the table (a subset of the
    systematics found in the file is fine; extra ones in the file are
    ignored). Each array gives, per event, the curve key (as produced by
    ``name_parser`` on the file's object names -- e.g. an ``int`` mode/bin
    tuple) that applies to that event, or ``None`` for an event this
    systematic does not affect (mapped to curve index 0, the identity
    response, matching :func:`~bananafit.splines.build_spline_table`'s
    convention). All arrays must have the same length ``N``.

    Knots are read fresh from ``path`` via :func:`read_tspline3_knots` and
    coefficients are refit locally with ``scipy`` at the given ``boundary``
    (default ``"natural"``; NOTE ROOT's ``TSpline3`` default is *not-a-knot*, which is what MaCh3 evaluates --
    pass ``boundary="not-a-knot"`` for that, and ``extrapolation="cubic"`` at ``build_spline_table`` for its
    end-segment evaluation outside the knots) -- the ROOT file's own spline
    coefficients are never used, only its knot positions/values.

    Within one systematic, every curve must share the *same* knot
    x-positions (``build_spline_table`` stores one knot grid per
    systematic, not per curve -- DESIGN.md Section 4); a systematic whose
    curves disagree raises ``ValueError`` naming the offending curves.
    Ragged knot *counts* are supported across *different* systematics (the
    usual ``build_spline_table`` NaN-padding).
    """
    if not event_curve_map:
        raise ValueError("bananafit.root_import: event_curve_map must not be empty")

    knots_by_syst = read_tspline3_knots(path, name_parser=name_parser)

    names = list(event_curve_map.keys())
    n_syst = len(names)

    lengths = {s: len(np.asarray(v, dtype=object)) for s, v in event_curve_map.items()}
    n_events = next(iter(lengths.values()))
    for s, n in lengths.items():
        if n != n_events:
            raise ValueError(
                "bananafit.root_import: event_curve_map arrays have "
                f"inconsistent lengths: {s!r} has {n}, expected {n_events} "
                f"(from {names[0]!r})"
            )

    per_syst_curve_index: dict[str, dict[CurveKey, int]] = {}
    per_syst_knot_x: list[np.ndarray] = []
    per_syst_knot_y: list[np.ndarray] = []  # (n_curves_in, n_knots) each, pre-pad
    max_n_knots = 0
    max_n_curves_in = 0

    for s in names:
        if s not in knots_by_syst:
            raise ValueError(
                f"bananafit.root_import: systematic {s!r} (from event_curve_map) "
                f"has no TSpline3 objects in {path!r}; found: "
                f"{sorted(knots_by_syst)}"
            )
        curves = knots_by_syst[s]
        if not curves:  # pragma: no cover - read_tspline3_knots never emits empty buckets
            raise ValueError(f"bananafit.root_import: systematic {s!r} has no curves")

        curve_keys = list(curves.keys())  # file traversal order -> stable numbering
        curve_index = {k: i + 1 for i, k in enumerate(curve_keys)}  # 0 reserved (identity)
        per_syst_curve_index[s] = curve_index

        ref_key = curve_keys[0]
        ref_x, _ = curves[ref_key]
        for k in curve_keys[1:]:
            x_k, _ = curves[k]
            if x_k.shape != ref_x.shape or not np.allclose(x_k, ref_x):
                raise ValueError(
                    f"bananafit.root_import: systematic {s!r}: curve {k!r} has "
                    f"knot x-positions {x_k.tolist()} that differ from curve "
                    f"{ref_key!r}'s {ref_x.tolist()} -- all curves of one "
                    "systematic must share a knot grid (DESIGN.md Section 4); "
                    "use a name_parser that splits this into separate "
                    "systematics instead if the grids are genuinely different"
                )

        y_stack = np.stack([curves[k][1] for k in curve_keys], axis=0)  # (n_curves_in, n_knots)
        per_syst_knot_x.append(ref_x)
        per_syst_knot_y.append(y_stack)
        max_n_knots = max(max_n_knots, ref_x.shape[0])
        max_n_curves_in = max(max_n_curves_in, y_stack.shape[0])

    knot_x = np.full((n_syst, max_n_knots), np.nan, dtype=np.float64)
    knot_y = np.zeros((n_syst, max_n_curves_in, max_n_knots), dtype=np.float64)
    for i, (x_row, y_stack) in enumerate(zip(per_syst_knot_x, per_syst_knot_y)):
        nk = x_row.shape[0]
        nc = y_stack.shape[0]
        knot_x[i, :nk] = x_row
        knot_y[i, :nc, :nk] = y_stack

    event_index = np.zeros((n_syst, n_events), dtype=np.int32)
    for i, s in enumerate(names):
        curve_index = per_syst_curve_index[s]
        arr = np.asarray(event_curve_map[s], dtype=object)
        row = np.zeros(n_events, dtype=np.int32)
        for e in range(n_events):
            key = arr[e]
            if key is None:
                continue  # identity, index 0
            if key not in curve_index:
                raise ValueError(
                    f"bananafit.root_import: systematic {s!r}: event {e} has "
                    f"curve key {key!r} not found among this systematic's "
                    f"ROOT curves {list(curve_index)}"
                )
            row[e] = curve_index[key]
        event_index[i] = row

    return build_spline_table(
        knot_x,
        knot_y,
        event_index,
        names=names,
        boundary=boundary,
        dtype=dtype,
        weight_clip=weight_clip,
    )
