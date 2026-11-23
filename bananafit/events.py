"""Event-sample data model (DESIGN.md Section 3).

``EventSample`` is a struct-of-arrays pytree over N reconstructed MC events for
one detection channel/selection (e.g. FHC nu_e-appearance). It is deliberately
"dumb": a frozen, registered-dataclass container with no behaviour of its own.
All the smarts -- sorting by neutrino/antineutrino, digitizing the reco
variable into analysis bins, and validating everything -- live in the builder
functions :func:`make_sample` / :func:`make_sample_prebinned`, which run at
Python/numpy build time (never on traced values; see the jaxnu
``_is_concrete`` convention this codebase follows elsewhere -- here it is
simpler still, since these builders only ever accept plain numpy input, so
every check is unconditionally eager).

Flavor convention matches ``jaxnu.oscillator.Flavor``: 0=e, 1=mu, 2=tau.
``flavor_from == flavor_to == -1`` marks an "unoscillated" event (e.g. an NC
background) that should get oscillation weight 1 later, in
``oscillation.py`` -- this module only validates that convention, it does not
apply it.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

_VALID_FLAVORS = (-1, 0, 1, 2)


def _static_field():
    return dataclasses.field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EventSample:
    """One channel's reconstructed MC events, sorted nu-then-antinu.

    Array leaves (all length ``N``, the total event count):

    - ``true_energy``:    (N,) f64, GeV
    - ``reco_bin``:       (N,) int32, in ``[0, n_bins]``; ``n_bins`` itself is
      the overflow slot (out-of-window events), dropped from the likelihood.
    - ``flavor_from``:    (N,) int32, jaxnu ``Flavor`` convention (0=e,1=mu,2=tau)
    - ``flavor_to``:      (N,) int32, ditto; ``-1`` on *both* from/to marks an
      unoscillated event (e.g. NC background) -- weight 1 later.
    - ``nominal_weight``: (N,) f64, flux x xsec x eff x POT (osc-independent)
    - ``w2_scale``:       optional (N,) f64, default ``None`` (every row is one
      MC event).  Set only by builders that merge events sharing *every*
      weight factor into one row (e.g. an exact compression of a large
      ND sample): a row
      standing for events with nominal weights ``a_i`` has ``nominal_weight
      = sum a_i`` and ``w2_scale = sum a_i**2 / (sum a_i)**2``, so that
      ``w_row**2 * w2_scale`` is the exact sum of squared event weights that
      ``ForwardModel.expected``'s ``w2`` needs.

    Static fields:

    - ``name``:       sample name, e.g. ``"FHC_numu_disappearance"``
    - ``n_bins``:      number of analysis bins for this sample
    - ``anti_split``:  events are sorted so ``[0:anti_split)`` are neutrinos
      and ``[anti_split:N)`` are antineutrinos (see ``nu_slice``/``antinu_slice``)
    - ``bin_edges``:   plain-numpy reporting metadata (tuple of floats), the
      edges used to build ``reco_bin`` -- *not* a traced leaf. ``None`` if the
      sample was built via :func:`make_sample_prebinned` without edges.

    Build via :func:`make_sample` / :func:`make_sample_prebinned` -- the raw
    constructor does no validation or sorting (same convention as
    ``jaxnu.OscParams``: the dataclass is a plain container, the builder does
    the work).
    """

    name: str = _static_field()
    n_bins: int = _static_field()
    anti_split: int = _static_field()
    bin_edges: tuple[float, ...] | None = _static_field()

    true_energy: jax.Array
    reco_bin: jax.Array
    flavor_from: jax.Array
    flavor_to: jax.Array
    nominal_weight: jax.Array
    w2_scale: jax.Array | None = None

    @property
    def n_events(self) -> int:
        """Total event count N (static under jit -- shapes are static)."""
        return self.true_energy.shape[0]

    @property
    def nu_slice(self) -> slice:
        """Python slice selecting the neutrino block ``[0, anti_split)``."""
        return slice(0, self.anti_split)

    @property
    def antinu_slice(self) -> slice:
        """Python slice selecting the antineutrino block ``[anti_split, N)``."""
        return slice(self.anti_split, self.n_events)


# --- validation helpers (build time only -- plain numpy, always concrete) ---


def _check_same_length(n_expected, name, arr):
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"{name} has length {arr.shape[0]}, expected {n_expected} "
            "(all per-event arrays passed to make_sample must have the same "
            "length)"
        )


def _check_finite(arr, name):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")


def _check_flavor_codes(flavor_from, flavor_to):
    if not np.all(np.isin(flavor_from, _VALID_FLAVORS)):
        raise ValueError(
            f"flavor_from has values outside {_VALID_FLAVORS} "
            "(jaxnu Flavor convention 0=e,1=mu,2=tau, or -1 for unoscillated)"
        )
    if not np.all(np.isin(flavor_to, _VALID_FLAVORS)):
        raise ValueError(
            f"flavor_to has values outside {_VALID_FLAVORS} "
            "(jaxnu Flavor convention 0=e,1=mu,2=tau, or -1 for unoscillated)"
        )
    from_unosc = flavor_from == -1
    to_unosc = flavor_to == -1
    if not np.array_equal(from_unosc, to_unosc):
        bad = np.nonzero(from_unosc != to_unosc)[0]
        raise ValueError(
            "flavor_from == -1 must always be paired with flavor_to == -1 "
            "(marking an unoscillated event) and vice versa -- mismatched at "
            f"event indices {bad.tolist()[:10]}{'...' if bad.size > 10 else ''}"
        )


def _check_increasing_edges(bin_edges):
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("bin_edges must be a 1-D array of length >= 2")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")
    return edges


def _check_anti(anti):
    arr = np.asarray(anti)
    if arr.dtype != bool:
        if not np.all(np.isin(arr, (0, 1))):
            raise ValueError("anti must be boolean, or 0/1-valued")
        arr = arr.astype(bool)
    return arr


def _check_reco_bin_range(reco_bin, n_bins):
    if np.any(reco_bin < 0) or np.any(reco_bin > n_bins):
        raise ValueError(
            f"reco_bin values must lie in [0, {n_bins}] ({n_bins} is the "
            "overflow slot)"
        )


def _digitize(reco_variable, edges):
    """Map ``reco_variable`` to bin index in ``[0, n_bins]``, ``n_bins`` =
    overflow slot (both under- and overflow land there).

    Half-open bins ``[edges[k], edges[k+1])`` -- i.e. ``np.digitize`` with
    ``right=False``: a value exactly on an inner edge belongs to the bin it
    starts; a value exactly on the *lower* outer edge belongs to bin 0; a
    value exactly on the *upper* outer edge is treated as >= that edge and
    therefore falls in the overflow slot (there is no bin starting there).
    """
    n_bins = edges.shape[0] - 1
    idx = np.digitize(reco_variable, edges, right=False) - 1
    return np.where(idx < 0, n_bins, idx).astype(np.int32)


# --- shared finalization: sort + pack into an EventSample -------------------


def _finalize(
    name,
    *,
    true_energy,
    reco_bin,
    n_bins,
    flavor_from,
    flavor_to,
    anti,
    nominal_weight,
    bin_edges,
):
    true_energy = np.asarray(true_energy, dtype=np.float64)
    reco_bin = np.asarray(reco_bin, dtype=np.int32)
    flavor_from = np.asarray(flavor_from, dtype=np.int32)
    flavor_to = np.asarray(flavor_to, dtype=np.int32)
    nominal_weight = np.asarray(nominal_weight, dtype=np.float64)

    n = true_energy.shape[0]
    for arr, arr_name in (
        (reco_bin, "reco_bin / reco_variable"),
        (flavor_from, "flavor_from"),
        (flavor_to, "flavor_to"),
        (nominal_weight, "nominal_weight"),
    ):
        _check_same_length(n, arr_name, arr)

    anti_bool = _check_anti(anti)
    _check_same_length(n, "anti", anti_bool)

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    _check_finite(true_energy, "true_energy")
    _check_finite(nominal_weight, "nominal_weight")
    _check_flavor_codes(flavor_from, flavor_to)
    _check_reco_bin_range(reco_bin, n_bins)

    # Stable sort so the neutrino block ([0:anti_split)) comes first, and
    # relative order within each block is preserved. `perm` is what callers
    # need to reorder any *other* per-event auxiliary array (spline
    # event_index columns, norm masks, ...) so it stays aligned with the
    # sorted events returned here -- see the module-level docstrings of
    # make_sample / make_sample_prebinned.
    perm = np.argsort(anti_bool, kind="stable")
    anti_split = int(np.count_nonzero(~anti_bool))

    true_energy = true_energy[perm]
    reco_bin = reco_bin[perm]
    flavor_from = flavor_from[perm]
    flavor_to = flavor_to[perm]
    nominal_weight = nominal_weight[perm]

    sample = EventSample(
        name=name,
        n_bins=n_bins,
        anti_split=anti_split,
        bin_edges=bin_edges,
        true_energy=jnp.asarray(true_energy, dtype=jnp.float64),
        reco_bin=jnp.asarray(reco_bin, dtype=jnp.int32),
        flavor_from=jnp.asarray(flavor_from, dtype=jnp.int32),
        flavor_to=jnp.asarray(flavor_to, dtype=jnp.int32),
        nominal_weight=jnp.asarray(nominal_weight, dtype=jnp.float64),
    )
    return sample, perm


# --- public builders ---------------------------------------------------------


def make_sample(
    name,
    *,
    true_energy,
    reco_variable,
    bin_edges,
    flavor_from,
    flavor_to,
    anti,
    nominal_weight,
) -> tuple[EventSample, np.ndarray]:
    """Build an :class:`EventSample` from plain-numpy per-event arrays.

    Sorts events by ``anti`` (stable sort -- the neutrino block comes first),
    digitizes ``reco_variable`` against ``bin_edges`` (under/overflow mapped
    to the overflow slot ``n_bins = len(bin_edges) - 1``), and validates
    shapes/finiteness/flavor codes/edge monotonicity before packing
    everything into the returned :class:`EventSample`.

    IMPORTANT -- return value is ``(sample, perm)``: ``perm`` is the integer
    permutation array (shape ``(N,)``) used to sort the *input* event order
    into the sample's sorted order. If you have any other per-event auxiliary
    array that must stay aligned with these events -- e.g. a
    ``SplineTable.event_index`` column, or a ``NormSet`` mask column, built
    against the same original event ordering -- you MUST reorder it yourself
    with ``aux_sorted = aux[..., perm]`` (or ``aux[perm]`` for a leading event
    axis) before using it together with the returned sample. The sample
    itself only carries the five canonical arrays; it has no way to reorder
    data it was never given.

    Parameters
    ----------
    name : str
        Sample name (static metadata).
    true_energy : (N,) array_like of float
        True neutrino energy, GeV.
    reco_variable : (N,) array_like of float
        The reconstructed observable to bin on (e.g. reco energy).
    bin_edges : (n_bins + 1,) array_like of float
        Strictly increasing bin edges for ``reco_variable``.
    flavor_from, flavor_to : (N,) array_like of int
        jaxnu ``Flavor`` codes (0=e, 1=mu, 2=tau), or -1 on *both* to mark an
        unoscillated event (e.g. NC background).
    anti : (N,) array_like of bool (or 0/1)
        True/1 for antineutrino events.
    nominal_weight : (N,) array_like of float
        flux x xsec x eff x POT, all oscillation-independent factors.

    Returns
    -------
    (EventSample, np.ndarray)
        The built sample, and the permutation array described above.
    """
    true_energy = np.asarray(true_energy, dtype=np.float64)
    reco_variable = np.asarray(reco_variable)
    _check_same_length(true_energy.shape[0], "reco_variable", reco_variable)

    edges = _check_increasing_edges(bin_edges)
    n_bins = edges.shape[0] - 1
    reco_bin = _digitize(reco_variable, edges)

    return _finalize(
        name,
        true_energy=true_energy,
        reco_bin=reco_bin,
        n_bins=n_bins,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
        bin_edges=tuple(float(e) for e in edges),
    )


def make_sample_prebinned(
    name,
    *,
    true_energy,
    reco_bin,
    n_bins,
    flavor_from,
    flavor_to,
    anti,
    nominal_weight,
    bin_edges=None,
) -> tuple[EventSample, np.ndarray]:
    """Build an :class:`EventSample` from inputs that already carry a bin index.

    Same sorting, validation, and return contract as :func:`make_sample`
    (see its docstring for the important note about the returned permutation
    array) -- but skips digitization: ``reco_bin`` is used as given, and must
    already lie in ``[0, n_bins]`` (``n_bins`` = overflow slot).

    Parameters
    ----------
    reco_bin : (N,) array_like of int
        Pre-computed bin index per event, in ``[0, n_bins]``.
    n_bins : int
        Number of analysis bins (excluding the overflow slot).
    bin_edges : (n_bins + 1,) array_like of float, optional
        If known, stored as reporting metadata on the returned sample (not
        used for any computation here).
    """
    true_energy = np.asarray(true_energy, dtype=np.float64)
    reco_bin = np.asarray(reco_bin)
    _check_same_length(true_energy.shape[0], "reco_bin", reco_bin)

    edges_tuple = None
    if bin_edges is not None:
        edges = _check_increasing_edges(bin_edges)
        if edges.shape[0] - 1 != n_bins:
            raise ValueError(
                f"bin_edges implies {edges.shape[0] - 1} bins but n_bins={n_bins}"
            )
        edges_tuple = tuple(float(e) for e in edges)

    return _finalize(
        name,
        true_energy=true_energy,
        reco_bin=reco_bin,
        n_bins=n_bins,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
        bin_edges=edges_tuple,
    )
