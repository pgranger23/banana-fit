"""Tests for bananafit.events (DESIGN.md Section 3)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bananafit.events import EventSample, make_sample, make_sample_prebinned


# --- helpers -----------------------------------------------------------------


def _toy_inputs(n=10, seed=0):
    rng = np.random.default_rng(seed)
    true_energy = rng.uniform(0.5, 5.0, size=n)
    reco_variable = rng.uniform(0.0, 6.0, size=n)
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # 5 bins
    flavor_from = rng.integers(0, 3, size=n)
    flavor_to = rng.integers(0, 3, size=n)
    anti = rng.integers(0, 2, size=n).astype(bool)
    nominal_weight = rng.uniform(0.1, 2.0, size=n)
    return dict(
        true_energy=true_energy,
        reco_variable=reco_variable,
        bin_edges=bin_edges,
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=anti,
        nominal_weight=nominal_weight,
    )


# --- round-trip: sorting, anti_split, permutation -----------------------------


def test_round_trip_sorting_and_anti_split():
    kwargs = _toy_inputs(n=12, seed=1)
    sample, perm = make_sample("toy", **kwargs)

    n = len(kwargs["true_energy"])
    assert sample.n_events == n
    assert perm.shape == (n,)
    assert sorted(perm.tolist()) == list(range(n))  # genuine permutation

    anti_bool = np.asarray(kwargs["anti"], dtype=bool)
    expected_anti_split = int(np.count_nonzero(~anti_bool))
    assert sample.anti_split == expected_anti_split

    # nu block first, antinu block second, matching the *sorted* anti array
    sorted_anti = anti_bool[perm]
    assert not np.any(sorted_anti[sample.nu_slice])
    assert np.all(sorted_anti[sample.antinu_slice])

    # slices partition all events, contiguous, no overlap
    assert sample.nu_slice == slice(0, expected_anti_split)
    assert sample.antinu_slice == slice(expected_anti_split, n)

    # sample arrays reflect the reordering by perm
    np.testing.assert_allclose(
        np.asarray(sample.true_energy), kwargs["true_energy"][perm]
    )
    np.testing.assert_allclose(
        np.asarray(sample.nominal_weight), kwargs["nominal_weight"][perm]
    )
    np.testing.assert_array_equal(
        np.asarray(sample.flavor_from), kwargs["flavor_from"][perm]
    )


def test_stable_sort_preserves_relative_order_within_block():
    # anti = [1, 0, 1, 0, 0] -> nu indices [1,3,4], antinu indices [0,2] in
    # original relative order preserved by stable sort.
    n = 5
    kwargs = dict(
        true_energy=np.arange(1.0, n + 1),
        reco_variable=np.array([0.5, 1.5, 2.5, 3.5, 0.2]),
        bin_edges=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        flavor_from=np.array([0, 1, 0, 1, 0]),
        flavor_to=np.array([0, 1, 0, 1, 0]),
        anti=np.array([1, 0, 1, 0, 0]),
        nominal_weight=np.ones(n),
    )
    sample, perm = make_sample("toy", **kwargs)
    np.testing.assert_array_equal(perm, np.array([1, 3, 4, 0, 2]))
    assert sample.anti_split == 3


def test_permutation_reorders_auxiliary_array_correctly():
    kwargs = _toy_inputs(n=8, seed=2)
    sample, perm = make_sample("toy", **kwargs)

    # simulate an external per-event auxiliary array built against the
    # ORIGINAL (pre-sort) event order, e.g. a spline event_index column.
    aux = np.arange(len(kwargs["true_energy"])) * 100
    aux_sorted = aux[perm]

    # aux_sorted[i] should correspond to the same original event as
    # sample.true_energy[i], i.e. aux_sorted[i] // 100 gives back the
    # original index whose true_energy matches.
    original_true_energy = kwargs["true_energy"]
    for i in range(len(perm)):
        orig_idx = aux_sorted[i] // 100
        assert np.asarray(sample.true_energy)[i] == pytest.approx(
            original_true_energy[orig_idx]
        )


# --- digitization edges -------------------------------------------------------


def test_digitize_below_first_and_above_last_go_to_overflow():
    edges = np.array([0.0, 1.0, 2.0, 3.0])  # n_bins = 3, overflow slot = 3
    reco_variable = np.array([-5.0, 10.0])
    n = len(reco_variable)
    sample, _ = make_sample(
        "toy",
        true_energy=np.ones(n),
        reco_variable=reco_variable,
        bin_edges=edges,
        flavor_from=np.zeros(n, dtype=int),
        flavor_to=np.zeros(n, dtype=int),
        anti=np.zeros(n, dtype=bool),
        nominal_weight=np.ones(n),
    )
    assert sample.n_bins == 3
    np.testing.assert_array_equal(np.asarray(sample.reco_bin), [3, 3])


def test_digitize_inner_edges_snap_to_bin_they_start():
    edges = np.array([0.0, 1.0, 2.0, 3.0])  # n_bins = 3
    # values exactly on the inner edges 1.0, 2.0 -> bins 1, 2 (half-open [a,b))
    reco_variable = np.array([1.0, 2.0])
    n = len(reco_variable)
    sample, _ = make_sample(
        "toy",
        true_energy=np.ones(n),
        reco_variable=reco_variable,
        bin_edges=edges,
        flavor_from=np.zeros(n, dtype=int),
        flavor_to=np.zeros(n, dtype=int),
        anti=np.zeros(n, dtype=bool),
        nominal_weight=np.ones(n),
    )
    np.testing.assert_array_equal(np.asarray(sample.reco_bin), [1, 2])


def test_digitize_outer_edges():
    edges = np.array([0.0, 1.0, 2.0, 3.0])  # n_bins = 3, overflow slot = 3
    # lower outer edge -> bin 0 (start of first half-open bin);
    # upper outer edge -> overflow (no bin starts there)
    reco_variable = np.array([0.0, 3.0])
    n = len(reco_variable)
    sample, _ = make_sample(
        "toy",
        true_energy=np.ones(n),
        reco_variable=reco_variable,
        bin_edges=edges,
        flavor_from=np.zeros(n, dtype=int),
        flavor_to=np.zeros(n, dtype=int),
        anti=np.zeros(n, dtype=bool),
        nominal_weight=np.ones(n),
    )
    np.testing.assert_array_equal(np.asarray(sample.reco_bin), [0, 3])


def test_digitize_interior_values_land_in_expected_bins():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    reco_variable = np.array([0.5, 1.5, 2.9])
    n = len(reco_variable)
    sample, _ = make_sample(
        "toy",
        true_energy=np.ones(n),
        reco_variable=reco_variable,
        bin_edges=edges,
        flavor_from=np.zeros(n, dtype=int),
        flavor_to=np.zeros(n, dtype=int),
        anti=np.zeros(n, dtype=bool),
        nominal_weight=np.ones(n),
    )
    np.testing.assert_array_equal(np.asarray(sample.reco_bin), [0, 1, 2])


def test_bin_edges_stored_as_static_metadata():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    n = 3
    sample, _ = make_sample(
        "toy",
        true_energy=np.ones(n),
        reco_variable=np.array([0.5, 1.5, 2.5]),
        bin_edges=edges,
        flavor_from=np.zeros(n, dtype=int),
        flavor_to=np.zeros(n, dtype=int),
        anti=np.zeros(n, dtype=bool),
        nominal_weight=np.ones(n),
    )
    assert sample.bin_edges == (0.0, 1.0, 2.0, 3.0)
    assert isinstance(sample.bin_edges, tuple)


# --- pre-binned path -----------------------------------------------------------


def test_make_sample_prebinned_round_trip():
    n = 6
    reco_bin = np.array([0, 2, 1, 3, 0, 1])  # n_bins=3, so 3 = overflow
    anti = np.array([True, False, True, False, False, True])
    sample, perm = make_sample_prebinned(
        "toy_prebinned",
        true_energy=np.linspace(1.0, 2.0, n),
        reco_bin=reco_bin,
        n_bins=3,
        flavor_from=np.zeros(n, dtype=int),
        flavor_to=np.zeros(n, dtype=int),
        anti=anti,
        nominal_weight=np.ones(n),
    )
    assert sample.n_bins == 3
    assert sample.bin_edges is None
    expected_anti_split = int(np.count_nonzero(~anti))
    assert sample.anti_split == expected_anti_split
    np.testing.assert_array_equal(np.asarray(sample.reco_bin), reco_bin[perm])


def test_make_sample_prebinned_rejects_out_of_range_bin():
    n = 2
    with pytest.raises(ValueError):
        make_sample_prebinned(
            "toy",
            true_energy=np.ones(n),
            reco_bin=np.array([0, 5]),  # n_bins=3 -> valid range [0,3]
            n_bins=3,
            flavor_from=np.zeros(n, dtype=int),
            flavor_to=np.zeros(n, dtype=int),
            anti=np.zeros(n, dtype=bool),
            nominal_weight=np.ones(n),
        )


# --- validation ---------------------------------------------------------------


def test_validation_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(5),
            reco_variable=np.ones(4),  # mismatched length
            bin_edges=np.array([0.0, 1.0, 2.0]),
            flavor_from=np.zeros(5, dtype=int),
            flavor_to=np.zeros(5, dtype=int),
            anti=np.zeros(5, dtype=bool),
            nominal_weight=np.ones(5),
        )


def test_validation_nan_weight_raises():
    n = 3
    weight = np.ones(n)
    weight[1] = np.nan
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(n),
            reco_variable=np.array([0.5, 1.5, 0.2]),
            bin_edges=np.array([0.0, 1.0, 2.0]),
            flavor_from=np.zeros(n, dtype=int),
            flavor_to=np.zeros(n, dtype=int),
            anti=np.zeros(n, dtype=bool),
            nominal_weight=weight,
        )


def test_validation_nonfinite_energy_raises():
    n = 3
    energy = np.array([1.0, np.inf, 2.0])
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=energy,
            reco_variable=np.array([0.5, 1.5, 0.2]),
            bin_edges=np.array([0.0, 1.0, 2.0]),
            flavor_from=np.zeros(n, dtype=int),
            flavor_to=np.zeros(n, dtype=int),
            anti=np.zeros(n, dtype=bool),
            nominal_weight=np.ones(n),
        )


def test_validation_flavor_minus_one_on_only_one_side_raises():
    n = 3
    flavor_from = np.array([-1, 0, 1])
    flavor_to = np.array([0, 0, 1])  # first event: from=-1 but to=0 -> invalid
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(n),
            reco_variable=np.array([0.5, 1.5, 0.2]),
            bin_edges=np.array([0.0, 1.0, 2.0]),
            flavor_from=flavor_from,
            flavor_to=flavor_to,
            anti=np.zeros(n, dtype=bool),
            nominal_weight=np.ones(n),
        )


def test_validation_flavor_minus_one_on_both_sides_is_valid():
    n = 3
    flavor_from = np.array([-1, 0, 1])
    flavor_to = np.array([-1, 0, 1])
    sample, _ = make_sample(
        "toy",
        true_energy=np.ones(n),
        reco_variable=np.array([0.5, 1.5, 0.2]),
        bin_edges=np.array([0.0, 1.0, 2.0]),
        flavor_from=flavor_from,
        flavor_to=flavor_to,
        anti=np.zeros(n, dtype=bool),
        nominal_weight=np.ones(n),
    )
    assert sample.n_events == n


def test_validation_invalid_flavor_code_raises():
    n = 2
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(n),
            reco_variable=np.array([0.5, 1.5]),
            bin_edges=np.array([0.0, 1.0, 2.0]),
            flavor_from=np.array([0, 5]),  # 5 is not a valid flavor code
            flavor_to=np.array([0, 0]),
            anti=np.zeros(n, dtype=bool),
            nominal_weight=np.ones(n),
        )


def test_validation_non_monotone_edges_raises():
    n = 2
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(n),
            reco_variable=np.array([0.5, 1.5]),
            bin_edges=np.array([0.0, 2.0, 1.0]),  # not increasing
            flavor_from=np.zeros(n, dtype=int),
            flavor_to=np.zeros(n, dtype=int),
            anti=np.zeros(n, dtype=bool),
            nominal_weight=np.ones(n),
        )


def test_validation_non_increasing_duplicate_edges_raises():
    n = 2
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(n),
            reco_variable=np.array([0.5, 1.5]),
            bin_edges=np.array([0.0, 1.0, 1.0]),  # not strictly increasing
            flavor_from=np.zeros(n, dtype=int),
            flavor_to=np.zeros(n, dtype=int),
            anti=np.zeros(n, dtype=bool),
            nominal_weight=np.ones(n),
        )


def test_validation_bad_anti_values_raises():
    n = 2
    with pytest.raises(ValueError):
        make_sample(
            "toy",
            true_energy=np.ones(n),
            reco_variable=np.array([0.5, 1.5]),
            bin_edges=np.array([0.0, 1.0, 2.0]),
            flavor_from=np.zeros(n, dtype=int),
            flavor_to=np.zeros(n, dtype=int),
            anti=np.array([0, 2]),  # 2 is not a valid anti flag
            nominal_weight=np.ones(n),
        )


# --- pytree behaviour ----------------------------------------------------------


def test_pytree_leaves_are_exactly_the_five_arrays():
    kwargs = _toy_inputs(n=6, seed=3)
    sample, _ = make_sample("toy", **kwargs)

    leaves = jax.tree_util.tree_leaves(sample)
    assert len(leaves) == 5

    leaf_shapes = sorted(np.asarray(x).shape for x in leaves)
    assert leaf_shapes == [(6,)] * 5

    # every leaf must round-trip through flatten/unflatten unchanged
    flat, treedef = jax.tree_util.tree_flatten(sample)
    rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
    assert rebuilt.name == sample.name
    assert rebuilt.n_bins == sample.n_bins
    assert rebuilt.anti_split == sample.anti_split
    assert rebuilt.bin_edges == sample.bin_edges
    np.testing.assert_array_equal(
        np.asarray(rebuilt.nominal_weight), np.asarray(sample.nominal_weight)
    )


def test_sample_passes_through_jit_unchanged():
    kwargs = _toy_inputs(n=7, seed=4)
    sample, _ = make_sample("toy", **kwargs)

    @jax.jit
    def total_weight(s: EventSample):
        return jnp.sum(s.nominal_weight)

    result = total_weight(sample)
    np.testing.assert_allclose(
        float(result), float(np.sum(kwargs["nominal_weight"]))
    )

    # a static field mismatch (different name) should NOT hit the same
    # compiled cache entry incorrectly -- functional sanity check that
    # jit recompiles/handles a second, differently-shaped sample correctly.
    kwargs2 = _toy_inputs(n=9, seed=5)
    sample2, _ = make_sample("toy2", **kwargs2)
    result2 = total_weight(sample2)
    np.testing.assert_allclose(
        float(result2), float(np.sum(kwargs2["nominal_weight"]))
    )


def test_dtypes_are_float64_and_int32():
    kwargs = _toy_inputs(n=5, seed=6)
    sample, _ = make_sample("toy", **kwargs)
    assert sample.true_energy.dtype == jnp.float64
    assert sample.nominal_weight.dtype == jnp.float64
    assert sample.reco_bin.dtype == jnp.int32
    assert sample.flavor_from.dtype == jnp.int32
    assert sample.flavor_to.dtype == jnp.int32
