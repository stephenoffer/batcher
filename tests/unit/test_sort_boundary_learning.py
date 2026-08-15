"""Metadata-driven range-sort boundaries: the quantile grids the SAMPLE barrier measures
are persisted by sort shape and reused on later runs, so a repeat sort range-partitions
without re-executing its mapped prefix over the whole input.

The loop is result-preserving — boundaries decide only which reducer a row lands on, and
the ordered concat is correct for any monotone boundary list — so these tests pin the
persistence semantics and the balance property. `tests/integration/` proves the sorted
relation itself is unchanged end to end.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.partition_io import merge_boundaries
from batcher.dist.sort_boundaries import (
    load_learned_grids,
    persist_grids,
    sort_key_identity,
    sort_key_is_string,
    sort_shape_key,
)

pytestmark = pytest.mark.unit


def test_sort_shape_key_is_stable_and_shape_specific():
    key = sort_shape_key("MAP_IR", "k")
    assert key == sort_shape_key("MAP_IR", "k")  # deterministic
    assert len(key) == 16  # short hash

    # The mapped prefix is part of the shape: the same column behind a different
    # predicate has a genuinely different key distribution and must not share a grid.
    assert sort_shape_key("MAP_IR_FILTERED", "k") != key
    # And so is the key column.
    assert sort_shape_key("MAP_IR", "other") != key


def test_learned_grids_round_trip_and_none_when_never_measured():
    key = sort_shape_key("MAP_IR_ROUNDTRIP", "k")

    # Never sampled → None, so the caller knows it must run the barrier.
    assert load_learned_grids(key) is None

    persist_grids(key, [([0.0, 5.0, 10.0], 100), ([2.0, 6.0, 12.0], 50)])
    assert load_learned_grids(key) == [([0.0, 5.0, 10.0], 100), ([2.0, 6.0, 12.0], 50)]


def test_grids_describing_no_rows_are_not_stored():
    """An empty split contributes nothing to the mixture, so storing it only inflates
    the payload a later run has to read back."""
    key = sort_shape_key("MAP_IR_EMPTY_SPLITS", "k")
    persist_grids(key, [([], 0), ([1.0, 2.0], 10), ([3.0, 4.0], 0)])
    assert load_learned_grids(key) == [([1.0, 2.0], 10)]

    # All-empty stores nothing at all, so the next run samples rather than reusing a
    # grid that describes no rows.
    all_empty = sort_shape_key("MAP_IR_ALL_EMPTY", "k")
    persist_grids(all_empty, [([], 0)])
    assert load_learned_grids(all_empty) is None


def test_learned_grids_recut_to_whatever_bucket_count_this_run_resolved():
    """Grids persist, not boundaries.

    The reducer count moves between runs (the learned shuffle fan-out and the
    `max_shuffle_partitions` cap both trim it), and `merge_boundaries` must emit exactly
    `n_buckets - 1` boundaries — more would route rows past the last bucket. Storing the
    grids keeps that re-cut available; storing a boundary list would not.
    """
    key = sort_shape_key("MAP_IR_RECUT", "k")
    persist_grids(key, [([float(i) for i in range(33)], 1_000)])
    grids = load_learned_grids(key)
    assert grids is not None

    for n_buckets in (2, 4, 8):
        boundaries = merge_boundaries(grids, n_buckets)
        assert len(boundaries) == n_buckets - 1
        # Monotone and deduplicated, which is what makes the ordered concat correct.
        assert boundaries == sorted(boundaries)
        assert len(set(boundaries)) == len(boundaries)


def test_a_stale_grid_still_yields_usable_monotone_boundaries():
    """Staleness costs balance, never correctness.

    A grid learned on one distribution and re-cut against a bucket count it never saw
    still produces monotone, deduplicated boundaries, which is the entire precondition
    the range partitioner needs. This is why the optimization is safe to apply without
    validating that the data still matches.
    """
    key = sort_shape_key("MAP_IR_STALE", "k")
    persist_grids(key, [([0.0, 1.0, 2.0, 3.0], 10)])
    boundaries = merge_boundaries(load_learned_grids(key), 4)
    assert boundaries == sorted(boundaries)
    assert len(boundaries) == 3


# --- the shape key must identify the RELATION and the TYPE, not just the column name ---
#
# A mapped prefix that is a bare scan serializes to `{"op": "scan", "source_id": 0}` -- a
# positional index into this plan's own source list, with no schema and no source identity.
# Every single-source sort in the process therefore hashed alike and shared one grid.
#
#   * wrong type -> it raises: a float boundary list reaches the string range partitioner as
#     `TypeError: argument 'boundaries': 'float' object cannot be converted to 'PyString'`,
#     inside a Ray task, after two retries;
#   * wrong relation -> it silently serializes the reduce: measured over 4,000 rows into 8
#     buckets, `[547, 479, 481, 485, 502, 473, 487, 546]` with the right grid against
#     `[0, 0, 0, 0, 0, 0, 0, 4000]` with another table's -- seven of eight reducers idle.
#
# `kyber.signature` fixed the same defect for learned statistics by putting the source's key
# in its scan token; `sort_key_identity` is that correction for this store.


def test_the_shape_key_separates_two_key_types_with_the_same_name():
    """Identical map IR, identical column name, different type."""
    scan = '{"op": "scan", "source_id": 0}'
    assert sort_shape_key(scan, "k", "src|double") != sort_shape_key(scan, "k", "src|string")


def test_the_shape_key_separates_two_relations_with_the_same_schema():
    """The silent half: same schema, same column, different table."""
    scan = '{"op": "scan", "source_id": 0}'
    assert sort_shape_key(scan, "k", "a.parquet|int64") != sort_shape_key(
        scan, "k", "b.parquet|int64"
    )


def test_an_absent_identity_keeps_the_untyped_digest():
    """A caller that cannot see the source still gets a working key, and the load-side check
    below is what protects it."""
    assert sort_shape_key("MAP_IR", "k") == sort_shape_key("MAP_IR", "k", None)


def test_the_identity_names_the_relation_and_the_type():
    import pyarrow as pa

    import batcher as bt

    strings = bt.from_pydict(
        {"k": ["a"], "v": [1]}, schema=pa.schema([("k", pa.string()), ("v", pa.int64())])
    )
    numbers = bt.from_pydict(
        {"k": [1.0], "v": [1]}, schema=pa.schema([("k", pa.float64()), ("v", pa.int64())])
    )
    assert sort_key_identity(strings._sources[0], "k").endswith("|string")
    assert sort_key_identity(numbers._sources[0], "k").endswith("|double")
    # Different relations, so the tokens must differ even though the column name matches.
    assert sort_key_identity(strings._sources[0], "k") != sort_key_identity(
        numbers._sources[0], "k"
    )


def test_the_same_file_keeps_one_identity_across_datasets():
    """The property the persistence exists for: a later run of the same query must hit.

    Without this, separating the shapes would have quietly disabled the optimization
    instead of fixing it -- the store would be written once per `Dataset` and never read.
    """
    import pathlib as _pathlib
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    import batcher as bt

    tmp = _pathlib.Path(tempfile.mkdtemp())
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    pq.write_table(pa.table({"k": [1, 2], "v": [1, 2]}, schema=schema), tmp / "a.parquet")
    pq.write_table(pa.table({"k": [1, 2], "v": [1, 2]}, schema=schema), tmp / "b.parquet")

    first = sort_key_identity(bt.read.parquet(str(tmp / "a.parquet"))._sources[0], "k")
    again = sort_key_identity(bt.read.parquet(str(tmp / "a.parquet"))._sources[0], "k")
    other = sort_key_identity(bt.read.parquet(str(tmp / "b.parquet"))._sources[0], "k")
    assert first == again, "a second Dataset over the same file must reuse the grid"
    assert first != other, "a different file must not"


def test_an_unreadable_source_yields_no_identity():
    """Best-effort: a source with no schema must not break the sort."""

    class _Opaque:
        pass

    assert sort_key_identity(_Opaque(), "k") is None
    assert sort_key_is_string(_Opaque(), "k") is None


# --- the load-side guard, for entries written under the old colliding digest -----------


def test_a_string_grid_is_refused_for_a_numeric_key():
    key = sort_shape_key("MAP_IR_TYPE_STR", "k")
    persist_grids(key, [(["a", "b", "c"], 100)])
    assert load_learned_grids(key, True) == [(["a", "b", "c"], 100)]
    assert load_learned_grids(key, False) is None


def test_a_numeric_grid_is_refused_for_a_string_key():
    key = sort_shape_key("MAP_IR_TYPE_NUM", "k")
    persist_grids(key, [([0.0, 5.0, 10.0], 100)])
    assert load_learned_grids(key, False) == [([0.0, 5.0, 10.0], 100)]
    assert load_learned_grids(key, True) is None


def test_no_type_opinion_accepts_whatever_was_stored():
    """`None` asks no question, so the pre-existing behavior is untouched."""
    key = sort_shape_key("MAP_IR_TYPE_NONE", "k")
    persist_grids(key, [([1.0, 2.0], 10)])
    assert load_learned_grids(key) == [([1.0, 2.0], 10)]
    assert load_learned_grids(key, None) == [([1.0, 2.0], 10)]


def test_a_mixed_store_is_refused_rather_than_partly_used():
    """One wrong-typed grid poisons the merge, so the whole entry is dropped."""
    key = sort_shape_key("MAP_IR_TYPE_MIXED", "k")
    persist_grids(key, [(["a", "b"], 10), ([1.0, 2.0], 10)])
    assert load_learned_grids(key, True) is None
    assert load_learned_grids(key, False) is None
