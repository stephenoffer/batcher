"""An ASOF join with `by` keys must run out-of-core, and agree with the in-memory answer.

`join_asof(..., by=...)` had a *distributed* decomposition and no bounded-memory one. The
cluster co-partitions both sides on the `by` keys, because an ASOF match only ever pairs rows
that share a `by` group -- so hashing on `by` puts every row that could match a given row in
that row's own bucket, and each bucket is an independent ASOF join whose union is the full
result (`dist.executor._distributed_asof`).

That argument is about cutting one relation into independently-joinable pieces. It does not
mention machines, and it holds identically when the pieces are visited one at a time from disk.
It had only ever been made on the cluster side, so `collect(spill=True)` and `iter_batches()`
both built the whole join in memory -- under the envelope whose purpose is to stop them.

The shape now rides `spill_breakers.stream_spilling_join`, the same grace pipeline the
equi-join uses, with `by` supplying the hash keys. What is checked here is that the answer is
unchanged for every setting that decides an ASOF match, because each is resolved *within* a
bucket and a decomposition that got that wrong would still return plausible rows:

* `direction` -- backward, forward and nearest;
* `tolerance` -- a match rejected for being too far away;
* `allow_exact_matches` -- whether an equal key counts.

A **keyless** ASOF is checked too, and for the opposite reason: it has no group to hash, so it
must *decline* and keep the in-memory kernel rather than co-partition on nothing.

Row order is not under test -- the spilled path emits bucket by bucket -- so each left row
carries a unique `rid` and both results are sorted by it outside the engine.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

_LEFT_ROWS = 1200
_RIGHT_ROWS = 500
#: Enough distinct `by` groups that the hash spreads over several buckets, few enough that
#: every bucket holds more than one group -- the arrangement where mixing two groups inside a
#: bucket would go unnoticed if the kernel did not respect `by`.
_GROUPS = 11


@pytest.fixture(scope="module")
def sides() -> tuple[pa.Table, pa.Table]:
    rng = random.Random(20260826)
    left = pa.table(
        {
            "rid": pa.array(range(_LEFT_ROWS), pa.int64()),
            "g": pa.array([f"s{i % _GROUPS}" for i in range(_LEFT_ROWS)]),
            "t": pa.array(sorted(rng.sample(range(100_000), _LEFT_ROWS)), pa.int64()),
            "v": pa.array([rng.random() for _ in range(_LEFT_ROWS)], pa.float64()),
        }
    )
    right = pa.table(
        {
            "g": pa.array([f"s{i % _GROUPS}" for i in range(_RIGHT_ROWS)]),
            "t": pa.array(sorted(rng.sample(range(100_000), _RIGHT_ROWS)), pa.int64()),
            "q": pa.array([rng.random() for _ in range(_RIGHT_ROWS)], pa.float64()),
        }
    )
    return left, right


#: Every knob that decides which right row an ASOF match picks. Each is resolved inside a
#: bucket, so a co-partition that respected `by` but mishandled one of these would return
#: plausible-looking rows rather than an error.
_SETTINGS = {
    "backward": {},
    "forward": {"direction": "forward"},
    "nearest": {"direction": "nearest"},
    "tolerance": {"tolerance": 500},
    "forward_tolerance": {"direction": "forward", "tolerance": 500},
    "no_exact_matches": {"allow_exact_matches": False},
}


def _by_rid(table: pa.Table) -> pa.Table:
    return table.sort_by([("rid", "ascending")])


@pytest.mark.parametrize("setting", sorted(_SETTINGS))
@pytest.mark.parametrize("partitions", [2, 6])
def test_a_spilled_asof_join_equals_the_in_memory_one(sides, setting, partitions):
    left, right = sides
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g", **_SETTINGS[setting])
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=partitions)),
        _by_rid(ds.collect()),
        ordered=True,
    )


@pytest.mark.parametrize("setting", sorted(_SETTINGS))
def test_a_streamed_asof_join_equals_the_in_memory_one(sides, setting):
    left, right = sides
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g", **_SETTINGS[setting])
    batches = list(ds.iter_batches())
    streamed = pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)
    assert_tables_equal(_by_rid(streamed), _by_rid(ds.collect()), ordered=True)


@pytest.mark.parametrize("partitions", [1, 2, 3, 8])
def test_a_by_group_with_no_right_rows_survives_null_extended(partitions):
    """The edge a co-partitioned ASOF is most likely to *drop* rather than get wrong.

    An ASOF join emits every left row, matched or not. Under co-partitioning, a `by` group
    present on the left and absent on the right lands in a bucket whose right side is empty —
    and a bucket pipeline that skipped an empty-right bucket, or that let the empty side come
    through untyped, would silently return fewer rows than the in-memory kernel. Every value
    in the result would still be correct, which is what makes it hard to notice.

    Parametrized over the partition count because how many groups share a bucket is exactly
    what the hash decides: at one partition nothing is empty, and the defect cannot appear.
    """
    left = pa.table(
        {
            "rid": pa.array(range(9), pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"]),
            "t": pa.array([1, 5, 9, 2, 6, 10, 3, 7, 11], pa.int64()),
        }
    )
    right = pa.table(
        {
            "g": pa.array(["b", "b"]),
            "t": pa.array([1, 6], pa.int64()),
            "q": pa.array([100, 200], pa.int64()),
        }
    )
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g")
    expected = ds.collect()
    assert expected.column("q").to_pylist() == [None, None, None, 100, 200, 200, None, None, None]
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=partitions)),
        _by_rid(expected),
        ordered=True,
    )


def test_an_empty_right_side_still_emits_every_left_row():
    """The degenerate case of the above: no bucket has any right rows at all."""
    left = pa.table(
        {
            "rid": pa.array(range(5), pa.int64()),
            "g": pa.array(["a", "b", "a", "b", "a"]),
            "t": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    )
    right = pa.table(
        {
            "g": pa.array([], pa.string()),
            "t": pa.array([], pa.int64()),
            "q": pa.array([], pa.int64()),
        }
    )
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g")
    spilled = ds.collect(spill=True, num_partitions=3)
    assert spilled.num_rows == 5
    assert_tables_equal(_by_rid(spilled), _by_rid(ds.collect()), ordered=True)


def test_a_keyless_asof_join_declines_the_grace_path_and_still_answers(sides):
    """No `by` means nothing to hash on, so this must fall back rather than co-partition.

    The cluster reaches a keyless ASOF by range-partitioning on `on` and lending each bucket
    the one boundary row that can match across the cut, which is a different decomposition.
    Hashing on an empty key set would put every row in one bucket -- correct but pointless --
    and a partial key set would be silently wrong, so the predicate refuses outright.
    """
    from batcher.dist.spill_breakers import supports_spilling_join

    left, right = sides
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t")
    assert supports_spilling_join(ds._plan) is False
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=4)), _by_rid(ds.collect()), ordered=True
    )
