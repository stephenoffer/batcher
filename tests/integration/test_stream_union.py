"""A UNION ALL over streams interleaves; over bounded inputs it still concatenates.

Concatenation needs every branch to end, which an unbounded one never does — so a union
of two topics could not stream at all: branch 0 would emit forever and branch 1 never, and
the router refused it with a `PlanError`. Two regions, two producer versions, a backfill
beside a live feed: the shape is ordinary, and Spark unions streaming DataFrames.

UNION ALL is a multiset union and makes no ordering claim, which is exactly what makes
interleaving sound rather than merely convenient. The bounded path keeps concatenating,
because there order *is* available for free and losing it would be a gratuitous change.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

_SCHEMA = pa.schema([("v", pa.int64())])


def _feed(values: list[int]):
    def gen():
        for value in values:
            yield pa.record_batch({"v": [value]}, schema=_SCHEMA)

    return gen


def _stream(values: list[int]):
    return bt.from_batches(_feed(values), _SCHEMA, bounded=False)


def _drain(dataset) -> list[int]:
    rows: list[int] = []
    for batch in dataset.iter_batches():
        rows.extend(batch.to_pydict()["v"])
    return rows


@pytest.mark.integration
def test_a_union_of_two_streams_yields_every_row_from_both():
    got = _drain(_stream([0, 1, 2]).union(_stream([10, 11])))
    assert sorted(got) == [0, 1, 2, 10, 11]


@pytest.mark.integration
def test_the_branches_are_interleaved_rather_than_concatenated():
    """The point of the change: a busy branch cannot starve a quiet one of its place,
    and an unbounded first branch cannot shut the second one out entirely."""
    got = _drain(_stream([0, 1, 2]).union(_stream([10, 11])))
    assert got[:2] == [0, 10], f"branch 1 waited for branch 0 to finish: {got}"


@pytest.mark.integration
def test_a_bounded_union_still_concatenates_in_order():
    """Order is free there, and losing it would be a gratuitous change."""
    left = bt.from_pydict({"v": [0, 1, 2]})
    right = bt.from_pydict({"v": [10, 11]})
    assert _drain(left.union(right)) == [0, 1, 2, 10, 11]


@pytest.mark.integration
def test_a_mixed_bounded_and_unbounded_union_still_streams():
    got = _drain(_stream([0, 1]).union(bt.from_pydict({"v": [10, 11]})))
    assert sorted(got) == [0, 1, 10, 11]


@pytest.mark.integration
def test_a_union_of_streams_writes_to_a_sink():
    query = (
        _stream([0, 1, 2])
        .union(_stream([10, 11]))
        .write.memory("union_sink", trigger=bt.Trigger.available_now())
    )
    assert query.await_termination(timeout=60) is True
    assert query.exception() is None
    assert sorted(bt.read_memory("union_sink").to_pydict()["v"]) == [0, 1, 2, 10, 11]


@pytest.mark.integration
def test_a_distinct_union_over_streams_is_still_refused():
    """`UNION` (distinct) needs a global dedup, which is exactly the whole-relation state
    this path does not have — so it must keep failing rather than silently emit duplicates."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        _drain(_stream([0, 1]).union(_stream([1, 2]), distinct=True))


@pytest.mark.integration
def test_a_union_sink_refuses_a_checkpoint_rather_than_ignoring_it():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="no checkpointable position"):
        _stream([0]).union(_stream([1])).write.memory(
            "union_ckpt", trigger=bt.Trigger.available_now(), checkpoint="/tmp/union-ckpt"
        )
