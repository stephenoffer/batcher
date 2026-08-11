"""The keyless distributed ASOF join returns exactly what one node does.

A `by`-keyed ASOF co-partitions by hash, because a match only ever pairs rows inside one
`by` group. A keyless one has no group: any left row may match any right row, and *which*
one is decided by a global order on `on`. So it range-partitions both sides on `on` through
one shared boundary list and lends each bucket the only out-of-bucket rows that can win —
the largest key below it (backward) and the smallest above it (forward).

That last claim is the whole correctness argument, and it is what these tests hold. They
run the real decomposition (`bucketize`, `_bucket_extremes`, `_asof_carry_rows`, and the
reducer's own IR through the engine) in one process, and compare the concatenation against
the single-node ASOF over the same rows. No Ray: what is under test is the algebra, and the
algebra is what a cluster would get wrong.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.native import engine
from batcher.dist.executor import _asof_carry_rows, _asof_reducer_ir, _bucket_extremes
from batcher.dist.executors.partition_io import bucketize, merge_boundaries, sample_key_grid
from batcher.dist.executors.partition_io.ranges import sample_probs
from batcher.kyber.optimizer import optimize_logical
from batcher.plan.logical import AsofJoin

pytestmark = pytest.mark.unit


def _asof_node(left: pa.Table, right: pa.Table, **kwargs) -> AsofJoin:
    """The optimized `AsofJoin` node for this query, and nothing above it."""
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), **kwargs)
    plan = optimize_logical(ds._plan)
    while not isinstance(plan, AsofJoin):
        plan = plan.input
    return plan


def _sorted_rows(table: pa.Table) -> list[tuple]:
    """The result as an order-independent multiset of rows.

    A distributed ASOF concatenates its buckets in key order and a single-node one emits in
    left-input order, so the two agree on rows and not on sequence — the same as every other
    distributed join. Comparing multisets is what the contract actually says.
    """
    cols = table.column_names
    rows = [tuple(row[c] for c in cols) for row in table.to_pylist()]
    return sorted(rows, key=lambda r: tuple((v is None, v) for v in r))


def _simulate(left: pa.Table, right: pa.Table, n_buckets: int, mappers: int, **kwargs):
    """Run the keyless ASOF's real decomposition in-process and return the assembled table.

    Mirrors `_distributed_asof_keyless` step for step — sample the left key, cut boundaries,
    bucketize both sides through them, measure each mapper's bucket extremes, fold them into
    per-bucket carries, and run the reducer's own IR per bucket — with Ray and the scratch
    files replaced by ordinary calls. Everything that decides a row is the shipped code.
    """
    asof = _asof_node(left, right, **kwargs)
    reducer_ir = json.dumps(_asof_reducer_ir(asof))
    nat = engine()

    def _shards(table: pa.Table) -> list[list[pa.RecordBatch]]:
        step = max(1, -(-table.num_rows // mappers))
        return [table.slice(i * step, step).to_batches() for i in range(mappers)]

    left_shards, right_shards = _shards(left), _shards(right)

    probs = sample_probs(n_buckets, len(left_shards))
    grids = [
        (sample_key_grid(shard, asof.left_on, probs), sum(b.num_rows for b in shard))
        for shard in left_shards
        if shard
    ]
    boundaries = merge_boundaries(grids, n_buckets)

    def _partition(shards, key, schema):
        """Bucketize each shard, seeding an empty bucket with a schema-only batch.

        The shipped task does exactly this, and for the reason the empty-right test pins:
        a bucket file with no columns reads back as "no input schema" rather than as an
        empty side.
        """
        out = []
        for shard in shards:
            buckets = bucketize(shard, key, boundaries, n_buckets, False, False)
            out.append([b or [pa.RecordBatch.from_pylist([], schema=schema)] for b in buckets])
        return out

    left_buckets = _partition(left_shards, asof.left_on, left.schema)
    right_buckets = _partition(right_shards, asof.right_on, right.schema)

    extremes = [
        table
        for table in (_bucket_extremes(b, asof.right_on) for b in right_buckets)
        if table is not None and table.num_rows
    ]
    carries = _asof_carry_rows(extremes, n_buckets, asof.direction, asof.right_on)

    out: list[pa.RecordBatch] = []
    for r in range(n_buckets):
        lhs = [b for shard in left_buckets for b in shard[r]]
        rhs = [b for shard in right_buckets for b in shard[r]]
        if carries[r] is not None:
            rhs = [*rhs, *carries[r].to_batches()]
        if not any(b.num_rows for b in lhs):
            continue
        out.extend(nat.execute_plan(reducer_ir, [lhs, rhs], ""))
    columns = [o.alias for o in asof.output]
    if not out:
        return pa.table({c: [] for c in columns})
    return pa.Table.from_batches([b for b in out if b.num_rows])


def _single_node(left: pa.Table, right: pa.Table, **kwargs) -> pa.Table:
    return bt.from_arrow(left).join_asof(bt.from_arrow(right), **kwargs).collect()


#: A left side whose keys straddle every boundary, with duplicates, a null, and keys that
#: fall before the first right row and after the last one.
LEFT = pa.table(
    {
        "t": pa.array([1, 5, 5, 9, 14, 20, 21, 33, 40, 41, 55, 60, 99, None], pa.int64()),
        "v": pa.array([float(i) for i in range(14)]),
    }
)
#: A right side deliberately sparser than the left, so most matches must cross a boundary.
RIGHT = pa.table(
    {
        "rt": pa.array([5, 12, 12, 30, 42, 58], pa.int64()),
        "q": pa.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
    }
)


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("n_buckets", [1, 2, 4, 8])
def test_the_bucketed_asof_equals_the_single_node_asof(direction, n_buckets):
    kwargs = {"left_on": "t", "right_on": "rt", "direction": direction}
    got = _simulate(LEFT, RIGHT, n_buckets=n_buckets, mappers=3, **kwargs)
    assert _sorted_rows(got) == _sorted_rows(_single_node(LEFT, RIGHT, **kwargs))


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
def test_it_holds_with_exact_matches_refused(direction):
    """The strict form a backtest uses: a right row stamped at the left row's own instant is
    information the left row did not have. The carry is the row *strictly* past the boundary,
    so refusing exact matches must not reach for a row that was left in another bucket."""
    kwargs = {
        "left_on": "t",
        "right_on": "rt",
        "direction": direction,
        "allow_exact_matches": False,
    }
    got = _simulate(LEFT, RIGHT, n_buckets=4, mappers=2, **kwargs)
    assert _sorted_rows(got) == _sorted_rows(_single_node(LEFT, RIGHT, **kwargs))


def test_it_holds_under_a_tolerance():
    """A tolerance turns a far carry into an unmatched row, and that decision is the
    engine's — the carry must still be *offered*, or a match inside tolerance is lost."""
    kwargs = {"left_on": "t", "right_on": "rt", "direction": "backward", "tolerance": 6}
    got = _simulate(LEFT, RIGHT, n_buckets=4, mappers=2, **kwargs)
    assert _sorted_rows(got) == _sorted_rows(_single_node(LEFT, RIGHT, **kwargs))


def test_an_empty_right_side_still_emits_every_left_row():
    """ASOF is left-style, so an empty right side is not an empty result."""
    empty = RIGHT.slice(0, 0)
    kwargs = {"left_on": "t", "right_on": "rt"}
    got = _simulate(LEFT, empty, n_buckets=4, mappers=2, **kwargs)
    assert got.num_rows == LEFT.num_rows
    assert _sorted_rows(got) == _sorted_rows(_single_node(LEFT, empty, **kwargs))


def test_the_carry_is_one_row_per_direction_not_a_bucket():
    """The cost claim, pinned: the carry is O(buckets), never O(rows)."""
    buckets = bucketize(RIGHT.to_batches(), "rt", [12.0, 30.0, 50.0], 4, False, False)
    extremes = _bucket_extremes(buckets, "rt")
    assert extremes is not None
    carries = _asof_carry_rows([extremes], 4, "nearest", "rt")
    assert [c.num_rows if c is not None else 0 for c in carries] == [1, 2, 2, 1]
    backward = _asof_carry_rows([extremes], 4, "backward", "rt")
    assert [c.num_rows if c is not None else 0 for c in backward] == [0, 1, 1, 1]


def test_the_carry_is_the_nearest_row_across_the_boundary():
    """Not merely *a* row from the other side of the boundary — the only one that can win."""
    buckets = bucketize(RIGHT.to_batches(), "rt", [12.0, 30.0, 50.0], 4, False, False)
    carries = _asof_carry_rows([_bucket_extremes(buckets, "rt")], 4, "nearest", "rt")
    # `bucketize` places a key by `searchsorted(side="right")`, so a key equal to a boundary
    # sits *above* it: with boundaries 12/30/50 the buckets hold {5}, {12, 12}, {30, 42},
    # {58}. Below bucket 2 the largest key is 12; above it the smallest is 58.
    assert sorted(carries[2].column("rt").to_pylist()) == [12, 58]


#: A right side where the match is always ambiguous by key and decided by position: three
#: rows share every `on` value, so picking the wrong member of the group returns the right
#: key with the wrong payload — a failure no schema or row-count check can see.
TIED_RIGHT = pa.table(
    {
        "rt": pa.array([12, 12, 12, 30, 30, 30], pa.int64()),
        "q": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
)
TIED_LEFT = pa.table({"t": pa.array([3, 20, 40], pa.int64())})


@pytest.mark.parametrize("direction", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("n_buckets", [2, 4, 6])
def test_a_tied_right_key_picks_the_same_row_one_node_picks(direction, n_buckets):
    """Backward takes the last of a tie group and forward the first, so a carry that keeps an
    arbitrary member is wrong even though its key is right."""
    kwargs = {"left_on": "t", "right_on": "rt", "direction": direction}
    got = _simulate(TIED_LEFT, TIED_RIGHT, n_buckets=n_buckets, mappers=2, **kwargs)
    assert _sorted_rows(got) == _sorted_rows(_single_node(TIED_LEFT, TIED_RIGHT, **kwargs))
