"""A global window's running *folds* must split into ordered buckets and match single-node.

A window with no ``PARTITION BY`` has one partition over every row, so it has no per-partition
seam. `dist.global_window.offsets` gives it a different one: range-partition on the single
``ORDER BY`` key into ordered buckets, window each independently, and fold the prior buckets'
accumulation into the next. `supports_ordered_bucket_offsets` decides which functions that
algebra covers, and a function it declines keeps the materializing kernel — correct, but the
whole relation on one node.

`sum` was covered; the bitwise and boolean running folds are the same arithmetic with a
different identity and were declined only because nobody had written theirs. This pins that
they now split, and that the split answer is the single-node answer.

What makes it non-trivial is the two things the offset has to get right and a single-node run
never exercises:

* a bucket that opens with NULLs and is not the first — the kernel's running value is NULL
  there, while the global value is the prior accumulation, so the identity has to fill before
  the fold (adding to NULL yields NULL under every one of these ops);
* the bucket's own contribution, which is a reduce over its **input** column and not the
  running column's last cell — the kernel hands a bucket's rows back in arrival order, so
  that cell is an arbitrary row's prefix. Writing it the wrong way passed at one partition
  and failed at seven, which is why the partition count is parametrized.

`product` is deliberately not here: it is the one running fold excluded from the algebra,
because over a few thousand values it overflows to `inf` and underflows to `0`, and the two
association orders disagree on `inf * 0`. See `_FOLDS`.

Row *order* is not the thing under test and the two paths legitimately differ on it: the
split path emits rows bucket by bucket (so, in order-key order) while `collect()` emits them
in input order, and an unpartitioned window promises neither. So each row carries a unique
`rid` and both results are sorted by it — in pyarrow, outside the engine — before an
**ordered** comparison. That is stronger than comparing the tables as multisets, which would
pass if a window value were attached to the wrong row and some other row happened to carry
the value this one should have had.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

#: Wide enough that a range partition produces genuinely different buckets at every count
#: below, with NULLs dense enough that a bucket opening on one is near-certain, and duplicate
#: order keys so peer groups straddle what would otherwise be a clean cut.
_N = 3000


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    rng = random.Random(20260826)
    return pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "k": pa.array([rng.randint(0, 250) for _ in range(_N)], pa.int64()),
            "v": pa.array(
                [None if i % 37 == 0 else rng.randint(-(1 << 40), 1 << 40) for i in range(_N)],
                pa.int64(),
            ),
            "b": pa.array(
                [None if i % 41 == 0 else bool(rng.getrandbits(1)) for i in range(_N)],
                pa.bool_(),
            ),
        }
    )


def _by_rid(table: pa.Table) -> pa.Table:
    """`table` in `rid` order — the row identity both paths carry, sorted outside the engine."""
    return table.sort_by([("rid", "ascending")])


@pytest.mark.parametrize(
    "func,column",
    [
        ("sum", "v"),
        ("bit_and", "v"),
        ("bit_or", "v"),
        ("bit_xor", "v"),
        ("bool_and", "b"),
        ("bool_or", "b"),
    ],
)
@pytest.mark.parametrize("partitions", [2, 3, 7, 13])
def test_a_split_global_fold_equals_the_single_node_one(rows, func, column, partitions):
    ds = bt.from_arrow(rows).window(order_by=["k"], functions={"w": (func, bt.col(column))})
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=partitions)),
        _by_rid(ds.collect()),
        ordered=True,
    )


@pytest.mark.parametrize("func,column", [("bit_and", "v"), ("bool_and", "b")])
def test_a_fold_over_only_nulls_stays_null_however_it_is_split(func, column):
    """The identity fills NULLs *for the offset*; it must never become an answer.

    A relation whose input is entirely NULL has no non-null value at any row, so every row's
    running fold is NULL. Filling with the identity and folding unconditionally would return
    the identity (`-1` for `bit_and`, `True` for `bool_and`) instead — a value that was never
    in the input, on the shape most likely to reach the offset with nothing accumulated.
    """
    table = pa.table(
        {
            "rid": pa.array(range(5), pa.int64()),
            "k": pa.array([1, 2, 3, 4, 5], pa.int64()),
            column: pa.array([None] * 5, pa.int64() if column == "v" else pa.bool_()),
        }
    )
    ds = bt.from_arrow(table).window(order_by=["k"], functions={"w": (func, bt.col(column))})
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=3)), _by_rid(ds.collect()), ordered=True
    )
    assert ds.collect().column("w").to_pylist() == [None] * 5


#: Every offsettable function, exercised over a **multi-key** `ORDER BY`. The offsets algebra
#: refused `len(order_keys) != 1`, and a global window is not a `_split_at` pass-through, so
#: nothing carried it up: `ORDER BY a, b` **raised** `PlanError` on distributed data instead of
#: declining to a slower path. All three drivers already cut on the leading key alone.
_MULTI_KEY_FUNCS = {
    "row_number": "row_number",
    "rank": "rank",
    "dense_rank": "dense_rank",
    "sum": ("sum", "v"),
    "avg": ("avg", "v"),
    "min": ("min", "v"),
    "max": ("max", "v"),
    "count": ("count", "v"),
    "first_value": ("first_value", "v"),
    "bit_xor": ("bit_xor", "v"),
}


@pytest.mark.parametrize("func", sorted(_MULTI_KEY_FUNCS))
@pytest.mark.parametrize("partitions", [2, 5])
def test_a_multi_key_global_window_splits_and_matches_single_node(rows, func, partitions):
    """The leading key alone may drive the cut, and the trailing keys may be anything.

    A peer group under a multi-key `ORDER BY` is a set of rows equal on *every* key, so it is
    contained in the set of rows equal on the leading key — which the range partitioner puts
    in one bucket. No peer group straddles a cut, and the buckets stay ordered relative to
    each other because an earlier bucket's rows have a strictly smaller leading key. `rank`
    and `dense_rank` are the two that would expose a broken peer group, so both are here.
    """
    spec = _MULTI_KEY_FUNCS[func]
    functions = {"w": spec if isinstance(spec, str) else (spec[0], bt.col(spec[1]))}
    ds = bt.from_arrow(rows).window(order_by=["k", "v"], functions=functions)
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=partitions)),
        _by_rid(ds.collect()),
        ordered=True,
    )


def test_a_computed_leading_order_key_is_still_refused(rows):
    """The relaxation is to the *trailing* keys only — the control.

    The leading key is the column the range partitioner reads values from, so it must remain
    a plain column of a type that partitioner can cut. Without this, a change that relaxed the
    leading key too would leave every test above green and fail inside the partitioner.
    """
    from batcher.dist.global_window import supports_ordered_bucket_offsets

    computed = bt.from_arrow(rows).window(
        order_by=[bt.col("k") + bt.col("v")], functions={"w": "row_number"}
    )
    assert supports_ordered_bucket_offsets(computed._plan) is False
    multi = bt.from_arrow(rows).window(order_by=["k", "v"], functions={"w": "row_number"})
    assert supports_ordered_bucket_offsets(multi._plan) is True
