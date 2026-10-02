"""Grouping, deduplicating and joining on a *clustered* sparse integer key matches DuckDB.

A clustered key repeats in runs without being sorted: a fact table loaded in its surrogate-key
order and then reshuffled by blocks, or the output of a stable hash partition of such a table.
Two kernels reuse the previous row's answer across a run instead of recomputing it -- the hash
path of ``bc_runtime::agg::group::assign::int_group_ids`` (the group id) and the single-integer
fast path of ``bc_runtime::shuffle::bucket_of_rows_salted`` (the partition bucket). Neither is
reached by a sorted key (the run path takes that) or a dense one (the direct map does), so the
fixtures here are deliberately unsorted, sparse, and large enough to shard (> 65,536 rows), with
nulls breaking runs and keys recurring after their run has ended.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col

# Past `MIN_ROWS_TO_SHARD` (4 morsels = 65,536 rows), so the parallel paths partition and shuffle.
_N = 120_000


def _clustered(n: int = _N) -> pa.Table:
    """Runs of 1-5 equal keys, sparse (span ~1e12), unsorted, with nulls and recurring keys."""
    keys: list[int | None] = []
    i = 0
    while len(keys) < n:
        run = 1 + (i % 5)
        # Blocks visited out of order so the key is clustered but not monotonic.
        block = (i * 7919) % 9001
        value = None if i % 41 == 0 else block * 104_729_993 - 5
        keys.extend([value] * run)
        i += 1
    keys = keys[:n]
    return pa.table(
        {
            "k": pa.array(keys, pa.int64()),
            "x": [float(j % 97) - 40.0 for j in range(n)],
            "j": list(range(n)),
        }
    )


def test_clustered_sparse_key_group_by(duck):
    t = _clustered()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("x").sum(), c=col("x").count(), lo=col("j").min())
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT k, sum(x) AS s, count(x) AS c, min(j) AS lo FROM t GROUP BY k")
    )


def test_clustered_sparse_key_distinct(duck):
    t = _clustered()
    duck.register("t", t)
    out = bt.from_arrow(t).select("k").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT k FROM t"))


def test_clustered_sparse_key_join_then_group_by(duck):
    """A join on the clustered key co-partitions both sides through the shuffle fast path."""
    t = _clustered()
    dim = pa.table(
        {
            "k": pa.array([b * 104_729_993 - 5 for b in range(0, 9001, 3)], pa.int64()),
            "w": [float(b % 13) for b in range(0, 9001, 3)],
        }
    )
    duck.register("t", t)
    duck.register("dim", dim)
    out = (
        bt.from_arrow(t)
        .join(bt.from_arrow(dim), on="k")
        .group_by("k")
        .agg(s=(col("x") * col("w")).sum(), n=col("j").count())
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT t.k AS k, sum(x * w) AS s, count(j) AS n FROM t JOIN dim USING (k) GROUP BY t.k"
        ),
    )


def test_clustered_sparse_key_having_ordered(duck):
    """An ordered result over the grouped key: compared in order, not as a multiset."""
    t = _clustered()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("x").sum())
        .filter(col("s") > 100.0)
        .sort("s", "k", descending=[True, False])
        .collect()
        .to_pydict()
    )
    want = duck.sql(
        "SELECT k, sum(x) AS s FROM t GROUP BY k HAVING sum(x) > 100 ORDER BY s DESC, k"
    ).to_arrow_table()
    assert out["k"] == want.column("k").to_pylist()
    assert out["s"] == want.column("s").to_pylist()
    assert len(out["k"]) > 0


def test_clustered_sparse_key_one_row_and_empty(duck):
    one = pa.table({"k": pa.array([123_456_789_012], pa.int64()), "x": [2.0]})
    duck.register("one", one)
    assert_same(
        bt.from_arrow(one).group_by("k").agg(s=col("x").sum()).collect(),
        duck.sql("SELECT k, sum(x) AS s FROM one GROUP BY k"),
    )
    empty = pa.table({"k": pa.array([], pa.int64()), "x": pa.array([], pa.float64())})
    duck.register("empty", empty)
    assert_same(
        bt.from_arrow(empty).group_by("k").agg(s=col("x").sum()).collect(),
        duck.sql("SELECT k, sum(x) AS s FROM empty GROUP BY k"),
    )
