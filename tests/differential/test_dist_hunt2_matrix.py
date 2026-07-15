"""Distributed scheduling correctness: single-node == distributed(2) == DuckDB.

Wave-2 bug-hunt coverage for the `dist` scheduling layer (partition assignment,
shard routing, the reduce/combine composition, spill routing). The invariant is that
the distributed executor is a pure *scheduling* concern over the same mergeable
primitives, so a two-worker result MUST equal the single-node result — and both must
equal DuckDB — across the edge-case cross-product the single-shape happy path misses:
``-0.0``/``NaN``/``NULL`` group keys, multi-column keys, a single hot key over a
multi-batch input, empty and single-row inputs, forced spill, and odd (non-divisor)
partition counts.

``num_workers`` is fixed at 2 (the schedulable fan-out in this environment); the
mergeable algebra makes 2-worker equivalence sufficient to exercise the shuffle
(partition -> combine -> finalize) that any higher fan-out reuses verbatim.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

_W = 2


def _both(ds):
    """(single-node table, distributed-2 table) for the same lazy plan."""
    return ds.collect(), ds.collect(distributed=True, num_workers=_W)


# A table whose keys exercise the float/null edges that broke grouping before:
# -0.0 vs 0.0 (bit-different, group-equal), NaN, and NULL, plus a second key column.
_EDGE = pa.table(
    {
        "k": [1, 2, None, 1, 2, 5, None, 4, 5, 1],
        "f": [-0.0, 0.0, float("nan"), 1.5, float("nan"), None, -0.0, 2.5, 0.0, 1.5],
        "g": [10, 20, 10, None, 20, 10, None, 20, 10, None],
        "s": ["a", "b", "", None, "b", "a", "", "c", "", "a"],
        "v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
)


def _edge_ds() -> bt.Dataset:
    return bt.from_arrow(_EDGE)


def test_group_by_int_key_with_nulls():
    ds = _edge_ds().group_by("k").agg(s=col("v").sum(), n=col("v").count(), mn=col("v").min())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(_EDGE).aggregate("k, sum(v) s, count(v) n, min(v) mn", "k")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_group_by_float_edge_key():
    # -0.0 and 0.0 collapse to ONE group; all NaNs collapse to one; NULL its own.
    ds = _edge_ds().group_by("f").agg(n=col("v").count())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(_EDGE).aggregate("f, count(v) n", "f")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_group_by_string_key_empty_and_null():
    ds = _edge_ds().group_by("s").agg(n=col("v").count(), sv=col("v").sum())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(_EDGE).aggregate("s, count(v) n, sum(v) sv", "s")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_group_by_multi_column_key():
    ds = _edge_ds().group_by("k", "g").agg(n=col("v").count())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(_EDGE).aggregate("k, g, count(v) n", "k, g")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_count_distinct_distributed():
    ds = _edge_ds().group_by("k").agg(cd=col("s").n_unique())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(_EDGE).aggregate("k, count(distinct s) cd", "k")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_global_aggregate_distributed():
    ds = _edge_ds().group_by().agg(s=col("v").sum(), n=col("v").count(), mn=col("v").min())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(_EDGE).aggregate("sum(v) s, count(v) n, min(v) mn")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_distinct_multi_and_float_column():
    for cols in (("k", "g"), ("f",), ("s",)):
        ds = _edge_ds().select(*cols).distinct()
        single, dist = _both(ds)
        assert_tables_equal(dist, single)
        import duckdb

        rel = duckdb.arrow(_EDGE).aggregate(", ".join(cols), ", ".join(cols))
        assert_same(single, rel)
        assert_same(dist, rel)


_RIGHT = pa.table({"k": [1, 2, 3, 4, 5, None], "rg": ["A", "B", "C", "D", "E", "Z"]})


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_join_types_distributed(how):
    ds = _edge_ds().join(bt.from_arrow(_RIGHT), left_on="k", right_on="k", how=how)
    single, dist = _both(ds)
    assert_tables_equal(dist, single)


def test_fused_join_group_by_join_key():
    # group by the join key -> exchange elimination (each reducer joins AND aggregates
    # its own co-partitioned bucket). Must still equal single-node and DuckDB.
    ds = (
        _edge_ds()
        .join(bt.from_arrow(_RIGHT), left_on="k", right_on="k", how="inner")
        .group_by("k")
        .agg(s=col("v").sum(), n=col("v").count())
    )
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    con = duckdb.connect()
    con.register("l", _EDGE)
    con.register("r", _RIGHT)
    rel = con.sql("SELECT l.k k, sum(l.v) s, count(l.v) n FROM l JOIN r ON l.k = r.k GROUP BY l.k")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_aggregate_over_join_non_key_group():
    # group by a RIGHT column that is NOT the join key -> the general partial/combine
    # aggregate-over-join path (not exchange elimination).
    ds = (
        _edge_ds()
        .join(bt.from_arrow(_RIGHT), left_on="k", right_on="k", how="inner")
        .group_by("rg")
        .agg(s=col("v").sum())
    )
    single, dist = _both(ds)
    assert_tables_equal(dist, single)


def test_window_partitioned_distributed():
    ds = _edge_ds().with_columns(ws=col("v").sum().over("k"))
    single, dist = _both(ds)
    assert_tables_equal(dist, single)


def test_global_window_distributed():
    ds = _edge_ds().with_columns(tot=col("v").sum().over())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)


def _big_table(n: int = 40000) -> pa.Table:
    rng = random.Random(7)
    return pa.table(
        {
            # key 0 is hot (~1/3 of rows) — stresses a single reducer / salting path.
            "k": [0 if i % 3 == 0 else rng.randint(1, 300) for i in range(n)],
            "v": [rng.randint(-500, 500) for _ in range(n)],
        }
    )


def test_hot_key_multibatch_group_by():
    t = _big_table()
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=col("v").count(), mx=col("v").max())
    single, dist = _both(ds)
    assert_tables_equal(dist, single)
    import duckdb

    rel = duckdb.arrow(t).aggregate("k, sum(v) s, count(v) n, max(v) mx", "k")
    assert_same(single, rel)
    assert_same(dist, rel)


def test_spill_matches_no_spill_distributed():
    t = _big_table()
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=col("v").count())
    no_spill = ds.collect(distributed=True, num_workers=_W)
    spilled = ds.collect(distributed=True, num_workers=_W, spill=True)
    assert_tables_equal(spilled, no_spill)


@pytest.mark.parametrize("np", [1, 7, 13])
def test_odd_partition_counts(np):
    # A partition count that does not divide the group cardinality must not lose,
    # duplicate, or mis-route any group.
    t = _big_table()
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum())
    single = ds.collect()
    dist = ds.collect(distributed=True, num_workers=_W, num_partitions=np)
    assert_tables_equal(dist, single)


def test_empty_and_single_row_distributed():
    empty = bt.from_arrow(_EDGE).filter(col("k") > 1000).group_by("k").agg(s=col("v").sum())
    assert_tables_equal(empty.collect(distributed=True, num_workers=_W), empty.collect())

    one = bt.from_arrow(pa.table({"k": [7], "v": [42]})).group_by("k").agg(s=col("v").sum())
    assert_tables_equal(one.collect(distributed=True, num_workers=_W), one.collect())


def test_asof_join_distributed():
    rng = random.Random(11)
    nl, nr = 1500, 1500
    left = pa.table(
        {
            "by": [rng.choice(["x", "y", "z"]) for _ in range(nl)],  # "z" only on the left
            "t": sorted(rng.randint(0, 800) for _ in range(nl)),
            "lv": list(range(nl)),
        }
    )
    right = pa.table(
        {
            "by": [rng.choice(["x", "y", "w"]) for _ in range(nr)],  # "w" only on the right
            "t": sorted(rng.randint(0, 800) for _ in range(nr)),
            "rv": list(range(nr)),
        }
    )
    for direction in ("backward", "forward"):
        ds = bt.from_arrow(left).join_asof(
            bt.from_arrow(right), on="t", by="by", direction=direction
        )
        single, dist = _both(ds)
        assert_tables_equal(dist, single)
