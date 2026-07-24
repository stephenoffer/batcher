"""Differential tests: a keyless aggregate over a LIMIT / top-N must not be answered
from whole-relation source statistics.

Regression for the metadata-aggregate fast path (`metadata_answer.aggregate`). A
`LIMIT` (or a `Sort` folded into a top-N with a limit) below a keyless aggregate
restricts *which* rows are aggregated, so no whole-relation statistic — sum, mean,
min, max — is the answer. The fast path used to derive `sum(a)` over `t ORDER BY a
DESC LIMIT 2` from the source's total column sum, returning the whole-column value
(e.g. 360 instead of 150) — a metadata answer that silently disagreed with both
execution and DuckDB. The gate now declines any keyless aggregate over a
row-limited subtree, so it executes.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher import col


def _t(duck):
    t = pa.table(
        {
            "k": pa.array([1, 2, 3, 4, 5, 6, 7, 8], pa.int64()),
            "a": pa.array([10, 20, 30, 40, 50, 60, 70, 80], pa.int64()),
        }
    )
    duck.register("t", t)
    return bt.from_arrow(t)


def test_sum_over_topn(duck):
    ds = _t(duck)
    out = ds.sort("a", descending=True).limit(2).agg(s=col("a").sum()).collect()
    assert_same_ordered(
        out, duck.sql("SELECT sum(a) AS s FROM (SELECT a FROM t ORDER BY a DESC LIMIT 2)")
    )


def test_mean_over_topn(duck):
    ds = _t(duck)
    out = ds.sort("a", descending=True).limit(2).agg(s=col("a").mean()).collect()
    assert_same_ordered(
        out, duck.sql("SELECT avg(a) AS s FROM (SELECT a FROM t ORDER BY a DESC LIMIT 2)")
    )


def test_min_over_topn(duck):
    ds = _t(duck)
    out = ds.sort("a", descending=True).limit(2).agg(s=col("a").min()).collect()
    assert_same_ordered(
        out, duck.sql("SELECT min(a) AS s FROM (SELECT a FROM t ORDER BY a DESC LIMIT 2)")
    )


def test_max_over_asc_topn(duck):
    ds = _t(duck)
    out = ds.sort("a").limit(3).agg(s=col("a").max()).collect()
    assert_same_ordered(
        out, duck.sql("SELECT max(a) AS s FROM (SELECT a FROM t ORDER BY a ASC LIMIT 3)")
    )


def test_sum_over_plain_limit(duck):
    ds = _t(duck)
    out = ds.limit(3).agg(s=col("a").sum()).collect()
    assert_same(out, duck.sql("SELECT sum(a) AS s FROM (SELECT a FROM t LIMIT 3)"))


def test_whole_relation_aggregate_still_exact(duck):
    """The fast path must still fire for a genuine whole-relation aggregate (no regression)."""
    ds = _t(duck)
    out = ds.agg(s=col("a").sum(), mn=col("a").min(), mx=col("a").max()).collect()
    assert_same(out, duck.sql("SELECT sum(a) AS s, min(a) AS mn, max(a) AS mx FROM t"))


def test_sorted_no_limit_aggregate_still_exact(duck):
    """A `Sort` with no limit is stat-preserving, so the aggregate answer stays valid."""
    ds = _t(duck)
    out = ds.sort("a").agg(s=col("a").sum()).collect()
    assert_same_ordered(out, duck.sql("SELECT sum(a) AS s FROM t"))


# --- scalar column shortcuts (ds.min / ds.max / ds.n_unique) over a top-N ------


def test_min_column_shortcut_over_topn(duck):
    """`ds.min(col)` over a top-N must reflect the surviving rows, not the global min."""
    ds = _t(duck)
    limited = ds.sort("a", descending=True).limit(2)  # rows: 80, 70
    got = limited.min("a")
    want = duck.sql("SELECT min(a) FROM (SELECT a FROM t ORDER BY a DESC LIMIT 2)").fetchone()[0]
    assert got == want == 70


def test_max_column_shortcut_over_topn(duck):
    ds = _t(duck)
    limited = ds.sort("a").limit(2)  # ascending → rows: 10, 20
    got = limited.max("a")
    want = duck.sql("SELECT max(a) FROM (SELECT a FROM t ORDER BY a ASC LIMIT 2)").fetchone()[0]
    assert got == want == 20


def test_n_unique_column_shortcut_over_limit(duck):
    t = pa.table({"a": pa.array([1, 1, 2, 2, 3, 3, 4, 4], pa.int64())})
    duck.register("u", t)
    ds = bt.from_arrow(t).limit(3)  # first 3 rows: 1, 1, 2 → 2 distinct
    got = ds.n_unique("a")
    want = duck.sql("SELECT count(DISTINCT a) FROM (SELECT a FROM u LIMIT 3)").fetchone()[0]
    assert got == want == 2


def test_min_column_shortcut_whole_relation_unaffected(duck):
    """No limiter → the metadata scalar shortcut still answers the true global min/max."""
    ds = _t(duck)
    assert ds.min("a") == duck.sql("SELECT min(a) FROM t").fetchone()[0] == 10
    assert ds.max("a") == duck.sql("SELECT max(a) FROM t").fetchone()[0] == 80
