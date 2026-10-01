"""DISTINCT and the set operations over wide integer ranges, and value-list merges, vs DuckDB.

Three engine paths these shapes reach that small fixtures do not:

* `DISTINCT` over one `Int64` column takes a **presence bitmap** when its value range is
  cheap at one bit per value — up to `64 x rows / workers` values, far past the million a
  group-id map allows. A `UNION`'s dedup, and the dedup that follows an `INTERSECT` or an
  `EXCEPT`, are this shape whenever the key is an id.
* `MEDIAN` / `QUANTILE_CONT` merge per-group value lists across partials; the merge must keep
  every value and only the values, whatever the partial count.
* A short string key (<= 7 bytes) is grouped on a packed `u64` built from an 8-byte load masked
  to the value's length; values that are prefixes of each other must stay distinct groups.

Every fixture is larger than `MIN_ROWS_TO_SHARD` (65,536 rows) so the parallel paths engage.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

N = 300_000


@pytest.fixture
def ids(duck):
    """Two id tables over a 3M-wide range (negative minimum), overlapping in part."""
    rng = np.random.default_rng(7)
    a = pa.table({"k": pa.array(rng.integers(-1_000_000, 2_000_000, size=N), pa.int64())})
    b = pa.table({"k": pa.array(rng.integers(0, 2_500_000, size=N // 3), pa.int64())})
    duck.register("a", a)
    duck.register("b", b)
    return {"a": a, "b": b}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT DISTINCT k FROM a",
        "SELECT k FROM a UNION SELECT k FROM b",
        "SELECT k FROM a INTERSECT SELECT k FROM b",
        "SELECT k FROM a EXCEPT SELECT k FROM b",
        "SELECT k FROM b EXCEPT SELECT k FROM a",
        "SELECT COUNT(*) AS n FROM (SELECT k FROM a UNION SELECT k FROM b) u",
        "SELECT COUNT(DISTINCT k) AS n FROM a",
    ],
)
def test_wide_range_distinct_and_set_ops_match_duckdb(duck, ids, sql):
    assert_same(bt.sql(sql, **ids).collect(), duck.sql(sql))


def test_wide_range_distinct_with_a_null_matches_duckdb(duck):
    """A null key is a DISTINCT group the bitmap has no slot for: the path must decline."""
    rng = np.random.default_rng(3)
    vals = rng.integers(0, 3_000_000, size=N).tolist()
    vals[1234] = None
    t = pa.table({"k": pa.array(vals, pa.int64())})
    duck.register("t", t)
    sql = "SELECT DISTINCT k FROM t"
    out = bt.sql(sql, t=t).collect()
    assert out.column("k").null_count == 1
    assert_same(out, duck.sql(sql))


@pytest.fixture
def lists(duck):
    """A float and an integer value column under short, prefix-sharing string keys."""
    rng = np.random.default_rng(11)
    keys = np.array(["", "a", "ab", "abc", "abcdefg", "b", "ba", "TRUCK", "REG AIR"])
    k = keys[rng.integers(0, len(keys), size=N)]
    v = np.round(rng.normal(100.0, 30.0, size=N), 4)
    v[::997] = np.nan
    i = rng.integers(-500, 500, size=N)
    t = pa.table(
        {
            "k": pa.array(k, pa.string()),
            "v": pa.array(v, pa.float64()),
            "i": pa.array(i, pa.int64()),
            "g": pa.array(rng.integers(0, 5_000, size=N), pa.int64()),
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k, count(*) AS n FROM t GROUP BY k",
        "SELECT k, median(i) AS m, quantile_cont(i, 0.9) AS q FROM t GROUP BY k",
        "SELECT k, median(v) AS m FROM t WHERE v = v GROUP BY k",
        "SELECT g, median(i) AS m, quantile_cont(i, 0.25) AS q FROM t GROUP BY g",
        "SELECT median(i) AS m, quantile_cont(i, 0.75) AS q FROM t",
    ],
)
def test_value_list_merges_and_short_keys_match_duckdb(duck, lists, sql):
    assert_same(bt.sql(sql, t=lists).collect(), duck.sql(sql))


def test_array_agg_keeps_every_value_across_partials(duck, lists):
    """The merged lists hold exactly each group's values, nulls included (sorted to compare)."""
    sql = "SELECT k, list_sort(array_agg(i)) AS xs, count(*) AS n FROM t GROUP BY k"
    got = bt.sql("SELECT k, array_agg(i) AS xs, count(*) AS n FROM t GROUP BY k", t=lists).collect()
    got = {r["k"]: (sorted(r["xs"]), r["n"]) for r in got.to_pylist()}
    want = {r["k"]: (r["xs"], r["n"]) for r in duck.sql(sql).to_arrow_table().to_pylist()}
    assert got == want
