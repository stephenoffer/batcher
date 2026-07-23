"""Differential tests for aggregate/group-by depth defects found in the wave-2 sweep.

Each test pins a specific bug: the query returned the wrong answer (or raised) before
the fix and matches DuckDB — and the in-memory path — after it. See
docs/internals/bug_hunt_ledger.md.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential


def test_approx_count_distinct_canonicalizes_signed_zero_and_nan(duck):
    """HLL approx_count_distinct counted -0.0/+0.0 (and NaN payloads) as distinct.

    The exact count(distinct) and DuckDB collapse -0.0==0.0 and all NaNs to one value;
    the HLL path row-encoded raw float bits, so `{-0.0, 0.0, NaN, NaN, 1.5}` estimated 4
    where the exact distinct count is 3. Small cardinalities are in HLL's exact
    linear-counting regime, so the estimate is exact here and can be asserted equal.
    """
    nan = float("nan")
    t = pa.table(
        {
            "g": ["a"] * 5,
            "f": pa.array([-0.0, 0.0, nan, nan, 1.5], pa.float64()),
        }
    )
    out = bt.from_arrow(t).group_by("g").agg(v=bt.col("f").approx_n_unique()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT g, approx_count_distinct(f) AS v FROM t GROUP BY g"))


def test_min_max_over_binary(duck):
    """min/max over a BLOB column raised "not supported"; DuckDB computes it bytewise."""
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b"],
            "x": pa.array([b"foo", b"bar", None, b"zzz", b"aaa"], pa.binary()),
        }
    )
    out = bt.from_arrow(t).group_by("g").agg(lo=bt.col("x").min(), hi=bt.col("x").max()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT g, min(x) AS lo, max(x) AS hi FROM t GROUP BY g"))


def test_min_max_over_large_string(duck):
    """min/max over a LargeUtf8 column raised "not supported"; must match DuckDB."""
    t = pa.table(
        {
            "g": ["a", "a", "a", "b"],
            "x": pa.array(["delta", "alpha", None, "omega"], pa.large_string()),
        }
    )
    out = bt.from_arrow(t).group_by("g").agg(lo=bt.col("x").min(), hi=bt.col("x").max()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT g, min(x) AS lo, max(x) AS hi FROM t GROUP BY g"))


def test_float_group_key_signed_zero_merges_under_spill(duck):
    """A -0.0/0.0 float group key must be ONE group under spill, matching collect + DuckDB.

    The spilling aggregate routed partial rows to hash partitions by a non-canonical float
    encoding, so partials that stored the same SQL group as -0.0 vs 0.0 landed in different
    partitions and were finalized as two groups.
    """
    keys = ([-0.0] * 500 + [0.0] * 500 + [1.0] * 300 + [2.0] * 200) * 3
    vals = list(range(len(keys)))
    t = pa.table({"g": pa.array(keys, pa.float64()), "v": pa.array(vals, pa.int64())})

    def build():
        return bt.from_arrow(t).group_by("g").agg(s=bt.col("v").sum(), n=bt.col("v").count())

    collected = build().collect()
    # Spill must equal the in-memory path exactly (invariant #7)...
    assert_tables_equal(build().collect(spill=True), collected)
    # ...and both must equal DuckDB (which merges -0.0 and 0.0 into one group).
    duck.register("t", t)
    assert_same(collected, duck.sql("SELECT g, sum(v) AS s, count(v) AS n FROM t GROUP BY g"))
