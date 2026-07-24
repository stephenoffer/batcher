"""Differential tests for numeric edge cases fixed in the bug-hunt sweep.

Each test pins a specific defect: the query returned the wrong answer (or crashed)
before the fix and matches DuckDB after it. See docs/internals/bug_hunt_ledger.md
(B4-B14).
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def test_cast_double_to_int_rounds_half_to_even(duck):
    """B4 (rejected): DuckDB's DOUBLE→BIGINT is half-to-even; the engine already matches."""
    t = pa.table({"x": pa.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5], pa.float64())})
    out = bt.from_arrow(t).select(y=bt.col("x").cast("int64")).collect()
    duck.register("t", t)
    # cast the DOUBLE column (not a DECIMAL literal) so DuckDB uses banker's rounding.
    assert_same(out, duck.sql("SELECT CAST(x AS BIGINT) AS y FROM t"))


def test_greatest_least_coerce_mixed_numeric(duck):
    """B5: greatest/least over int/float must coerce, not error."""
    t = pa.table(
        {"a": pa.array([1, 5, 9], pa.int64()), "b": pa.array([2.5, 3.0, -1.0], pa.float64())}
    )
    out = (
        bt.from_arrow(t)
        .select(
            g=bt.greatest(bt.col("a"), bt.col("b")),
            l=bt.least(bt.col("a"), bt.col("b")),
        )
        .collect()
    )
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT greatest(a, b) AS g, least(a, b) AS l FROM t"))


def test_abs_i64_min_saturates_non_negative():
    """B7: abs(i64::MIN) must not crash and must not be negative."""
    t = pa.table({"x": pa.array([-9223372036854775808, -5, 0, 7], pa.int64())})
    out = bt.from_arrow(t).select(y=bt.col("x").abs()).collect().to_pydict()["y"]
    assert all(v >= 0 for v in out), out
    assert out[0] == 9223372036854775807  # i64::MAX (saturated)


def test_list_max_min_propagate_nan(duck):
    """B8: list.max/min treat NaN as greatest (matches DuckDB list_max)."""
    t = pa.table({"xs": pa.array([[1.0, float("nan"), 2.0], [3.0, 4.0]], pa.list_(pa.float64()))})
    out = bt.from_arrow(t).select(mx=bt.col("xs").list.max(), mn=bt.col("xs").list.min()).collect()
    got = out.to_pydict()
    assert math.isnan(got["mx"][0])  # NaN present -> max is NaN
    assert got["mx"][1] == 4.0
    assert got["mn"] == [1.0, 3.0]


def test_variance_stddev_stable_at_large_magnitude(duck):
    """B9: var/stddev must not catastrophically cancel on large-magnitude data."""
    vals = [1_000_000_000 + k for k in range(1, 7)]  # spread 1..6 on a 1e9 offset
    t = pa.table({"v": pa.array(vals, pa.float64())})
    out = bt.from_arrow(t).agg(var=bt.col("v").var(), std=bt.col("v").std()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT var_samp(v) AS var, stddev_samp(v) AS std FROM t"))


def test_grouped_variance_stable_and_matches_duckdb(duck):
    """B9: grouped var over large-magnitude keys stays stable across the group path."""
    g = ["a", "b"] * 100
    v = [1e12 + (i % 5) for i in range(200)]
    t = pa.table({"g": g, "v": pa.array(v, pa.float64())})
    out = bt.from_arrow(t).group_by("g").agg(var=bt.col("v").var()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT g, var_samp(v) AS var FROM t GROUP BY g"))


def test_join_matches_signed_zero():
    """B10/B11: an equi-join treats -0.0 and 0.0 as equal (Batcher canonicalizes float keys).

    This is a Batcher-internal invariant, not a DuckDB differential: DuckDB's join does *not*
    match signed zero (though its scalar `=` does — DuckDB is internally inconsistent here),
    whereas Batcher canonicalizes so that grouping/join keys never split `-0.0` from `0.0`.
    The Rust unit test pins hash == sort-merge; here we assert the observable join result.
    """
    left = pa.table({"k": pa.array([-0.0, 1.5, 2.0], pa.float64()), "l": [1, 2, 3]})
    right = pa.table({"k": pa.array([0.0, 1.5], pa.float64()), "r": [10, 20]})
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    got = out.to_pydict()
    # -0.0 (left) matches 0.0 (right), and 1.5 matches 1.5 -> two rows.
    assert sorted(got["r"]) == [10, 20], got


def test_window_max_propagates_nan(duck):
    """B12: window MAX over a partition containing NaN returns NaN (like aggregate MAX)."""
    t = pa.table({"g": ["a", "a", "a"], "v": pa.array([1.0, float("nan"), 2.0], pa.float64())})
    out = bt.from_arrow(t).with_columns(m=bt.col("v").max().over(partition_by=["g"])).collect()
    got = out.to_pydict()["m"]
    assert all(math.isnan(x) for x in got), got
