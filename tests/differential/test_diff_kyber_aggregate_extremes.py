"""The `aggregate_algebra.extremes` rewrites must match DuckDB after optimization.

The extreme-percentile and self-ordered-`arg_min` collapses change *which reduction the
engine runs*, so the oracle is the only proof the answer is unchanged. The fixture carries
a group with nulls, a group with a single row, a group whose values are all equal (the tie
case `arg_min` turns on), and a group that is entirely null.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.aggregate_algebra
from _harness import assert_same
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "c", "c", "d", "d"], type=pa.string()),
            "f": pa.array([1.5, None, -2.5, 7.0, 4.0, 4.0, None, None], type=pa.float64()),
            "i": pa.array([1, 2, 3, 4, 5, 6, 7, 8], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "g": pa.array([], type=pa.string()),
            "f": pa.array([], type=pa.float64()),
            "i": pa.array([], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_quantile_at_zero_matches_duckdb(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("f").quantile(0.0)).collect()
    assert_same(out, duck.sql("SELECT g, quantile_cont(f, 0.0) AS r FROM t GROUP BY g"))


def test_quantile_at_one_matches_duckdb(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("f").quantile(1.0)).collect()
    assert_same(out, duck.sql("SELECT g, quantile_cont(f, 1.0) AS r FROM t GROUP BY g"))


def test_approx_quantile_at_the_extremes_matches_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(lo=col("f").approx_quantile(0.0), hi=col("f").approx_quantile(1.0))
        .collect()
    )
    assert_same(out, duck.sql("SELECT g, min(f) AS lo, max(f) AS hi FROM t GROUP BY g"))


def test_interior_quantile_is_untouched_and_still_matches(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("f").quantile(0.5)).collect()
    assert_same(out, duck.sql("SELECT g, quantile_cont(f, 0.5) AS r FROM t GROUP BY g"))


def test_quantile_over_an_integer_column_is_untouched_and_still_matches(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("i").quantile(0.0)).collect()
    assert_same(out, duck.sql("SELECT g, quantile_cont(i, 0.0) AS r FROM t GROUP BY g"))


def test_self_ordered_arg_min_matches_duckdb(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("f").arg_min(col("f"))).collect()
    assert_same(out, duck.sql("SELECT g, min_by(f, f) AS r FROM t GROUP BY g"))


def test_self_ordered_arg_max_matches_duckdb(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("f").arg_max(col("f"))).collect()
    assert_same(out, duck.sql("SELECT g, max_by(f, f) AS r FROM t GROUP BY g"))


def test_arg_min_ordered_by_another_column_is_untouched_and_still_matches(duck, t):
    out = bt.from_arrow(t).group_by("g").agg(r=col("f").arg_min(col("i"))).collect()
    assert_same(out, duck.sql("SELECT g, min_by(f, i) AS r FROM t GROUP BY g"))


def test_extremes_over_an_empty_input_match_duckdb(duck, empty):
    out = (
        bt.from_arrow(empty)
        .group_by("g")
        .agg(a=col("f").quantile(0.0), b=col("f").arg_max(col("f")))
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT g, quantile_cont(f, 0.0) AS a, max_by(f, f) AS b FROM t GROUP BY g"),
    )


def test_global_extreme_quantile_matches_duckdb(duck, t):
    out = bt.from_arrow(t).agg(r=col("f").quantile(1.0)).collect()
    assert_same(out, duck.sql("SELECT quantile_cont(f, 1.0) AS r FROM t"))
