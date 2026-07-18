"""Differential tests for the newly added composable expression functions.

Every function here composes existing IR nodes (no new wire shape), so the proof it is
correct is the oracle: build it in Batcher, compute the same thing in DuckDB on identical
input — including NULLs and the domain edges of the inverse-hyperbolics — and compare.

Covers the `Expr` math methods (`square`, `log1p`, `expm1`, `asinh`, `acosh`, `atanh`),
the rolling moments (`rolling_var` / `rolling_std`), the `.str` additions (`zfill`,
`contains_any`), and the horizontal reducers (`count_horizontal`, `product_horizontal`,
`reduce_horizontal`, `fold_horizontal`).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit
from conftest import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


@pytest.fixture
def nums(duck):
    """Numeric input whose values satisfy every method's domain, plus a NULL."""
    tbl = pa.table(
        {
            # log1p/expm1/asinh/square: defined everywhere; acosh needs >= 1;
            # atanh needs (-1, 1). Separate columns keep each in-domain.
            "x": [0.5, 2.0, 3.5, None],
            "xc": [1.0, 2.0, 3.5, None],
            "xt": [0.1, -0.5, 0.9, None],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_math_methods_match_duckdb(duck, nums):
    out = bt.from_arrow(nums).select(
        sq=col("x").square(),
        l1p=col("x").log1p(),
        em1=col("x").expm1(),
        ash=col("x").asinh(),
        ach=col("xc").acosh(),
        ath=col("xt").atanh(),
    )
    expected = duck.sql(
        "SELECT x*x AS sq, ln(1+x) AS l1p, exp(x)-1 AS em1, asinh(x) AS ash, "
        "acosh(xc) AS ach, atanh(xt) AS ath FROM t"
    )
    assert_same(out.to_arrow(), expected)


@pytest.fixture
def series(duck):
    """An ordered series for the trailing-window moments."""
    tbl = pa.table(
        {
            "i": [1, 2, 3, 4, 5],
            "v": [1.0, 3.0, 6.0, 10.0, 15.0],
        }
    )
    duck.register("s", tbl)
    return tbl


def test_rolling_var_std_match_duckdb_window(duck, series):
    # min_periods == window so leading partial frames are NULL in both engines
    # (DuckDB's var_samp/stddev_samp over a 1-row frame is NULL, matching the guard).
    out = bt.from_arrow(series).select(
        i=col("i"),
        rv=col("v").rolling_var(2, min_periods=2, order_by=["i"]),
        rs=col("v").rolling_std(2, min_periods=2, order_by=["i"]),
    )
    expected = duck.sql(
        "SELECT i, "
        "var_samp(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS rv, "
        "stddev_samp(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS rs "
        "FROM s"
    )
    assert_same_ordered(out.sort("i").to_arrow(), expected.order("i"))


@pytest.fixture
def strs(duck):
    tbl = pa.table({"s": ["7", "42", "1000", None], "w": ["cat", "dog", "bird", None]})
    duck.register("t", tbl)
    return tbl


def test_str_zfill_and_contains_any_match_duckdb(duck, strs):
    out = bt.from_arrow(strs).select(
        padded=col("s").str.zfill(4),
        hit=col("w").str.contains_any(["ca", "ir"]),
    )
    expected = duck.sql(
        "SELECT lpad(s, 4, '0') AS padded, (w LIKE '%ca%' OR w LIKE '%ir%') AS hit FROM t"
    )
    assert_same(out.to_arrow(), expected)


@pytest.fixture
def cols3(duck):
    tbl = pa.table(
        {
            "a": [1, 2, None, 4],
            "b": [10, None, 30, 40],
            "c": [100, 200, 300, None],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_horizontal_reducers_match_duckdb(duck, cols3):
    a, b, c = col("a"), col("b"), col("c")
    out = bt.from_arrow(cols3).select(
        cnt=bt.count_horizontal(a, b, c),
        prod=bt.product_horizontal(a, b, c),
        red=bt.reduce_horizontal(lambda x, y: x + y, a, b, c),
        fold=bt.fold_horizontal(lit(0), lambda s, x: s + x, a, b, c),
    )
    expected = duck.sql(
        "SELECT "
        "(CASE WHEN a IS NULL THEN 0 ELSE 1 END + CASE WHEN b IS NULL THEN 0 ELSE 1 END "
        " + CASE WHEN c IS NULL THEN 0 ELSE 1 END) AS cnt, "
        "coalesce(a, 1) * coalesce(b, 1) * coalesce(c, 1) AS prod, "
        "(a + b + c) AS red, "
        "(0 + a + b + c) AS fold "
        "FROM t"
    )
    assert_same(out.to_arrow(), expected)
