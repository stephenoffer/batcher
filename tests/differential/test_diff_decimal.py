"""Decimal128 columns: exact sum/min/max and numeric-literal comparisons vs DuckDB."""

from __future__ import annotations

import decimal as D

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, lit


@pytest.fixture
def t(duck):
    prices = pa.array(
        [
            D.Decimal("1.50"),
            D.Decimal("2.25"),
            D.Decimal("3.75"),
            D.Decimal("1.50"),
            D.Decimal("9.99"),
        ],
        pa.decimal128(10, 2),
    )
    tbl = pa.table({"k": [1, 2, 1, 2, 1], "price": prices})
    duck.register("t", tbl)
    return tbl


def test_decimal_grouped_sum_min_max(duck, t):
    out = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("price").sum(), mn=col("price").min(), mx=col("price").max())
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT k, SUM(price) s, MIN(price) mn, MAX(price) mx FROM t GROUP BY k")
    )


def test_decimal_global_sum(duck, t):
    out = bt.from_arrow(t).group_by().agg(s=col("price").sum()).collect()
    assert_same(out, duck.sql("SELECT SUM(price) s FROM t"))


def test_decimal_filter_against_numeric_literal(duck, t):
    out = bt.from_arrow(t).filter(col("price") > lit(2.0)).select("price").collect()
    assert_same(out, duck.sql("SELECT price FROM t WHERE price > 2.0"))


def test_decimal_sort_and_distinct(duck, t):
    assert_same(
        bt.from_arrow(t).select("price").distinct().collect(),
        duck.sql("SELECT DISTINCT price FROM t"),
    )


@pytest.fixture
def mixed(duck):
    tbl = pa.table(
        {
            "d": pa.array(
                [D.Decimal("1.00"), D.Decimal("2.00"), D.Decimal("3.00")], pa.decimal128(10, 2)
            ),
            "f": pa.array([0.3333333333, 1.5, 0.1], pa.float64()),
        }
    )
    duck.register("m", tbl)
    return tbl


def test_float_column_plus_decimal_promotes_to_double(duck, mixed):
    """A DOUBLE column against a DECIMAL promotes to DOUBLE (DuckDB), keeping the float's
    sub-scale precision — the old path narrowed the float into the decimal's scale, so
    `0.3333333333 + 1.00` collapsed to `1.33`."""
    out = bt.from_arrow(mixed).select((col("f") + col("d")).alias("v")).collect()
    assert_same(out, duck.sql("SELECT f + d AS v FROM m"))


def test_decimal_true_division_is_double(duck, mixed):
    """`a / b` over decimals is true (double) division in DuckDB; the engine used to
    re-narrow the float numerator back to a truncated scale-6 decimal quotient."""
    out = bt.from_arrow(mixed).select((col("d") / col("f")).alias("v")).collect()
    assert_same(out, duck.sql("SELECT d / f AS v FROM m"))


def test_decimal_to_integer_rounds_half_away(duck):
    """`CAST(DECIMAL AS BIGINT)` rounds half-away-from-zero (DuckDB), not toward zero:
    2.5 -> 3, -2.5 -> -3, 0.5 -> 1, -0.5 -> -1."""
    tbl = pa.table(
        {
            "d": pa.array(
                [
                    D.Decimal("2.5"),
                    D.Decimal("3.5"),
                    D.Decimal("-2.5"),
                    D.Decimal("0.5"),
                    D.Decimal("-0.5"),
                    D.Decimal("2.4"),
                    D.Decimal("2.6"),
                ],
                pa.decimal128(10, 1),
            )
        }
    )
    duck.register("d1", tbl)
    out = bt.from_arrow(tbl).select(col("d").cast("int64").alias("v")).collect()
    assert_same(out, duck.sql("SELECT CAST(d AS BIGINT) AS v FROM d1"))


def test_decimal_compare_different_scales(duck):
    """Comparing decimals of different scale (1.0 vs 1.00) compares by value in DuckDB;
    the bare kernel raised "Invalid comparison operation" on the mismatched types."""
    a = [D.Decimal("1.0"), D.Decimal("1.5"), D.Decimal("-2.0")]
    b = [D.Decimal("1.00"), D.Decimal("1.40"), D.Decimal("-2.00")]
    tbl = pa.table(
        {
            "a": pa.array(a, pa.decimal128(10, 1)),
            "b": pa.array(b, pa.decimal128(12, 2)),
        }
    )
    duck.register("c2", tbl)
    out = (
        bt.from_arrow(tbl)
        .select(
            (col("a") == col("b")).alias("eq"),
            (col("a") > col("b")).alias("gt"),
        )
        .collect()
    )
    assert_same(out, duck.sql("SELECT a = b AS eq, a > b AS gt FROM c2"))
