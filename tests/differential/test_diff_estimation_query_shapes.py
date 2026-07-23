"""Differential vs DuckDB for the query shapes Kyber's estimation layer reads specially.

The cardinality estimator was sharpened for a family of predicate/join shapes (bounded
ranges, LIKE patterns, date-part filters, `IN`/`NOT IN`, `COALESCE`/`NULLIF`, disjoint and
skewed joins). Those changes are estimation-only and must not perturb *results* — this file
runs each shape end to end against DuckDB, over data with nulls, duplicates, and boundary
values, so a regression anywhere on the path (not just the estimate) is caught.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "x": pa.array([1, 2, 5, 8, 10, 20, 50, None, 5, 8], type=pa.int64()),
            "y": pa.array([5, 2, 5, 3, 10, 20, 7, 4, 5, None], type=pa.int64()),
            "s": pa.array(
                ["AIR", "AIR MAIL", "RAIL", "SHIP", "TRUCK", None, "AIR", "air", "RAIL", ""],
            ),
            "d": pa.array(
                [datetime.date(2020, m, 1) for m in (1, 3, 6, 6, 9, 12, 6, 1, 3, 12)],
                type=pa.date32(),
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


def _same(out, expected):
    assert_same(out, expected)


# --- bounded ranges / BETWEEN -----------------------------------------------------


def test_between_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("x").between(5, 10)).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x BETWEEN 5 AND 10"))


def test_open_range_conjunction_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter((col("x") > 2) & (col("x") < 20)).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x > 2 AND x < 20"))


def test_out_of_range_equality_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("x") == 999).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x = 999"))


# --- column = column residual -----------------------------------------------------


def test_column_equality_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("x") == col("y")).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x = y"))


# --- LIKE patterns ----------------------------------------------------------------


def test_like_prefix_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("AIR%")).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'AIR%'"))


def test_like_no_wildcard_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("RAIL")).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'RAIL'"))


# --- date-part filters ------------------------------------------------------------


def test_month_filter_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("d").dt.month() == 6).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE month(d) = 6"))


def test_month_in_list_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("d").dt.month().is_in([1, 3])).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE month(d) IN (1, 3)"))


# --- IN / NOT IN (with nulls) -----------------------------------------------------


def test_in_list_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("x").is_in([5, 8, 999])).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x IN (5, 8, 999)"))


def test_not_in_list_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(~col("x").is_in([5, 8])).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x NOT IN (5, 8)"))


# --- OR of same-column equalities -------------------------------------------------


def test_or_of_equalities_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter((col("x") == 5) | (col("x") == 8) | (col("x") == 1)).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x = 5 OR x = 8 OR x = 1"))


# --- COALESCE / NULLIF cleaning ---------------------------------------------------


def test_coalesce_equality_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("x").fill_null(0) == 0).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE coalesce(x, 0) = 0"))


def test_nullif_then_filter_vs_duckdb(duck, t):
    out = bt.from_arrow(t).with_columns(xc=bt.nullif(col("x"), 5)).filter(col("xc") > 7).collect()
    _same(out, duck.sql("SELECT *, nullif(x, 5) AS xc FROM t WHERE nullif(x, 5) > 7"))


# --- aggregates / distinct --------------------------------------------------------


def test_group_by_two_keys_vs_duckdb(duck, t):
    out = bt.from_arrow(t).group_by("x", "y").agg(n=col("s").count()).collect()
    _same(out, duck.sql("SELECT x, y, count(s) AS n FROM t GROUP BY x, y"))


def test_distinct_vs_duckdb(duck, t):
    out = bt.from_arrow(t).select("x", "y").distinct().collect()
    _same(out, duck.sql("SELECT DISTINCT x, y FROM t"))


# --- joins (inner / semi / anti) over keys with duplicates and nulls --------------


@pytest.fixture
def dim(duck):
    tbl = pa.table(
        {"x": pa.array([5, 8, 100], type=pa.int64()), "label": ["five", "eight", "hundred"]}
    )
    duck.register("dim", tbl)
    return tbl


def test_inner_join_vs_duckdb(duck, t, dim):
    out = bt.from_arrow(t).join(bt.from_arrow(dim), on="x").collect()
    _same(out, duck.sql("SELECT * FROM t JOIN dim USING (x)"))


def test_semi_join_vs_duckdb(duck, t, dim):
    out = bt.from_arrow(t).join(bt.from_arrow(dim), on="x", how="semi").collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x IN (SELECT x FROM dim)"))


def test_anti_join_vs_duckdb(duck, t, dim):
    # NOT EXISTS is the null-safe anti-join equivalent (`NOT IN` mishandles nulls).
    out = bt.from_arrow(t).join(bt.from_arrow(dim), on="x", how="anti").collect()
    _same(out, duck.sql("SELECT t.* FROM t WHERE NOT EXISTS (SELECT 1 FROM dim WHERE dim.x = t.x)"))


# --- greatest / least in a filter -------------------------------------------------


def test_greatest_filter_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter(bt.greatest(col("x"), col("y")) > 8).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE greatest(x, y) > 8"))


# --- empty relation edge case -----------------------------------------------------


def test_empty_after_contradiction_vs_duckdb(duck, t):
    out = bt.from_arrow(t).filter((col("x") > 100) & (col("x") < 0)).collect()
    _same(out, duck.sql("SELECT * FROM t WHERE x > 100 AND x < 0"))
