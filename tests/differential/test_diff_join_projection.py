"""`push_projection_through_join` preserves results vs DuckDB.

Pushing a derived column through a join changes the plan, never the answer. The edges: a
single-side push, an overwrite (the item's alias collides with a join-output name), a
mixed-side expression that must NOT push, a non-inner join that must NOT push, and a
boolean/comparison push — each compared end to end against DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col


@pytest.fixture
def tables(duck):
    left = pa.table(
        {
            "k": pa.array([1, 2, 3, 4], type=pa.int64()),
            "a": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
            "b": pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float64()),
        }
    )
    right = pa.table(
        {
            "k": pa.array([1, 2, 3, 5], type=pa.int64()),
            "grp": pa.array(["x", "y", "x", "z"]),
            "c": pa.array([5.0, 6.0, 7.0, 8.0], type=pa.float64()),
        }
    )
    duck.register("l", left)
    duck.register("r", right)
    return bt.from_arrow(left), bt.from_arrow(right)


def test_single_side_push_then_group(tables, duck):
    left, right = tables
    ds = (
        left.join(right, on="k", how="inner")
        .with_columns(rev=col("a") * (1 - col("b")))
        .group_by("grp")
        .agg(s=col("rev").sum())
    )
    assert_same(
        ds.collect(), duck.sql("SELECT grp, sum(a*(1-b)) AS s FROM l JOIN r USING(k) GROUP BY grp")
    )


def test_overwrite_existing_column(tables, duck):
    left, right = tables
    ds = left.join(right, on="k", how="inner").with_columns(a=col("a") * 2).select("grp", "a")
    assert_same(ds.collect(), duck.sql("SELECT grp, a*2 AS a FROM l JOIN r USING(k)"))


def test_mixed_side_stays_correct(tables, duck):
    left, right = tables
    ds = (
        left.join(right, on="k", how="inner")
        .with_columns(mix=col("a") + col("c"))
        .select("grp", "mix")
    )
    assert_same(ds.collect(), duck.sql("SELECT grp, a+c AS mix FROM l JOIN r USING(k)"))


def test_left_join_stays_correct(tables, duck):
    left, right = tables
    ds = (
        left.join(right, on="k", how="left")
        .with_columns(rev=col("a") * (1 - col("b")))
        .select("k", "rev")
    )
    assert_same(ds.collect(), duck.sql("SELECT k, a*(1-b) AS rev FROM l LEFT JOIN r USING(k)"))


def test_comparison_push(tables, duck):
    left, right = tables
    ds = left.join(right, on="k", how="inner").with_columns(big=col("a") > 15).select("grp", "big")
    assert_same(ds.collect(), duck.sql("SELECT grp, (a>15) AS big FROM l JOIN r USING(k)"))
