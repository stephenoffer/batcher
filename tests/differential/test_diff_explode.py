"""Differential coverage for `Dataset.explode` (SQL UNNEST) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _lists():
    return pa.table(
        {
            "a": pa.array([[1, 2, 3], [4], [], None], type=pa.list_(pa.int64())),
            "b": pa.array([10, 20, 30, 40], type=pa.int64()),
        }
    )


def test_explode_matches_duckdb_unnest(duck):
    from conftest import assert_same

    out = bt.from_arrow(_lists()).explode("a").collect()
    duck.register("t", _lists())
    # DuckDB UNNEST drops null/empty lists (no row), matching explode.
    assert_same(out, duck.sql("SELECT UNNEST(a) AS a, b FROM t"))


def test_explode_then_filter(duck):
    from conftest import assert_same

    out = bt.from_arrow(_lists()).explode("a").filter(col("a") > 1).collect()
    duck.register("t", _lists())
    assert_same(
        out,
        duck.sql("SELECT * FROM (SELECT UNNEST(a) AS a, b FROM t) WHERE a > 1"),
    )


def test_explode_with_alias(duck):
    from conftest import assert_same

    out = bt.from_arrow(_lists()).explode("a", alias="x").collect()
    duck.register("t", _lists())
    assert_same(out, duck.sql("SELECT UNNEST(a) AS x, b FROM t"))
