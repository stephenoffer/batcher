"""Differential coverage for the frame-level sugar (fill_null / drop_nulls / cast).

These lower to existing expressions (`select`/`with_columns`/`filter`), so the result
must match DuckDB's equivalent SQL.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "a": pa.array([1, None, 3, None], type=pa.int64()),
            "b": pa.array([10, 20, None, 40], type=pa.int64()),
        }
    )


def test_fill_null_all_columns(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).fill_null(0).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT COALESCE(a, 0) AS a, COALESCE(b, 0) AS b FROM t"))


def test_fill_null_per_column(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).fill_null({"a": -1}).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT COALESCE(a, -1) AS a, b FROM t"))


def test_drop_nulls_subset(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).drop_nulls(["a"]).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT * FROM t WHERE a IS NOT NULL"))


def test_drop_nulls_all(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).drop_nulls().collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT * FROM t WHERE a IS NOT NULL AND b IS NOT NULL"))


def test_cast_per_column(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).cast({"a": "float64"}).drop_nulls(["a"]).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT CAST(a AS DOUBLE) AS a, b FROM t WHERE a IS NOT NULL"))


def test_agg_global(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).agg(s=col("b").sum(), n=bt.count()).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT SUM(b) AS s, COUNT(*) AS n FROM t"))
