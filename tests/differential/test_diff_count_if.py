"""Differential coverage for `count_if` (desugars to sum of a 0/1 indicator)."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count_if

pytestmark = pytest.mark.differential


def _data():
    return pa.table(
        {
            "g": ["a", "a", "b", "b", "b"],
            "v": pa.array([150000, 50000, 200000, None, 300000], type=pa.int64()),
        }
    )


def test_count_if_grouped_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    out = bt.from_arrow(_data()).group_by("g").agg(n=count_if(col("v") > 100000)).collect()
    # A NULL predicate is treated as false (not counted), matching DuckDB.
    assert_same(out, duck.sql("SELECT g, count_if(v > 100000) AS n FROM t GROUP BY g"))


def test_count_if_global_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    out = bt.from_arrow(_data()).agg(n=count_if(col("v") > 100000)).collect()
    assert_same(out, duck.sql("SELECT count_if(v > 100000) AS n FROM t"))
