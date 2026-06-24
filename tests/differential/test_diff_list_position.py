"""Differential coverage for `list.position` (DuckDB `list_position`)."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _data():
    return pa.table({"a": pa.array([[10, 20, 30], [5, 5, 7], [], None], type=pa.list_(pa.int64()))})


def test_list_position_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    out = (
        bt.from_arrow(_data())
        .select(p=col("a").list.position(20), q=col("a").list.position(5))
        .collect()
    )
    assert_same(out, duck.sql("SELECT list_position(a, 20) AS p, list_position(a, 5) AS q FROM t"))
