"""Differential coverage for `Dataset.explode` (SQL UNNEST) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
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
    out = bt.from_arrow(_lists()).explode("a").collect()
    duck.register("t", _lists())
    # DuckDB UNNEST drops null/empty lists (no row), matching explode.
    assert_same(out, duck.sql("SELECT UNNEST(a) AS a, b FROM t"))


def test_explode_then_filter(duck):
    out = bt.from_arrow(_lists()).explode("a").filter(col("a") > 1).collect()
    duck.register("t", _lists())
    assert_same(
        out,
        duck.sql("SELECT * FROM (SELECT UNNEST(a) AS a, b FROM t) WHERE a > 1"),
    )


def test_explode_with_alias(duck):
    out = bt.from_arrow(_lists()).explode("a", alias="x").collect()
    duck.register("t", _lists())
    assert_same(out, duck.sql("SELECT UNNEST(a) AS x, b FROM t"))


def test_explode_fixed_size_list(duck):
    """`explode` of a fixed-size-list column expands each row into its elements, like a
    variable-length list — previously it errored though the planner advertised a schema
    for it. Null rows drop (UNNEST semantics)."""
    t = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "xs": pa.array([[1, 2], None, [3, 4]], type=pa.list_(pa.int64(), 2)),
        }
    )
    out = bt.from_arrow(t).explode("xs").collect()
    assert out.num_rows == 4  # row 1 (null) drops; rows 0 and 2 give two elems each
    # DuckDB reads the fixed-size list as a regular list for UNNEST.
    reg = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "xs": pa.array([[1, 2], None, [3, 4]], type=pa.list_(pa.int64())),
        }
    )
    duck.register("t", reg)
    assert_same(out, duck.sql("SELECT id, UNNEST(xs) AS xs FROM t"))


def test_explode_alias_collision_raises():
    """`explode(col, alias=other)` where `other` is an existing column must raise —
    otherwise the output carries two same-named columns and silently drops one."""
    from batcher._internal.errors import PlanError

    t = pa.table({"a": [1, 2], "xs": pa.array([[10], [20]])})
    with pytest.raises(PlanError, match="collides"):
        bt.from_arrow(t).explode("xs", alias="a")
