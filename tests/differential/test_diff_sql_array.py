"""Array literals `[...]` (per-row List construction) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table({"id": [1, 2, 3], "x": [10, 20, 30], "y": [1, 2, 3]})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        "SELECT [1, 2, 3] a",
        "SELECT ['a', 'b', 'c'] a",
        "SELECT id, [x, y] a FROM t",
        "SELECT id, [x + 1, y * 2] a FROM t",
        "SELECT [x, y, id] a FROM t WHERE id > 1",
    ],
)
def test_array_literal(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_array_dataframe_roundtrip(duck, t):
    """The Array node round-trips through the IR to the engine."""
    from batcher import col
    from batcher.plan.expr_ir import Array
    from conftest import assert_same

    out = bt.from_arrow(t).select("id", a=Array([col("x"), col("y")])).collect()
    assert_same(out, duck.sql("SELECT id, [x, y] a FROM t"))
