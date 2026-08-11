"""Declaring what a row callback reads must not change what it returns.

`input_columns` on `map`/`flat_map`/`filter` is a promise to the optimizer, and the optimizer
acts on it by narrowing the scan. That makes it the dangerous kind of declaration: get it
wrong and the answer changes rather than the runtime. Each case below runs the same callback
declared and undeclared, and holds both against DuckDB.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same

_ROWS = "(1,10,'aa'),(2,20,'bb'),(3,30,'cc'),(4,40,'dd')"


def _t() -> bt.Dataset:
    return bt.from_arrow(
        pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40], "s": ["aa", "bb", "cc", "dd"]})
    )


def _register(duck) -> None:
    duck.execute(f"CREATE TABLE t AS SELECT * FROM (VALUES {_ROWS}) AS x(id, v, s)")


def _doubled(row: dict) -> dict:
    return {"id": row["id"], "doubled": row["v"] * 2}


def test_declared_map_matches_duckdb(duck):
    ds = _t().ml.map(_doubled, input_columns=["id", "v"], output_columns=["id", "doubled"])
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT id, v * 2 AS doubled FROM t"))


def test_declared_and_undeclared_map_agree(duck):
    declared = _t().ml.map(_doubled, input_columns=["id", "v"], output_columns=["id", "doubled"])
    undeclared = _t().ml.map(_doubled, output_columns=["id", "doubled"])
    _register(duck)
    expected = duck.sql("SELECT id, v * 2 AS doubled FROM t")
    assert_same(declared.collect(), expected)
    assert_same(undeclared.collect(), expected)


def test_declared_flat_map_matches_duckdb(duck):
    ds = _t().ml.flat_map(
        lambda row: [{"id": row["id"]}, {"id": row["id"]}],
        input_columns=["id"],
        output_columns=["id"],
    )
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT id FROM t UNION ALL SELECT id FROM t"))


def test_a_declared_row_callback_composes_with_a_projection(duck):
    """The shape that makes pruning bite: the wide column is read by nobody above the stage."""
    ds = (
        _t()
        .ml.map(_doubled, input_columns=["id", "v"], output_columns=["id", "doubled"])
        .select("doubled")
    )
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT v * 2 AS doubled FROM t"))
