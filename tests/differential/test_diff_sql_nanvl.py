"""Correctness vs DuckDB for the SQL `nanvl` spelling.

`bt.nanvl` was reachable from Python and from the Anonymous-function table, but sqlglot
parses `nanvl(...)` into a *typed* `exp.Nanvl` node, so the SQL spelling never reached the
Anonymous table and raised `unsupported SQL expression: Nanvl` instead.

DuckDB has no `nanvl`, so the oracle is its explicit equivalent:
``CASE WHEN isnan(x) THEN fallback ELSE x END``. The distinction that matters is NaN
against NULL -- `nanvl` replaces only NaN, and a NULL input stays NULL, which is what
separates it from `coalesce`.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same


def _t(duck, name="nv"):
    t = pa.table(
        {
            "k": pa.array([1, 2, 3, 4, 5], pa.int64()),
            # 1.5, NaN, NULL, -0.0, inf -- the edges nanvl must tell apart.
            "x": pa.array([1.5, float("nan"), None, -0.0, float("inf")], pa.float64()),
        }
    )
    duck.register(name, t)
    return bt.from_arrow(t)


def test_sql_nanvl_matches_duckdbs_isnan_case(duck):
    ds = _t(duck, "nv1")
    out = ds.sql("SELECT k, nanvl(x, 99.0) AS r FROM self").collect()
    assert_same(
        out,
        duck.sql("SELECT k, CASE WHEN isnan(x) THEN 99.0 ELSE x END AS r FROM nv1"),
    )


def test_sql_nanvl_leaves_null_null(duck):
    """The NaN/NULL distinction: `coalesce` would replace the NULL, `nanvl` must not."""
    ds = _t(duck, "nv2")
    out = ds.sql("SELECT k, nanvl(x, 99.0) AS r FROM self WHERE k = 3").collect()
    assert_same(
        out,
        duck.sql("SELECT k, CASE WHEN isnan(x) THEN 99.0 ELSE x END AS r FROM nv2 WHERE k = 3"),
    )
    assert out.column("r").to_pylist() == [None], "a NULL input must stay NULL, not become 99.0"


def test_sql_nanvl_agrees_with_the_python_function(duck):
    """The SQL spelling and `bt.nanvl` must lower to the same expression."""
    ds = _t(duck, "nv3")
    via_sql = ds.sql("SELECT nanvl(x, 99.0) AS r FROM self").collect()
    via_py = ds.select(r=bt.nanvl(bt.col("x"), bt.lit(99.0))).collect()
    assert via_sql.column("r").to_pylist() == via_py.column("r").to_pylist()
