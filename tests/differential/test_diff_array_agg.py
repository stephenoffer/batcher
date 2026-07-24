"""Differential coverage for the fluent ``Expr.array_agg()`` aggregate vs DuckDB.

``col("x").array_agg()`` collects each group's values (INCLUDING nulls, matching
DuckDB ``array_agg(x)`` / ``list(x)``) into a ``List``. Without an ORDER BY the
element order is arrival-dependent (as in DuckDB), so per-group lists are compared
as multisets, not as ordered lists.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "g": pa.array([1, 1, 2, 2, 2], type=pa.int64()),
            "name": ["a", "b", "c", "d", "e"],
            "v": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def _nkey(x):
    # Null-safe sort key: nulls sort last, then by (type-name, value) so a list of
    # mixed None/int (array_agg keeps nulls) sorts without a None-vs-int TypeError.
    return (x is None, type(x).__name__, x) if x is not None else (True,)


def _as_sets(rows):
    out = []
    for r in rows:
        norm = {}
        for k, v in r.items():
            norm[k] = tuple(sorted(v, key=_nkey)) if isinstance(v, list) else v
        out.append(tuple(sorted(norm.items(), key=lambda kv: kv[0])))
    return sorted(out, key=repr)


def test_array_agg_numeric_grouped(t, duck):
    got = bt.from_arrow(t).group_by("g").agg(a=col("v").array_agg()).collect().to_pylist()
    exp = duck.sql("SELECT g, array_agg(v) a FROM t GROUP BY g").to_arrow_table().to_pylist()
    assert _as_sets(got) == _as_sets(exp)


def test_array_agg_string_grouped(t, duck):
    got = bt.from_arrow(t).group_by("g").agg(a=col("name").array_agg()).collect().to_pylist()
    exp = duck.sql("SELECT g, array_agg(name) a FROM t GROUP BY g").to_arrow_table().to_pylist()
    assert _as_sets(got) == _as_sets(exp)


def test_array_agg_global_no_group(t, duck):
    got = bt.from_arrow(t).agg(a=col("v").array_agg()).collect().to_pylist()
    exp = duck.sql("SELECT array_agg(v) a FROM t").to_arrow_table().to_pylist()
    assert _as_sets(got) == _as_sets(exp)


def test_array_agg_keeps_nulls(duck):
    # array_agg keeps null elements (DuckDB semantics): group 1 → [10, None, 30],
    # group 2 → [None]. (Previously Batcher's list_agg dropped nulls — ledger B112.)
    tbl = pa.table(
        {
            "g": pa.array([1, 1, 1, 2], type=pa.int64()),
            "v": pa.array([10, None, 30, None], type=pa.int64()),
        }
    )
    duck.register("t2", tbl)
    got = bt.from_arrow(tbl).group_by("g").agg(a=col("v").array_agg()).collect().to_pylist()
    exp = duck.sql("SELECT g, array_agg(v) a FROM t2 GROUP BY g").to_arrow_table().to_pylist()
    assert _as_sets(got) == _as_sets(exp)


def test_array_agg_over_empty_input_is_null(duck):
    # A global array_agg over ZERO rows is NULL in DuckDB, not an empty list `[]`.
    # (Spark's collect_list returns []; our oracle is DuckDB.) This is the only way a
    # non-null empty list can arise — a real GROUP BY group always has >=1 element.
    tbl = pa.table({"v": pa.array([], type=pa.int64())})
    duck.register("e", tbl)
    got = bt.from_arrow(tbl).agg(a=col("v").array_agg()).collect().to_pylist()
    exp = duck.sql("SELECT array_agg(v) a FROM e").to_arrow_table().to_pylist()
    assert got == exp == [{"a": None}]


def test_array_agg_over_filtered_to_empty_is_null(duck):
    # array_agg over a group filtered down to zero rows is NULL, matching DuckDB.
    tbl = pa.table({"v": pa.array([1, 2, 3], type=pa.int64())})
    duck.register("f", tbl)
    got = (
        bt.from_arrow(tbl).filter(col("v") > 100).agg(a=col("v").array_agg()).collect().to_pylist()
    )
    exp = duck.sql("SELECT array_agg(v) a FROM f WHERE v > 100").to_arrow_table().to_pylist()
    assert got == exp == [{"a": None}]
