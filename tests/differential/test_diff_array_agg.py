"""Differential coverage for the fluent ``Expr.array_agg()`` aggregate vs DuckDB.

``col("x").array_agg()`` collects each group's non-null values into a ``List`` —
SQL ``array_agg(x)`` / Spark ``collect_list``. Without an ORDER BY the element
order is arrival-dependent (as in DuckDB), so per-group lists are compared as
multisets, not as ordered lists.
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


def _as_sets(rows):
    out = []
    for r in rows:
        norm = {}
        for k, v in r.items():
            norm[k] = tuple(sorted(v)) if isinstance(v, list) else v
        out.append(tuple(sorted(norm.items())))
    return sorted(out)


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


def test_array_agg_with_nulls_dropped(duck):
    tbl = pa.table(
        {
            "g": pa.array([1, 1, 1, 2], type=pa.int64()),
            "v": pa.array([10, None, 30, None], type=pa.int64()),
        }
    )
    duck.register("t2", tbl)
    got = bt.from_arrow(tbl).group_by("g").agg(a=col("v").array_agg()).collect().to_pylist()
    # DuckDB array_agg keeps nulls; Batcher's list_agg drops them. Compare against
    # the non-null reference so the documented "non-null values" contract holds.
    exp = (
        duck.sql("SELECT g, array_agg(v) a FROM t2 WHERE v IS NOT NULL GROUP BY g")
        .to_arrow_table()
        .to_pylist()
    )
    got_nonempty = [r for r in got if r["a"]]
    assert _as_sets(got_nonempty) == _as_sets(exp)
