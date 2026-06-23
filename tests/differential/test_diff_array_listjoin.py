"""Public DataFrame API for `array(...)` and `.list.join(sep)` vs DuckDB.

The `Array` / `ListJoin` IR nodes were already reachable via SQL (`[a, b]`,
`string_agg`); these cover the expression-API surface (`bt.array`, `.list.join`)
that lowers to the same nodes, checked against DuckDB's `list_value` /
`array_to_string`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import array, col


@pytest.fixture
def t(duck):
    tbl = pa.table({"id": [1, 2, 3], "x": [10, 20, 30], "y": [1, 2, 3]})
    duck.register("t", tbl)
    return tbl


# --- array(...) ------------------------------------------------------------


def test_array_of_columns(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select("id", a=array(col("x"), col("y"))).collect()
    assert_same(out, duck.sql("SELECT id, [x, y] a FROM t"))


def test_array_of_expressions(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(a=array(col("x") + 1, col("y") * 2)).collect()
    assert_same(out, duck.sql("SELECT [x + 1, y * 2] a FROM t"))


def test_array_requires_an_element():
    with pytest.raises(ValueError, match="at least one element"):
        array()


# --- .list.join(sep) -------------------------------------------------------


def test_list_join_after_split(duck):
    from conftest import assert_same

    tbl = pa.table({"s": pa.array(["a,b,c", "x", "p,q", None])})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select(j=col("s").str.split(",").list.join("-")).collect()
    # DuckDB: array_to_string(string_split(s, ','), '-'); null input → null.
    assert_same(out, duck.sql("SELECT array_to_string(string_split(s, ','), '-') j FROM t"))


def test_list_join_of_array_literal(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(j=array(col("x"), col("y")).list.join("|")).collect()
    assert_same(out, duck.sql("SELECT array_to_string([x, y], '|') j FROM t"))
