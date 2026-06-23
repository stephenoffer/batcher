"""INTERSECT / EXCEPT NULL semantics vs DuckDB.

Set operations treat NULLs as equal (a wholly-null row in both inputs intersects,
and is excluded by EXCEPT) — unlike an equi-join, which drops NULL keys. These
tests pin that against DuckDB across single- and multi-column inputs.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _ab(duck):
    a = pa.table({"x": [1, 2, None, 4], "y": [10, 20, None, 40]})
    b = pa.table({"x": [2, None, 4, 5], "y": [20, None, 40, 50]})
    duck.register("a", a)
    duck.register("b", b)
    return bt.from_arrow(a), bt.from_arrow(b)


def test_intersect_with_nulls(duck):
    from conftest import assert_same

    a, b = _ab(duck)
    assert_same(a.intersect(b).collect(), duck.sql("SELECT * FROM a INTERSECT SELECT * FROM b"))


def test_except_with_nulls(duck):
    from conftest import assert_same

    a, b = _ab(duck)
    assert_same(a.except_(b).collect(), duck.sql("SELECT * FROM a EXCEPT SELECT * FROM b"))


def test_intersect_dedups(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 1, 2, 2, 3]})
    b = pa.table({"x": [1, 2, 2, 4]})
    duck.register("a2", a)
    duck.register("b2", b)
    assert_same(
        bt.from_arrow(a).intersect(bt.from_arrow(b)).collect(),
        duck.sql("SELECT * FROM a2 INTERSECT SELECT * FROM b2"),
    )


def test_except_single_column_nulls(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, None, 3]})
    b = pa.table({"x": [2, None]})
    duck.register("a3", a)
    duck.register("b3", b)
    assert_same(
        bt.from_arrow(a).except_(bt.from_arrow(b)).collect(),
        duck.sql("SELECT * FROM a3 EXCEPT SELECT * FROM b3"),
    )


def test_sql_intersect_except_with_nulls(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, None, 4]})
    b = pa.table({"x": [2, None, 5]})
    duck.register("sa", a)
    duck.register("sb", b)
    assert_same(
        bt.sql("SELECT x FROM a INTERSECT SELECT x FROM b", a=a, b=b).collect(),
        duck.sql("SELECT x FROM sa INTERSECT SELECT x FROM sb"),
    )
    assert_same(
        bt.sql("SELECT x FROM a EXCEPT SELECT x FROM b", a=a, b=b).collect(),
        duck.sql("SELECT x FROM sa EXCEPT SELECT x FROM sb"),
    )
