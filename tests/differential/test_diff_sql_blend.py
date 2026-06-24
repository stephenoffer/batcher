"""Differential tests for the SQL/Python blend: sessions, dialects, DDL, joins, functions."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same


@pytest.fixture
def joins(duck):
    a = pa.table({"k": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    b = pa.table({"k": [1, 2, 3, 4], "y": [15, 15, 35, 5]})
    duck.register("a", a)
    duck.register("b", b)
    return a, b


@pytest.mark.differential
def test_equi_plus_residual_join(duck, joins):
    a, b = joins
    query = "SELECT a.k AS k, a.x AS x, b.y AS y FROM a JOIN b ON a.k = b.k AND a.x < b.y"
    assert_same(bt.sql(query, a=a, b=b).collect(), duck.sql(query))


@pytest.mark.differential
def test_regexp_replace(duck):
    t = pa.table({"s": ["a1b2", "xyz", "c3"]})
    duck.register("t", t)
    query = "SELECT regexp_replace(s, '[0-9]', '#') AS r FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_log_arbitrary_base(duck):
    t = pa.table({"x": [8.0, 27.0, 100.0]})
    duck.register("t", t)
    query = "SELECT log(3, x) AS l FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_dialect_override(duck):
    # `STRPOS` is a Postgres spelling of position(); read it under the postgres dialect.
    t = pa.table({"s": ["hello", "world"]})
    duck.register("t", t)
    out = bt.sql("SELECT STRPOS(s, 'o') AS p FROM t", t=t, dialect="postgres").collect()
    assert out.to_pydict() == {"p": [5, 2]}


@pytest.mark.differential
def test_create_table_as_and_select(duck):
    s = bt.Session()
    src = pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    s.register("src", src)
    s.sql("CREATE TABLE big AS SELECT id, v FROM src WHERE v > 20")
    duck.register("src", src)
    assert_same(
        s.sql("SELECT * FROM big").collect(), duck.sql("SELECT id, v FROM src WHERE v > 20")
    )


@pytest.mark.differential
def test_create_or_replace_view(duck):
    s = bt.Session()
    src = pa.table({"id": [1, 2, 3], "v": [5, 6, 7]})
    s.register("src", src)
    s.sql("CREATE VIEW v1 AS SELECT id FROM src WHERE v >= 6")
    s.sql("CREATE OR REPLACE VIEW v1 AS SELECT id FROM src WHERE v >= 7")
    duck.register("src", src)
    assert_same(s.sql("SELECT * FROM v1").collect(), duck.sql("SELECT id FROM src WHERE v >= 7"))
