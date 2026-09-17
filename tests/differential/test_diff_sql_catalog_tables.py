"""Catalog SQL against DuckDB: schemas, qualified tables, INSERT, SHOW, USE, current_*.

The same script runs through a `bt.Session` and a DuckDB connection, and every result is
compared. A fresh session starts in catalog ``memory``, namespace ``main``, which are
DuckDB's names, so ``current_catalog()`` and ``current_schema()`` are asserted *equal*
rather than merely present.

One deliberate divergence is pinned rather than hidden: an unqualified ``CREATE TABLE t AS``
in Batcher registers a session view (the behaviour it had before catalogs existed), where
DuckDB creates ``memory.main.t``. Both list it in ``SHOW TABLES`` and both answer
``SELECT * FROM t``; only a *qualified* create reaches the catalog, and that is what this
file exercises.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _both(session: bt.Session, duck, *statements: str) -> None:
    for statement in statements:
        session.sql(statement)
        duck.sql(statement)


def test_create_schema_create_table_as_insert_and_select(duck):
    session = bt.Session()
    _both(
        session,
        duck,
        "CREATE SCHEMA sales",
        "CREATE TABLE sales.orders AS SELECT * FROM "
        "(VALUES (1, 'a', 10.5), (2, 'b', NULL)) v(id, name, total)",
        "INSERT INTO sales.orders VALUES (3, 'c', 7.0)",
        "INSERT INTO sales.orders (name, id) VALUES ('d', 4)",
        "INSERT INTO sales.orders SELECT id + 10, name, total FROM sales.orders WHERE id <= 2",
    )
    query = "SELECT id, name, total FROM sales.orders"
    assert_same(session.sql(query).collect(), duck.sql(query))
    query = "SELECT name, count(*) AS n, sum(total) AS s FROM sales.orders GROUP BY name"
    assert_same(session.sql(query).collect(), duck.sql(query))


def test_a_qualified_table_joins_a_session_view(duck):
    session = bt.Session()
    _both(
        session,
        duck,
        "CREATE SCHEMA dim",
        "CREATE TABLE dim.people AS SELECT 1 AS id, 'ann' AS who",
    )
    rows = bt.from_pydict({"id": [1, 1, 2]})
    session.register("events", rows)
    duck.register("events", rows.collect())
    query = "SELECT e.id, p.who FROM events e LEFT JOIN dim.people p ON e.id = p.id"
    assert_same(session.sql(query).collect(), duck.sql(query))


def test_show_tables_in_the_current_and_a_named_schema(duck):
    session = bt.Session()
    _both(
        session,
        duck,
        "CREATE SCHEMA raw",
        "CREATE TABLE raw.events AS SELECT 1 AS e",
        "CREATE TABLE raw.clicks AS SELECT 2 AS c",
        "CREATE TABLE main.orders AS SELECT 3 AS o",
    )
    for statement in ("SHOW TABLES", "SHOW TABLES FROM raw"):
        got = session.sql(statement)
        expected = duck.sql(statement)
        assert got.columns == list(expected.columns)
        assert sorted(got.to_pydict()["name"]) == sorted(r[0] for r in expected.fetchall())


def test_show_databases(duck):
    got = bt.Session().sql("SHOW DATABASES")
    expected = duck.sql("SHOW DATABASES")
    assert got.columns == list(expected.columns)
    assert got.to_pydict()["database_name"] == [r[0] for r in expected.fetchall()]


def test_current_catalog_and_schema_follow_use(duck):
    session = bt.Session()
    query = "SELECT current_catalog() AS c, current_schema() AS s, current_database() AS d"
    assert_same(session.sql(query).collect(), duck.sql(query))
    _both(session, duck, "CREATE SCHEMA staging", "USE staging")
    assert_same(session.sql(query).collect(), duck.sql(query))


def test_use_changes_where_an_unqualified_name_resolves(duck):
    session = bt.Session()
    _both(
        session,
        duck,
        "CREATE SCHEMA staging",
        "CREATE TABLE staging.t AS SELECT 42 AS x",
        "USE staging",
    )
    assert_same(session.sql("SELECT x FROM t").collect(), duck.sql("SELECT x FROM t"))


def test_drop_table_and_drop_schema_cascade(duck):
    session = bt.Session()
    _both(
        session,
        duck,
        "CREATE SCHEMA tmp",
        "CREATE TABLE tmp.a AS SELECT 1 AS x",
        "CREATE TABLE tmp.b AS SELECT 2 AS x",
        "DROP TABLE tmp.a",
    )
    assert sorted(session.sql("SHOW TABLES FROM tmp").to_pydict()["name"]) == ["b"]
    with pytest.raises(bt.PlanError, match="still holds"):
        session.sql("DROP SCHEMA tmp")
    with pytest.raises(Exception, match="depend"):
        duck.sql("DROP SCHEMA tmp")
    _both(session, duck, "DROP SCHEMA tmp CASCADE", "DROP SCHEMA IF EXISTS tmp")
    assert not session.catalog.has_namespace("tmp")
    listed = [
        r[0] for r in duck.sql("SELECT schema_name FROM information_schema.schemata").fetchall()
    ]
    assert "tmp" not in listed


def test_create_or_replace_and_if_not_exists(duck):
    session = bt.Session()
    _both(
        session,
        duck,
        "CREATE SCHEMA s",
        "CREATE TABLE s.t AS SELECT 1 AS x",
        "CREATE TABLE IF NOT EXISTS s.t AS SELECT 2 AS x",
    )
    assert_same(session.sql("SELECT x FROM s.t").collect(), duck.sql("SELECT x FROM s.t"))
    _both(session, duck, "CREATE OR REPLACE TABLE s.t AS SELECT 'z' AS y")
    assert_same(session.sql("SELECT * FROM s.t").collect(), duck.sql("SELECT * FROM s.t"))
    with pytest.raises(bt.PlanError, match="already exists"):
        session.sql("CREATE TABLE s.t AS SELECT 3 AS x")
    with pytest.raises(Exception, match="already exists"):
        duck.sql("CREATE TABLE s.t AS SELECT 3 AS x")
