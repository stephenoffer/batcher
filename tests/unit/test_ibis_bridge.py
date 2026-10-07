"""The Ibis bridge against a stand-in ``ibis`` module, running SQL in Ibis's own shape.

Ibis is not installed here, so a stand-in module provides the three public calls the bridge
uses (``ibis.to_sql``, ``ibis.table``, ``ibis.Schema.from_pyarrow``). ``to_sql`` returns SQL
written the way Ibis 9's DuckDB compiler writes it: quoted ``t0`` aliases, ``CAST`` on
literals, ``GROUP BY 1`` and a nested select. That proves the bridge's wiring and that
Batcher translates those idioms; it does not prove what a given Ibis release emits, which is
what `tests/integration/live/test_live_ibis.py` checks with Ibis installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pyarrow as pa
import pytest

import batcher as bt
from batcher.integrations import ibis as bt_ibis

pytestmark = pytest.mark.unit

_FILTER = (
    'SELECT "t0"."id" FROM "orders" AS "t0" WHERE "t0"."amount" > CAST(6 AS TINYINT) '
    'ORDER BY "t0"."id" ASC'
)
_AGGREGATE = (
    'SELECT "t1"."g", "t1"."total" FROM (SELECT "t0"."g", SUM("t0"."v") AS "total" '
    'FROM "t" AS "t0" GROUP BY 1) AS "t1" ORDER BY "t1"."g" ASC'
)
_JOIN = (
    'SELECT "t2"."id", "t3"."label" FROM "orders" AS "t2" '
    'INNER JOIN "labels" AS "t3" ON "t2"."id" = "t3"."id" ORDER BY "t2"."id" ASC'
)


class _Expr:
    def __init__(self, sql: str) -> None:
        self.sql = sql


@pytest.fixture
def fake_ibis(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    calls: dict[str, Any] = {}
    module = types.ModuleType("ibis")

    def to_sql(expr: _Expr, dialect: str | None = None) -> str:
        calls["dialect"] = dialect
        return expr.sql

    def table(schema: Any, name: str) -> Any:
        return types.SimpleNamespace(schema=schema, name=name)

    module.to_sql = to_sql
    module.table = table
    module.Schema = types.SimpleNamespace(from_pyarrow=lambda s: ("ibis-schema", s))
    module.calls = calls
    monkeypatch.setitem(sys.modules, "ibis", module)
    return module


@pytest.fixture
def session() -> bt.Session:
    s = bt.Session()
    s.register("orders", bt.from_pydict({"id": [3, 1, 2], "amount": [9.0, 5.0, 7.5]}))
    s.register("t", bt.from_pydict({"g": ["b", "a", "a", None], "v": [2, 1, 3, 4]}))
    s.register("labels", bt.from_pydict({"id": [1, 2], "label": ["one", "two"]}))
    return s


def test_ibis_sql_runs_lazily_and_compiles_for_duckdb(fake_ibis, session) -> None:
    ds = bt_ibis.to_dataset(_Expr(_FILTER), session)
    assert isinstance(ds, bt.Dataset)
    assert fake_ibis.calls["dialect"] == "duckdb"
    assert ds.to_pydict() == {"id": [2, 3]}


def test_aggregate_and_join_idioms(fake_ibis, session) -> None:
    got = bt_ibis.to_dataset(_Expr(_AGGREGATE), session).to_pydict()
    assert got == {"g": ["a", "b", None], "total": [4, 2, 4]}
    assert bt_ibis.to_dataset(_Expr(_JOIN), session).to_pydict() == {
        "id": [1, 2],
        "label": ["one", "two"],
    }


def test_the_result_stays_in_arrow(fake_ibis, session) -> None:
    table = bt_ibis.to_dataset(_Expr(_FILTER), session).to_arrow()
    assert isinstance(table, pa.Table)


def test_table_describes_a_session_table(fake_ibis, session) -> None:
    unbound = bt_ibis.table("orders", session)
    assert unbound.name == "orders"
    tag, schema = unbound.schema
    assert tag == "ibis-schema"
    assert schema.names == ["id", "amount"]


def test_the_default_session_is_the_current_one(fake_ibis) -> None:
    session = bt.Session()
    session.register("orders", bt.from_pydict({"id": [7], "amount": [10.0]}))
    with session.activate():
        assert bt_ibis.to_dataset(_Expr(_FILTER)).to_pydict() == {"id": [7]}


def test_unsupported_sql_raises_the_typed_refusal(fake_ibis, session) -> None:
    with pytest.raises(bt.SQLUnsupportedError):
        bt_ibis.to_dataset(_Expr('SELECT no_such_fn("t0"."id") FROM "orders" AS "t0"'), session)


def test_a_non_session_is_refused(fake_ibis) -> None:
    with pytest.raises(bt.PlanError, match="Session"):
        bt_ibis.to_dataset(_Expr(_FILTER), session="nope")


def test_without_ibis_the_error_names_the_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ibis", None)
    with pytest.raises(ImportError, match=r"batcher-engine\[ibis\]"):
        bt_ibis.to_dataset(_Expr(_FILTER))
