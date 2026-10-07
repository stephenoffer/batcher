"""The dbt adapter pilot against a stand-in for dbt-adapters, and its macros against Jinja.

dbt-core is not installed in this environment, so these tests inject minimal stand-ins for
the dbt-adapters classes the adapter subclasses (``SQLAdapter``, ``SQLConnectionManager``,
``BaseRelation``, ``Credentials``, ``AdapterPlugin``) and drive the adapter's *own* code:
opening a connection over the shared session, the catalog operations a ``dbt run`` calls,
the relation rendering rule, the refusals, and the SQL the macros emit, executed for real on
Batcher. What the stand-ins cannot prove is that dbt-core calls these the way it does in a
release; `tests/integration/live/test_live_dbt.py` runs a real project when dbt is present.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from batcher import dbapi

pytestmark = pytest.mark.unit

_ADAPTER = "batcher.integrations.dbt.adapter"


class _Available:
    """``dbt.adapters.base.meta.available``: a decorator with ``parse*`` variants."""

    def __call__(self, fn: Any) -> Any:
        return fn

    def parse(self, _default: Any) -> Any:
        return lambda fn: fn

    parse_list = parse_none = staticmethod(lambda fn: fn)


class _Policy:
    identifier = True


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class _BaseRelation:
    database: str | None = None
    schema: str | None = None
    identifier: str | None = None
    type: str | None = None

    quote_policy = _Policy()

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        kwargs.pop("quote_policy", None)
        return cls(**kwargs)

    @classmethod
    def create_from(cls, quoting: Any, relation_config: Any, **kwargs: Any) -> Any:
        return cls.create(
            database=relation_config.database,
            schema=relation_config.schema,
            identifier=relation_config.identifier,
            **kwargs,
        )

    @property
    def is_view(self) -> bool:
        return self.type == "view"

    def quoted(self, identifier: str) -> str:
        return f'"{identifier}"'

    def render(self) -> str:
        parts = (self.database, self.schema, self.identifier)
        return ".".join(self.quoted(p) for p in parts if p is not None)

    def __str__(self) -> str:
        return self.render()


class _ConnectionManager:
    def __init__(self) -> None:
        self.connection: Any = None

    def get_thread_connection(self) -> Any:
        return self.connection

    def execute(self, sql: str) -> Any:
        cursor = self.connection.handle.cursor()
        with self.exception_handler(sql):
            cursor.execute(sql)
        return self.get_response(cursor)


class _Cache:
    def __init__(self) -> None:
        self.dropped: list[Any] = []

    def drop_schema(self, database: str, schema: str) -> None:
        self.dropped.append((database, schema))


class _Column:
    @classmethod
    def create(cls, name: str, dtype: str) -> tuple[str, str]:
        return (name, dtype)


class _SQLAdapter:
    ConnectionManager: Any = None
    Column = _Column

    def __init__(self, config: Any) -> None:
        self.config = config
        self.connections = self.ConnectionManager()
        self.cache = _Cache()

    def cache_dropped(self, relation: Any) -> None:
        self.cache.dropped.append(relation)


@dataclasses.dataclass
class _Credentials:
    database: str
    schema: str


@dataclasses.dataclass
class _AdapterResponse:
    _message: str
    rows_affected: int


class _DbtDatabaseError(Exception):
    pass


class _DbtRuntimeError(Exception):
    pass


class _AdapterPlugin:
    def __init__(self, adapter: Any, credentials: Any, include_path: str) -> None:
        self.adapter, self.credentials, self.include_path = adapter, credentials, include_path


@pytest.fixture
def dbt(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """The adapter module, imported over stand-in dbt modules."""

    def module(name: str, **attrs: Any) -> None:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    module("dbt")
    module("dbt.adapters")
    module("dbt.adapters.base", AdapterPlugin=_AdapterPlugin)
    module("dbt.adapters.base.meta", available=_Available())
    module(
        "dbt.adapters.base.relation",
        BaseRelation=_BaseRelation,
        RelationType=types.SimpleNamespace(View="view", Table="table"),
    )
    module("dbt.adapters.sql", SQLAdapter=_SQLAdapter, SQLConnectionManager=_ConnectionManager)
    module("dbt.adapters.contracts")
    module(
        "dbt.adapters.contracts.connection",
        Credentials=_Credentials,
        AdapterResponse=_AdapterResponse,
    )
    module("dbt_common")
    module(
        "dbt_common.exceptions",
        DbtDatabaseError=_DbtDatabaseError,
        DbtRuntimeError=_DbtRuntimeError,
    )
    monkeypatch.delitem(sys.modules, _ADAPTER, raising=False)
    yield importlib.import_module(_ADAPTER)
    sys.modules.pop(_ADAPTER, None)


@pytest.fixture
def adapter(dbt: types.ModuleType, tmp_path: Path) -> Any:
    """An adapter with an open connection to a directory-catalog target."""
    credentials = dbt.BatcherCredentials(database="wh", schema="analytics", path=str(tmp_path))
    config = types.SimpleNamespace(credentials=credentials)
    adapter = dbt.BatcherAdapter(config)
    connection = types.SimpleNamespace(state="init", credentials=credentials, handle=None)
    adapter.connections.connection = dbt.BatcherConnectionManager.open(connection)
    return adapter


def _relation(dbt: types.ModuleType, identifier: str, kind: str = "table") -> Any:
    return dbt.BatcherRelation.create(
        database="wh", schema="analytics", identifier=identifier, type=kind
    )


def _macros() -> Any:
    jinja2 = pytest.importorskip("jinja2")
    path = Path(__file__).resolve().parents[2] / "python/batcher/integrations/dbt/include"
    source = (path / "macros/adapters.sql").read_text()
    return jinja2.Environment().from_string(source).module


def test_credentials_name_the_adapter(dbt) -> None:
    credentials = dbt.BatcherCredentials(database="wh", schema="s", path="/tmp/x")
    assert credentials.type == "batcher"
    assert credentials.unique_field == "/tmp/x"
    assert credentials._connection_keys() == ("database", "schema", "path")


def test_open_wraps_the_targets_shared_session(dbt, adapter, tmp_path) -> None:
    handle = adapter.connections.connection.handle
    assert isinstance(handle, dbapi.Connection)
    assert adapter.connections.connection.state == "open"
    other = types.SimpleNamespace(state="init", credentials=adapter.config.credentials, handle=None)
    assert dbt.BatcherConnectionManager.open(other).handle.session is handle.session
    assert handle.session.catalog.has_catalog("wh")


def test_a_run_builds_a_table_and_a_view_and_tests_them(dbt, adapter) -> None:
    """The macros' SQL, executed for real: what ``dbt run`` and ``dbt test`` send."""
    macros = _macros()
    adapter.create_schema(_relation(dbt, None))
    assert adapter.check_schema_exists("wh", "analytics")

    table = _relation(dbt, "orders")
    view = _relation(dbt, "big_orders", "view")
    adapter.connections.execute(
        macros.batcher__create_table_as(
            False, table, "select 1 as id, 5.0 as amount union all select 2, 9.0"
        )
    )
    adapter.connections.execute(
        macros.batcher__create_view_as(view, f"select id from {table} where amount > 6")
    )
    listed = {
        (r.identifier, r.type) for r in adapter.list_relations_without_caching(_relation(dbt, None))
    }
    assert listed == {("orders", "table"), ("big_orders", "view")}
    assert adapter.get_columns_in_relation(table) == [("id", "int64"), ("amount", "double")]

    cursor = adapter.connections.connection.handle.cursor()
    failures = cursor.execute(
        "select count(*) as failures from (select id from "
        f"{view} where id is null) dbt_internal_test"
    ).fetchone()
    assert failures == (0,)
    assert cursor.execute(f"select id from {view}").fetchall() == [(2,)]


def test_a_table_persists_in_the_directory_catalog(dbt, adapter, tmp_path) -> None:
    import batcher as bt

    adapter.create_schema(_relation(dbt, None))
    adapter.connections.execute(
        _macros().batcher__create_table_as(False, _relation(dbt, "t"), "select 7 as x")
    )
    fresh = bt.Session()
    fresh.catalog.attach(bt.Catalog.from_directory(str(tmp_path), name="wh"))
    assert fresh.sql("SELECT x FROM wh.analytics.t").to_pydict() == {"x": [7]}


def test_drop_truncate_and_drop_schema(dbt, adapter) -> None:
    adapter.create_schema(_relation(dbt, None))
    table = _relation(dbt, "t")
    adapter.connections.execute(_macros().batcher__create_table_as(False, table, "select 1 as x"))
    adapter.truncate_relation(table)
    session = adapter.connections.connection.handle.session
    assert session.sql("SELECT * FROM wh.analytics.t").to_pydict() == {"x": []}
    adapter.drop_relation(table)
    assert table in adapter.cache.dropped
    assert not session.catalog.has_table("wh.analytics.t")
    adapter.drop_schema(_relation(dbt, None))
    assert not adapter.check_schema_exists("wh", "analytics")
    assert ("wh", "analytics") in adapter.cache.dropped


def test_rename_is_refused_rather_than_faked(dbt, adapter) -> None:
    with pytest.raises(_DbtRuntimeError, match="cannot rename"):
        adapter.rename_relation(_relation(dbt, "a"), _relation(dbt, "b"))


def test_an_engine_error_becomes_a_dbt_database_error(dbt, adapter) -> None:
    with pytest.raises(_DbtDatabaseError, match="no_such_table"):
        adapter.connections.execute("select * from no_such_table")


def test_the_response_reports_an_unknown_row_count(dbt, adapter) -> None:
    response = adapter.connections.execute("select 1")
    assert response.rows_affected == -1


def test_a_view_renders_bare_and_a_table_fully_qualified(dbt) -> None:
    assert _relation(dbt, "t").render() == '"wh"."analytics"."t"'
    assert _relation(dbt, "v", "view").render() == '"v"'
    node = types.SimpleNamespace(
        database="wh",
        schema="analytics",
        identifier="v",
        config=types.SimpleNamespace(materialized="view"),
    )
    assert dbt.BatcherRelation.create_from(None, node).render() == '"v"'


def test_no_transaction_statements_are_sent(dbt, adapter) -> None:
    manager = adapter.connections
    assert manager.add_begin_query() is None
    assert manager.add_commit_query() is None
    assert not dbt.BatcherAdapter.is_cancelable()


def test_the_plugin_points_at_the_shipped_macros(dbt) -> None:
    include = Path(dbt.Plugin.include_path)
    assert (include / "dbt_project.yml").is_file()
    materializations = (include / "macros/materializations.sql").read_text()
    assert "materialization table, adapter='batcher'" in materializations
    assert "materialization view, adapter='batcher'" in materializations
    assert dbt.Plugin.adapter is dbt.BatcherAdapter


def test_the_dbt_namespace_shim_exports_the_plugin(dbt) -> None:
    root = Path(__file__).resolve().parents[2] / "python"
    source = (root / "dbt/adapters/batcher/__init__.py").read_text()
    assert "from batcher.integrations.dbt.adapter import Plugin" in source
    assert not (root / "dbt/__init__.py").exists()  # a namespace package, as dbt requires
    assert not (root / "dbt/adapters/__init__.py").exists()
