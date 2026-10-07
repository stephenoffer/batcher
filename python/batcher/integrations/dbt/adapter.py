"""The dbt adapter classes: credentials, connection manager, relation, adapter, and plugin.

Written against dbt-adapters 1.x (``dbt.adapters.sql.SQLAdapter`` and
``SQLConnectionManager``, ``dbt.adapters.contracts``) and dbt-common's exceptions. A
connection's handle is a `batcher.dbapi` connection over the target's shared session
(`batcher.integrations.dbt.sessions`).

**What the pilot covers.** The ``table`` and ``view`` materializations (``include/macros``),
``dbt run``, and ``dbt test``'s generic and singular tests, which are ``SELECT`` statements.
Catalog operations a run needs (listing schemas and relations, columns, creating and
dropping schemas and relations) are answered here in Python from the session's catalog
rather than by SQL macros, because ``information_schema`` does not list catalog tables.

**Where relations live.** A table is a catalog table, ``database.schema.identifier``, so with
a ``path`` it persists between dbt invocations. A view is a session view: Batcher stores
views in the session, not in a catalog, so a view renders as its bare identifier, is visible
only inside the dbt process that built it, and two views of one name in different schemas
collide. ``dbt build`` runs and tests in one process, so a view is tested there. A separate
``dbt test`` process sees the tables but not the views.

**No transactions.** ``BEGIN``/``COMMIT`` are not sent; each statement takes effect when it
runs. Seeds, snapshots, incremental models, ``rename``, and ``dbt docs generate`` are not
part of the pilot.

Not yet verified against a live dbt-core installation; see tests/PENDING_VERIFICATION.md.

This is the `integrations` layer.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from typing_extensions import override

from batcher import dbapi
from batcher._internal.optional import require
from batcher.integrations.dbt.sessions import session_for


def _dbt(module: str) -> Any:
    return require(module, feature="The Batcher dbt adapter", provides="dbt-core", extra="dbt")


_base = _dbt("dbt.adapters.base")
_meta = _dbt("dbt.adapters.base.meta")
_relation = _dbt("dbt.adapters.base.relation")
_sql = _dbt("dbt.adapters.sql")
_contracts = _dbt("dbt.adapters.contracts.connection")
_exceptions = _dbt("dbt_common.exceptions")
available = _meta.available

__all__ = [
    "BatcherAdapter",
    "BatcherConnectionManager",
    "BatcherCredentials",
    "BatcherRelation",
    "Plugin",
]

#: The adapter type a ``profiles.yml`` target names.
ADAPTER_TYPE = "batcher"
#: Where the materializations and macros live (``dbt_project.yml`` plus ``macros/``).
INCLUDE_PATH = os.path.join(os.path.dirname(__file__), "include")


@dataclass
class BatcherCredentials(_contracts.Credentials):
    """A ``type: batcher`` target: ``database`` names the catalog, ``path`` stores it."""

    path: str | None = None

    @property
    def type(self) -> str:
        return ADAPTER_TYPE

    @property
    def unique_field(self) -> str:
        return self.path or self.database

    def _connection_keys(self) -> tuple[str, ...]:
        return ("database", "schema", "path")


class BatcherConnectionManager(_sql.SQLConnectionManager):
    """Opens `batcher.dbapi` connections over the target's shared session."""

    TYPE = ADAPTER_TYPE

    @classmethod
    @override
    def open(cls, connection: Any) -> Any:
        if connection.state == "open":
            return connection
        credentials = connection.credentials
        try:
            connection.handle = dbapi.connect(session_for(credentials.path, credentials.database))
        except Exception as exc:
            connection.handle = None
            connection.state = "fail"
            raise _exceptions.DbtDatabaseError(str(exc)) from exc
        connection.state = "open"
        return connection

    @classmethod
    @override
    def get_response(cls, cursor: Any) -> Any:
        return _contracts.AdapterResponse(_message="OK", rows_affected=cursor.rowcount)

    @override
    def cancel(self, connection: Any) -> None:
        """Nothing to cancel: `BatcherAdapter.is_cancelable` is False."""

    @contextmanager
    @override
    def exception_handler(self, sql: str) -> Any:
        try:
            yield
        except dbapi.Error as exc:
            raise _exceptions.DbtDatabaseError(str(exc)) from exc

    @override
    def add_begin_query(self) -> None:
        """Send no ``BEGIN``: Batcher has no transactions."""

    @override
    def add_commit_query(self) -> None:
        """Send no ``COMMIT``: each statement already took effect."""


@dataclass(frozen=True, eq=False, repr=False)
class BatcherRelation(_relation.BaseRelation):
    """Renders a table as ``database.schema.identifier`` and a view as its identifier alone."""

    @classmethod
    @override
    def create_from(cls, quoting: Any, relation_config: Any, **kwargs: Any) -> Any:
        config = getattr(relation_config, "config", None)
        if kwargs.get("type") is None and getattr(config, "materialized", None) == "view":
            kwargs["type"] = _relation.RelationType.View
        return super().create_from(quoting, relation_config, **kwargs)

    @override
    def render(self) -> str:
        if self.is_view and self.identifier:
            quote = self.quote_policy.identifier
            return self.quoted(self.identifier) if quote else self.identifier
        return super().render()


class BatcherAdapter(_sql.SQLAdapter):
    """The dbt adapter for Batcher; see the module docstring for what it covers."""

    ConnectionManager = BatcherConnectionManager
    Relation = BatcherRelation

    @classmethod
    @override
    def date_function(cls) -> str:
        return "current_timestamp"

    @classmethod
    @override
    def is_cancelable(cls) -> bool:
        return False

    def _session(self) -> Any:
        return self.connections.get_thread_connection().handle.session

    def _catalog(self, database: str | None) -> Any:
        session = self._session()
        return session.catalog.get_catalog(database or self.config.credentials.database)

    @override
    def list_schemas(self, database: str) -> list[str]:
        return self._catalog(database).list_namespaces()

    @available.parse(lambda *_a, **_k: False)
    @override
    def check_schema_exists(self, database: str, schema: str) -> bool:
        return self._catalog(database).has_namespace(schema)

    @override
    def create_schema(self, relation: Any) -> None:
        self._catalog(relation.database).create_namespace(relation.schema, if_not_exists=True)

    @override
    def drop_schema(self, relation: Any) -> None:
        self._catalog(relation.database).drop_namespace(
            relation.schema, if_exists=True, cascade=True
        )
        self.cache.drop_schema(relation.database, relation.schema)

    @override
    def list_relations_without_caching(self, schema_relation: Any) -> list[Any]:
        database, schema = schema_relation.database, schema_relation.schema
        catalog = self._catalog(database)
        tables = (
            (name.rsplit(".", 1)[-1] for name in catalog.list_tables(f"{schema}.*"))
            if catalog.has_namespace(schema)
            else ()
        )
        views = sorted(self._session_views())
        return [
            self.Relation.create(database=database, schema=schema, identifier=name, type=kind)
            for kind, names in (("table", tables), ("view", views))
            for name in names
        ]

    def _session_views(self) -> set[str]:
        cursor = self.connections.get_thread_connection().handle.cursor()
        rows = cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
        ).fetchall()
        return {row[0] for row in rows}

    @available.parse_list
    @override
    def get_columns_in_relation(self, relation: Any) -> list[Any]:
        schema = self._session().sql(f"SELECT * FROM {relation.render()}").schema
        return [self.Column.create(field.name, str(field.type)) for field in schema]

    @available.parse_none
    @override
    def drop_relation(self, relation: Any) -> None:
        self.cache_dropped(relation)
        kind = "VIEW" if relation.is_view else "TABLE"
        self.connections.execute(f"DROP {kind} IF EXISTS {relation.render()}")

    @available.parse_none
    @override
    def truncate_relation(self, relation: Any) -> None:
        self._catalog(relation.database).truncate_table(f"{relation.schema}.{relation.identifier}")

    @available.parse_none
    @override
    def rename_relation(self, from_relation: Any, to_relation: Any) -> None:
        raise _exceptions.DbtRuntimeError(
            "the Batcher adapter cannot rename a relation; its materializations replace in "
            "place with CREATE OR REPLACE instead"
        )


Plugin = _base.AdapterPlugin(
    adapter=BatcherAdapter, credentials=BatcherCredentials, include_path=INCLUDE_PATH
)
