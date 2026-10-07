"""The SQLAlchemy dialect: Batcher's PEP 249 adapter, its SQL, and its catalog reflection.

``create_engine("batcher://")`` connects through `batcher.dbapi.connect`, so every pooled
connection runs on the same `Session`, the one `bt.sql` uses. To run on another session,
pass it through ``connect_args={"session": s}``. The URL takes no host, database or query
options: a session is a Python object, and a URL cannot name one.

**Statements.** SQLAlchemy's default compiler emits ANSI SQL with ``?`` placeholders, which
Batcher parses in its default DuckDB dialect and binds as typed literals. An ``INSERT`` from
`executemany` runs once per row, each a plan rewrite of the target table.

**No transactions.** Every statement takes effect when it runs. `do_begin` and `do_commit`
do nothing, and `do_rollback` delegates to `Connection.rollback`, which returns quietly when
nothing was written since the last commit and raises `NotSupportedError` when something was.
So a read-only ``engine.connect()`` block closes cleanly, ``engine.begin()`` commits, and a
block that writes and then rolls back is told the write was not undone.

**Reflection** reads ``information_schema``: `has_table`, `get_table_names`,
`get_view_names`, `get_schema_names` and `get_columns`. Batcher has no primary keys, foreign
keys or indexes, so those report none. ``information_schema.tables`` lists the session's own
tables and views, so a table in an attached catalog is not reflected here.

This is the `integrations` layer.
"""

from __future__ import annotations

import re
from typing import Any

from typing_extensions import override

from batcher._internal.optional import require

_sa = require(
    "sqlalchemy",
    feature="The Batcher SQLAlchemy dialect",
    provides="SQLAlchemy",
    extra="sqlalchemy",
)
_default = require(
    "sqlalchemy.engine.default",
    feature="The Batcher SQLAlchemy dialect",
    provides="SQLAlchemy",
    extra="sqlalchemy",
)
_types = _sa.types

__all__ = ["BatcherDialect", "sqlalchemy_type"]

#: The schema Batcher reports for a session's own tables in ``information_schema``.
_DEFAULT_SCHEMA = "main"

_SIMPLE: dict[str, Any] = {
    "bool": _types.Boolean,
    "int8": _types.SmallInteger,
    "int16": _types.SmallInteger,
    "int32": _types.Integer,
    "int64": _types.BigInteger,
    "uint8": _types.SmallInteger,
    "uint16": _types.Integer,
    "uint32": _types.BigInteger,
    "uint64": _types.BigInteger,
    "halffloat": _types.Float,
    "float": _types.Float,
    "double": _types.Double,
    "string": _types.String,
    "large_string": _types.String,
    "string_view": _types.String,
    "binary": _types.LargeBinary,
    "large_binary": _types.LargeBinary,
    "binary_view": _types.LargeBinary,
    "date32[day]": _types.Date,
    "date64[ms]": _types.Date,
    "null": _types.NullType,
}
_DECIMAL = re.compile(r"decimal(?:32|64|128|256)?\((\d+), (-?\d+)\)")


def sqlalchemy_type(data_type: str) -> Any:
    """The SQLAlchemy type for an Arrow type string as ``information_schema`` reports it.

    Args:
        data_type: The ``data_type`` column of ``information_schema.columns``, such as
            ``"int64"`` or ``"timestamp[us, tz=UTC]"``.

    Returns:
        A SQLAlchemy type instance; `NullType` for a nested or unmapped type, which
        SQLAlchemy reflects as "type unknown" rather than guessing.
    """
    simple = _SIMPLE.get(data_type)
    if simple is not None:
        return simple()
    if data_type.startswith("timestamp["):
        return _types.DateTime(timezone="tz=" in data_type)
    if data_type.startswith(("time32[", "time64[")):
        return _types.Time()
    if data_type.startswith("duration["):
        return _types.Interval()
    decimal = _DECIMAL.fullmatch(data_type)
    if decimal is not None:
        return _types.Numeric(precision=int(decimal[1]), scale=int(decimal[2]))
    return _types.NullType()


class BatcherDialect(_default.DefaultDialect):
    """The SQLAlchemy dialect behind ``batcher://`` URLs.

    Not yet verified against a live SQLAlchemy application beyond this repository's tests;
    see tests/PENDING_VERIFICATION.md.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import sqlalchemy as sa
            >>> from sqlalchemy.dialects import registry
            >>> registry.register("batcher", "batcher.integrations.sqlalchemy", "BatcherDialect")
            >>> s = bt.Session()
            >>> _ = s.register("t", bt.from_pydict({"v": [1, 2, 3]}))
            >>> engine = sa.create_engine("batcher://", connect_args={"session": s})
            >>> with engine.connect() as conn:
            ...     query = sa.text("SELECT SUM(v) AS s FROM t WHERE v > :lo")
            ...     conn.execute(query, {"lo": 1}).all()
            [(5,)]
            >>> sa.inspect(engine).get_table_names()
            ['t']
    """

    name = "batcher"
    driver = "dbapi"
    default_paramstyle = "qmark"
    supports_statement_cache = True
    supports_native_boolean = True
    supports_native_decimal = True
    supports_alter = False
    supports_sequences = False
    supports_default_values = False
    supports_default_metavalue = False
    supports_empty_insert = False
    supports_multivalues_insert = True
    supports_comments = False
    postfetch_lastrowid = False
    insert_returning = False
    update_returning = False
    delete_returning = False

    @classmethod
    def import_dbapi(cls) -> Any:
        """The PEP 249 module this dialect drives: `batcher.dbapi`.

        Returns:
            The module.
        """
        from batcher import dbapi

        return dbapi

    @override
    def create_connect_args(self, url: Any) -> tuple[list[Any], dict[str, Any]]:
        """Refuse any URL part a session cannot be named by; there are no connect arguments.

        Args:
            url: The parsed engine URL.

        Returns:
            Empty positional and keyword arguments for `batcher.dbapi.connect`.

        Raises:
            ArgumentError: The URL names a host, port, database, user or query option.
        """
        named = [
            part
            for part in ("username", "password", "host", "port", "database")
            if getattr(url, part, None)
        ]
        if url.query:
            named.append("query options " + ", ".join(sorted(url.query)))
        if named:
            raise _sa.exc.ArgumentError(
                f"a batcher:// URL names no {', '.join(named)}: it connects to "
                "bt.current_session(). Pass connect_args={'session': s} for another session."
            )
        return [], {}

    # --- no transactions: see the module docstring ------------------------------
    @override
    def do_begin(self, dbapi_connection: Any) -> None:
        """Begin nothing: each statement takes effect when it runs."""

    @override
    def do_commit(self, dbapi_connection: Any) -> None:
        """Acknowledge the writes so far; they already took effect."""
        dbapi_connection.commit()

    @override
    def do_rollback(self, dbapi_connection: Any) -> None:
        """Delegate to `Connection.rollback`, which refuses only when a write would be lost."""
        dbapi_connection.rollback()

    @override
    def get_isolation_level(self, dbapi_connection: Any) -> str:
        """Report ``AUTOCOMMIT``, which is what every Batcher statement does."""
        return "AUTOCOMMIT"

    @override
    def get_default_isolation_level(self, dbapi_conn: Any) -> str:
        """Report ``AUTOCOMMIT``."""
        return "AUTOCOMMIT"

    @override
    def set_isolation_level(self, dbapi_connection: Any, level: str) -> None:
        """Accept ``AUTOCOMMIT``, the one level `get_isolation_level_values` offers.

        SQLAlchemy refuses any other level before calling this, naming the levels offered.
        """

    @override
    def get_isolation_level_values(self, dbapi_conn: Any) -> list[str]:
        """The one level available."""
        return ["AUTOCOMMIT"]

    @override
    def _get_default_schema_name(self, connection: Any) -> str:
        return _DEFAULT_SCHEMA

    @override
    def _get_server_version_info(self, connection: Any) -> tuple[int, ...]:
        from batcher import __version__

        return tuple(int(p) for p in re.findall(r"\d+", __version__)[:3])

    # --- reflection over information_schema -------------------------------------
    def _rows(self, connection: Any, query: str, params: dict[str, Any]) -> list[Any]:
        return list(connection.exec_driver_sql(query, tuple(params.values())))

    def _names(self, connection: Any, kind: str, schema: str | None) -> list[str]:
        rows = self._rows(
            connection,
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = ? AND table_schema = ? ORDER BY table_name",
            {"kind": kind, "schema": schema or _DEFAULT_SCHEMA},
        )
        return [row[0] for row in rows]

    @override
    def has_table(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> bool:
        """Whether ``information_schema.tables`` lists `table_name`, as a table or a view."""
        rows = self._rows(
            connection,
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema = ?",
            {"name": table_name, "schema": schema or _DEFAULT_SCHEMA},
        )
        return bool(rows[0][0])

    @override
    def get_table_names(self, connection: Any, schema: str | None = None, **kw: Any) -> list[str]:
        """The session's base tables, sorted."""
        return self._names(connection, "BASE TABLE", schema)

    @override
    def get_view_names(self, connection: Any, schema: str | None = None, **kw: Any) -> list[str]:
        """The session's views, sorted."""
        return self._names(connection, "VIEW", schema)

    @override
    def get_schema_names(self, connection: Any, **kw: Any) -> list[str]:
        """The schema names ``information_schema.schemata`` lists, sorted and distinct."""
        rows = self._rows(
            connection,
            "SELECT DISTINCT schema_name FROM information_schema.schemata ORDER BY schema_name",
            {},
        )
        return [row[0] for row in rows]

    @override
    def get_columns(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        """The columns of `table_name`, in order, with their reflected SQLAlchemy types.

        Raises:
            NoSuchTableError: `table_name` has no columns in ``information_schema``.
        """
        rows = self._rows(
            connection,
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = ? ORDER BY ordinal_position",
            {"name": table_name, "schema": schema or _DEFAULT_SCHEMA},
        )
        if not rows:
            raise _sa.exc.NoSuchTableError(table_name)
        return [
            {
                "name": name,
                "type": sqlalchemy_type(data_type),
                "nullable": nullable == "YES",
                "default": default,
            }
            for name, data_type, nullable, default in rows
        ]

    @override
    def get_pk_constraint(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        """No primary key: Batcher declares and enforces none."""
        return {"constrained_columns": [], "name": None}

    @override
    def get_foreign_keys(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        """No foreign keys: Batcher declares and enforces none."""
        return []

    @override
    def get_indexes(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        """No indexes: Batcher has none to report."""
        return []
