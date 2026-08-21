"""Dialect-aware DML for the DB-API write path — the statement each write mode runs.

Reading a database needs one statement shape: a ``SELECT`` the server plans. Writing one
needs five, and every one of them is spelled differently by every dialect. This module is
that spelling, kept apart from `sink` for the same reason `_arrow` is kept apart from
`source`: connection handling and SQL generation fail in unrelated ways, and the SQL is
the half that can be tested without a database.

## Why an upsert cannot be one statement

``INSERT``, ``UPDATE`` and ``DELETE`` are ANSI SQL and portable. *Upsert* — insert this
row, or update the one already holding its key — is not. Three incompatible spellings
cover essentially every operational database in use:

* ``INSERT … ON CONFLICT (key) DO UPDATE SET …`` — PostgreSQL 9.5+, SQLite 3.24+, DuckDB,
  and the PostgreSQL-wire databases that implement it.
* ``INSERT … ON DUPLICATE KEY UPDATE …`` — MySQL and its wire-protocol family.
* ``MERGE INTO … USING … WHEN MATCHED / WHEN NOT MATCHED`` — SQL:2003, which SQL Server,
  Oracle, Snowflake, BigQuery and Redshift implement.

A dialect outside those three is **refused**, naming ``mode="delete_insert"`` — the
portable two-statement form, which the sink runs inside one transaction. Guessing a
fourth spelling would produce a syntax error at best; guessing the *conflict target*
would silently update the wrong rows, which is worse.

## The one semantic difference worth knowing

``ON DUPLICATE KEY`` has no conflict target: MySQL matches on **any** unique index, not on
the columns you named. A table with a second unique index therefore updates rows that a
PostgreSQL ``ON CONFLICT (id)`` would have inserted. That is MySQL's semantics rather than
a translation defect, and it is stated here because it is invisible in the SQL Batcher
emits.

## Placeholders

PEP 249 lets a driver pick any of five ``paramstyle`` spellings, and a statement built for
the wrong one fails with the driver's own parse error rather than anything Batcher could
explain. Every statement here is built against the driver's declared `paramstyle`, and the
name-based styles bind through synthetic ``p0…pN`` names rather than column names, because
a column called ``order by`` is a legal identifier and an illegal parameter name.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.errors import BackendError
from batcher.io.formats.sql.uri import quote_identifier

__all__ = [
    "PARAMSTYLES",
    "Statement",
    "delete",
    "dialect_for_driver",
    "insert",
    "qualified_table",
    "quote",
    "truncate",
    "update",
    "upsert",
    "upsert_style",
]

#: The five ``paramstyle`` values PEP 249 defines. A driver declaring anything else is
#: not a PEP 249 driver, and is refused rather than guessed at.
PARAMSTYLES = frozenset({"qmark", "numeric", "format", "pyformat", "named"})

#: Styles that bind a *sequence* per row; the rest bind a mapping.
_POSITIONAL_STYLES = frozenset({"qmark", "numeric", "format"})

#: Dialects spelling an upsert ``INSERT … ON CONFLICT (key) DO UPDATE SET …``.
#:
#: Deliberately narrower than the set of PostgreSQL-wire databases `uri` can route.
#: Speaking the wire protocol does not imply implementing ``ON CONFLICT``, and a dialect
#: that is merely *probably* fine belongs in `delete_insert` rather than in a statement
#: the user cannot see before it runs.
_ON_CONFLICT: frozenset[str] = frozenset(
    {
        "postgresql",
        "postgres",
        "sqlite",
        "duckdb",
        "cockroachdb",
        "cockroach",
        "timescaledb",
        "alloydb",
        "yugabytedb",
        "yugabyte",
    }
)

#: Dialects spelling an upsert ``INSERT … ON DUPLICATE KEY UPDATE …``.
_ON_DUPLICATE_KEY: frozenset[str] = frozenset(
    {"mysql", "mariadb", "tidb", "singlestore", "memsql", "percona"}
)

#: Dialects spelling an upsert as SQL:2003 ``MERGE INTO``.
_MERGE: frozenset[str] = frozenset(
    {"mssql", "sqlserver", "oracle", "snowflake", "bigquery", "redshift"}
)

#: Dialects whose ``MERGE`` source subquery needs a ``FROM`` clause (Oracle's ``dual``).
_MERGE_NEEDS_FROM_DUAL: frozenset[str] = frozenset({"oracle"})

#: Dialects whose ``MERGE`` statement must be terminated by a semicolon (SQL Server).
_MERGE_NEEDS_TERMINATOR: frozenset[str] = frozenset({"mssql", "sqlserver"})

#: Dialects that reject ``AS`` before a table alias (Oracle).
_NO_AS_ALIAS: frozenset[str] = frozenset({"oracle"})

#: Driver module name → the dialect it speaks, so ``module="sqlite3"`` needs no ``uri=``.
#:
#: Only unambiguous drivers are listed. ``pyodbc`` and ``jaydebeapi`` are deliberately
#: absent: a DSN names a *driver*, not a dialect, so inferring one from the module would
#: be a coin flip that decides how identifiers are quoted and how an upsert is spelled.
_DRIVER_DIALECTS: dict[str, str] = {
    "sqlite3": "sqlite",
    "duckdb": "duckdb",
    "psycopg": "postgresql",
    "psycopg2": "postgresql",
    "pg8000": "postgresql",
    "asyncpg": "postgresql",
    "pymysql": "mysql",
    "MySQLdb": "mysql",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "oracledb": "oracle",
    "cx_Oracle": "oracle",
    "snowflake": "snowflake",
    "clickhouse_driver": "clickhouse",
    "trino": "trino",
}


@dataclass(frozen=True, slots=True)
class Statement:
    """One executable statement plus the per-row parameter order it expects.

    `columns` is the order values are bound in, which is *not* the table's column order:
    an ``UPDATE`` binds the assigned columns first and the key columns last. `names` is
    non-empty only for a name-based `paramstyle`, in which case a bound row is a mapping
    from these synthetic names rather than a sequence.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._statements import insert
            >>> stmt = insert("t", ("id", "amt"), dialect="postgresql", paramstyle="qmark")
            >>> stmt.sql
            'INSERT INTO "t" ("id", "amt") VALUES (?, ?)'
            >>> stmt.columns
            ('id', 'amt')
    """

    sql: str
    columns: tuple[str, ...]
    names: tuple[str, ...] = ()

    @property
    def positional(self) -> bool:
        """Whether a bound row is a sequence (rather than a mapping keyed by `names`)."""
        return not self.names


def dialect_for_driver(module: str) -> str | None:
    """The dialect a PEP 249 driver module speaks, or None when it is ambiguous.

    Args:
        module: The importable driver module name, e.g. ``"psycopg"``.

    Returns:
        A dialect name `quote_identifier` and `upsert_style` understand, or None.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._statements import dialect_for_driver
            >>> dialect_for_driver("psycopg2")
            'postgresql'
            >>> dialect_for_driver("pyodbc") is None
            True
    """
    return _DRIVER_DIALECTS.get(module.split(".")[0])


def upsert_style(dialect: str | None) -> str | None:
    """Which upsert spelling `dialect` accepts, or None when Batcher knows of none.

    Args:
        dialect: A connection-URI scheme or driver-inferred dialect name.

    Returns:
        ``"on_conflict"``, ``"on_duplicate_key"``, ``"merge"``, or None.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._statements import upsert_style
            >>> upsert_style("sqlite")
            'on_conflict'
            >>> upsert_style("mysql")
            'on_duplicate_key'
            >>> upsert_style("informix") is None
            True
    """
    if dialect is None:
        return None
    normalized = dialect.split("+")[0].strip().lower()
    if normalized in _ON_CONFLICT:
        return "on_conflict"
    if normalized in _ON_DUPLICATE_KEY:
        return "on_duplicate_key"
    if normalized in _MERGE:
        return "merge"
    return None


def quote(name: str, dialect: str | None) -> str:
    """`name` delimited for `dialect`, or unchanged when the dialect is unknown."""
    return quote_identifier(name, dialect) if dialect else name


def qualified_table(table: str, dialect: str | None) -> str:
    """Quote a possibly schema-qualified table name one part at a time.

    ``public.orders`` must become ``"public"."orders"``, not ``"public.orders"`` — the
    latter names a single table whose name contains a dot, which does not exist. Splitting
    on the separator is safe here because a table name holding a literal dot cannot be
    expressed in the unquoted form this parameter accepts.
    """
    return ".".join(quote(part, dialect) for part in table.split(".") if part)


def _placeholders(paramstyle: str, count: int, start: int = 0) -> tuple[list[str], tuple[str, ...]]:
    """`count` placeholders in `paramstyle`, plus the synthetic names they bind by.

    `start` offsets the numbering so a statement binding two groups (``MERGE``'s source
    row, then nothing else) keeps ``numeric``'s absolute positions correct.
    """
    if paramstyle not in PARAMSTYLES:
        raise BackendError(
            f"driver declares paramstyle={paramstyle!r}, which is not one of PEP 249's "
            f"{sorted(PARAMSTYLES)}. Batcher cannot build a statement it cannot bind."
        )
    if paramstyle == "qmark":
        return ["?"] * count, ()
    if paramstyle == "format":
        return ["%s"] * count, ()
    if paramstyle == "numeric":
        return [f":{start + i + 1}" for i in range(count)], ()
    names = tuple(f"p{start + i}" for i in range(count))
    if paramstyle == "named":
        return [f":{n}" for n in names], names
    return [f"%({n})s" for n in names], names


def _validate(table: str, columns: tuple[str, ...], keys: tuple[str, ...] = ()) -> None:
    """Reject a statement request that cannot produce valid SQL, before building it."""
    if not table:
        raise BackendError("a SQL write needs a destination table name")
    if not columns:
        raise BackendError(f"a write to {table!r} needs at least one column")
    missing = [k for k in keys if k not in columns]
    if missing:
        raise BackendError(
            f"key_columns {missing} are not columns of the data being written "
            f"({list(columns)}). A key column names a column of the *frame*, which is "
            "matched against the same-named column of the target table."
        )


def insert(
    table: str, columns: tuple[str, ...], *, dialect: str | None, paramstyle: str
) -> Statement:
    """``INSERT INTO table (columns) VALUES (…)`` — one bound row per parameter set.

    Args:
        table: The destination table, optionally schema-qualified.
        columns: The columns written, in frame order.
        dialect: The dialect whose identifier quoting applies, or None to leave names bare.
        paramstyle: The driver's declared PEP 249 ``paramstyle``.

    Returns:
        The statement and its per-row parameter order.
    """
    _validate(table, columns)
    marks, names = _placeholders(paramstyle, len(columns))
    cols = ", ".join(quote(c, dialect) for c in columns)
    sql = f"INSERT INTO {qualified_table(table, dialect)} ({cols}) VALUES ({', '.join(marks)})"
    return Statement(sql, columns, names)


def upsert(
    table: str,
    columns: tuple[str, ...],
    key_columns: tuple[str, ...],
    *,
    dialect: str | None,
    paramstyle: str,
) -> Statement:
    """The dialect's insert-or-update statement, keyed on `key_columns`.

    Columns outside `key_columns` are the ones an existing row has overwritten; a frame
    holding only key columns degrades to "insert if absent, otherwise leave alone".

    Args:
        table: The destination table, optionally schema-qualified.
        columns: The columns written, in frame order.
        key_columns: The columns identifying an existing row.
        dialect: The dialect whose upsert spelling applies.
        paramstyle: The driver's declared PEP 249 ``paramstyle``.

    Returns:
        The statement and its per-row parameter order.

    Raises:
        BackendError: If `key_columns` is empty, names a column absent from `columns`, or
            `dialect` has no upsert spelling Batcher knows.
    """
    _validate(table, columns, key_columns)
    if not key_columns:
        raise BackendError(
            "mode='upsert' needs key_columns= — the columns that identify an existing "
            "row. Without them an upsert is an append; use mode='append'."
        )
    style = upsert_style(dialect)
    if style is None:
        raise BackendError(
            f"Batcher knows no upsert spelling for dialect {dialect!r}. Use "
            "mode='delete_insert', which deletes the source's keys and re-inserts them "
            "inside one transaction using only ANSI SQL, or pass dialect= naming a "
            "dialect whose upsert syntax this backend accepts."
        )
    updated = tuple(c for c in columns if c not in key_columns)
    if style == "merge":
        return _merge(table, columns, key_columns, updated, dialect=dialect, paramstyle=paramstyle)
    marks, names = _placeholders(paramstyle, len(columns))
    cols = ", ".join(quote(c, dialect) for c in columns)
    head = f"INSERT INTO {qualified_table(table, dialect)} ({cols}) VALUES ({', '.join(marks)})"
    if style == "on_conflict":
        target = ", ".join(quote(c, dialect) for c in key_columns)
        if not updated:
            return Statement(f"{head} ON CONFLICT ({target}) DO NOTHING", columns, names)
        sets = ", ".join(f"{quote(c, dialect)} = EXCLUDED.{quote(c, dialect)}" for c in updated)
        return Statement(f"{head} ON CONFLICT ({target}) DO UPDATE SET {sets}", columns, names)
    # ON DUPLICATE KEY has no conflict target — MySQL matches any unique index. With
    # nothing to update, assigning a key column to itself is the idiomatic no-op; the
    # clause is not optional.
    assigned = updated or key_columns
    sets = ", ".join(f"{quote(c, dialect)} = VALUES({quote(c, dialect)})" for c in assigned)
    return Statement(f"{head} ON DUPLICATE KEY UPDATE {sets}", columns, names)


def _merge(
    table: str,
    columns: tuple[str, ...],
    key_columns: tuple[str, ...],
    updated: tuple[str, ...],
    *,
    dialect: str | None,
    paramstyle: str,
) -> Statement:
    """SQL:2003 ``MERGE INTO``, with the three per-dialect deviations applied.

    Oracle needs ``FROM dual`` under the source subquery and rejects ``AS`` before a table
    alias; SQL Server requires the statement to be terminated by a semicolon. Everything
    else is identical across the five dialects that implement ``MERGE``.
    """
    marks, names = _placeholders(paramstyle, len(columns))
    alias = "" if dialect in _NO_AS_ALIAS else "AS "
    from_dual = " FROM dual" if dialect in _MERGE_NEEDS_FROM_DUAL else ""
    selected = ", ".join(f"{m} AS {quote(c, dialect)}" for m, c in zip(marks, columns, strict=True))
    on = " AND ".join(f"t.{quote(c, dialect)} = s.{quote(c, dialect)}" for c in key_columns)
    cols = ", ".join(quote(c, dialect) for c in columns)
    values = ", ".join(f"s.{quote(c, dialect)}" for c in columns)
    sql = (
        f"MERGE INTO {qualified_table(table, dialect)} {alias}t "
        f"USING (SELECT {selected}{from_dual}) {alias}s ON ({on}) "
    )
    if updated:
        sets = ", ".join(f"t.{quote(c, dialect)} = s.{quote(c, dialect)}" for c in updated)
        sql += f"WHEN MATCHED THEN UPDATE SET {sets} "
    sql += f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({values})"
    if dialect in _MERGE_NEEDS_TERMINATOR:
        sql += ";"
    return Statement(sql, columns, names)


def update(
    table: str,
    columns: tuple[str, ...],
    key_columns: tuple[str, ...],
    *,
    dialect: str | None,
    paramstyle: str,
) -> Statement:
    """``UPDATE table SET … WHERE key = …`` — assigned columns bind before key columns.

    Args:
        table: The destination table, optionally schema-qualified.
        columns: The columns present in the frame; those outside `key_columns` are assigned.
        key_columns: The columns matched in the ``WHERE`` clause.
        dialect: The dialect whose identifier quoting applies.
        paramstyle: The driver's declared PEP 249 ``paramstyle``.

    Returns:
        The statement and its per-row parameter order — assigned columns, then keys.

    Raises:
        BackendError: If `key_columns` is empty or leaves no column to assign.
    """
    _validate(table, columns, key_columns)
    if not key_columns:
        raise BackendError(
            "mode='update' needs key_columns= — without a WHERE clause every row of the "
            "target would be rewritten by every row of the source."
        )
    assigned = tuple(c for c in columns if c not in key_columns)
    if not assigned:
        raise BackendError(
            f"mode='update' has nothing to set: every column {list(columns)} is a key "
            "column. Add the columns to change, or use mode='upsert'."
        )
    order = assigned + key_columns
    marks, names = _placeholders(paramstyle, len(order))
    sets = ", ".join(
        f"{quote(c, dialect)} = {m}" for c, m in zip(assigned, marks[: len(assigned)], strict=True)
    )
    where = " AND ".join(
        f"{quote(c, dialect)} = {m}"
        for c, m in zip(key_columns, marks[len(assigned) :], strict=True)
    )
    return Statement(
        f"UPDATE {qualified_table(table, dialect)} SET {sets} WHERE {where}", order, names
    )


def delete(
    table: str, key_columns: tuple[str, ...], *, dialect: str | None, paramstyle: str
) -> Statement:
    """``DELETE FROM table WHERE key = …`` — one bound row deletes one key.

    Args:
        table: The destination table, optionally schema-qualified.
        key_columns: The columns matched in the ``WHERE`` clause.
        dialect: The dialect whose identifier quoting applies.
        paramstyle: The driver's declared PEP 249 ``paramstyle``.

    Returns:
        The statement and its per-row parameter order.

    Raises:
        BackendError: If `key_columns` is empty — an unqualified delete empties the table.
    """
    if not key_columns:
        raise BackendError(
            "mode='delete' needs key_columns=. A DELETE with no WHERE clause empties the "
            "table; use mode='overwrite' if that is genuinely what you want."
        )
    _validate(table, key_columns, key_columns)
    marks, names = _placeholders(paramstyle, len(key_columns))
    where = " AND ".join(
        f"{quote(c, dialect)} = {m}" for c, m in zip(key_columns, marks, strict=True)
    )
    return Statement(
        f"DELETE FROM {qualified_table(table, dialect)} WHERE {where}", key_columns, names
    )


def truncate(table: str, *, dialect: str | None) -> str:
    """The statement that empties `table` without dropping it.

    ``DELETE FROM`` rather than ``TRUNCATE``: truncation is DDL on several engines and so
    commits the surrounding transaction implicitly, which would publish an overwrite's
    empty state before its rows were written. A crash between the two would then have
    destroyed the table's contents. ``DELETE`` is transactional everywhere.

    Args:
        table: The table to empty, optionally schema-qualified.
        dialect: The dialect whose identifier quoting applies.

    Returns:
        The statement text.
    """
    return f"DELETE FROM {qualified_table(table, dialect)}"
