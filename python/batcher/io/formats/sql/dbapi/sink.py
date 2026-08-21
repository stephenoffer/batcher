"""The row-level SQL write path — ``INSERT``, ``UPSERT``, ``UPDATE``, ``DELETE``.

Batcher could already *bulk-ingest* into a database: `ADBCSink` hands Arrow to a driver
that appends it, which is the right shape for loading a warehouse and the wrong shape for
almost everything else. An operational table is not loaded, it is *maintained* — a batch of
orders is upserted onto the keys it already has, a set of expired sessions is deleted, a
scored column is updated in place. None of that is expressible as an append, and none of it
was reachable from Batcher before this sink.

It is also the only SQL write path that reaches a database with no ADBC driver. ADBC covers
PostgreSQL, SQLite, DuckDB, Snowflake, BigQuery and FlightSQL; MySQL, MariaDB, Oracle, SQL
Server and the rest of the operational estate have a PEP 249 driver and nothing else, so
``ds.write.sql("orders", uri="mysql://…")`` had no way to run at all.

## The concessions, stated plainly

**Rows become Python objects.** A DB-API cursor is row-shaped and there is no way to hand a
driver a column, so this path pays object materialization per value exactly as `DBAPISource`
does on the way in. The conversion is column-wise and per chunk (`_bind`), never per row, and
`ADBCSink` remains the faster answer for a plain append to a database ADBC covers. That is
why `ds.write.sql` still routes an append to ADBC and comes here for everything else.

**A write is idempotent only where the mode is.** `with_retry` re-runs a transaction the
server rolled back, which is safe because the rollback restored the pre-write state. An
``append`` that the *client* lost the connection on after the server committed is the one
case no retry can decide, and it is the same ambiguity every at-least-once writer has: an
upsert absorbs the repeat, an append duplicates it. Prefer ``upsert`` where keys exist.

## Transactions

One `write` call is one transaction: every chunk of every statement, then a single commit.
A shard that fails rolls back whole, so a partial batch is never visible. That is the
property an operational table needs and the reason ``overwrite`` is a ``DELETE`` rather than
a ``TRUNCATE`` — see `_statements.truncate`.

A connection the **caller** supplied is never committed and never rolled back unless they
ask, because they may be mid-transaction and this write may be one statement inside a larger
unit of work they intend to commit themselves. `commit_writes` states which it is. For the
same reason it is never *probed*: PostgreSQL aborts a whole transaction on any failed
statement, so asking whether a table exists destroys the caller's work when the answer is no.
A borrowed connection therefore writes into a table that already exists.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.logging import get_logger, log_kv
from batcher.io.base._transient import with_retry
from batcher.io.formats.base import SINKS
from batcher.io.formats.sql._common import schema_probe
from batcher.io.formats.sql.dbapi import _ddl, _statements
from batcher.io.formats.sql.dbapi import source as _source
from batcher.io.formats.sql.dbapi._bind import null_key_rows, parameter_chunks
from batcher.io.formats.sql.dbapi._statements import Statement
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.plan.types import logical_bytes

__all__ = ["WRITE_MODES", "DBAPISink"]

#: The write modes this sink implements.
#:
#: ``append`` and ``overwrite`` are the two save modes that mean something for a table, so
#: the vocabulary is a genuine superset of `SAVE_MODES` rather than a second spelling of it.
#: The other four are row-level DML, which no save mode can express.
WRITE_MODES = ("append", "overwrite", "upsert", "update", "delete", "delete_insert")

#: Modes that discard rows the write itself did not supply. Ruinous once per shard.
_DESTRUCTIVE_MODES = frozenset({"overwrite"})

#: Modes whose statement is keyed on `key_columns`.
_KEYED_MODES = frozenset({"upsert", "update", "delete", "delete_insert"})

#: Rows bound into one ``executemany``.
#:
#: 1,000 rather than `bc_arrow::Morsel`'s 16,384 because the ceiling here is the *driver's*,
#: not the engine's: PostgreSQL's wire protocol allows 65,535 parameters per statement, so a
#: 10-column insert overflows at 6,554 rows and a 60-column one at 1,093. A chunk is a
#: round trip, so smaller costs latency; overflowing costs a hard protocol error. This is
#: the largest round number under the tightest common ceiling, and `rows_per_statement`
#: raises it for a narrow table.
DEFAULT_ROWS_PER_STATEMENT = 1_000

_LOGGER = get_logger("io.sql")


@SINKS.register("dbapi")
@dataclass(frozen=True, slots=True)
class DBAPISink:
    """Write Arrow rows into a SQL table through any PEP 249 driver.

    Args:
        module: The importable driver module, e.g. ``"psycopg"``, ``"pymysql"``,
            ``"sqlite3"``. Derived from `uri` when omitted.
        connect_kwargs: Keyword arguments passed to the driver's ``connect()``. String
            values may be ``env:``/``file:`` references, resolved on the worker so no
            secret is pickled. Derived from `uri` when omitted.
        connection: An already-open PEP 249 connection to write through, the way
            ``pandas.to_sql(name, con)`` accepts one. Single-node only, **never closed by
            Batcher** (see `commit_writes` for who commits it), and never probed: the
            destination table must already exist, because asking whether it does aborts the
            caller's transaction on PostgreSQL when the answer is no.
        uri: A standard connection URI (``mysql://host/shop``), the same one `bt.read.sql`
            takes, resolved to a driver and its connect kwargs.
        password: The password, as a literal or an ``env:``/``file:`` reference.
        dialect: Overrides the SQL dialect, which otherwise comes from `uri`'s scheme or
            from `module`. It decides identifier quoting, column types and the upsert
            spelling, so name it when writing through a driver Batcher cannot place —
            ``pyodbc`` reaches SQL Server, Oracle and half a dozen others.
        paramstyle: Overrides the driver's declared PEP 249 ``paramstyle``.
        mode: One of `WRITE_MODES`.
        key_columns: The columns identifying a row, required by the keyed modes.
        create_table: Create the destination table when it does not exist, using
            `key_columns` as its primary key. A keyed mode against a table with no such
            key silently degrades to an append, so this is on by default. Ignored for a
            `connection` the caller supplied.
        rows_per_statement: Rows bound into one ``executemany`` call.
        retries: Extra attempts after a transient failure — a deadlock, a serialization
            failure, a dropped connection. `0` disables retrying.
        retry_backoff_s: The first retry's backoff ceiling, doubled and jittered per round.
        commit_writes: Whether Batcher commits. `None` (the default) commits a connection
            Batcher opened and leaves a `connection=` the caller supplied alone, so this
            write can be one statement inside their larger transaction.

    Raises:
        BackendError: If no connection is derivable, `mode` is not a write mode, or a keyed
            mode is given no `key_columns`.
    """

    #: Row-level DML, so `mode` is this sink's own vocabulary rather than a save mode.
    dml_modes: ClassVar[tuple[str, ...]] = WRITE_MODES

    module: str = ""
    connect_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    connection: Any = field(default=None, repr=False, compare=False)
    uri: str | None = None
    password: str | None = field(default=None, repr=False)
    dialect: str | None = None
    paramstyle: str | None = None
    mode: str = "append"
    key_columns: tuple[str, ...] = ()
    create_table: bool = True
    rows_per_statement: int = DEFAULT_ROWS_PER_STATEMENT
    retries: int = 3
    retry_backoff_s: float = 0.25
    commit_writes: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_columns", tuple(self.key_columns or ()))
        if self.mode not in WRITE_MODES:
            raise BackendError(
                f"unknown SQL write mode {self.mode!r}; expected one of {list(WRITE_MODES)}."
            )
        if self.mode in _KEYED_MODES and not self.key_columns:
            raise BackendError(
                f"mode={self.mode!r} needs key_columns= — the columns that identify a row "
                "in the target table."
            )
        if self.uri is not None and not self.module:
            self._resolve_uri()
        if not self.module and self.connection is None:
            raise BackendError(
                "a SQL write needs uri= (e.g. 'mysql://host/shop'), module= with "
                "connect_kwargs=, or an already-open connection="
            )
        if self.rows_per_statement < 1:
            raise BackendError(f"rows_per_statement must be >= 1, got {self.rows_per_statement}")
        if self.retries < 0:
            raise BackendError(f"retries must be >= 0, got {self.retries}")

    def _resolve_uri(self) -> None:
        """Fill `module`, `connect_kwargs` and `dialect` in from the connection URI."""
        from batcher.io.formats.sql.dbapi._dsn import resolve_connection

        driver, kwargs, safe_uri, scheme = resolve_connection(
            str(self.uri),
            password=self.password,
            connect_kwargs=self.connect_kwargs,
        )
        object.__setattr__(self, "module", driver)
        object.__setattr__(self, "connect_kwargs", kwargs)
        object.__setattr__(self, "uri", safe_uri)
        if self.dialect is None:
            object.__setattr__(self, "dialect", scheme)

    # --- resolution -------------------------------------------------------------------

    def _driver(self) -> Any:
        """The driver module, whether it was named or inferred from a borrowed connection."""
        name = self.module
        if not name and self.connection is not None:
            name = type(_source._as_dbapi_connection(self.connection)).__module__.split(".")[0]
        return _source._import_driver(name)

    def _resolved_dialect(self) -> str | None:
        """The dialect to build SQL for, from the explicit setting, the URI, or the driver."""
        if self.dialect is not None:
            return self.dialect
        name = self.module or type(self.connection).__module__
        return _statements.dialect_for_driver(name) if name else None

    def _resolved_paramstyle(self) -> str:
        """The driver's declared ``paramstyle``, or the explicit override.

        PEP 249 requires the attribute, so its absence means the module is not a driver.
        Defaulting to ``qmark`` instead would build statements the driver cannot parse and
        fail with its syntax error rather than this one.
        """
        if self.paramstyle is not None:
            return self.paramstyle
        style = getattr(self._driver(), "paramstyle", None)
        if not isinstance(style, str):
            raise BackendError(
                f"{self.module or 'the connection'} declares no PEP 249 paramstyle, so "
                "Batcher cannot tell how to bind parameters to it. Pass paramstyle= "
                f"naming one of {sorted(_statements.PARAMSTYLES)}."
            )
        return style

    # --- planning ---------------------------------------------------------------------

    def _plan(self, schema: pa.Schema, table_name: str) -> list[Statement | str]:
        """The statements one write runs, in order, inside one transaction."""
        dialect = self._resolved_dialect()
        paramstyle = self._resolved_paramstyle()
        columns = tuple(schema.names)
        keys = self.key_columns
        build = {
            "append": lambda: [
                _statements.insert(table_name, columns, dialect=dialect, paramstyle=paramstyle)
            ],
            "overwrite": lambda: [
                _statements.truncate(table_name, dialect=dialect),
                _statements.insert(table_name, columns, dialect=dialect, paramstyle=paramstyle),
            ],
            "upsert": lambda: [
                _statements.upsert(
                    table_name, columns, keys, dialect=dialect, paramstyle=paramstyle
                )
            ],
            "update": lambda: [
                _statements.update(
                    table_name, columns, keys, dialect=dialect, paramstyle=paramstyle
                )
            ],
            "delete": lambda: [
                _statements.delete(table_name, keys, dialect=dialect, paramstyle=paramstyle)
            ],
            "delete_insert": lambda: [
                _statements.delete(table_name, keys, dialect=dialect, paramstyle=paramstyle),
                _statements.insert(table_name, columns, dialect=dialect, paramstyle=paramstyle),
            ],
        }
        return build[self.mode]()

    # --- execution --------------------------------------------------------------------

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Apply `table` to the database table named by `path`, as one transaction.

        Args:
            table: The rows to apply.
            path: The destination table name, optionally schema-qualified.

        Returns:
            A `WrittenFile` recording the rows applied, and the server's affected-row count
            under ``stats["affected_rows"]`` where the driver reports one.
        """
        if table.num_rows == 0 and self.mode != "overwrite":
            # An empty frame in a keyed mode is a no-op, and an empty `overwrite` is not:
            # it still means "leave the table holding exactly these rows", i.e. none.
            return WrittenFile(path=path, rows=0, bytes=0)
        if self.key_columns:
            self._warn_null_keys(table, path)
        affected = with_retry(
            lambda: self._apply(table, path),
            attempts=self.retries + 1,
            backoff_base_s=self.retry_backoff_s,
        )
        stats = {} if affected is None else {"affected_rows": affected}
        return WrittenFile(path=path, rows=table.num_rows, bytes=logical_bytes(table), stats=stats)

    def _warn_null_keys(self, table: pa.Table, path: str) -> None:
        """Say so when rows carry a null key, which matches nothing on any database."""
        nulls = null_key_rows(table, self.key_columns)
        if nulls:
            log_kv(
                _LOGGER,
                30,  # logging.WARNING, without importing logging for one constant
                "sql write: rows with a null key cannot match a target row",
                table=path,
                mode=self.mode,
                key_columns=list(self.key_columns),
                rows=nulls,
            )

    def _apply(self, table: pa.Table, path: str) -> int | None:
        """Run the whole write on one connection, committing once or rolling back whole."""
        borrowed = self.connection is not None
        conn = (
            _source._as_dbapi_connection(self.connection)
            if borrowed
            else _source._connect(self.module, self.connect_kwargs)
        )
        commit = not borrowed if self.commit_writes is None else self.commit_writes
        try:
            if self.create_table and self.mode != "delete" and not borrowed:
                self._ensure_table(conn, table.schema, path)
            affected = self._run(conn, table, path)
            if commit:
                conn.commit()
            return affected
        except Exception:
            if commit:
                # Only unwind a transaction this sink owns. Rolling back a caller's
                # connection would discard work they did before handing it over.
                try:
                    conn.rollback()
                except Exception as rollback_failure:  # pragma: no cover - driver-specific
                    raise BackendError(
                        f"sql write to {path!r} failed and the rollback failed too: "
                        f"{rollback_failure}"
                    ) from rollback_failure
            raise
        finally:
            if not borrowed:
                conn.close()

    def _run(self, conn: Any, table: pa.Table, path: str) -> int | None:
        """Execute every statement of the plan; return the summed affected-row count."""
        affected: int | None = None
        cursor = conn.cursor()
        try:
            for step in self._plan(table.schema, path):
                if isinstance(step, str):
                    self._execute(cursor, step, None, path)
                    continue
                for chunk in parameter_chunks(
                    table, step, rows_per_statement=self.rows_per_statement
                ):
                    self._execute(cursor, step.sql, chunk, path, step.columns)
                    reported = getattr(cursor, "rowcount", -1)
                    if isinstance(reported, int) and reported >= 0:
                        affected = reported + (affected or 0)
        finally:
            cursor.close()
        return affected

    def _execute(
        self,
        cursor: Any,
        sql: str,
        chunk: list[Any] | None,
        path: str,
        columns: tuple[str, ...] = (),
    ) -> None:
        """Run one statement, re-raising a driver failure with the statement attached.

        A driver's error says what the *server* objected to and nothing about how the
        statement came to exist. ``ON CONFLICT clause does not match any PRIMARY KEY or
        UNIQUE constraint`` is the one that matters most here, because its cause is almost
        always the same and is invisible from the message: the table was created by an
        earlier ``append`` that had no `key_columns`, so it has no key for the upsert to
        conflict on. On MySQL the same mistake raises nothing at all and duplicates every
        row instead, which is why the hint is worth carrying.

        The original message is kept verbatim inside the new one, so `is_transient` still
        classifies a deadlock or a serialization failure as retryable through the wrap.
        """
        try:
            if chunk is None:
                cursor.execute(sql)
            else:
                cursor.executemany(sql, chunk)
        except Exception as exc:
            raise BackendError(
                f"sql write to {path!r} (mode={self.mode!r}) failed: {exc}."
                f"{self._diagnose(exc, path, columns)}\n{sql}"
            ) from exc

    def _diagnose(self, exc: Exception, path: str, columns: tuple[str, ...]) -> str:
        """The sentence that turns a driver's message into something actionable, or none.

        Two failures on this path have a cause the driver's own message cannot state, and
        both are ordinary rather than exotic.

        A conflict-target error means the target table has no key to conflict on — almost
        always because an earlier ``mode="append"`` created it without `key_columns`. On
        MySQL the same mistake raises nothing at all and duplicates every row instead,
        which is why it is worth naming here.

        A binding error means the *driver* refuses a Python type Arrow produced, most often
        `decimal.Decimal` — SQLite has no decimal type and its driver will not adapt one.
        The driver names a parameter position, which is meaningless to someone holding a
        frame, so the columns being bound and their Arrow types are named instead.

        A key violation under ``delete_insert`` means the frame holds two rows for one key.
        That is the one mode where a repeated key is an error rather than a last-writer-wins
        collapse, and the target's message says only that a constraint failed.
        """
        text = str(exc).lower()
        if self.mode == "delete_insert" and ("unique" in text or "duplicate" in text):
            return (
                f" The rows being written hold more than one row per key "
                f"{list(self.key_columns)}. mode='delete_insert' deletes the keys and then "
                "inserts every row, so a repeated key becomes a repeated row and the "
                "target's own key constraint rejects it. mode='upsert' collapses them "
                "instead, keeping the last row for each key; or deduplicate first with "
                "ds.drop_duplicates(subset=key_columns)."
            )
        if self.mode == "upsert" and "conflict" in text:
            return (
                f" The target table {path!r} needs a PRIMARY KEY or UNIQUE constraint on "
                f"{list(self.key_columns)} for an upsert to detect a conflict. A table "
                "Batcher created for an earlier mode='append' write has none unless that "
                "write also passed key_columns=."
            )
        if "bind" in text or "not supported" in text or "unsupported type" in text:
            return (
                f" The driver would not bind one of the columns being written "
                f"({list(columns)}). A PEP 249 driver accepts the Python types it knows, "
                "and Arrow produces types some of them do not — decimal.Decimal is the "
                "usual one. Cast the column to a type the driver takes "
                "(ds.cast({'amount': 'float64'})), or register an adapter with your driver."
            )
        return ""

    def _ensure_table(self, conn: Any, schema: pa.Schema, path: str) -> None:
        """Create the destination table when it is absent, tolerating a concurrent create.

        Existence is probed with the same zero-row ``WHERE 1 = 0`` query every source in
        this package uses for schema inference, rather than with a ``CREATE TABLE IF NOT
        EXISTS`` that SQL Server and Oracle do not have.

        Two shards of a distributed write can probe as absent at the same moment and both
        issue the ``CREATE``. The loser's error is not a failure — the table it wanted now
        exists — so it is swallowed only after a second probe proves that.

        Never called for a connection the caller supplied. See `_apply`.
        """
        dialect = self._resolved_dialect()
        if self._table_exists(conn, path, dialect):
            return
        ddl = _ddl.create_table(path, schema, dialect=dialect, key_columns=self.key_columns)
        cursor = conn.cursor()
        try:
            cursor.execute(ddl)
        except Exception:
            if not self._table_exists(conn, path, dialect):
                raise
        finally:
            cursor.close()

    @staticmethod
    def _table_exists(conn: Any, path: str, dialect: str | None) -> bool:
        """Whether `path` names a table this connection can select from.

        The rollback is load-bearing rather than tidy. PostgreSQL aborts the *whole*
        transaction when a statement fails, and every statement after that returns
        ``current transaction is aborted, commands ignored until end of transaction
        block``. So a probe for a table that does not exist — the one case this function is
        asked about — poisons the connection, and the ``CREATE TABLE`` that follows fails
        with an error naming neither the table nor the probe. Rolling back the failed probe
        is what makes "ask, then create" work at all on the database it matters most for.

        `DBAPISource.catalog_session` does the same thing for the same reason.
        """
        cursor = conn.cursor()
        try:
            cursor.execute(schema_probe(None, table=_statements.qualified_table(path, dialect)))
            cursor.fetchall()
            return True
        except Exception:
            with suppress(Exception):
                conn.rollback()
            return False
        finally:
            cursor.close()

    # --- the Sink protocol ------------------------------------------------------------

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - a table has no Hive layout
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write one shard; every shard targets the same table.

        A file sink gives each shard its own ``part-N`` file, so shards cannot collide. A
        database table has no such luxury, and a mode that discards rows the shard did not
        write is applied by *every* shard independently: six rows across three shards leave
        two, each shard having deleted what the one before it just wrote. It is invisible
        single-node, where there is only ever one shard, and appears at cluster scale as a
        wrong answer rather than an error — so it is refused.

        ``upsert``, ``update``, ``delete`` and ``append`` are all safe across shards: each
        one only ever touches the keys its own rows name.

        Raises:
            BackendError: If a destructive `mode` meets a multi-shard write.
        """
        if file_index > 0 and self.mode in _DESTRUCTIVE_MODES:
            raise BackendError(
                f"mode={self.mode!r} cannot be used for a distributed write to table "
                f"{path!r}: every shard would apply it to the same table, so each one "
                "would discard the shards before it. Use mode='upsert' with key_columns=, "
                "or mode='append' having emptied the table beforehand."
            )
        return [self.write(table, path)]

    def commit(self, manifest: WriteManifest, path: str) -> None:
        """No-op: each shard commits its own transaction as it writes."""
