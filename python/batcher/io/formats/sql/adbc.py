"""ADBC / FlightSQL source + sink — Arrow-native database connectivity.

ADBC (Arrow Database Connectivity) is the zero-copy, Arrow-first driver layer:
results arrive as Arrow streams without a row-by-row Python materialization.
This module is the generic entry point for any ADBC driver (SQLite, PostgreSQL,
Snowflake-ADBC, DuckDB, …) and is the *only* backend with true, shippable
distributed partitions — via FlightSQL's ``adbc_execute_partitions``.

Single-submission contract:

- A FlightSQL driver returns ``(partition_descriptors, schema)`` from a single
  ``adbc_execute_partitions(sql)`` call. We build one `_ADBCPartitionSplit` per
  opaque descriptor; each split carries the descriptor bytes plus the driver +
  ``db_kwargs`` needed to rebuild a *fresh* connection on the worker, then reads
  its slice with ``adbc_read_partition(desc).fetch_arrow_table()``.
- A non-partitioning driver yields a single `_ADBCQuerySplit` that streams the
  whole query once with ``fetch_record_batch``.

Credentials live only in ``db_kwargs``/``conn_kwargs`` carried on the split as
plain values; they are never logged. Connections are never pickled.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.sql._common import apply_projection, require_module
from batcher.io.manifest import WrittenFile

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["ADBCSink", "ADBCSource"]

_EXTRA = "sql"
_MODULE = "adbc_driver_manager.dbapi"


def _connect(driver: str, db_kwargs: dict[str, Any], conn_kwargs: dict[str, Any] | None) -> Any:
    """Open a fresh DBAPI connection for `driver` (rebuilt per worker)."""
    dbapi = require_module(_MODULE, extra=_EXTRA)
    return dbapi.connect(driver=driver, db_kwargs=db_kwargs, **(conn_kwargs or {}))


@dataclass(frozen=True, slots=True)
class _ADBCQuerySplit:
    """A single streaming read of one logical query over a fresh connection."""

    driver: str
    db_kwargs: dict[str, Any]
    conn_kwargs: dict[str, Any] | None
    sql: str

    def _table(self) -> pa.Table:
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self.sql)
            return cur.fetch_arrow_table()
        finally:
            conn.close()

    def schema(self) -> pa.Schema:
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self.sql)
            reader = cur.fetch_record_batch()
            for batch in reader:
                yield batch.select(projection) if projection is not None else batch
        finally:
            conn.close()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"adbc:{self.driver}:{self.sql}"


@dataclass(frozen=True, slots=True)
class _ADBCPartitionSplit:
    """One FlightSQL partition descriptor, read via a fresh worker connection."""

    driver: str
    db_kwargs: dict[str, Any]
    conn_kwargs: dict[str, Any] | None
    descriptor: bytes
    index: int

    def _table(self) -> pa.Table:
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            reader = cur.adbc_read_partition(self.descriptor)
            return reader.fetch_arrow_table()
        finally:
            conn.close()

    def schema(self) -> pa.Schema:
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"adbc-part:{self.driver}:{self.index}"


@SOURCES.register("adbc")
@dataclass(frozen=True, slots=True)
class ADBCSource:
    """A relation read through an ADBC driver, optionally FlightSQL-partitioned.

    Args:
        query: The SQL to run. Mutually exclusive with `table`.
        table: A table name to read in full (``SELECT * FROM table``).
        driver: The ADBC driver to load (e.g. ``"adbc_driver_sqlite"`` or a
            FlightSQL driver path).
        db_kwargs: Driver/database connection kwargs (DSN, uri, credentials).
            Carried on splits verbatim and never logged.
        conn_kwargs: Extra ``connect()`` kwargs (autocommit, etc).
        partition: If True, attempt FlightSQL ``adbc_execute_partitions`` to
            produce one split per server-side partition (one query submission).

    Raises:
        BackendError: If `adbc_driver_manager` is not installed, or neither
            `query` nor `table` is given.
    """

    # Predicate pushdown: Kyber's pushed predicate → an appended SQL WHERE (the
    # server filters before returning Arrow). Class var, not a dataclass field.
    supports_predicate: ClassVar[bool] = True

    driver: str
    db_kwargs: dict[str, Any]
    query: str | None = None
    table: str | None = None
    conn_kwargs: dict[str, Any] | None = None
    partition: bool = False

    def __post_init__(self) -> None:
        if self.query is None and self.table is None:
            from batcher._internal.errors import BackendError

            raise BackendError("ADBCSource requires either query= or table=")

    def _sql(self, projection: list[str] | None = None) -> str:
        return apply_projection(self.query, projection, table=self.table)

    def schema(self) -> pa.Schema:
        return self.splits()[0].schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                sql = f"SELECT * FROM ({self._sql(projection)}) AS _bq_pred WHERE {where}"
                return _ADBCQuerySplit(self.driver, self.db_kwargs, self.conn_kwargs, sql).read(
                    projection
                )
        out: list[pa.RecordBatch] = []
        for split in self.splits():
            out.extend(split.read(projection))
        return out

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                sql = f"SELECT * FROM ({self._sql(projection)}) AS _bq_pred WHERE {where}"
                yield from _ADBCQuerySplit(
                    self.driver, self.db_kwargs, self.conn_kwargs, sql
                ).iter_batches(projection)
                return
        for split in self.splits():
            yield from split.iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"adbc:{self.driver}:{self.query or self.table}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        sql = self._sql()
        if self.partition:
            parts = self._execute_partitions(sql)
            if parts is not None:
                return parts
        return [_ADBCQuerySplit(self.driver, self.db_kwargs, self.conn_kwargs, sql)]

    def _execute_partitions(self, sql: str) -> list[Split] | None:
        """ONE submission → opaque descriptors. None if the driver can't partition."""
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            descriptors, _schema, _rows = cur.adbc_execute_partitions(sql)
        except (AttributeError, NotImplementedError):
            return None
        finally:
            conn.close()
        return [
            _ADBCPartitionSplit(self.driver, self.db_kwargs, self.conn_kwargs, bytes(desc), i)
            for i, desc in enumerate(descriptors)
        ]


@SINKS.register("adbc")
@dataclass(frozen=True, slots=True)
class ADBCSink:
    """Bulk-ingest Arrow tables into a database table via ADBC.

    Args:
        driver: The ADBC driver to load.
        db_kwargs: Driver/database connection kwargs (never logged).
        conn_kwargs: Extra ``connect()`` kwargs.
        mode: Ingest disposition passed to ``adbc_ingest`` (``"create"``,
            ``"append"``, ``"replace"``, ``"create_append"``).
    """

    driver: str
    db_kwargs: dict[str, Any]
    conn_kwargs: dict[str, Any] | None = None
    mode: str = "create_append"

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Ingest `table` into the destination table named by `path`."""
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.adbc_ingest(path, table, mode=self.mode)
            conn.commit()
        finally:
            conn.close()
        return WrittenFile(path=path, rows=table.num_rows, bytes=0)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - DB ingest is unpartitioned
        file_index: int = 0,  # noqa: ARG002
    ) -> list[WrittenFile]:
        """Ingest one shard; each worker appends to the same destination table."""
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No-op: ADBC ingests are committed per shard on write."""
