"""Snowflake source + sink — one query submission, N shippable result chunks.

Snowflake's Python connector exposes ``cursor.get_result_batches()``: after a
*single* query execution it returns a list of ``ResultBatch`` objects, each an
opaque, pickle-serializable handle to one chunk of the result set in cloud
storage. That is exactly Batcher's split model — one query submission, then one
independently-readable, picklable split per chunk that a worker fetches as Arrow
(``batch.to_arrow()``, zero-copy) without re-running the query.

Credentials live only in ``connection_kwargs`` carried on the split; they are
never logged. Connections are rebuilt per worker and never pickled.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.sql._common import require_module
from batcher.io.manifest import WrittenFile

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["SnowflakeSink", "SnowflakeSource"]

_EXTRA = "snowflake"
_MODULE = "snowflake.connector"


def _connect(connection_kwargs: dict[str, Any]) -> Any:
    """Open a fresh Snowflake connection (rebuilt per worker)."""
    sf = require_module(_MODULE, extra=_EXTRA)
    return sf.connect(**connection_kwargs)


@dataclass(frozen=True, slots=True)
class _SnowflakeBatchSplit:
    """One Snowflake ``ResultBatch`` chunk, fetched as Arrow on a worker.

    The ``result_batch`` is the connector's own pickle-serializable handle; it
    carries no live connection, so the split ships cleanly to a remote worker.
    """

    result_batch: Any
    index: int

    def _table(self) -> pa.Table:
        table = self.result_batch.to_arrow()
        # Some connector versions hand back a RecordBatch; normalize to a Table.
        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        return table

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
        rows = getattr(self.result_batch, "rowcount", None)
        return rows if isinstance(rows, int) else None

    def identity(self) -> str:
        return f"snowflake-chunk:{self.index}"


@SOURCES.register("snowflake")
@dataclass(frozen=True, slots=True)
class SnowflakeSource:
    """A relation read from Snowflake as Arrow result chunks.

    Args:
        query: The single SQL query to execute.
        connection_kwargs: ``snowflake.connector.connect`` kwargs (account,
            user, credentials, warehouse, database, schema, …). Carried on
            splits verbatim and never logged.

    Raises:
        BackendError: If `snowflake-connector-python` is not installed.
    """

    # Predicate pushdown: Kyber's pushed predicate → an appended SQL WHERE (the
    # warehouse filters before returning Arrow). Class var, not a dataclass field.
    supports_predicate: ClassVar[bool] = True

    query: str
    connection_kwargs: dict[str, Any]

    def _result_batches(self, predicate: dict | None = None) -> list[Any]:
        """ONE submission → the connector's list of pickle-serializable chunks."""
        conn = _connect(self.connection_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self._query(predicate))
            return list(cur.get_result_batches() or [])
        finally:
            conn.close()

    def schema(self) -> pa.Schema:
        return self.splits()[0].schema()

    def _query(self, predicate: dict | None) -> str:
        """The query, wrapped in a pushdown WHERE when `predicate` is pushable."""
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                return f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}"
        return self.query

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        conn = _connect(self.connection_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self._query(predicate))
            table = cur.fetch_arrow_all()
            if table is None:
                return []
            if projection is not None:
                table = table.select(projection)
            return table.to_batches()
        finally:
            conn.close()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        for i, rb in enumerate(self._result_batches(predicate)):
            yield from _SnowflakeBatchSplit(rb, i).iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"snowflake:{self.query}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [_SnowflakeBatchSplit(rb, i) for i, rb in enumerate(self._result_batches())]


@SINKS.register("snowflake")
@dataclass(frozen=True, slots=True)
class SnowflakeSink:
    """Ingest Arrow tables into a Snowflake table.

    Args:
        connection_kwargs: ``snowflake.connector.connect`` kwargs (never logged).
        mode: ``"append"`` (default) or ``"overwrite"`` the destination table.
    """

    connection_kwargs: dict[str, Any]
    mode: str = "append"

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Ingest `table` into the Snowflake table named by `path`."""
        write_pandas = require_module("snowflake.connector.pandas_tools", extra=_EXTRA).write_pandas
        conn = _connect(self.connection_kwargs)
        try:
            success, _chunks, nrows, _ = write_pandas(
                conn,
                table.to_pandas(),
                table_name=path,
                auto_create_table=True,
                overwrite=(self.mode == "overwrite"),
            )
            if not success:
                from batcher._internal.errors import BackendError

                raise BackendError(f"Snowflake write_pandas failed for table {path!r}")
        finally:
            conn.close()
        return WrittenFile(path=path, rows=nrows, bytes=0)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - warehouse table, unpartitioned
        file_index: int = 0,  # noqa: ARG002
    ) -> list[WrittenFile]:
        """Ingest one shard; each worker appends to the same destination table."""
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No-op: Snowflake ingests are committed per shard on write."""
