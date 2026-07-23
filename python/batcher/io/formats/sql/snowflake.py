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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.sql._common import (
    connection_fingerprint,
    probe_is_typed,
    require_module,
    schema_probe,
)
from batcher.io.manifest import WrittenFile

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["SnowflakeSink", "SnowflakeSource"]

_EXTRA = "snowflake"
_MODULE = "snowflake.connector"


def _connect(connection_kwargs: dict[str, Any]) -> Any:
    """Open a fresh Snowflake connection (rebuilt per worker).

    Credential-bearing kwargs are resolved here, on the worker: `connection_kwargs` rides
    a pickled split, so a reference must still be a reference until this point."""
    sf = require_module(_MODULE, extra=_EXTRA)
    resolved = {
        k: (resolve_secret(v, what=f"Snowflake {k}") if isinstance(v, str) else v)
        for k, v in connection_kwargs.items()
    }
    return sf.connect(**resolved)


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
        """This chunk's column types — which cost a download of the chunk.

        `ResultBatch.to_arrow` is the only Arrow accessor the connector exposes on a vended
        chunk, and it fetches the whole chunk from cloud storage; there is no header or
        metadata call to take a schema from. So `SnowflakeSource.schema` avoids reaching here
        at all, via a ``WHERE 1 = 0`` probe whose chunk is empty, and this stays the documented
        fallback rather than being dressed up as cheap.
        """
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Chunk an already-downloaded result chunk — the chunk itself is atomic.

        This looks like the ``yield from self.read(...)`` anti-pattern and is deliberately not
        being changed. A `ResultBatch` *is* Snowflake's streaming unit: the connector vends one
        handle per cloud-storage chunk and `to_arrow()` fetches that chunk whole, with no
        incremental reader beneath it. The bound on memory is therefore the chunk, and it is
        applied one level up — `SnowflakeSource.iter_batches` walks the chunk handles in turn,
        so only one chunk is resident at a time. Splitting a chunk finer is not something the
        driver offers, and pretending otherwise here would just move the materialization.
        """
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
    connection_kwargs: dict[str, Any] = field(repr=False)

    def _chunks(self, sql: str) -> list[Any]:
        """ONE submission of `sql` → the connector's list of pickle-serializable chunks."""
        conn = _connect(self.connection_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return list(cur.get_result_batches() or [])
        finally:
            conn.close()

    def _result_batches(self, predicate: dict | None = None) -> list[Any]:
        """The chunks of the real read, with `predicate` pushed into the query."""
        return self._chunks(self._query(predicate))

    def schema(self) -> pa.Schema:
        """The relation's columns, from a zero-row probe where the warehouse allows it.

        This used to run the user's query in full and then *download chunk 0 from cloud
        storage* just to read its column names — a warehouse-credit charge and a transfer
        for a result that is thrown away, incurred before the real read even starts. See
        `schema_probe`.

        It also indexed `splits()[0]` unguarded, so a query returning no rows — which is
        ordinary, not exceptional — raised `IndexError` instead of reporting a schema.

        The probe is best-effort: if Snowflake vends no chunk for an empty result there is
        no Arrow schema to take, so this falls back to the original full-query path rather
        than failing. Worst case is what happened before; usually it is free.
        """
        probe = self._chunks(schema_probe(self.query))
        if probe:
            probed = probe[0].to_arrow().schema
            if probe_is_typed(probed):
                return probed
        full = self.splits()
        if not full:
            raise BackendError(
                f"Snowflake returned no result chunks for {self.query!r}, so its schema "
                f"cannot be determined"
            )
        return full[0].schema()

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
        """The learned-statistics key: the connection *and* the query, never the query alone.

        `connection_fingerprint` folds in account/user/warehouse/database/schema while skipping
        the credential kwargs, so the same query against two Snowflake accounts — or two
        databases on one account — is two relations, and rotating a password does not orphan
        the statistics a relation has accumulated.
        """
        return f"snowflake:{connection_fingerprint(self.connection_kwargs)}:{self.query}"

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature)
        predicate: dict | None = None,
        projection: list[str] | None = None,  # noqa: ARG002 (see below)
    ) -> list[Split]:
        """One split per Snowflake result chunk, from a query that already carries the predicate.

        `_result_batches` submits the query **once** and returns one picklable `ResultBatch` per
        cloud-storage chunk — the right shape. But it was submitted *unfiltered*: the predicate
        was applied only on the single-node path, so a distributed read materialized the whole
        result in Snowflake and shipped every chunk. Pushing it into `_query` means the warehouse
        does the filtering and the chunks that never match are never produced.

        Projection is not pushed here: the chunks are already materialized by the time they are
        vended, so `Split.read(projection)` slices them client-side, which is where it has to
        happen. (`_query` could take it too — that is a separate change to `_query`'s callers.)
        """
        return [_SnowflakeBatchSplit(rb, i) for i, rb in enumerate(self._result_batches(predicate))]


@SINKS.register("snowflake")
@dataclass(frozen=True, slots=True)
class SnowflakeSink:
    """Ingest Arrow tables into a Snowflake table.

    Args:
        connection_kwargs: ``snowflake.connector.connect`` kwargs (never logged).
        mode: ``"append"`` (default) or ``"overwrite"`` the destination table.
    """

    connection_kwargs: dict[str, Any] = field(repr=False)
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
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Ingest one shard; each worker appends to the same destination table.

        Every shard writes into **one** table, so an overwriting disposition is applied
        once per shard rather than once per write: shard 2 overwrites what shard 1 just
        loaded, and the table ends up holding only whichever shard finished last. No
        error is raised, and single-node runs never show it because there is only ever
        one shard.

        This matters more here than for ADBC because ``overwrite`` is the *default* —
        `Writer.__call__` passes ``mode="overwrite"`` and Snowflake is in
        `_MODE_AWARE_SINKS`, so a plain ``ds.write.snowflake("ORDERS")`` that happened to
        run distributed silently kept a fraction of the rows.

        Doing it properly needs the overwrite to happen exactly once before any shard
        writes; shards run concurrently, so no shard can be the one to do it. There is no
        driver-side prepare hook on the `Sink` protocol, so this refuses instead.

        Raises:
            BackendError: If ``mode="overwrite"`` meets a multi-shard write.
        """
        if file_index > 0 and self.mode == "overwrite":
            raise BackendError(
                f"mode='overwrite' cannot be used for a distributed write to table "
                f"{path!r}: every shard would overwrite the shards before it, leaving "
                "only one shard's rows. Use mode='append', truncating the table "
                "beforehand if you need it empty."
            )
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No-op: Snowflake ingests are committed per shard on write."""
