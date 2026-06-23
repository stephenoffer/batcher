"""Couchbase connector — Columnar (analytics) SDK to Arrow.

The ``couchbase-columnar`` SDK runs SQL++ over Couchbase's analytics service and
can stream results; `CouchbaseSource` issues a SQL++ query and assembles the
returned documents into Arrow at batch granularity. Parallel reads partition the
result with ``LIMIT``/``OFFSET`` windows — one `Split` per window, a disjoint and
exhaustive cover of an ordered query.

The ``couchbase_columnar`` import is deferred; a missing driver raises
`BackendError` with the ``couchbase`` extra hint. Connection kwargs (connstr,
credentials) are stored verbatim and never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.formats.nosql.base import (
    PartitionSpec,
    ScanSource,
    require_driver,
    rows_to_batches,
)

__all__ = ["CouchbaseSource"]

# An offset-window locator: ``(offset, limit)`` over an ordered SQL++ result.
_Window = tuple[int, int]

# Rows per offset window when the source is partitioned.
_WINDOW_ROWS = 100_000


@SOURCES.register("couchbase")
class CouchbaseSource(ScanSource):
    """A Couchbase analytics (Columnar) collection read via SQL++.

    Args:
        connstr: A Couchbase connection string (``couchbases://…``); never logged.
        username: The Columnar username; never logged.
        password: The Columnar password; never logged.
        database: The analytics database name.
        scope: The analytics scope name.
        collection: The analytics collection name.
        partition_spec: Optional parallelism hint; ``segments`` sets how many
            ``OFFSET``/``LIMIT`` windows to split the result into (default 1).
    """

    format_name = "couchbase"
    # Predicate pushdown: a pushed predicate → a SQL++ WHERE clause (server-side).
    supports_predicate = True

    __slots__ = ()

    def __init__(
        self,
        *,
        connstr: str,
        username: str,
        password: str,
        database: str,
        scope: str,
        collection: str,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            connstr=connstr,
            username=username,
            password=password,
            database=database,
            scope=scope,
            collection=collection,
        )

    def _cluster(self) -> Any:
        columnar = require_driver("couchbase_columnar.cluster", "couchbase")
        credential = require_driver("couchbase_columnar.credential", "couchbase")
        kw = self._conn_kwargs
        cred = credential.Credential.from_username_and_password(kw["username"], kw["password"])
        return columnar.Cluster.create_instance(kw["connstr"], cred)

    def _from_clause(self) -> str:
        kw = self._conn_kwargs
        return f"`{kw['database']}`.`{kw['scope']}`.`{kw['collection']}`"

    def _identity_suffix(self) -> str:
        kw = self._conn_kwargs
        return f"{kw['database']}.{kw['scope']}.{kw['collection']}"

    def _infer_schema(self) -> pa.Schema:
        cluster = self._cluster()
        stmt = f"SELECT VALUE c FROM {self._from_clause()} c LIMIT 1"
        rows = list(cluster.execute_query(stmt).rows())
        if not rows:
            return pa.schema([])
        return pa.RecordBatch.from_pylist([rows[0]]).schema

    def _enumerate_partitions(self) -> list[_Window]:
        segments = max(1, self._partition_spec.segments)
        if segments == 1:
            return [(0, 0)]  # 0 limit = unbounded single window.
        return [(i * _WINDOW_ROWS, _WINDOW_ROWS) for i in range(segments)]

    @staticmethod
    def _sql_where(predicate: dict | None) -> str | None:
        if predicate is None:
            return None
        from batcher.io.predicate import to_sql_where

        return to_sql_where(predicate)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        where = self._sql_where(predicate)
        for partition in self._enumerate_partitions():
            yield from self._read_partition(partition, projection, where)

    def _read_partition(
        self, partition: _Window, projection: list[str] | None, where: str | None = None
    ) -> Iterator[pa.RecordBatch]:
        offset, limit = partition
        select = ", ".join(f"c.`{c}`" for c in projection) if projection else "VALUE c"
        stmt = f"SELECT {select} FROM {self._from_clause()} c"
        if where:
            stmt += f" WHERE {where}"
        if limit:
            stmt += f" ORDER BY META(c).id LIMIT {limit} OFFSET {offset}"
        cluster = self._cluster()
        schema = self.schema() if not projection else None
        rows = (
            row if isinstance(row, dict) else dict(row)
            for row in cluster.execute_query(stmt).rows()
        )
        yield from rows_to_batches(rows, schema=schema)
