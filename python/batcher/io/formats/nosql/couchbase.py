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
    offset_windows,
    require_driver,
    rows_to_batches,
)

__all__ = ["CouchbaseSource"]

# An offset-window locator: ``(offset, limit)`` over an ordered SQL++ result. A ``limit`` of 0
# is unbounded — the tail of the cover, which is what makes it reach the end of the result.
_Window = tuple[int, int]

# SQL++ makes ``OFFSET`` a sub-clause of ``LIMIT`` (``LIMIT expr [OFFSET expr]``), so an
# offset can only be expressed alongside a limit. The unbounded tail window therefore uses
# this astronomically large limit — larger than any real result — so it reaches the end while
# still applying its offset. No dataset approaches 2**63 rows, so this never truncates.
_UNBOUNDED_LIMIT = (1 << 63) - 1


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
        """Offset windows that cover the whole collection — see `offset_windows`.

        Sized from an actual ``COUNT(*)`` so the windows are balanced, and terminated by an
        unbounded tail so the cover is exhaustive even if rows land after the count. If the
        count cannot be obtained, `offset_windows` degrades to one serial unbounded window,
        which is slow and right rather than fast and short.
        """
        segments = max(1, self._partition_spec.segments)
        if segments == 1:
            return [(0, 0)]  # one unbounded window: no count needed, and no connection made
        return offset_windows(self._total_rows(), segments)

    def _total_rows(self) -> int | None:
        """The collection's row count via SQL++ ``COUNT(*)``, or None if it cannot be had.

        Analytics answers this from metadata; it is one cheap query against the service that
        is about to be read, and it is what lets the offset windows be a *balanced* cover
        rather than a serial read.
        """
        stmt = f"SELECT VALUE COUNT(*) FROM {self._from_clause()} c"
        try:
            rows = list(self._cluster().execute_query(stmt).rows())
        except Exception:  # a count that fails must not fail the read — fall back to serial
            return None
        if not rows:
            return None
        value = rows[0]
        if isinstance(value, dict):
            value = next(iter(value.values()), None)
        return int(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _sql_where(predicate: dict | None) -> str | None:
        if predicate is None:
            return None
        from batcher.io.predicate import to_sql_where

        return to_sql_where(predicate)

    def _read_partition(
        self,
        partition: _Window,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        # Translate here, not in `iter_batches`: a worker rebuilt from a pickled split never
        # runs `iter_batches`, so a `where` computed there would never reach the query.
        where = self._sql_where(predicate)
        offset, limit = partition
        select = ", ".join(f"c.`{c}`" for c in projection) if projection else "VALUE c"
        stmt = f"SELECT {select} FROM {self._from_clause()} c"
        if where:
            stmt += f" WHERE {where}"
        # A window is ``(offset, limit)`` where ``limit == 0`` means *unbounded* — the tail of
        # the cover, which must still honour its ``offset`` and run to the end. Guarding on
        # ``limit`` alone dropped the ``OFFSET`` for that tail window, so it re-read the whole
        # collection and every prior window's rows came back a second time. The single serial
        # window ``(0, 0)`` still emits no ORDER BY/LIMIT/OFFSET (a plain full scan).
        if limit or offset:
            effective_limit = limit if limit else _UNBOUNDED_LIMIT
            stmt += f" ORDER BY META(c).id LIMIT {effective_limit} OFFSET {offset}"
        cluster = self._cluster()
        schema = self.schema() if not projection else None
        rows = (
            row if isinstance(row, dict) else dict(row)
            for row in cluster.execute_query(stmt).rows()
        )
        yield from rows_to_batches(rows, schema=schema)
