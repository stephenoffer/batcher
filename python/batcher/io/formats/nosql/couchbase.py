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
from contextlib import contextmanager, suppress
from typing import Any

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher.io.formats.base import SOURCES
from batcher.io.formats.nosql.base import (
    PartitionSpec,
    ScanSource,
    offset_windows,
    require_driver,
    rows_to_batches,
    schema_from_rows,
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


@contextmanager
def _closing_cluster(cluster: Any) -> Iterator[Any]:
    """Yield `cluster`, closing it afterwards if this SDK build has a `close()`.

    The Columnar SDK documents `Cluster.close()`, but the driver is an optional extra we
    cannot import here to confirm it, and older builds managed the connection purely by
    refcount. A bare `contextlib.closing` therefore risks raising `AttributeError` from
    inside the teardown — which would *replace* whatever real exception the body raised,
    turning a clear database error into a confusing one. Closing when we can, and leaving
    it to the collector when we cannot, keeps the leak fix without that hazard.
    """
    try:
        yield cluster
    finally:
        with suppress(AttributeError):
            cluster.close()


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
        cred = credential.Credential.from_username_and_password(
            kw["username"], self._secret("password")
        )
        return columnar.Cluster.create_instance(kw["connstr"], cred)

    def _from_clause(self) -> str:
        kw = self._conn_kwargs
        return f"`{kw['database']}`.`{kw['scope']}`.`{kw['collection']}`"

    def _identity_suffix(self) -> str:
        kw = self._conn_kwargs
        # No `_fingerprint_material` override is needed here, and that is worth stating
        # rather than leaving to be rediscovered: Couchbase keeps its credential in a field
        # literally named ``password``, which is exactly what `connection_fingerprint`
        # drops by name. The connector's own connstr/username stay in the digest, so two
        # clusters sharing this collection path remain distinct relations.
        return f"{kw['database']}.{kw['scope']}.{kw['collection']}"

    def _infer_schema(self) -> pa.Schema:
        stmt = f"SELECT VALUE c FROM {self._from_clause()} c LIMIT 1"
        with _closing_cluster(self._cluster()) as cluster:
            rows = list(cluster.execute_query(stmt).rows())
        return schema_from_rows([rows[0]] if rows else [])

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
            # `closing` matters especially here: the bare `self._cluster()` leaked a cluster
            # on *both* paths — on success it was simply never closed, and on the exception
            # this method exists to swallow it was dropped mid-flight. Sizing the offset
            # windows is a routine step of every parallel read, so the leak recurred per read.
            with _closing_cluster(self._cluster()) as cluster:
                rows = list(cluster.execute_query(stmt).rows())
        except Exception as exc:  # a count that fails must not fail the read — fall back to serial
            note_suppressed("io", "count rows for a parallel read", exc)
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
        # Resolve the schema before opening this window's cluster: `self.schema()` falls
        # through to `_infer_schema`, which builds a cluster of its own, so leaving it below
        # held two connections open at once on the first window of every read.
        schema = self.schema() if not projection else None
        # The SDK's `rows()` is a streaming result, so this genuinely reads incrementally —
        # but the cluster was never closed at all, on any path. Holding it for the life of
        # the iteration is required (closing early would kill the stream mid-read), so it is
        # closed when the generator finishes *or is closed*, which is what `closing` around
        # the yields buys over a bare call.
        with _closing_cluster(self._cluster()) as cluster:
            rows = (
                row if isinstance(row, dict) else dict(row)
                for row in cluster.execute_query(stmt).rows()
            )
            yield from rows_to_batches(rows, schema=schema)
