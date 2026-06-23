"""Cassandra / ScyllaDB connector — token-range parallel scan to Arrow.

Cassandra distributes rows around a token ring; the natural parallel unit is a
*token range*. `CassandraSource` enumerates contiguous ranges over the full
``Murmur3`` token space and emits one `Split` per range, each issuing a
``SELECT … WHERE token(pk) >= ? AND token(pk) < ?``. The ranges partition the
ring disjointly and exhaustively, so concatenating every split equals a full
scan. Rows are assembled into Arrow at batch granularity via `rows_to_batches`.

The ``cassandra-driver`` import is deferred; a missing driver raises
`BackendError` with the ``cassandra`` extra hint. Registered as both
``"cassandra"`` and ``"scylla"`` (Scylla is wire-compatible). Connection kwargs
(contact points, auth) are stored verbatim and never logged.
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

__all__ = ["CassandraSource"]

# The inclusive bounds of the Murmur3 partitioner token space.
_MIN_TOKEN = -(2**63)
_MAX_TOKEN = 2**63 - 1

# A token-range locator: a half-open ``[start, end)`` interval of partition tokens.
_TokenRange = tuple[int, int]


class _CassandraSourceBase(ScanSource):
    """A Cassandra/Scylla table scanned in parallel by token range.

    Args:
        contact_points: Cluster seed hosts (list of host strings); never logged.
        keyspace: The keyspace to query.
        table: The table to scan.
        partition_key: The partition-key column name(s) for the ``token(...)``
            predicate. A single name or a tuple for composite keys.
        port: The native-protocol port (default 9042).
        auth: Optional ``{"username", "password"}`` mapping; never logged.
        partition_spec: Optional parallelism hint; ``segments`` sets the number
            of token ranges (default 64 — one per typical vnode count).
    """

    # Predicate pushdown: Kyber's pushed predicate → a CQL ``WHERE … ALLOW
    # FILTERING`` clause AND-merged with the token-range predicate. ``ALLOW
    # FILTERING`` can be slow (Cassandra may scan rows it then discards) but is
    # always correct, and only simple comparisons are pushed. ``<>`` / ``!=`` is
    # not valid CQL, so a predicate translating to one is skipped entirely; the
    # engine's `Filter` re-check then drops the rows. See `_pushed_cql`.
    supports_predicate = True

    __slots__ = ("_pushed_cql",)

    def __init__(
        self,
        *,
        contact_points: list[str],
        keyspace: str,
        table: str,
        partition_key: str | tuple[str, ...],
        port: int = 9042,
        auth: dict[str, str] | None = None,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec or PartitionSpec(segments=64),
            contact_points=list(contact_points),
            keyspace=keyspace,
            table=table,
            partition_key=partition_key,
            port=port,
            auth=auth,
        )
        # The CQL WHERE fragment for the active read (no token-range part), or None.
        # Set per-read by `read`/`iter_batches`; consumed in `_read_partition`.
        self._pushed_cql: str | None = None

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        self._pushed_cql = _pushed_cql(predicate)
        for partition in self._enumerate_partitions():
            yield from self._read_partition(partition, projection)

    def _session(self) -> tuple[Any, Any]:
        cassandra_cluster = require_driver("cassandra.cluster", "cassandra")
        kw = self._conn_kwargs
        auth_provider = None
        if kw["auth"]:
            cassandra_auth = require_driver("cassandra.auth", "cassandra")
            auth_provider = cassandra_auth.PlainTextAuthProvider(
                username=kw["auth"]["username"], password=kw["auth"]["password"]
            )
        cluster = cassandra_cluster.Cluster(
            contact_points=kw["contact_points"], port=kw["port"], auth_provider=auth_provider
        )
        session = cluster.connect(kw["keyspace"])
        return cluster, session

    def _pk_expr(self) -> str:
        pk = self._conn_kwargs["partition_key"]
        cols = pk if isinstance(pk, tuple) else (pk,)
        return ", ".join(cols)

    def _identity_suffix(self) -> str:
        return f"{self._conn_kwargs['keyspace']}.{self._conn_kwargs['table']}"

    def _infer_schema(self) -> pa.Schema:
        kw = self._conn_kwargs
        cluster, session = self._session()
        try:
            stmt = f"SELECT * FROM {kw['table']} LIMIT 1"
            rows = list(session.execute(stmt))
        finally:
            cluster.shutdown()
        if not rows:
            return pa.schema([])
        return pa.RecordBatch.from_pylist([dict(rows[0]._asdict())]).schema

    def _enumerate_partitions(self) -> list[_TokenRange]:
        return _token_ranges(max(1, self._partition_spec.segments))

    def _read_partition(
        self, partition: _TokenRange, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        kw = self._conn_kwargs
        start, end = partition
        cols = ", ".join(projection) if projection else "*"
        pk = self._pk_expr()
        cluster, session = self._session()
        try:
            stmt = (
                f"SELECT {cols} FROM {kw['table']} "
                f"WHERE token({pk}) >= {start} AND token({pk}) < {end}"
            )
            if self._pushed_cql is not None:
                stmt += f" AND {self._pushed_cql} ALLOW FILTERING"
            schema = self.schema()
            rows = (dict(row._asdict()) for row in session.execute(stmt))
            yield from rows_to_batches(rows, schema=schema if not projection else None)
        finally:
            cluster.shutdown()


def _pushed_cql(predicate: dict | None) -> str | None:
    """The CQL WHERE fragment for `predicate`, or None if it can't be pushed.

    Translates via the shared `to_sql_where` (CQL shares ``= < > <= >=`` syntax),
    then rejects the whole predicate if it produced a ``<>`` inequality — CQL has
    no ``<>`` operator, so pushing it would be a syntax error. The engine's
    `Filter` keeps the result correct when pushdown is skipped.
    """
    if predicate is None:
        return None
    from batcher.io.predicate import to_sql_where

    where = to_sql_where(predicate)
    if where is None or "<>" in where:
        return None
    return where


def _token_ranges(segments: int) -> list[_TokenRange]:
    """Divide the Murmur3 token ring into `segments` disjoint half-open ranges.

    The first range starts at ``_MIN_TOKEN`` and the last ends one past
    ``_MAX_TOKEN`` so every possible token falls in exactly one range.
    """
    span = (_MAX_TOKEN - _MIN_TOKEN + 1) // segments
    ranges: list[_TokenRange] = []
    start = _MIN_TOKEN
    for i in range(segments):
        end = _MAX_TOKEN + 1 if i == segments - 1 else start + span
        ranges.append((start, end))
        start = end
    return ranges


@SOURCES.register("cassandra")
class CassandraSource(_CassandraSourceBase):
    """A Cassandra table scanned in parallel by token range (see base)."""

    format_name = "cassandra"

    __slots__ = ()


@SOURCES.register("scylla")
class ScyllaSource(_CassandraSourceBase):
    """A ScyllaDB table — wire-compatible with Cassandra (see base)."""

    format_name = "scylla"

    __slots__ = ()
