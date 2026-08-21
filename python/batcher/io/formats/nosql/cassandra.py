"""Cassandra / ScyllaDB connector — token-range parallel scan to Arrow.

Cassandra distributes rows around a token ring; the natural parallel unit is a
*token range*. `CassandraSource` enumerates contiguous ranges over the full
``Murmur3`` token space and emits one `Split` per range, each issuing a
``SELECT … WHERE token(pk) >= ? AND token(pk) < ?``. The ranges partition the
ring disjointly and exhaustively, so concatenating every split equals a full
scan. Rows are assembled into Arrow at batch granularity via `rows_to_batches`.

## The scan is the wrong shape for a point read

Sixty-four token-range queries with ``ALLOW FILTERING`` is how you read a *table*. It is
also what ``ds.filter(col("user_id") == "u-42")`` used to do to read **one partition**:
every replica scanned its ranges, discarded almost everything, and the coordinator merged
the remains — the query shape Cassandra operators specifically warn against, because it
turns a partition read into a cluster-wide scan.

A predicate that pins every partition-key column to a literal cannot match a row outside
that one partition, so this source drops the token predicate and the fan-out and issues a
single ``SELECT``, which the driver routes straight to the owning replicas. Anything it
cannot prove that of — a partition key only partly pinned, a top-level ``OR``, a range on
the key — falls back to the token-range scan it always did.

`CassandraSink` is the write half. CQL has no distinct insert and update: an
``INSERT`` *is* an upsert, replacing whatever the primary key already held, which is
why this sink implements ``upsert`` and ``delete`` and declines ``append``. Writes go
through one prepared statement executed concurrently across the batch, which is the
shape the driver is built for — a logged ``BATCH`` across partitions is slower and
loads the coordinator, and is not the batching primitive it looks like.

The ``cassandra-driver`` import is deferred; a missing driver raises
`BackendError` with the ``cassandra`` extra hint. Registered as both
``"cassandra"`` and ``"scylla"`` (Scylla is wire-compatible). Connection kwargs
(contact points, auth) are stored verbatim and never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import (
    BulkSink,
    PartitionSpec,
    ScanSource,
    _ScanSplit,
    require_driver,
    rows_to_batches,
    schema_from_rows,
)
from batcher.io.predicate import pinned_columns

__all__ = ["CassandraSink", "CassandraSource", "ScyllaSource"]

# The inclusive bounds of the Murmur3 partitioner token space.
_MIN_TOKEN = -(2**63)
_MAX_TOKEN = 2**63 - 1

# A token-range locator: a half-open ``[start, end)`` interval of partition tokens.
_TokenRange = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _PartitionRead:
    """A locator for one partition, read with no token predicate and no fan-out.

    Picklable and connection-free like every other locator here: it carries the CQL
    ``WHERE`` fragment the predicate translated to, and nothing else.
    """

    where: str


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

    __slots__ = ()

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

    def _fingerprint_material(self) -> dict[str, Any]:
        """`_conn_kwargs` with the auth block reduced to its non-secret half.

        `ScanSource.identity()` drops credentials by *key name*, and Cassandra's do not
        have one: the password sits inside ``auth={"username", "password"}``, under a key
        called ``auth``. So the raw password was being digested into `identity()`, which is
        **persisted** as the learned-statistics key.

        The digest is one-way, so this is not a plaintext leak — it is a stability bug.
        Every rotation of the Cassandra password changes the key, and the table's
        accumulated cardinalities are orphaned with nothing to indicate it; Kyber quietly
        goes back to cold estimates. Keeping the username preserves the part that actually
        distinguishes two connections to the same ring.
        """
        auth = self._conn_kwargs.get("auth")
        return {
            **self._conn_kwargs,
            "auth": auth.get("username") if isinstance(auth, dict) else auth,
        }

    def _infer_schema(self) -> pa.Schema:
        kw = self._conn_kwargs
        cluster, session = self._session()
        try:
            stmt = f"SELECT * FROM {kw['table']} LIMIT 1"
            rows = list(session.execute(stmt))
        finally:
            cluster.shutdown()
        return schema_from_rows([dict(rows[0]._asdict())] if rows else [])

    def _pk_columns(self) -> tuple[str, ...]:
        """The partition-key column names, whether declared as one name or a tuple."""
        pk = self._conn_kwargs["partition_key"]
        return pk if isinstance(pk, tuple) else (pk,)

    def splits(
        self,
        target_size: int | None = None,
        predicate: dict | None = None,
    ) -> list[Any]:
        """One partition read when the predicate pins the whole partition key; else the ring.

        Args:
            target_size: Ignored; a store splits by its own partitions.
            predicate: The predicate Kyber pushed to this scan.

        Returns:
            The splits to read, which is a single partition read when one is provable.
        """
        where = _single_partition_where(predicate, self._pk_columns())
        if where is None:
            return super().splits(target_size, predicate)
        return [
            _ScanSplit(
                source_cls=type(self),
                conn_kwargs=dict(self._conn_kwargs),
                partition=_PartitionRead(where),
                identity_prefix=self.identity(),
                predicate=predicate,
            )
        ]

    def _enumerate_partitions(self) -> list[_TokenRange]:
        return _token_ranges(max(1, self._partition_spec.segments))

    def _read_partition(
        self,
        partition: _TokenRange | _PartitionRead,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        kw = self._conn_kwargs
        cols = ", ".join(projection) if projection else "*"
        if isinstance(partition, _PartitionRead):
            # No token predicate: the partition key is pinned, so the driver's
            # token-aware routing sends this straight to the owning replicas.
            stmt = f"SELECT {cols} FROM {kw['table']} WHERE {partition.where} ALLOW FILTERING"
        else:
            start, end = partition
            pushed = _pushed_cql(predicate)
            pk = self._pk_expr()
            stmt = (
                f"SELECT {cols} FROM {kw['table']} "
                f"WHERE token({pk}) >= {start} AND token({pk}) < {end}"
            )
            if pushed is not None:
                stmt += f" AND {pushed} ALLOW FILTERING"
        # Resolve the schema *before* opening this partition's session. `self.schema()`
        # falls through to `_infer_schema`, which builds and shuts down a cluster of its
        # own — done inside the `try` below, that stood up a second connection to the ring
        # while this one was live, on the first partition of every read. The result is
        # cached, so hoisting it costs nothing and the nesting simply disappears.
        schema = self.schema() if not projection else None
        cluster, session = self._session()
        try:
            # `session.execute` returns a paging ResultSet, so the rows genuinely stream:
            # the driver fetches a page at a time and `rows_to_batches` converts at batch
            # granularity. Nothing here materializes the token range.
            rows = (dict(row._asdict()) for row in session.execute(stmt))
            yield from rows_to_batches(rows, schema=schema)
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


def _single_partition_where(
    predicate: dict[str, Any] | None, partition_key: tuple[str, ...]
) -> str | None:
    """The CQL ``WHERE`` for a single-partition read, or None to scan the ring.

    Sound exactly when every partition-key column is pinned to a literal by a top-level
    conjunct: each such term is true of every matching row, so no matching row can live
    outside the partition those values name. A partly-pinned composite key is not enough —
    Cassandra hashes the *whole* key, so fixing one of two columns names no partition at
    all — and neither is a range, for the same reason.

    Args:
        predicate: The pushed predicate, or None.
        partition_key: The table's partition-key column names.

    Returns:
        The ``WHERE`` fragment to read one partition with, or None.
    """
    if predicate is None or not partition_key:
        return None
    if not set(partition_key) <= pinned_columns(predicate):
        return None
    return _pushed_cql(predicate)


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


#: Statements in flight at once against the cluster.
#:
#: Cassandra's throughput comes from spreading writes across coordinators, not from
#: batching them: a ``BATCH`` spanning partitions makes one node responsible for fanning
#: every mutation out, which is slower than the same mutations sent independently and is
#: the most common way a Cassandra write path is accidentally serialized. So the sink
#: prepares one statement and runs it concurrently, bounded here.
_CASSANDRA_CONCURRENCY = 64


@SINKS.register("cassandra")
class CassandraSink(BulkSink):
    """Write rows into a Cassandra or ScyllaDB table with one prepared statement.

    ``append`` is declined because CQL cannot express it: an ``INSERT`` replaces the row
    holding the same primary key, so an "append" would silently be an upsert. ``INSERT …
    IF NOT EXISTS`` does express it, but only through a Paxos round trip per row, which
    is a different operation with a different cost and belongs behind a different name.

    ``overwrite`` is declined for the same reason DynamoDB's is: emptying the table means
    ``TRUNCATE``, which is a cluster-wide schema operation rather than a write, and
    reaching it by passing a string is not a thing a write API should offer.

    Args:
        contact_points: Cluster seed hosts; never logged.
        keyspace: The target keyspace.
        table: The target table, when it is not the write's destination name.
        key_columns: The primary-key columns, required by ``delete``.
        port: The native-protocol port.
        auth: Optional ``{"username", "password"}`` mapping; never logged.
        concurrency: Statements in flight at once.
        mode: ``"upsert"`` (default) or ``"delete"``.
    """

    format_name = "cassandra"
    supported_modes = ("upsert", "delete")

    __slots__ = ("concurrency", "key_columns", "table")

    def __init__(
        self,
        *,
        contact_points: list[str] | tuple[str, ...],
        keyspace: str,
        table: str | None = None,
        key_columns: str | list[str] | tuple[str, ...] = (),
        port: int = 9042,
        auth: dict[str, str] | None = None,
        concurrency: int = _CASSANDRA_CONCURRENCY,
        mode: str = "upsert",
    ) -> None:
        keys = (key_columns,) if isinstance(key_columns, str) else tuple(key_columns)
        super().__init__(
            key_field=keys[0] if keys else "id",
            mode=mode,
            contact_points=list(contact_points),
            keyspace=keyspace,
            port=port,
            auth=auth,
        )
        if mode == "delete" and not keys:
            from batcher._internal.errors import BackendError

            raise BackendError(
                "mode='delete' needs key_columns= — the primary-key columns of the target "
                "table. CQL cannot delete by a non-key predicate."
            )
        self.table = table
        self.key_columns = keys
        self.concurrency = concurrency

    def _session(self) -> tuple[Any, Any]:
        """A cluster and session, opened here so the credential is resolved on the worker."""
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
        return cluster, cluster.connect(kw["keyspace"])

    def _statement(self, table: str, columns: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        """The CQL to prepare, and the per-row column order it binds."""
        if self.mode == "delete":
            where = " AND ".join(f"{c} = ?" for c in self.key_columns)
            return f"DELETE FROM {table} WHERE {where}", self.key_columns
        placeholders = ", ".join("?" for _ in columns)
        names = ", ".join(columns)
        return f"INSERT INTO {table} ({names}) VALUES ({placeholders})", columns

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Prepare once, then execute concurrently over the batch."""
        from cassandra.concurrent import execute_concurrent_with_args

        columns = tuple(rows[0])
        cql, order = self._statement(self.table or path, columns)
        cluster, session = self._session()
        try:
            prepared = session.prepare(cql)
            parameters = [tuple(row.get(name) for name in order) for row in rows]
            results = execute_concurrent_with_args(
                session,
                prepared,
                parameters,
                concurrency=self.concurrency,
                raise_on_first_error=True,
            )
            failed = [outcome for success, outcome in results if not success]
            if failed:
                from batcher._internal.errors import BackendError

                raise BackendError(
                    f"cassandra {self.mode} to {self.table or path!r}: {len(failed)} of "
                    f"{len(parameters)} statements failed; the first was {failed[0]}"
                )
        finally:
            cluster.shutdown()
