"""ConnectorX source — the parallel relational reader for the long tail.

ConnectorX is a fast, Arrow-native query engine for relational databases
(MySQL/MariaDB, SQL Server, Azure Synapse, Oracle, Redshift, Trino, …). It reads
straight into Arrow and, given a ``partition_on`` integer column and a
``num_partitions``, issues that many balanced ranged sub-queries in parallel.

Single-submission contract: there is exactly one *logical* query. When
``partition_on`` is set we build one split per partition, each carrying the same
query plus the (column, num, index) it must materialize — ConnectorX itself
computes the min/max bounds internally as part of the partitioned read, so we do
not issue a separate bound probe from the control plane. Without partitioning we
build a single split. Splits carry only the connection URI (which embeds, and
never logs, credentials) and the serialized query.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pyarrow as pa

from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import (
    apply_projection,
    connection_fingerprint,
    require_module,
)
from batcher.io.formats.sql._source_base import SingleResultQuerySource
from batcher.io.formats.sql.uri import redact_uri

__all__ = ["ConnectorXSource"]

_EXTRA = "connectorx"
_MODULE = "connectorx"


def _uri_fingerprint(conn_uri: str) -> str:
    """A stable, non-secret discriminator for a ConnectorX connection URI.

    ConnectorX takes its whole connection as one URI, so unlike the other backends here the
    credential is *inside* the identifying value rather than beside it in a named kwarg that
    `connection_fingerprint` knows to skip. `redact_uri` removes the inline password first, so
    the digest covers driver/user/host/port/database and nothing else — which is what makes
    rotating a password preserve a relation's accumulated statistics instead of orphaning them.
    """
    return connection_fingerprint({"conn_uri": redact_uri(conn_uri)})


def _read_arrow(
    conn_uri: str,
    query: str,
    *,
    partition_on: str | None,
    num_partitions: int,
) -> pa.Table:
    """Run one (possibly partitioned) ConnectorX read into an Arrow table.

    The URI embeds credentials, so an `env:`/`file:` reference is resolved here — on the
    worker that opens the connection — never on the driver that built the split."""
    conn_uri = resolve_secret(conn_uri, what="ConnectorX conn_uri") or conn_uri
    cx = require_module(_MODULE, extra=_EXTRA)
    if partition_on is not None and num_partitions > 1:
        return cx.read_sql(
            conn_uri,
            query,
            return_type="arrow",
            partition_on=partition_on,
            partition_num=num_partitions,
        )
    return cx.read_sql(conn_uri, query, return_type="arrow")


@dataclass(frozen=True, slots=True)
class _ConnectorXSplit:
    """A picklable ConnectorX read: a URI + the rewritten query (no live conn)."""

    conn_uri: str = field(repr=False)
    query: str
    partition_on: str | None
    num_partitions: int

    def _table(self, projection: list[str] | None) -> pa.Table:
        sql = apply_projection(self.query, projection)
        table = _read_arrow(
            self.conn_uri,
            sql,
            partition_on=self.partition_on,
            num_partitions=self.num_partitions,
        )
        return table

    def schema(self) -> pa.Schema:
        """The query's column types — which ConnectorX can only report by running it.

        Every other backend here can take a schema off a stream header or a session response
        without downloading rows. ConnectorX cannot: `read_sql` is a single Rust call that
        returns a finished `pyarrow.Table`, and the library exposes no metadata-only or lazy
        entry point to take a schema from. So this genuinely materializes, and the cheap path
        is `ConnectorXSource.schema`'s ``WHERE 1 = 0`` probe, which makes the materialized
        relation empty. Do not "fix" this into a streaming read — there is no stream to read.
        """
        return self._table(None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Chunk an already-materialized table — ConnectorX offers no incremental read.

        This looks like the ``yield from self.read(...)`` anti-pattern and is deliberately not
        being changed. `read_sql(..., return_type="arrow")` fans the query into its partitioned
        sub-queries *inside Rust* and hands back one complete `pyarrow.Table`; there is no
        record-batch reader, cursor, or page iterator to consume incrementally. Yielding the
        table's batches is therefore an honest description of what the driver can do, and
        `iter_batches` here bounds the caller's per-batch working set but **not** peak memory.

        The memory bound that does exist is `partition_on`/`num_partitions` at the source
        level: more partitions means more, smaller sub-queries. A caller that needs a hard
        memory ceiling on this backend wants ADBC or ODBC, which do stream.
        """
        yield from self._table(projection).to_batches()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return (
            f"connectorx:{_uri_fingerprint(self.conn_uri)}:"
            f"{self.query}:{self.partition_on}:{self.num_partitions}"
        )


@SOURCES.register("connectorx")
@dataclass(frozen=True, slots=True)
class ConnectorXSource(SingleResultQuerySource):
    """A relation read in parallel through ConnectorX.

    Args:
        query: The single logical SQL query to read.
        conn_uri: A ConnectorX/SQLAlchemy-style connection URI. Embeds
            credentials; carried on splits verbatim and never logged.
        partition_on: An integer column ConnectorX range-partitions on for
            parallel reads. ``None`` reads in a single partition.
        num_partitions: Number of balanced partitions (ignored if
            `partition_on` is None).

    Raises:
        BackendError: If `connectorx` is not installed.
    """

    # One split, but for a different reason than the base's other backends. They are one
    # split because the server vends no addressable result handles; ConnectorX *does*
    # range-partition, only internally — a partitioned read fans the one logical query into
    # `num_partitions` balanced sub-queries and merges them into a single Arrow table. That is
    # driver parallelism, not independent slices the engine could each ship to a different
    # worker without re-deriving disjoint ranges, which would mean the extra bound probes the
    # design forbids. So the base's `splits` is right here, and `_probe_split_for` below is
    # the one place the internal partitioning has to be suppressed.

    query: str
    conn_uri: str = field(repr=False)
    partition_on: str | None = None
    num_partitions: int = 1

    def _split_for(self, sql: str) -> _ConnectorXSplit:
        """The split for a real read, carrying ConnectorX's own range-partitioning."""
        return _ConnectorXSplit(self.conn_uri, sql, self.partition_on, self.num_partitions)

    def _probe_split_for(self, sql: str) -> _ConnectorXSplit:
        """Probe unpartitioned: fanning a `WHERE 1 = 0` into N sub-queries buys nothing.

        The only override of this hook in the tree, and the reason it exists: every other
        backend's split is already one query, so its probe is too.
        """
        return _ConnectorXSplit(self.conn_uri, sql, None, 1)

    def identity(self) -> str:
        """The learned-statistics key: the connection *and* the query, never the query alone.

        Keyed on the query alone, ``SELECT * FROM orders`` against production and against
        staging is one relation, so Kyber plans the thousand-row table with the billion-row
        table's cardinalities. Nothing errors — it is simply the wrong plan, from good code.
        """
        return f"connectorx:{_uri_fingerprint(self.conn_uri)}:{self.query}"
