"""ConnectorX source — the parallel relational reader for the long tail.

ConnectorX is a fast, Arrow-native query engine for relational databases
(MySQL/MariaDB, SQL Server, Azure Synapse, Oracle, Redshift, Trino, …). It reads
straight into Arrow and, given a ``partition_on`` integer column and a
``partition_num``, issues that many balanced ranged sub-queries in parallel.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import apply_projection, require_module

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["ConnectorXSource"]

_EXTRA = "connectorx"
_MODULE = "connectorx"


def _read_arrow(
    conn_uri: str,
    query: str,
    *,
    partition_on: str | None,
    partition_num: int,
) -> pa.Table:
    """Run one (possibly partitioned) ConnectorX read into an Arrow table."""
    cx = require_module(_MODULE, extra=_EXTRA)
    if partition_on is not None and partition_num > 1:
        return cx.read_sql(
            conn_uri,
            query,
            return_type="arrow",
            partition_on=partition_on,
            partition_num=partition_num,
        )
    return cx.read_sql(conn_uri, query, return_type="arrow")


@dataclass(frozen=True, slots=True)
class _ConnectorXSplit:
    """A picklable ConnectorX read: a URI + the rewritten query (no live conn)."""

    conn_uri: str
    query: str
    partition_on: str | None
    partition_num: int

    def _table(self, projection: list[str] | None) -> pa.Table:
        sql = apply_projection(self.query, projection)
        table = _read_arrow(
            self.conn_uri,
            sql,
            partition_on=self.partition_on,
            partition_num=self.partition_num,
        )
        return table

    def schema(self) -> pa.Schema:
        return self._table(None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection).to_batches()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"connectorx:{self.query}:{self.partition_on}:{self.partition_num}"


@SOURCES.register("connectorx")
@dataclass(frozen=True, slots=True)
class ConnectorXSource:
    """A relation read in parallel through ConnectorX.

    Args:
        query: The single logical SQL query to read.
        conn_uri: A ConnectorX/SQLAlchemy-style connection URI. Embeds
            credentials; carried on splits verbatim and never logged.
        partition_on: An integer column ConnectorX range-partitions on for
            parallel reads. ``None`` reads in a single partition.
        partition_num: Number of balanced partitions (ignored if
            `partition_on` is None).

    Raises:
        BackendError: If `connectorx` is not installed.
    """

    # Predicate pushdown: Kyber's pushed predicate → an appended SQL WHERE around
    # the base query (the database filters before returning Arrow). Class var, not
    # a dataclass field.
    supports_predicate: ClassVar[bool] = True

    query: str
    conn_uri: str
    partition_on: str | None = None
    partition_num: int = 1

    def _split(self, query: str | None = None) -> _ConnectorXSplit:
        return _ConnectorXSplit(
            self.conn_uri, query or self.query, self.partition_on, self.partition_num
        )

    def schema(self) -> pa.Schema:
        return self._split().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                query = f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}"
                return self._split(query).read(projection)
        return self._split().read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                query = f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}"
                yield from self._split(query).iter_batches(projection)
                return
        yield from self._split().iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"connectorx:{self.query}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """The independently-readable slices of this source.

        ConnectorX owns range-partitioning itself: a partitioned read fans the
        one logical query into ``partition_num`` balanced sub-queries and merges
        them into a single Arrow table. That is internal parallelism, not
        independent slices we can each ship to a different worker without
        re-deriving disjoint ranges (which would mean extra bound probes we
        explicitly forbid). So the source is a single split that delegates its
        parallelism to ConnectorX.
        """
        return [self._split()]
