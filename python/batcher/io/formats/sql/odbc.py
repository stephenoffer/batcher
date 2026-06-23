"""ODBC source — Arrow reads via turbodbc, for the enterprise tail.

turbodbc speaks ODBC and returns Arrow directly (``cursor.fetchallarrow()``),
covering the enterprise long tail with no first-party Arrow driver: DB2,
Teradata, SAP HANA, Vertica, and any ODBC-reachable system. ODBC exposes no
shippable result partitions, so the single logical query is one split that
fetches the result as Arrow; connection details (a DSN or full connection
string, which may embed credentials) ride on the split as plain values and are
never logged. Connections are rebuilt per worker and never pickled.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import require_module

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["ODBCSource"]

_EXTRA = "odbc"
_MODULE = "turbodbc"


def _connect(dsn: str | None, connection_string: str | None) -> Any:
    """Open a fresh turbodbc connection (rebuilt per worker)."""
    turbodbc = require_module(_MODULE, extra=_EXTRA)
    if connection_string is not None:
        return turbodbc.connect(connection_string=connection_string)
    return turbodbc.connect(dsn=dsn)


@dataclass(frozen=True, slots=True)
class _ODBCSplit:
    """A picklable ODBC read: DSN/connection string + the query (no live conn)."""

    dsn: str | None
    connection_string: str | None
    query: str

    def _table(self) -> pa.Table:
        conn = _connect(self.dsn, self.connection_string)
        try:
            cur = conn.cursor()
            cur.execute(self.query)
            result = cur.fetchallarrow()
            if isinstance(result, pa.RecordBatch):
                result = pa.Table.from_batches([result])
            return result
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
        return f"odbc:{self.dsn or self.connection_string}:{self.query}"


@SOURCES.register("odbc")
@dataclass(frozen=True, slots=True)
class ODBCSource:
    """A relation read over ODBC via turbodbc.

    Args:
        query: The single SQL query to execute.
        dsn: A configured ODBC data-source name. Mutually exclusive with
            `connection_string`.
        connection_string: A full ODBC connection string (may embed
            credentials). Carried on the split and never logged.

    Raises:
        BackendError: If `turbodbc` is not installed, or neither `dsn` nor
            `connection_string` is given.
    """

    # Predicate pushdown: Kyber's pushed predicate → an appended SQL WHERE (the
    # server filters before returning Arrow). Class var, not a dataclass field.
    supports_predicate: ClassVar[bool] = True

    query: str
    dsn: str | None = None
    connection_string: str | None = None

    def __post_init__(self) -> None:
        if self.dsn is None and self.connection_string is None:
            raise BackendError("ODBCSource requires either dsn= or connection_string=")

    def _split(self) -> _ODBCSplit:
        return _ODBCSplit(self.dsn, self.connection_string, self.query)

    def schema(self) -> pa.Schema:
        return self._split().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                sql = f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}"
                return _ODBCSplit(self.dsn, self.connection_string, sql).read(projection)
        return self._split().read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                sql = f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}"
                yield from _ODBCSplit(self.dsn, self.connection_string, sql).iter_batches(
                    projection
                )
                return
        yield from self._split().iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"odbc:{self.dsn or self.connection_string}:{self.query}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [self._split()]
