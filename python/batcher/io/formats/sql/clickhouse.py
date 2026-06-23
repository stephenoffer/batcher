"""ClickHouse source — Arrow reads via clickhouse-connect.

ClickHouse's official ``clickhouse-connect`` client reads queries directly into
Arrow with ``client.query_arrow(query)``. ClickHouse has no externally-shippable
result-partition handles (the server streams a single result), so the single
logical query is one split that streams in bounded chunks; the parallelism that
matters is server-side. Connection parameters (carrying credentials) ride on the
split as plain values and are never logged; connections are rebuilt per worker.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import require_module

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["ClickHouseSource"]

_EXTRA = "clickhouse"
_MODULE = "clickhouse_connect"


def _client(params: dict[str, Any]) -> Any:
    """Open a fresh clickhouse-connect client (rebuilt per worker)."""
    ch = require_module(_MODULE, extra=_EXTRA)
    return ch.get_client(**params)


@dataclass(frozen=True, slots=True)
class _ClickHouseSplit:
    """A picklable ClickHouse read: connection params + the query (no live conn)."""

    params: dict[str, Any]
    query: str

    def _table(self) -> pa.Table:
        client = _client(self.params)
        try:
            return client.query_arrow(self.query)
        finally:
            client.close()

    def schema(self) -> pa.Schema:
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        client = _client(self.params)
        try:
            with client.query_arrow_stream(self.query) as reader:
                for batch in reader:
                    yield batch.select(projection) if projection is not None else batch
        finally:
            client.close()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"clickhouse:{self.params.get('host')}:{self.query}"


@SOURCES.register("clickhouse")
@dataclass(frozen=True, slots=True)
class ClickHouseSource:
    """A relation read from ClickHouse as Arrow.

    Args:
        query: The single SQL query to execute.
        host: ClickHouse server host.
        port: Server port (driver default if None).
        username: Auth user (default ``"default"``).
        password: Auth password. Carried on the split and never logged.
        database: Default database for the query.
        client_kwargs: Any additional ``clickhouse_connect.get_client`` kwargs
            (secure, settings, …).

    Raises:
        BackendError: If `clickhouse-connect` is not installed.
    """

    # Predicate pushdown: Kyber's pushed predicate → an appended SQL WHERE (the
    # server filters before returning Arrow). Class var, not a dataclass field.
    supports_predicate: ClassVar[bool] = True

    query: str
    host: str
    port: int | None = None
    username: str = "default"
    password: str = ""
    database: str | None = None
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    def _params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"host": self.host, "username": self.username}
        if self.port is not None:
            params["port"] = self.port
        if self.password:
            params["password"] = self.password
        if self.database is not None:
            params["database"] = self.database
        params.update(self.client_kwargs)
        return params

    def _split(self) -> _ClickHouseSplit:
        return _ClickHouseSplit(self._params(), self.query)

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
                return _ClickHouseSplit(self._params(), sql).read(projection)
        return self._split().read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                sql = f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}"
                yield from _ClickHouseSplit(self._params(), sql).iter_batches(projection)
                return
        yield from self._split().iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"clickhouse:{self.host}:{self.query}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [self._split()]
