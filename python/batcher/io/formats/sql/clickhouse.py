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

from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import (
    probe_is_typed,
    push_down,
    require_module,
    schema_probe,
)

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["ClickHouseSource"]

_EXTRA = "clickhouse"
_MODULE = "clickhouse_connect"


def _client(params: dict[str, Any]) -> Any:
    """Open a fresh clickhouse-connect client (rebuilt per worker).

    The password is resolved *here*, not when the source was built: `params` is carried on
    a pickled split, so an `env:`/`file:` reference must still be a reference at that point
    and only becomes the secret on the machine that dials the server."""
    ch = require_module(_MODULE, extra=_EXTRA)
    if params.get("password"):
        params = {
            **params,
            "password": resolve_secret(params["password"], what="ClickHouse password"),
        }
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
        """The query's column types, taken off the stream without draining it.

        `query_arrow` downloads the entire result to read a schema the Arrow IPC stream
        already carries in its header. That is normally hidden because `ClickHouseSource.schema`
        asks a ``WHERE 1 = 0`` probe, which returns nothing — but the fallback for a driver
        whose probe comes back untyped runs the *real* query, and there materializing a whole
        relation to learn its column names is the difference between a metadata lookup and
        an OOM.
        """
        client = _client(self.params)
        try:
            with client.query_arrow_stream(self.query) as reader:
                return reader.schema
        finally:
            client.close()

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
    password: str = field(default="", repr=False)
    database: str | None = None
    # `client_kwargs` can carry auth material too (e.g. a password or TLS settings).
    client_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)

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

    def _split(
        self, predicate: dict | None = None, projection: list[str] | None = None
    ) -> _ClickHouseSplit:
        """The split, with the pushdown already folded into its SQL (see `push_down`)."""
        return _ClickHouseSplit(self._params(), push_down(self.query, predicate, projection))

    def schema(self) -> pa.Schema:
        """The relation's columns, from a zero-row probe rather than the whole query.

        See `schema_probe`: this used to execute the full query and discard every row.
        """
        probed = _ClickHouseSplit(self._params(), schema_probe(self.query)).schema()
        return probed if probe_is_typed(probed) else self._split().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._split(predicate, projection).read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._split(predicate, projection).iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"clickhouse:{self.host}:{self.query}"

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature)
        predicate: dict | None = None,
        projection: list[str] | None = None,
    ) -> list[Split]:
        """One split, whose SQL already carries the pushdown — so the worker's query is filtered.

        Without this, the worker rebuilt an *unfiltered* query from the split and ClickHouse
        streamed the whole table for the engine's `Filter` to discard.
        """
        return [self._split(predicate, projection)]
