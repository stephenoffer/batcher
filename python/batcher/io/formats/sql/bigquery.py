"""BigQuery source — multi-stream Arrow reads via the Storage Read API.

The BigQuery Storage Read API is Arrow-native and natively parallel: a single
``create_read_session(data_format=ARROW, max_stream_count=N)`` call returns up to
``N`` independent read streams over the table in ONE submission. Each stream is a
picklable string name that a worker turns into Arrow with
``read_rows(stream).to_arrow(session)``.

Single-submission contract:

- A direct table read creates one read session (one call) → one split per
  stream. Server-side projection (``selected_fields``) and predicate
  (``row_restriction``) are pushed into the session at build time.
- Arbitrary SQL has no direct Storage API; we run the query once into a
  destination table, then open a read session on that table. Still one logical
  query submission plus the (cheap, metadata-only) session creation.

Credentials are taken from the ambient ``google.auth`` environment or an
explicit client; nothing credential-bearing is logged.
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

__all__ = ["BigQuerySource"]

_EXTRA = "bigquery"
_STORAGE_MODULE = "google.cloud.bigquery_storage_v1"
_BQ_MODULE = "google.cloud.bigquery"


def _read_client() -> Any:
    """A fresh BigQuery Storage Read client (rebuilt per worker)."""
    storage = require_module(_STORAGE_MODULE, extra=_EXTRA)
    return storage.BigQueryReadClient()


@dataclass(frozen=True, slots=True)
class _BigQueryStreamSplit:
    """One Storage Read API stream, fetched as Arrow on a worker.

    Carries only the stream name (a string) — no client, no session object — so
    it ships cleanly to a remote worker which rebuilds its own read client.
    """

    stream_name: str
    index: int

    def _table(self) -> pa.Table:
        client = _read_client()
        reader = client.read_rows(self.stream_name)
        return reader.to_arrow()

    def schema(self) -> pa.Schema:
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        client = _read_client()
        for page in client.read_rows(self.stream_name).rows().pages:
            batch = page.to_arrow()
            yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"bigquery-stream:{self.stream_name}:{self.index}"


@SOURCES.register("bigquery")
@dataclass(frozen=True, slots=True)
class BigQuerySource:
    """A relation read from BigQuery via the parallel Storage Read API.

    Args:
        query: Arbitrary SQL. Routed to a destination table, then stream-read.
            Mutually exclusive with `table`.
        table: A fully-qualified ``project.dataset.table`` to read directly.
        project: The billing/parent project for the read session.
        max_streams: Requested number of parallel Arrow streams (the server may
            return fewer). Each becomes one split.
        selected_fields: Server-side column projection pushed into the session.
        row_restriction: Server-side predicate (``WHERE``-style) pushed into the
            session (table reads only).

    Raises:
        BackendError: If the BigQuery client libraries are not installed, or
            neither `query` nor `table` is given.
    """

    # Predicate pushdown: Kyber's pushed predicate → the Storage Read API's
    # ``row_restriction`` (a SQL boolean string filtered server-side) for table
    # reads, or an appended WHERE around a query read. Class var, not a field.
    supports_predicate: ClassVar[bool] = True

    project: str
    query: str | None = None
    table: str | None = None
    max_streams: int = 8
    selected_fields: tuple[str, ...] | None = None
    row_restriction: str | None = None

    def __post_init__(self) -> None:
        if self.query is None and self.table is None:
            raise BackendError("BigQuerySource requires either query= or table=")

    @staticmethod
    def _pushed_where(predicate: dict | None) -> str | None:
        """The pushable SQL WHERE for `predicate`, or None if unpushable/absent."""
        if predicate is None:
            return None
        from batcher.io.predicate import to_sql_where

        return to_sql_where(predicate)

    def _table_ref(self, predicate: dict | None = None) -> str:
        """The table to stream-read (running query into a temp table if needed).

        For a query read, a pushable predicate is appended as a WHERE so the
        materialization job filters server-side before the read session opens.
        """
        if self.table is not None:
            return self.table
        return self._materialize_query(predicate)

    def _materialize_query(self, predicate: dict | None = None) -> str:
        """Run the SQL once into a destination table; return its identifier."""
        bq = require_module(_BQ_MODULE, extra=_EXTRA)
        client = bq.Client(project=self.project)
        where = self._pushed_where(predicate)
        query = f"SELECT * FROM ({self.query}) AS _bq_pred WHERE {where}" if where else self.query
        job = client.query(query)
        job.result()  # one submission; wait for completion
        dest = job.destination
        return f"{dest.project}.{dest.dataset_id}.{dest.table_id}"

    def _row_restriction(self, predicate: dict | None) -> str | None:
        """The effective Storage API ``row_restriction`` for a table read.

        Combines the constructor `row_restriction` with a pushable predicate via
        SQL ``AND``; a query read pushes through `_materialize_query` instead, so
        the predicate is not re-applied at the session there.
        """
        pushed = self._pushed_where(predicate) if self.table is not None else None
        if self.row_restriction and pushed:
            return f"({self.row_restriction}) AND ({pushed})"
        return pushed or self.row_restriction

    def _create_session(self, predicate: dict | None = None) -> tuple[Any, list[str]]:
        """ONE create_read_session call → (session, stream names)."""
        storage = require_module(_STORAGE_MODULE, extra=_EXTRA)
        types = require_module(f"{_STORAGE_MODULE}.types", extra=_EXTRA)
        client = storage.BigQueryReadClient()

        table_path = "projects/{}/datasets/{}/tables/{}".format(
            *self._table_ref(predicate).split(".")
        )
        read_options = types.ReadSession.TableReadOptions()
        if self.selected_fields:
            read_options.selected_fields.extend(self.selected_fields)
        row_restriction = self._row_restriction(predicate)
        if row_restriction:
            read_options.row_restriction = row_restriction
        session = types.ReadSession(
            table=table_path,
            data_format=types.DataFormat.ARROW,
            read_options=read_options,
        )
        session = client.create_read_session(
            parent=f"projects/{self.project}",
            read_session=session,
            max_stream_count=self.max_streams,
        )
        return session, [s.name for s in session.streams]

    def schema(self) -> pa.Schema:
        return self.splits()[0].schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        _session, streams = self._create_session(predicate)
        for i, name in enumerate(streams):
            out.extend(_BigQueryStreamSplit(name, i).read(projection))
        return out

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        _session, streams = self._create_session(predicate)
        for i, name in enumerate(streams):
            yield from _BigQueryStreamSplit(name, i).iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"bigquery:{self.project}:{self.query or self.table}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        _session, streams = self._create_session()
        return [_BigQueryStreamSplit(name, i) for i, name in enumerate(streams)]
