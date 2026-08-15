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

from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import connection_fingerprint, require_module
from batcher.io.formats.sql._source_base import SingleResultQuerySource

__all__ = ["ODBCSource"]

_EXTRA = "odbc"
_MODULE = "turbodbc"


#: ODBC connection-string keys holding a password. Case-insensitive; drivers accept both.
_ODBC_SECRET_KEYS = frozenset({"pwd", "password"})


def redact_connection_string(value: str | None) -> str | None:
    """An ODBC connection string with its password masked.

    ODBC does not use a URI — it uses ``KEY=VALUE;KEY=VALUE``, and the password sits in
    it as ``PWD=`` alongside the server and user. So unlike a URI there is no component
    to lift out; the string has to be rewritten field by field.

    This matters more than a `repr` guard, because this string was going into
    `identity()` — the key learned statistics are *persisted* under. A password there is
    not merely printed, it is written to the metadata store and outlives the process.

    Args:
        value: An ODBC connection string, or None.

    Returns:
        The string with every password field masked, or None.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.odbc import redact_connection_string
            >>> redact_connection_string("SERVER=db;UID=admin;PWD=hunter2")
            'SERVER=db;UID=admin;PWD=***'
    """
    if not value:
        return value
    fields = []
    for field_ in value.split(";"):
        key, sep, _ = field_.partition("=")
        fields.append(f"{key}=***" if sep and key.strip().lower() in _ODBC_SECRET_KEYS else field_)
    return ";".join(fields)


def _connection_key(dsn: str | None, connection_string: str | None) -> str:
    """The connection's contribution to `identity()` — fingerprinted, never the raw string.

    The raw connection string used to go straight into `identity()`, which is *persisted*
    as the learned-statistics key, so an ODBC password was being written to the metadata
    store and outliving the process. Masking before fingerprinting also keeps a password
    rotation from changing the key and orphaning everything already learned.
    """
    return connection_fingerprint(
        {"dsn": dsn, "connection_string": redact_connection_string(connection_string)}
    )


def _connect(dsn: str | None, connection_string: str | None) -> Any:
    """Open a fresh turbodbc connection (rebuilt per worker).

    The connection string may embed credentials, so it resolves here — on the worker that
    dials — leaving the split carrying only an `env:`/`file:` reference."""
    turbodbc = require_module(_MODULE, extra=_EXTRA)
    if connection_string is not None:
        connection_string = resolve_secret(connection_string, what="ODBC connection_string")
        return turbodbc.connect(connection_string=connection_string)
    return turbodbc.connect(dsn=dsn)


def _as_batches(chunk: Any) -> Iterable[pa.RecordBatch]:
    """Normalize one turbodbc Arrow chunk to record batches.

    ``fetcharrowbatches`` is documented to yield `RecordBatch`, but builds differ on whether
    a chunk arrives as a batch or a single-chunk `Table`. Normalizing here keeps the caller
    from having to care, and keeps a version difference from turning into a type error on a
    machine we cannot test against."""
    if isinstance(chunk, pa.Table):
        return chunk.to_batches()
    return [chunk]


@dataclass(frozen=True, slots=True)
class _ODBCSplit:
    """A picklable ODBC read: DSN/connection string + the query (no live conn)."""

    dsn: str | None = field(repr=False)
    connection_string: str | None = field(repr=False)
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
        """The result's column types, from its first batch rather than the whole result.

        `fetchallarrow` pulls every row across the wire to read column names. That is normally
        hidden because `ODBCSource.schema` asks a ``WHERE 1 = 0`` probe, but the fallback for a
        driver whose probe comes back untyped runs the *real* query — and on the enterprise
        systems ODBC exists to reach, materializing a whole relation for its schema is exactly
        the OOM the probe was added to avoid.

        `closing` matters here: this abandons the generator after its first batch, and without
        it the ``finally`` that closes the connection would only run whenever the collector got
        around to the orphaned frame.
        """
        with closing(self.iter_batches()) as batches:
            for batch in batches:
                return batch.schema
        # An empty result yields no batch to read a schema off; only then pay for the fetch.
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the result set, rather than materializing it and then chunking it.

        This was ``yield from self.read(...)``, so the "streaming" entry point called
        `fetchallarrow` and pulled the entire result into memory before yielding its first
        batch — silently defeating every caller that chose `iter_batches` precisely to keep
        memory bounded.

        `fetcharrowbatches` arrived in turbodbc after `fetchallarrow`; a build without it falls
        back to the materializing fetch rather than failing, so this is never worse than the
        behavior it replaces.
        """
        conn = _connect(self.dsn, self.connection_string)
        try:
            cur = conn.cursor()
            cur.execute(self.query)
            stream = getattr(cur, "fetcharrowbatches", None)
            if stream is None:
                result = cur.fetchallarrow()
                chunks: Iterable[Any] = _as_batches(result)
            else:
                chunks = stream()
            for chunk in chunks:
                for batch in _as_batches(chunk):
                    yield batch.select(projection) if projection is not None else batch
        finally:
            conn.close()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"odbc:{_connection_key(self.dsn, self.connection_string)}:{self.query}"


@SOURCES.register("odbc")
@dataclass(frozen=True, slots=True)
class ODBCSource(SingleResultQuerySource):
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

    query: str
    dsn: str | None = field(default=None, repr=False)
    connection_string: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.dsn is None and self.connection_string is None:
            raise BackendError("ODBCSource requires either dsn= or connection_string=")

    def _split_for(self, sql: str) -> _ODBCSplit:
        return _ODBCSplit(self.dsn, self.connection_string, sql)

    def identity(self) -> str:
        # Fingerprinted, and password-masked before fingerprinting: the raw string went
        # straight into the persisted stats key, so an ODBC password was being written to
        # the metadata store. Masking first also keeps a rotation from orphaning the
        # statistics, since the digest no longer depends on the credential.
        return f"odbc:{_connection_key(self.dsn, self.connection_string)}:{self.query}"
