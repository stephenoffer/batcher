"""A Flight SQL service over a Batcher `Session`: statements, bound parameters, cancellation.

`serve` starts a pyarrow Flight server that answers the Flight SQL commands a client needs to
run SQL and read Arrow back:

* ``CommandStatementQuery`` (``GetFlightInfo``, then ``DoGet`` on its ticket), for a query;
* ``CommandStatementUpdate`` (``DoPut``), for a statement run for its effect;
* ``CreatePreparedStatement`` / ``ClosePreparedStatement`` (``DoAction``), with parameters
  bound by ``DoPut`` on ``CommandPreparedStatementQuery`` and run by ``GetFlightInfo`` /
  ``DoGet``, or ``DoPut`` on ``CommandPreparedStatementUpdate``;
* ``CancelFlightInfo`` (``DoAction``), which stops a result stream at its next batch.

Parameters bind through `Session.sql`'s ``params=``: the first row of the bound batch fills
the statement's ``?`` placeholders by position. Results stream from `Dataset.iter_batches`,
one Arrow batch per Flight message, so the server never holds a whole result.

**What it does not do.** There are no transactions: ``BeginTransaction`` and any command
carrying a ``transaction_id`` are refused. Catalog-metadata commands (``GetSqlInfo``,
``GetTables``, ``GetCatalogs`` and the rest) are refused as unimplemented; query
``information_schema`` instead. An update reports a record count of -1, Flight SQL's
"unknown", because a Batcher DML statement does not count rows when it runs.

**Authentication** is one shared bearer token, checked on every call against the
``authorization: Bearer <token>`` header with a constant-time comparison. The token may be a
secret reference (``env:NAME``, ``file:PATH``), resolved once when the server starts. Use TLS
(a ``grpc+tls://`` location with a certificate) for anything beyond localhost: a bearer
token over plaintext gRPC is readable by anyone on the path.

Not yet verified against a live Flight SQL client such as the ADBC or JDBC driver; see
tests/PENDING_VERIFICATION.md.

This is the `integrations` layer.
"""

from __future__ import annotations

import hmac
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from typing_extensions import override

from batcher._internal.errors import BatcherError, PlanError, SQLUnsupportedError
from batcher._internal.optional import require
from batcher.integrations.flightsql import proto

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session import Session

flight = require(
    "pyarrow.flight",
    feature="The Flight SQL service",
    provides="pyarrow with Flight",
    extra="flightsql",
)

__all__ = ["serve"]

#: Flight SQL's ``CancelStatus`` values (``Flight.proto``).
_CANCELLED, _NOT_CANCELLABLE = 1, 3


@dataclass
class _Prepared:
    """A prepared statement: its text and the parameter row bound to it, if any."""

    query: str
    params: list[Any] | None = None


@dataclass
class _Run:
    """One ticket the server issued: what it reads and the flag that cancels it."""

    query: str
    params: list[Any] | None
    dataset: Dataset | None = None
    started: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)


class _BearerAuthFactory(flight.ServerMiddlewareFactory):
    """Refuse every call whose ``authorization`` header is not ``Bearer <token>``."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._expected = f"Bearer {token}".encode()

    @override
    def start_call(self, info: Any, headers: dict[str, list[Any]]) -> None:
        values = headers.get("authorization") or headers.get("Authorization") or []
        for value in values:
            given = value.encode() if isinstance(value, str) else bytes(value)
            if hmac.compare_digest(given, self._expected):
                return None
        raise flight.FlightUnauthenticatedError("a valid bearer token is required")


def _flight_error(exc: Exception) -> Exception:
    """The Flight error a client should see for an engine failure, carrying its message."""
    if isinstance(exc, SQLUnsupportedError | NotImplementedError):
        return pa.ArrowNotImplementedError(str(exc))
    if isinstance(exc, PlanError):
        return pa.ArrowInvalid(str(exc))
    if isinstance(exc, BatcherError):
        return flight.FlightServerError(str(exc))
    return flight.FlightInternalError(f"{type(exc).__name__}: {exc}")


def _refuse_transaction(fields: dict[int, Any]) -> None:
    if fields.get(2):
        raise pa.ArrowNotImplementedError(
            "Batcher has no transactions; send the command without a transaction_id"
        )


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else ""


class _Server(flight.FlightServerBase):
    """The Flight SQL service; built by `serve`."""

    def __init__(self, session: Session, location: str, token: str | None, **kwargs: Any):
        middleware = {"auth": _BearerAuthFactory(token)} if token else None
        super().__init__(location, middleware=middleware, **kwargs)
        self._session = session
        # A `Session` is not locked; Flight calls arrive on many threads. Building a plan is
        # serialized here, and the batches stream outside the lock.
        self._session_lock = threading.Lock()
        self._lock = threading.Lock()
        self._runs: dict[bytes, _Run] = {}
        self._prepared: dict[bytes, _Prepared] = {}

    # --- statements -------------------------------------------------------------
    def _sql(self, query: str, params: list[Any] | None) -> Dataset:
        try:
            with self._session_lock:
                return self._session.sql(query, params=params or None)
        except Exception as exc:
            raise _flight_error(exc) from exc

    @override
    def get_flight_info(self, context: Any, descriptor: Any) -> Any:
        name, fields = self._command(descriptor)
        if name == "CommandStatementQuery":
            _refuse_transaction(fields)
            run = _Run(_text(fields.get(1)), None)
        elif name == "CommandPreparedStatementQuery":
            prepared = self._prepared_for(fields.get(1))
            run = _Run(prepared.query, prepared.params)
        else:
            raise pa.ArrowNotImplementedError(f"Flight SQL command {name} is not supported")
        run.dataset = self._sql(run.query, run.params)
        ticket = proto.pack_any("TicketStatementQuery", {1: uuid.uuid4().bytes})
        with self._lock:
            self._runs[ticket] = run
        endpoint = flight.FlightEndpoint(ticket, [])
        return flight.FlightInfo(run.dataset.schema, descriptor, [endpoint], -1, -1)

    @override
    def do_get(self, context: Any, ticket: Any) -> Any:
        key = ticket.ticket
        with self._lock:
            run = self._runs.get(key)
            if run is None or run.dataset is None or run.started:
                raise pa.ArrowInvalid("unknown or already-read ticket; call GetFlightInfo first")
            run.started = True
        if run.cancelled.is_set():
            self._forget(key)
            raise flight.FlightCancelledError("the statement was cancelled")
        schema = run.dataset.schema
        return flight.GeneratorStream(schema, self._stream(key, run, schema))

    def _forget(self, key: bytes) -> None:
        with self._lock:
            self._runs.pop(key, None)

    def _stream(self, key: bytes, run: _Run, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
        """Yield the run's batches, checking its cancel flag before each one.

        The run stays registered while it streams, so `CancelFlightInfo` can still find
        it, and is forgotten however the stream ends.
        """
        batches = iter(run.dataset.iter_batches()) if run.dataset is not None else iter(())
        try:
            for batch in batches:
                if run.cancelled.is_set():
                    raise flight.FlightCancelledError("the statement was cancelled")
                if batch.schema.equals(schema):
                    yield batch
                else:
                    yield from pa.Table.from_batches([batch]).cast(schema).to_batches()
        except flight.FlightCancelledError:
            raise
        except Exception as exc:
            raise _flight_error(exc) from exc
        finally:
            self._forget(key)
            close = getattr(batches, "close", None)
            if close is not None:
                close()

    @override
    def do_put(self, context: Any, descriptor: Any, reader: Any, writer: Any) -> None:
        name, fields = self._command(descriptor)
        if name == "CommandStatementUpdate":
            _refuse_transaction(fields)
            self._sql(_text(fields.get(1)), None)
            writer.write(pa.py_buffer(proto.encode({1: -1})))
        elif name == "CommandPreparedStatementQuery":
            prepared = self._prepared_for(fields.get(1))
            prepared.params = self._first_row(reader)
            writer.write(pa.py_buffer(proto.encode({1: fields.get(1)})))
        elif name == "CommandPreparedStatementUpdate":
            prepared = self._prepared_for(fields.get(1))
            rows = self._rows(reader) or [None]
            for row in rows:
                self._sql(prepared.query, row)
            writer.write(pa.py_buffer(proto.encode({1: -1})))
        else:
            raise pa.ArrowNotImplementedError(f"Flight SQL command {name} is not supported")

    # --- actions ------------------------------------------------------------------
    @override
    def list_actions(self, context: Any) -> list[tuple[str, str]]:
        return [
            ("CreatePreparedStatement", "Prepare a statement with ? placeholders."),
            ("ClosePreparedStatement", "Release a prepared statement."),
            ("CancelFlightInfo", "Stop a result stream at its next batch."),
        ]

    @override
    def do_action(self, context: Any, action: Any) -> Iterator[Any]:
        body = action.body.to_pybytes() if action.body is not None else b""
        if action.type == "CreatePreparedStatement":
            _, fields = proto.unpack_any(body)
            _refuse_transaction(fields)
            handle = uuid.uuid4().bytes
            with self._lock:
                self._prepared[handle] = _Prepared(_text(fields.get(1)))
            result = proto.pack_any("ActionCreatePreparedStatementResult", {1: handle})
            return iter([flight.Result(pa.py_buffer(result))])
        if action.type == "ClosePreparedStatement":
            _, fields = proto.unpack_any(body)
            with self._lock:
                self._prepared.pop(bytes(fields.get(1) or b""), None)
            return iter([])
        if action.type == "CancelFlightInfo":
            info = flight.FlightInfo.deserialize(proto.decode(body).get(1, b""))
            status = self._cancel(info)
            return iter([flight.Result(pa.py_buffer(proto.encode({1: status})))])
        if action.type in ("BeginTransaction", "EndTransaction", "BeginSavepoint"):
            raise pa.ArrowNotImplementedError("Batcher has no transactions")
        raise pa.ArrowNotImplementedError(f"action {action.type!r} is not supported")

    def _cancel(self, info: Any) -> int:
        cancelled = False
        with self._lock:
            for endpoint in info.endpoints:
                run = self._runs.get(endpoint.ticket.ticket)
                if run is not None:
                    run.cancelled.set()
                    cancelled = True
        return _CANCELLED if cancelled else _NOT_CANCELLABLE

    # --- helpers -------------------------------------------------------------------
    @staticmethod
    def _command(descriptor: Any) -> tuple[str, dict[int, Any]]:
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Flight SQL expects a command descriptor, not a path")
        try:
            return proto.unpack_any(descriptor.command)
        except ValueError as exc:
            raise pa.ArrowInvalid(str(exc)) from exc

    def _prepared_for(self, handle: Any) -> _Prepared:
        with self._lock:
            prepared = self._prepared.get(bytes(handle or b""))
        if prepared is None:
            raise pa.ArrowInvalid("unknown prepared statement handle")
        return prepared

    @staticmethod
    def _rows(reader: Any) -> list[list[Any]]:
        table = reader.read_all()
        columns = [column.to_pylist() for column in table.columns]
        return [list(row) for row in zip(*columns, strict=True)]

    def _first_row(self, reader: Any) -> list[Any] | None:
        rows = self._rows(reader)
        if len(rows) > 1:
            raise pa.ArrowInvalid(
                f"a prepared query binds one parameter row; {len(rows)} were sent"
            )
        return rows[0] if rows else None


def serve(
    session: Session | None = None,
    location: str = "grpc://127.0.0.1:0",
    *,
    auth: str | None = None,
    **kwargs: Any,
) -> Any:
    """Start a Flight SQL service for `session` at `location`, and return the running server.

    The server is listening when this returns. Its `port` attribute is the bound port (use
    port 0 to let the OS pick one), `serve()` on it blocks until it stops, and `shutdown()`
    stops it. Every client runs against the one `session`, so tables registered on it are
    visible to every client. Not yet verified against a live Flight SQL client; see
    tests/PENDING_VERIFICATION.md.

    Args:
        session: The session to serve. None serves `bt.current_session()`.
        location: The gRPC URI to listen on, such as ``"grpc://0.0.0.0:31337"`` or a
            ``grpc+tls://`` URI together with ``tls_certificates=`` in `kwargs`.
        auth: A bearer token every call must present, or a secret reference to one
            (``env:NAME``, ``file:PATH``). None accepts unauthenticated calls, which is only
            appropriate on a trusted interface.
        **kwargs: Passed to `pyarrow.flight.FlightServerBase`, such as
            ``tls_certificates``.

    Returns:
        The running `pyarrow.flight.FlightServerBase`.

    Raises:
        PlanError: `session` is not a `bt.Session`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.integrations import flightsql
            >>> s = bt.Session()
            >>> _ = s.register("t", bt.from_pydict({"v": [1, 2, 3]}))
            >>> server = flightsql.serve(s, auth="s3cret")
            >>> server.port > 0
            True
            >>> server.shutdown()
    """
    from batcher.api.session.sql import current_session
    from batcher.api.sql_session import Session
    from batcher.io.credentials import resolve_secret

    if session is None:
        session = current_session()
    if not isinstance(session, Session):
        raise PlanError(f"serve() expects a bt.Session, got {type(session).__name__}")
    token = resolve_secret(auth, what="Flight SQL bearer token")
    return _Server(session, location, token, **kwargs)
