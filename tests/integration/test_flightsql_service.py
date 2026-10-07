"""A localhost round trip through the Flight SQL service, spoken by a pyarrow Flight client.

The client side here builds Flight SQL commands with the same codec the server reads, so
these tests prove the *service* (statements, binding, streaming, cancellation, auth, the
transaction refusal) and `tests/unit/test_flightsql_proto.py` proves the codec against the
protobuf runtime. Neither is a substitute for a real Flight SQL driver, which is what
`tests/integration/live/test_live_flightsql.py` runs when one is installed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pytest

import batcher as bt
from batcher.integrations import flightsql
from batcher.integrations.flightsql import proto

flight = pytest.importorskip("pyarrow.flight")

pytestmark = pytest.mark.integration

_TOKEN = "test-token"
_AUTH = flight.FlightCallOptions(headers=[(b"authorization", f"Bearer {_TOKEN}".encode())])


@pytest.fixture
def session() -> bt.Session:
    s = bt.Session()
    s.register("orders", bt.from_pydict({"id": [1, 2, 3, 4], "amount": [5.0, 7.5, None, 2.0]}))
    return s


@pytest.fixture
def client(session: bt.Session) -> Iterator[flight.FlightClient]:
    server = flightsql.serve(session, auth=_TOKEN)
    client = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")
    try:
        yield client
    finally:
        client.close()
        server.shutdown()


def _command(message: str, fields: dict) -> flight.FlightDescriptor:
    return flight.FlightDescriptor.for_command(proto.pack_any(message, fields))


def _query(client: flight.FlightClient, sql: str) -> pa.Table:
    info = client.get_flight_info(_command("CommandStatementQuery", {1: sql}), _AUTH)
    return client.do_get(info.endpoints[0].ticket, _AUTH).read_all()


def _action(client: flight.FlightClient, kind: str, body: bytes) -> list[bytes]:
    return [r.body.to_pybytes() for r in client.do_action(flight.Action(kind, body), _AUTH)]


def test_a_statement_streams_its_result_with_the_advertised_schema(client) -> None:
    info = client.get_flight_info(
        _command("CommandStatementQuery", {1: "SELECT id, amount FROM orders ORDER BY id"}),
        _AUTH,
    )
    table = client.do_get(info.endpoints[0].ticket, _AUTH).read_all()
    assert table.schema.equals(info.schema)
    assert table.to_pydict() == {"id": [1, 2, 3, 4], "amount": [5.0, 7.5, None, 2.0]}


def test_an_empty_result_keeps_its_schema(client) -> None:
    table = _query(client, "SELECT id FROM orders WHERE id > 100")
    assert table.num_rows == 0
    assert table.schema.names == ["id"]


def test_a_ticket_is_read_once(client) -> None:
    info = client.get_flight_info(_command("CommandStatementQuery", {1: "SELECT 1 AS x"}), _AUTH)
    client.do_get(info.endpoints[0].ticket, _AUTH).read_all()
    with pytest.raises(pa.ArrowInvalid, match="unknown or already-read ticket"):
        client.do_get(info.endpoints[0].ticket, _AUTH).read_all()


def test_a_prepared_statement_binds_parameters(client) -> None:
    body = proto.pack_any(
        "ActionCreatePreparedStatementRequest", {1: "SELECT id FROM orders WHERE amount > ?"}
    )
    (result,) = _action(client, "CreatePreparedStatement", body)
    name, fields = proto.unpack_any(result)
    assert name == "ActionCreatePreparedStatementResult"
    handle = fields[1]
    descriptor = _command("CommandPreparedStatementQuery", {1: handle})

    params = pa.table({"p": [4.0]})
    writer, reader = client.do_put(descriptor, params.schema, _AUTH)
    writer.write_table(params)
    writer.done_writing()
    assert proto.decode(reader.read().to_pybytes()) == {1: handle}
    writer.close()

    info = client.get_flight_info(descriptor, _AUTH)
    table = client.do_get(info.endpoints[0].ticket, _AUTH).read_all()
    assert sorted(table.column("id").to_pylist()) == [1, 2]

    _action(
        client,
        "ClosePreparedStatement",
        proto.pack_any("ActionClosePreparedStatementRequest", {1: handle}),
    )
    with pytest.raises(pa.ArrowInvalid, match="unknown prepared statement"):
        client.get_flight_info(descriptor, _AUTH)


def test_a_bound_value_is_a_value_not_sql(client) -> None:
    """A parameter holding SQL text is compared as a string; nothing is spliced."""
    body = proto.pack_any(
        "ActionCreatePreparedStatementRequest",
        {1: "SELECT COUNT(*) AS n FROM orders WHERE CAST(id AS VARCHAR) = ?"},
    )
    handle = proto.unpack_any(_action(client, "CreatePreparedStatement", body)[0])[1][1]
    descriptor = _command("CommandPreparedStatementQuery", {1: handle})
    params = pa.table({"p": ["1' OR '1'='1"]})
    writer, reader = client.do_put(descriptor, params.schema, _AUTH)
    writer.write_table(params)
    writer.done_writing()
    reader.read()
    writer.close()
    info = client.get_flight_info(descriptor, _AUTH)
    assert client.do_get(info.endpoints[0].ticket, _AUTH).read_all().to_pydict() == {"n": [0]}


def test_an_update_runs_and_reports_an_unknown_count(client, session) -> None:
    descriptor = _command("CommandStatementUpdate", {1: "CREATE TABLE big AS SELECT 1 AS k"})
    writer, reader = client.do_put(descriptor, pa.schema([]), _AUTH)
    writer.done_writing()
    assert proto.decode(reader.read().to_pybytes()) == {1: (1 << 64) - 1}  # int64 -1
    writer.close()
    assert session.sql("SELECT k FROM big").to_pydict() == {"k": [1]}


def test_cancel_stops_a_ticket_before_it_streams(client) -> None:
    info = client.get_flight_info(_command("CommandStatementQuery", {1: "SELECT 1 AS x"}), _AUTH)
    (result,) = _action(client, "CancelFlightInfo", proto.encode({1: info.serialize()}))
    assert proto.decode(result) == {1: 1}  # CANCELLED
    with pytest.raises(flight.FlightCancelledError):
        client.do_get(info.endpoints[0].ticket, _AUTH).read_all()
    (again,) = _action(client, "CancelFlightInfo", proto.encode({1: info.serialize()}))
    assert proto.decode(again) == {1: 3}  # NOT_CANCELLABLE: already gone


def test_a_transaction_is_refused(client) -> None:
    with pytest.raises(pa.ArrowNotImplementedError, match="no transactions"):
        _action(client, "BeginTransaction", b"")
    with pytest.raises(pa.ArrowNotImplementedError, match="no transactions"):
        client.get_flight_info(
            _command("CommandStatementQuery", {1: "SELECT 1", 2: b"tx-1"}), _AUTH
        )


def test_an_engine_refusal_reaches_the_client_typed(client) -> None:
    with pytest.raises(pa.ArrowInvalid, match="no_such_table"):
        _query(client, "SELECT * FROM no_such_table")


def test_a_call_without_the_token_is_refused(client) -> None:
    with pytest.raises(flight.FlightUnauthenticatedError):
        client.get_flight_info(_command("CommandStatementQuery", {1: "SELECT 1"}))
    wrong = flight.FlightCallOptions(headers=[(b"authorization", b"Bearer nope")])
    with pytest.raises(flight.FlightUnauthenticatedError):
        client.get_flight_info(_command("CommandStatementQuery", {1: "SELECT 1"}), wrong)


def test_the_token_may_be_a_secret_reference(session, monkeypatch) -> None:
    monkeypatch.setenv("BATCHER_TEST_FLIGHT_TOKEN", _TOKEN)
    server = flightsql.serve(session, auth="env:BATCHER_TEST_FLIGHT_TOKEN")
    client = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")
    try:
        assert _query(client, "SELECT 2 AS x").to_pydict() == {"x": [2]}
    finally:
        client.close()
        server.shutdown()


def test_cancel_stops_a_stream_already_running(client) -> None:
    """A stream cancelled after its first batch ends in an error, never a short result."""
    info = client.get_flight_info(
        _command("CommandStatementQuery", {1: "SELECT i FROM range(2000000) t(i)"}), _AUTH
    )
    reader = client.do_get(info.endpoints[0].ticket, _AUTH)
    first = reader.read_chunk().data
    assert 0 < first.num_rows < 2_000_000
    (result,) = _action(client, "CancelFlightInfo", proto.encode({1: info.serialize()}))
    assert proto.decode(result) == {1: 1}
    with pytest.raises(flight.FlightCancelledError):
        while True:
            reader.read_chunk()
