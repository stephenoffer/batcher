"""The hand-written Flight SQL codec agrees with the protobuf runtime's own encoder.

`batcher.integrations.flightsql.proto` encodes the handful of Flight SQL messages the
service needs without the generated classes, which pyarrow does not ship. A wrong field
number there would round-trip through the codec perfectly and still be unreadable to a real
client, so these tests check it against `google.protobuf` itself: each message is declared
here with the field numbers from Apache Arrow's ``FlightSql.proto`` and ``Flight.proto``,
and the two encoders must read each other's bytes.
"""

from __future__ import annotations

import pytest

from batcher.integrations.flightsql import proto

pytestmark = pytest.mark.unit

descriptor_pb2 = pytest.importorskip("google.protobuf.descriptor_pb2")
message_factory = pytest.importorskip("google.protobuf.message_factory")
descriptor_pool = pytest.importorskip("google.protobuf.descriptor_pool")
any_pb2 = pytest.importorskip("google.protobuf.any_pb2")

_F = descriptor_pb2.FieldDescriptorProto
#: Message -> [(name, number, type)], copied from Arrow's FlightSql.proto / Flight.proto.
_MESSAGES = {
    "CommandStatementQuery": [("query", 1, _F.TYPE_STRING), ("transaction_id", 2, _F.TYPE_BYTES)],
    "CommandStatementUpdate": [("query", 1, _F.TYPE_STRING), ("transaction_id", 2, _F.TYPE_BYTES)],
    "TicketStatementQuery": [("statement_handle", 1, _F.TYPE_BYTES)],
    "ActionCreatePreparedStatementRequest": [
        ("query", 1, _F.TYPE_STRING),
        ("transaction_id", 2, _F.TYPE_BYTES),
    ],
    "ActionCreatePreparedStatementResult": [
        ("prepared_statement_handle", 1, _F.TYPE_BYTES),
        ("dataset_schema", 2, _F.TYPE_BYTES),
        ("parameter_schema", 3, _F.TYPE_BYTES),
    ],
    "CommandPreparedStatementQuery": [("prepared_statement_handle", 1, _F.TYPE_BYTES)],
    "DoPutUpdateResult": [("record_count", 1, _F.TYPE_INT64)],
    "CancelFlightInfoResult": [("status", 1, _F.TYPE_INT32)],
}


@pytest.fixture(scope="module")
def classes() -> dict[str, type]:
    """Real protobuf message classes for the declared messages."""
    file = descriptor_pb2.FileDescriptorProto(
        name="batcher_test_flightsql.proto", package="arrow.flight.protocol.sql", syntax="proto3"
    )
    for name, fields in _MESSAGES.items():
        message = file.message_type.add(name=name)
        for field_name, number, kind in fields:
            message.field.add(name=field_name, number=number, type=kind, label=_F.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return {
        name: message_factory.GetMessageClass(
            pool.FindMessageTypeByName(f"arrow.flight.protocol.sql.{name}")
        )
        for name in _MESSAGES
    }


def test_a_command_packed_by_protobuf_unpacks_here(classes: dict[str, type]) -> None:
    """What a client sends is what the service reads."""
    command = classes["CommandStatementQuery"](query="SELECT 1", transaction_id=b"tx")
    wrapped = any_pb2.Any()
    wrapped.Pack(command, type_url_prefix="type.googleapis.com/")
    name, fields = proto.unpack_any(wrapped.SerializeToString())
    assert name == "CommandStatementQuery"
    assert fields == {1: b"SELECT 1", 2: b"tx"}


def test_a_result_packed_here_parses_in_protobuf(classes: dict[str, type]) -> None:
    """What the service sends is what a client reads."""
    raw = proto.pack_any("ActionCreatePreparedStatementResult", {1: b"handle", 2: b"schema-bytes"})
    wrapped = any_pb2.Any.FromString(raw)
    assert wrapped.type_url == proto.SQL_PREFIX + "ActionCreatePreparedStatementResult"
    result = classes["ActionCreatePreparedStatementResult"].FromString(wrapped.value)
    assert result.prepared_statement_handle == b"handle"
    assert result.dataset_schema == b"schema-bytes"
    assert result.parameter_schema == b""


def test_an_unknown_record_count_encodes_as_minus_one(classes: dict[str, type]) -> None:
    """-1 is Flight SQL's "unknown" count, and an int64 encodes it as ten varint bytes."""
    result = classes["DoPutUpdateResult"].FromString(proto.encode({1: -1}))
    assert result.record_count == -1


def test_a_cancel_status_reads_back(classes: dict[str, type]) -> None:
    """CANCELLED is 1 in Flight.proto's CancelStatus."""
    assert classes["CancelFlightInfoResult"].FromString(proto.encode({1: 1})).status == 1


def test_a_non_flight_sql_any_is_refused() -> None:
    """An Any naming another package is not silently read as a command."""
    raw = proto.encode({1: "type.googleapis.com/other.Message", 2: b""})
    with pytest.raises(ValueError, match="not a Flight SQL command"):
        proto.unpack_any(raw)


def test_a_truncated_message_is_refused() -> None:
    """A length prefix running past the end raises rather than reading garbage."""
    with pytest.raises(ValueError, match="truncated"):
        proto.decode(b"\x0a\x05ab")
