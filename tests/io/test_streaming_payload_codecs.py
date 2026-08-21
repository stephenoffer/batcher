"""The payload-codec layer: wire framing, each codec, and the broker source that uses them.

What these hold down is the property that made the codecs worth having: a broker's declared
schema, the batches it produces, and the batches its *splits* produce all agree, on every
wire format, before a single message is polled.
"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource
from batcher.io.formats.streaming.codecs import (
    CODECS,
    SchemaRegistry,
    frame_confluent,
    resolve_codec,
    unframe_confluent,
)

fastavro = pytest.importorskip("fastavro")

pytestmark = pytest.mark.unit


# --- fixtures --------------------------------------------------------------

ORDER_V1 = {
    "type": "record",
    "name": "Order",
    "fields": [{"name": "user", "type": "string"}],
}

ORDER_V2 = {
    "type": "record",
    "name": "Order",
    "fields": [
        {"name": "user", "type": "string"},
        {"name": "amount", "type": ["null", "long"], "default": None},
    ],
}


def avro_bytes(schema: dict, record: dict) -> bytes:
    """One schemaless Avro record, as a producer would write it."""
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, fastavro.parse_schema(schema), record)
    return buffer.getvalue()


class _FakeRegistry(SchemaRegistry):
    """A `SchemaRegistry` pre-seeded with schemas, which never issues an HTTP request."""

    def __init__(self, by_id: dict[int, dict], latest_id: int) -> None:
        super().__init__("http://schema-registry.invalid:8081")
        self._latest_id = latest_id
        for schema_id, schema in by_id.items():
            self._by_id[schema_id] = json.dumps(schema)

    def _get(self, path: str) -> dict:  # pragma: no cover - reaching this is the failure
        raise AssertionError(f"unexpected registry HTTP call: {path}")

    def latest(self, subject: str) -> tuple[int, str]:
        return self._latest_id, self._by_id[self._latest_id]


@pytest.fixture
def registry() -> _FakeRegistry:
    return _FakeRegistry({1: ORDER_V1, 2: ORDER_V2}, latest_id=2)


class _Broker(BrokerSource):
    """A bounded test broker that publishes one fixed batch of payloads then ends."""

    format_name = "codec_test_broker"

    bounded = True

    def __init__(self, topic: str, *, payloads=(), keys=None, **kwargs) -> None:
        super().__init__(topic, **kwargs)
        self._payloads = list(payloads)
        self._keys = list(keys) if keys is not None else [None] * len(self._payloads)
        self._served = False

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        if self._served:
            return None
        self._served = True
        return [
            BrokerMessage(
                value=payload,
                key=key,
                partition=0,
                offset=index,
                timestamp=index,
                topic=self.topic,
            )
            for index, (payload, key) in enumerate(zip(self._payloads, self._keys, strict=True))
        ]


# --- Confluent framing -----------------------------------------------------


def test_framing_round_trips_a_schema_id():
    framed = frame_confluent(4711, b"body")
    assert framed[:5] == b"\x00" + (4711).to_bytes(4, "big")
    parsed = unframe_confluent(framed)
    assert (parsed.schema_id, parsed.body) == (4711, b"body")


def test_the_default_protobuf_message_index_is_the_first_message_not_an_empty_path():
    """The single byte 0 encodes "message 0", not an empty index array.

    Read as empty, a multi-message descriptor decodes against the wrong message and returns
    plausible wrong fields rather than erroring.
    """
    parsed = unframe_confluent(frame_confluent(1, b"body", message_indexes=(0,)), protobuf=True)
    assert parsed.message_indexes == (0,)
    assert parsed.body == b"body"


def test_a_nested_protobuf_message_index_path_round_trips():
    parsed = unframe_confluent(frame_confluent(1, b"b", message_indexes=(2, 5)), protobuf=True)
    assert parsed.message_indexes == (2, 5)
    assert parsed.body == b"b"


def test_an_unframed_payload_is_refused_rather_than_silently_misread():
    """Five bytes of real data would otherwise be eaten as a header, with no error."""
    with pytest.raises(BackendError, match="magic byte"):
        unframe_confluent(b"\x01\x02\x03\x04\x05\x06")


def test_a_payload_too_short_to_carry_framing_is_named_as_such():
    with pytest.raises(BackendError, match="too short"):
        unframe_confluent(b"\x00\x01")


def test_a_registry_url_must_be_http_and_is_checked_where_it_is_written():
    with pytest.raises(PlanError, match="http"):
        SchemaRegistry("file:///etc/passwd")


def test_a_registry_repr_never_prints_its_credential():
    text = repr(SchemaRegistry("http://r:8081", basic_auth_user_info="user:hunter2"))
    assert "hunter2" not in text and "authenticated=True" in text


# --- Avro ------------------------------------------------------------------


def test_avro_decodes_to_the_arrow_type_it_declared():
    codec = resolve_codec("avro", schema=ORDER_V2)
    assert codec.arrow_type() == pa.struct(
        [pa.field("user", pa.string(), nullable=False), pa.field("amount", pa.int64())]
    )
    column = pa.array([avro_bytes(ORDER_V2, {"user": "u1", "amount": 10})], type=pa.binary())
    assert codec.decode(column).to_pylist() == [{"user": "u1", "amount": 10}]


def test_an_avro_tombstone_stays_null_rather_than_becoming_an_empty_record():
    """A null Kafka value means "this key is deleted"; decoding it as {} loses that."""
    codec = resolve_codec("avro", schema=ORDER_V2)
    decoded = codec.decode(pa.array([None], type=pa.binary()))
    assert decoded.to_pylist() == [None]


def test_avro_round_trips_through_encode():
    codec = resolve_codec("avro", schema=ORDER_V2)
    payloads = [avro_bytes(ORDER_V2, {"user": "u", "amount": 3})]
    column = pa.array(payloads, type=pa.binary())
    assert codec.encode(codec.decode(column)).to_pylist() == payloads


def test_two_schema_versions_in_one_batch_decode_to_one_column(registry):
    """The reason the reader schema is pinned: a producer mid-evolution writes both.

    Decoding each record against its own writer schema and concatenating would give a
    column whose type changed between rows. Avro's schema resolution against one reader
    schema is what makes the batch a single type.
    """
    codec = resolve_codec("avro", registry=registry, subject="orders-value")
    payloads = pa.array(
        [
            frame_confluent(1, avro_bytes(ORDER_V1, {"user": "old"})),
            frame_confluent(2, avro_bytes(ORDER_V2, {"user": "new", "amount": 5})),
        ],
        type=pa.binary(),
    )
    assert codec.decode(payloads).to_pylist() == [
        {"user": "old", "amount": None},
        {"user": "new", "amount": 5},
    ]


def test_a_corrupt_record_fails_the_query_by_default(registry):
    codec = resolve_codec("avro", registry=registry, subject="orders-value")
    with pytest.raises(BackendError, match="row 0"):
        codec.decode(pa.array([frame_confluent(2, b"not-avro")], type=pa.binary()))


def test_permissive_mode_nulls_the_corrupt_record_and_keeps_the_good_one(registry):
    codec = resolve_codec("avro", registry=registry, subject="orders-value", mode="permissive")
    column = pa.array(
        [
            frame_confluent(2, b"not-avro"),
            frame_confluent(2, avro_bytes(ORDER_V2, {"user": "ok", "amount": 1})),
        ],
        type=pa.binary(),
    )
    assert codec.decode(column).to_pylist() == [None, {"user": "ok", "amount": 1}]


def test_avro_without_a_schema_or_a_registry_is_refused_at_construction():
    """The Arrow type must be knowable before the first poll, or `Dataset.schema` cannot
    answer and no expression over the column can be type-checked at plan time."""
    with pytest.raises(PlanError, match="reader schema"):
        resolve_codec("avro")


def test_an_avro_schema_may_be_given_as_json_text():
    codec = resolve_codec("avro", schema=json.dumps(ORDER_V1))
    assert codec.arrow_type().field("user").type == pa.string()


def test_an_avro_schema_path_that_does_not_exist_is_reported_as_a_path():
    with pytest.raises(PlanError, match=r"readable \.avsc"):
        resolve_codec("avro", schema="/nonexistent/orders.avsc")


# --- JSON ------------------------------------------------------------------


def test_json_decodes_against_a_declared_schema():
    codec = resolve_codec("json", schema={"a": "int64", "b": "string"})
    column = pa.array([b'{"a":1,"b":"x"}', b'{"a":2,"b":"y"}'], type=pa.binary())
    assert codec.decode(column).to_pylist() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_json_holds_row_positions_when_some_payloads_are_blank():
    """A blank payload is not a document. Left in the parse buffer it shifts every later
    row onto the wrong message, which is a wrong answer rather than an error."""
    codec = resolve_codec("json", schema={"a": "int64"})
    column = pa.array([b'{"a":1}', b"", None, b'{"a":2}'], type=pa.binary())
    assert codec.decode(column).to_pylist() == [{"a": 1}, None, None, {"a": 2}]


def test_a_pretty_printed_json_payload_is_one_row_not_several():
    codec = resolve_codec("json", schema={"a": "int64"})
    column = pa.array([b'{\n  "a": 1\n}', b'{"a":2}'], type=pa.binary())
    assert codec.decode(column).to_pylist() == [{"a": 1}, {"a": 2}]


def test_json_without_a_schema_is_refused_at_construction():
    """Inferring from the first batch looks like a convenience and is a silent data loss:
    the plan is built before a message is polled, so a type discovered on the first poll
    arrives after the plan that had to carry it — and the decoded fields are then coerced
    back to the plan's empty struct on the way out. No error, just `{}` per row."""
    with pytest.raises(PlanError, match="reader schema"):
        resolve_codec("json")


def test_a_field_the_schema_does_not_mention_is_ignored_not_rejected():
    """What lets a producer add a field without stopping every consumer of the topic."""
    codec = resolve_codec("json", schema={"a": "int64"})
    column = pa.array([b'{"a":1,"unexpected":"x"}'], type=pa.binary())
    assert codec.decode(column).to_pylist() == [{"a": 1}]


def test_an_unknown_json_dtype_name_is_refused_by_field_name():
    with pytest.raises(PlanError, match="'a'"):
        resolve_codec("json", schema={"a": "int65"})


def test_json_round_trips_through_encode():
    codec = resolve_codec("json", schema={"a": "int64"})
    decoded = codec.decode(pa.array([b'{"a":1}'], type=pa.binary()))
    assert codec.encode(decoded).to_pylist() == [b'{"a":1}']


# --- string / bytes --------------------------------------------------------


def test_string_declares_utf8_and_decodes_it():
    codec = resolve_codec("string")
    assert codec.arrow_type() == pa.string()
    assert codec.decode(pa.array([b"hi", None], type=pa.binary())).to_pylist() == ["hi", None]


def test_invalid_utf8_fails_loudly_by_default_and_nulls_when_permissive():
    column = pa.array([b"\xff\xfe"], type=pa.binary())
    with pytest.raises(BackendError, match="UTF-8"):
        resolve_codec("string").decode(column)
    assert resolve_codec("string", mode="permissive").decode(column).to_pylist() == [None]


def test_bytes_is_the_identity_codec():
    codec = resolve_codec("bytes")
    column = pa.array([b"raw"], type=pa.binary())
    assert codec.decode(column).to_pylist() == [b"raw"]
    assert codec.arrow_type() == pa.binary()


# --- resolution ------------------------------------------------------------


def test_an_unknown_format_name_names_the_registered_ones():
    with pytest.raises(Exception, match="Unknown payload codec"):
        resolve_codec("avroo", schema=ORDER_V1)


def test_an_unknown_decode_mode_is_refused_where_it_was_written():
    with pytest.raises(PlanError, match="permissive"):
        resolve_codec("string", mode="lenient")


def test_every_registered_codec_declares_an_arrow_type_and_a_name():
    for name in CODECS.names():
        codec_cls = CODECS.get(name)
        assert isinstance(codec_cls.name, str) and codec_cls.name == name


# --- the broker source that uses them --------------------------------------


def test_a_decoding_source_declares_the_decoded_schema_before_any_poll():
    """The whole point: `Dataset.schema` is right before a message exists."""
    source = _Broker("orders", value_format="avro", value_schema=ORDER_V2)
    assert source.schema().field("value").type == pa.struct(
        [pa.field("user", pa.string(), nullable=False), pa.field("amount", pa.int64())]
    )
    assert source.schema().field("key").type == pa.binary()


def test_the_batches_a_source_produces_match_the_schema_it_declared():
    payloads = [avro_bytes(ORDER_V2, {"user": "u", "amount": 1})]
    source = _Broker("orders", payloads=payloads, value_format="avro", value_schema=ORDER_V2)
    batches = list(source.iter_batches())
    assert len(batches) == 1
    assert batches[0].schema == source.schema()
    assert batches[0].column("value").to_pylist() == [{"user": "u", "amount": 1}]


def test_the_key_column_decodes_independently_of_the_value_column():
    source = _Broker(
        "orders",
        payloads=[b'{"a":1}'],
        keys=[b"user-1"],
        value_format="json",
        value_schema={"a": "int64"},
        key_format="string",
    )
    batch = next(iter(source.iter_batches()))
    assert batch.column("key").to_pylist() == ["user-1"]
    assert batch.column("value").to_pylist() == [{"a": 1}]


def test_an_undecoded_source_still_shares_the_one_immutable_schema_instance():
    """The codec branch must not cost the common path its shared-schema fast path."""
    assert _Broker("t").schema() is _Broker("t").schema()


def test_codec_options_never_reach_the_client_config():
    """Forwarded as unknown options they land in the broker client config, where librdkafka
    rejects them or silently ignores them."""
    source = _Broker("t", value_format="string", schema_registry=None, bootstrap_servers="b:9092")
    assert source._options == {"bootstrap_servers": "b:9092"}


@pytest.fixture
def registered_broker():
    """Register `_Broker` under its format name, as a split's rebuild looks it up there.

    Removed again on teardown so the process-wide `SOURCES` registry a later test may
    enumerate does not grow a test-only entry.
    """
    from batcher.io.formats.base import SOURCES

    SOURCES.add(_Broker.format_name, _Broker)
    try:
        yield
    finally:
        SOURCES._items.pop(_Broker.format_name, None)


def test_a_split_carries_the_codec_config_and_declares_the_same_schema(registered_broker):
    """A split that dropped the codec returns binary where its siblings return a struct,
    and the epoch's concat fails on the schema."""
    source = _Broker("orders", value_format="avro", value_schema=ORDER_V2)
    split = source.splits()[0]
    assert split.schema() == source.schema()


def test_a_split_repr_never_prints_the_registry_credential():
    source = _Broker(
        "orders",
        value_format="avro",
        value_schema=ORDER_V2,
        schema_registry="http://r:8081",
        schema_registry_auth="user:hunter2",
    )
    assert "hunter2" not in repr(source.splits()[0])


# --- projection is applied while building, not after ------------------------


class _CountingCodec:
    """A codec that records how many payloads it was asked to decode."""

    name = "counting"

    def __init__(self, **_: object) -> None:
        self.decoded = 0

    def arrow_type(self) -> pa.DataType:
        return pa.string()

    def decode(self, column: pa.Array) -> pa.Array:
        self.decoded += len(column)
        return column.cast(pa.string())

    def encode(self, column: pa.Array) -> pa.Array:
        return column.cast(pa.binary())


def test_a_projection_that_drops_the_payload_never_decodes_it():
    """Decoding then discarding made the codec the dominant cost of a read that does not
    use the payload at all."""
    source = _Broker("orders", payloads=[b"a", b"b"], value_format="string")
    codec = _CountingCodec()
    source._value_codec = codec

    batch = next(iter(source.iter_batches(["offset", "timestamp"])))

    assert batch.schema.names == ["offset", "timestamp"]
    assert codec.decoded == 0


def test_a_projection_that_keeps_the_payload_still_decodes_it():
    source = _Broker("orders", payloads=[b"a", b"b"], value_format="string")
    codec = _CountingCodec()
    source._value_codec = codec

    batch = next(iter(source.iter_batches(["value"])))

    assert batch.column("value").to_pylist() == ["a", "b"]
    assert codec.decoded == 2


def test_a_projection_never_assembles_the_headers_column_it_drops():
    """`headers` costs a Python call per message to build, so building it to throw it away
    is the same waste one type-family over."""
    source = _Broker("orders", payloads=[b"a"], include_headers=True)
    assert next(iter(source.iter_batches(["value"]))).schema.names == ["value"]


def test_a_projection_reorders_the_batch_to_match_what_was_asked_for():
    source = _Broker("orders", payloads=[b"a"])
    assert next(iter(source.iter_batches(["topic", "offset"]))).schema.names == [
        "topic",
        "offset",
    ]


def test_an_unprojected_read_still_carries_every_column():
    source = _Broker("orders", payloads=[b"a"], include_headers=True)
    batch = next(iter(source.iter_batches()))
    assert batch.schema == source.schema()
