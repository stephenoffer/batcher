"""Edge cases of the streaming payload codecs: row preservation, per-row permissive nulls,
registry failures, framing, and read-then-write round trips.

Each of these was reproduced against the previous implementation before it was fixed. The
two that mattered most returned a wrong answer rather than an error: a JSON batch with a
whitespace-only message, or a message carrying two documents, came back with a different row
count, so every later row sat on the wrong message; and a registry outage under
``permissive`` nulled every record after one GET per row, reporting success while
delivering nothing.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.streaming.codecs import SchemaRegistry, frame_confluent, resolve_codec
from batcher.io.formats.streaming.codecs import protobuf as protobuf_mod
from batcher.io.formats.streaming.codecs.base import build_payload_codecs
from batcher.io.formats.streaming.codecs.wire import SchemaNotFoundError

pytestmark = pytest.mark.io

fastavro = pytest.importorskip("fastavro")


class _Registry(SchemaRegistry):
    """A registry answering from a dict, counting requests; `down` makes every GET fail."""

    def __init__(self, by_id: dict[int, str], latest_id: int | None = None, *, down=False):
        super().__init__("http://registry.invalid")
        self.by_id, self.latest_id, self.down, self.gets = by_id, latest_id, down, 0

    def _get(self, path: str) -> dict:
        self.gets += 1
        if self.down:
            raise BackendError("schema registry http://registry.invalid is unreachable")
        if path.startswith("/schemas/ids/"):
            schema_id = int(path.rsplit("/", 1)[1])
            if schema_id not in self.by_id:
                raise SchemaNotFoundError(f"no schema {schema_id}")
            return {"schema": self.by_id[schema_id]}
        return {"id": self.latest_id, "schema": self.by_id[self.latest_id]}


def _avro(schema: dict, record: dict) -> bytes:
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, fastavro.parse_schema(schema), record)
    return buffer.getvalue()


_JSON_SCHEMA = {"a": "int64", "s": "string"}


def _json(mode: str = "fail", **kw):
    return resolve_codec("json", schema=kw.pop("schema", _JSON_SCHEMA), mode=mode, **kw)


# --- JSON: one message is one row ----------------------------------------------------


@pytest.mark.parametrize(
    "middle", [b"   ", b"\t\r\n", b""], ids=["spaces", "whitespace-mix", "empty"]
)
def test_a_blank_json_message_is_a_null_in_its_own_row(middle):
    out = _json().decode(pa.array([b'{"a":1}', middle, b'{"a":3}']))
    assert out.to_pylist() == [{"a": 1, "s": None}, None, {"a": 3, "s": None}]


@pytest.mark.parametrize(
    "double", [b'{"a":1}{"a":2}', b'{"a":1}\n{"a":2}', b'{"a":1} {"a":2}'], ids=repr
)
def test_a_message_with_two_documents_is_malformed_not_two_rows(double):
    column = pa.array([double, b'{"a":3}'])
    with pytest.raises(BackendError, match=r"row 0 .* holds 2 JSON documents"):
        _json().decode(column)
    out = _json("permissive").decode(column)
    assert out.to_pylist() == [None, {"a": 3, "s": None}]


def test_permissive_json_nulls_only_the_bad_row():
    column = pa.array([b'{"a":1}', b"{bad", b'{"a":"x"}', b"5", None, b'{"a":6,"s":"k"}'])
    out = _json("permissive").decode(column)
    assert out.to_pylist() == [{"a": 1, "s": None}, None, None, None, None, {"a": 6, "s": "k"}]


def test_fail_mode_json_names_the_bad_row():
    with pytest.raises(BackendError, match="row 2 of the batch"):
        _json().decode(pa.array([b'{"a":1}', b'{"a":2}', b"{bad"]))


def test_a_clean_json_batch_is_parsed_in_one_pass(monkeypatch):
    """The per-message fallback is for bad batches only: a clean one never reaches it."""
    from batcher.io.formats.streaming.codecs import json as json_mod

    calls = []
    real = json_mod.JsonCodec._decode_each
    monkeypatch.setattr(json_mod.JsonCodec, "_decode_each", lambda *a: calls.append(1) or real(*a))
    codec = _json("permissive")
    codec.decode(pa.array([b'{"a":1}', None, b'{"a":2}']))
    assert calls == []
    codec.decode(pa.array([b'{"a":1}', b"{bad"]))  # the control: a bad batch does reach it
    assert calls == [1]


def test_a_string_payload_column_decodes_like_binary():
    assert _json().decode(pa.array(['{"a":1}'])).to_pylist() == [{"a": 1, "s": None}]


def test_json_round_trips_through_encode():
    codec = _json()
    column = pa.array([{"a": 1, "s": "x"}, None, {"a": None, "s": "y"}], type=codec.arrow_type())
    assert codec.decode(codec.encode(column)).equals(column)


# --- JSON: Confluent framing and JSON Schema translation -----------------------------

_REGISTERED = json.dumps(
    {"type": "object", "properties": {"a": {"type": "integer"}, "s": {"type": "string"}}}
)


def test_json_with_a_registry_strips_and_writes_confluent_framing():
    registry = _Registry({4: _REGISTERED}, latest_id=4)
    codec = resolve_codec("json", registry=registry, subject="t-value")
    column = pa.array([frame_confluent(4, b'{"a":1,"s":"x"}'), None])
    assert codec.decode(column).to_pylist() == [{"a": 1, "s": "x"}, None]
    encoded = codec.encode(codec.decode(column))
    assert encoded[0].as_py() == frame_confluent(4, b'{"a":1,"s":"x"}')


def test_an_unframed_message_on_a_registry_json_topic_is_malformed():
    registry = _Registry({4: _REGISTERED}, latest_id=4)
    column = pa.array([b'{"a":1}', frame_confluent(4, b'{"a":2}')])
    with pytest.raises(BackendError, match=r"row 0 .*magic byte"):
        resolve_codec("json", registry=registry, subject="t-value").decode(column)
    permissive = resolve_codec("json", registry=registry, subject="t-value", mode="permissive")
    assert permissive.decode(column).to_pylist() == [None, {"a": 2, "s": None}]


@pytest.mark.parametrize(
    ("spec", "named"),
    [
        ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, "oneOf"),
        ({"anyOf": [{"type": "string"}]}, "anyOf"),
        ({"$ref": "#/definitions/x"}, r"\$ref"),
        ({"type": ["string", "integer"]}, "type"),
        ({"type": "object", "properties": {"inner": {"allOf": []}}}, "allOf"),
    ],
    ids=["oneOf", "anyOf", "ref", "type-union", "nested-allOf"],
)
def test_a_json_schema_with_no_single_arrow_type_is_refused(spec, named):
    text = json.dumps({"type": "object", "properties": {"ok": {"type": "string"}, "p": spec}})
    registry = _Registry({1: text}, latest_id=1)
    with pytest.raises(PlanError, match=rf"property 'p(\.inner)?' uses .*{named}"):
        resolve_codec("json", registry=registry, subject="t-value")


def test_a_nullable_json_schema_type_still_translates():
    """The control for the refusal above: `["null", T]` is one type, not a union."""
    text = json.dumps({"type": "object", "properties": {"p": {"type": ["null", "integer"]}}})
    codec = resolve_codec("json", registry=_Registry({1: text}, latest_id=1), subject="t")
    assert codec.arrow_type() == pa.struct([("p", pa.int64())])


# --- Avro: registry failures, framing, trailing bytes --------------------------------

_V1 = {"type": "record", "name": "R", "fields": [{"name": "a", "type": "long"}]}


@pytest.mark.parametrize("mode", ["fail", "permissive"])
def test_a_registry_outage_fails_the_batch_in_every_mode(mode):
    registry = _Registry({}, down=True)
    codec = resolve_codec("avro", schema=_V1, registry=registry, mode=mode)
    column = pa.array([frame_confluent(1, _avro(_V1, {"a": i})) for i in range(50)])
    with pytest.raises(BackendError, match="unreachable"):
        codec.decode(column)
    assert registry.gets == 1  # it failed on the first lookup, not after fifty


def test_an_unknown_schema_id_nulls_under_permissive_with_one_lookup_per_batch():
    registry = _Registry({1: json.dumps(_V1)})
    codec = resolve_codec("avro", schema=_V1, registry=registry, mode="permissive")
    good = frame_confluent(1, _avro(_V1, {"a": 7}))
    column = pa.array([frame_confluent(99, b"\x02")] * 20 + [good])
    assert codec.decode(column).to_pylist() == [None] * 20 + [{"a": 7}]
    assert registry.gets == 2  # id 99 once (then cached as unknown), id 1 once


def test_an_unknown_schema_id_fails_by_row_under_fail():
    registry = _Registry({1: json.dumps(_V1)})
    codec = resolve_codec("avro", schema=_V1, registry=registry)
    with pytest.raises(BackendError, match="row 1 of the batch"):
        codec.decode(pa.array([frame_confluent(1, _avro(_V1, {"a": 1})), frame_confluent(9, b"")]))


_V2 = {
    "type": "record",
    "name": "R",
    "fields": [{"name": "a", "type": "long"}, {"name": "b", "type": "string"}],
}


def test_a_framed_payload_read_without_a_registry_is_refused_not_misread():
    framed = frame_confluent(7, _avro(_V2, {"a": 3, "b": "hi"}))
    with pytest.raises(BackendError, match=r"remain after the record.*schema_registry="):
        resolve_codec("avro", schema=_V2).decode(pa.array([framed]))
    permissive = resolve_codec("avro", schema=_V2, mode="permissive")
    good = _avro(_V2, {"a": 1, "b": "x"})
    assert permissive.decode(pa.array([framed, good])).to_pylist() == [None, {"a": 1, "b": "x"}]


def test_trailing_bytes_after_an_avro_record_are_an_error():
    with pytest.raises(BackendError, match="3 byte"):
        resolve_codec("avro", schema=_V1).decode(pa.array([_avro(_V1, {"a": 1}) + b"xyz"]))


# --- Avro: read -> write round trip over the shapes decode produces ------------------

_RICH = {
    "type": "record",
    "name": "Rich",
    "fields": [
        {"name": "u", "type": ["int", "string"]},
        {"name": "nu", "type": ["null", "int", "string"], "default": None},
        {"name": "m", "type": {"type": "map", "values": "long"}},
        {
            "name": "nested",
            "type": {
                "type": "record",
                "name": "N",
                "fields": [{"name": "mm", "type": {"type": "map", "values": "string"}}],
            },
        },
        {"name": "arr", "type": {"type": "array", "items": {"type": "map", "values": "int"}}},
    ],
}


def test_avro_union_and_map_fields_round_trip_through_encode():
    records = [
        {
            "u": 5,
            "nu": None,
            "m": {"a": 1},
            "nested": {"mm": {"k": "v"}},
            "arr": [{"x": 1}],
        },
        {
            "u": "s",
            "nu": "t",
            "m": {},
            "nested": {"mm": {}},
            "arr": [],
        },
        {
            "u": 0,
            "nu": 3,
            "m": {"b": 2, "c": 3},
            "nested": {"mm": {}},
            "arr": [{}, {"y": 2}],
        },
    ]
    codec = resolve_codec("avro", schema=_RICH)
    decoded = codec.decode(pa.array([_avro(_RICH, r) for r in records] + [None]))
    assert decoded.to_pylist()[0]["u"] == {"member0": 5, "member1": None}  # the shape encoded
    again = codec.decode(codec.encode(decoded))
    assert again.equals(decoded)
    # And the bytes are what fastavro itself writes, so another consumer reads them too.
    assert codec.encode(decoded).to_pylist()[:3] == [_avro(_RICH, r) for r in records]


# --- options reach the codec ---------------------------------------------------------


def test_codec_options_pass_through_to_the_codec():
    value, _ = build_payload_codecs(
        "t", {"value_format": "string", "value_codec_options": {"encoding": "latin-1"}}
    )
    assert value.decode(pa.array([b"caf\xe9"])).to_pylist() == ["café"]
    # The control: without the option the same bytes are not UTF-8.
    plain, _ = build_payload_codecs("t", {"value_format": "string"})
    with pytest.raises(BackendError, match="UTF-8"):
        plain.decode(pa.array([b"caf\xe9"]))


def test_codec_options_must_be_a_dict():
    with pytest.raises(PlanError, match="value_codec_options must be a dict"):
        build_payload_codecs("t", {"value_format": "string", "value_codec_options": "latin-1"})


# --- Protobuf: the Confluent message index is checked ---------------------------------


class _Message:
    """A stand-in generated class: records the bytes it was parsed from."""

    DESCRIPTOR = object()

    def ParseFromString(self, body: bytes) -> None:
        self.body = body


@pytest.fixture
def stub_protarrow(monkeypatch):
    """`protarrow` is optional and not installed here; the index check runs before it."""
    fake = SimpleNamespace(
        message_type_to_schema=lambda cls: pa.schema([("body", pa.binary())]),
        messages_to_table=lambda msgs, cls: pa.table({"body": [m.body for m in msgs]}),
    )
    monkeypatch.setattr(protobuf_mod, "_require_protarrow", lambda: fake)


def test_a_protobuf_payload_for_another_message_index_is_refused(stub_protarrow):
    registry = _Registry({3: "syntax = 'proto3';"}, latest_id=3)
    value, _ = build_payload_codecs(
        "t",
        {
            "value_format": "protobuf",
            "value_schema": _Message,
            "schema_registry": registry,
            "value_codec_options": {"message_indexes": (1,)},
        },
    )
    ours = frame_confluent(3, b"\x08\x01", message_indexes=(1,))
    theirs = frame_confluent(3, b"\x08\x02", message_indexes=(0,))
    assert value.decode(pa.array([ours])).to_pylist() == [{"body": b"\x08\x01"}]
    with pytest.raises(BackendError, match=r"message index is \[0\].*decodes \[1\]"):
        value.decode(pa.array([ours, theirs]))
    permissive = resolve_codec(
        "protobuf", schema=_Message, registry=registry, mode="permissive", message_indexes=(1,)
    )
    assert permissive.decode(pa.array([ours, theirs])).to_pylist() == [
        {"body": b"\x08\x01"},
        None,
    ]


def test_protobuf_round_trips_with_the_real_library():
    pytest.importorskip("protarrow")
    from google.protobuf import struct_pb2

    codec = resolve_codec("protobuf", schema=struct_pb2.ListValue)
    column = codec.decode(pa.array([struct_pb2.ListValue().SerializeToString()]))
    assert codec.decode(codec.encode(column)).equals(column)
