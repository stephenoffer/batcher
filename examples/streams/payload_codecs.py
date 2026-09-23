"""Payload codecs: turning broker message bytes into typed columns, and back.

A broker source named with `value_format=` runs one of these codecs over each micro-batch's
`value` column. The codec is usable on its own, which is what this script does, so it needs no
broker: encode a column of records, decode the bytes back, and see what a malformed message
costs under each decode mode.

    python examples/streams/payload_codecs.py
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher.io.formats.streaming.codecs import frame_confluent, resolve_codec


def json_round_trip() -> None:
    codec = resolve_codec("json", schema={"user": "string", "amount": "int64"})
    records = pa.array(
        [{"user": "u1", "amount": 10}, None, {"user": "u3", "amount": None}],
        type=codec.arrow_type(),
    )
    payloads = codec.encode(records)
    print(payloads.to_pylist())
    assert payloads[0].as_py() == b'{"user":"u1","amount":10}'
    assert payloads[1].as_py() is None  # a null record stays a null message (a tombstone)
    assert codec.decode(payloads).equals(records)


def json_permissive_nulls_one_row() -> None:
    payloads = pa.array(
        [
            b'{"user":"u1","amount":10}',
            b"{not json",  # malformed
            b'{"user":"u2"}{"user":"u3"}',  # two documents in one message: also malformed
            b"   ",  # blank: a null, not a skipped line
            b'{"user":"u4","amount":4}',
        ]
    )
    strict = resolve_codec("json", schema={"user": "string", "amount": "int64"})
    try:
        strict.decode(payloads)
    except bt.BackendError as exc:
        print("fail mode:", str(exc).split(":")[0])
        assert "row 1 of the batch" in str(exc)
    else:
        raise AssertionError("fail mode must raise on the malformed message")

    lenient = resolve_codec("json", schema={"user": "string", "amount": "int64"}, mode="permissive")
    decoded = lenient.decode(payloads)
    print(decoded.to_pylist())
    # One row per message, always: only the bad messages' own rows are null.
    assert decoded.to_pylist() == [
        {"user": "u1", "amount": 10},
        None,
        None,
        None,
        {"user": "u4", "amount": 4},
    ]


def avro_round_trip() -> None:
    schema = {
        "type": "record",
        "name": "Order",
        "fields": [
            {"name": "id", "type": "long"},
            {"name": "tag", "type": ["int", "string"]},
            {"name": "attrs", "type": {"type": "map", "values": "long"}},
        ],
    }
    codec = resolve_codec("avro", schema=schema)
    payloads = codec.encode(
        pa.array(
            [
                {"id": 1, "tag": {"member0": 7, "member1": None}, "attrs": [("a", 1)]},
                {"id": 2, "tag": {"member0": None, "member1": "vip"}, "attrs": []},
            ],
            type=codec.arrow_type(),
        )
    )
    decoded = codec.decode(payloads)
    print(decoded.to_pylist())
    # A union decodes to a struct with one `memberN` per branch, and a map to key/value
    # pairs; encoding takes the same shapes back, so a record read can be written again.
    assert codec.decode(codec.encode(decoded)).equals(decoded)
    assert decoded.to_pylist()[1]["tag"] == {"member0": None, "member1": "vip"}

    # A Confluent-framed payload read without a registry is refused, not misread: the
    # framing would otherwise decode as field data and leave the real record unread.
    framed = frame_confluent(42, payloads[0].as_py())
    lenient = resolve_codec("avro", schema=schema, mode="permissive")
    assert lenient.decode(pa.array([framed, payloads[1].as_py()])).to_pylist()[0] is None


def main() -> None:
    json_round_trip()
    json_permissive_nulls_one_row()
    avro_round_trip()
    print("codecs ok")


if __name__ == "__main__":
    main()
