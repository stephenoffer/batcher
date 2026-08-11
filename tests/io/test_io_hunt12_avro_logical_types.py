"""Avro logical-type read correctness (hunt 12).

`AvroSource.schema()` mapped Avro logical types (date/time/timestamp/decimal) to their
underlying int/long instead of the Arrow logical type the native `arrow-avro` reader
decodes. Two consequences, both pinned here:

* the advertised ``schema()`` disagreed with the batches ``read()`` actually returns
  (``ts: int64`` advertised, ``timestamp[ms]`` decoded) — a lying schema; and
* the row-by-row `fastavro` fallback *crashed*, because it could not coerce the
  ``datetime``/``date``/``Decimal`` values fastavro yields into an int column.

These run only when `fastavro` is installed (the Avro extra).
"""

from __future__ import annotations

import datetime
import decimal

import pyarrow as pa
import pytest


def _logical_avro(path: str) -> None:
    """Write an Avro file exercising every scalar logical type + a nullable union."""
    fastavro = pytest.importorskip("fastavro")
    avro_schema = {
        "type": "record",
        "name": "r",
        "fields": [
            {"name": "id", "type": "long"},
            {"name": "ts_ms", "type": {"type": "long", "logicalType": "timestamp-millis"}},
            {"name": "ts_us", "type": {"type": "long", "logicalType": "timestamp-micros"}},
            {"name": "day", "type": {"type": "int", "logicalType": "date"}},
            {"name": "tm", "type": {"type": "int", "logicalType": "time-millis"}},
            {
                "name": "amt",
                "type": {"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 2},
            },
            {"name": "maybe", "type": ["null", "long"]},
        ],
    }
    records = [
        {
            "id": 1,
            "ts_ms": datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
            "ts_us": datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
            "day": datetime.date(2020, 1, 1),
            "tm": datetime.time(1, 2, 3),
            "amt": decimal.Decimal("3.14"),
            "maybe": 5,
        },
        {
            "id": 2,
            "ts_ms": datetime.datetime(2021, 6, 15, 12, tzinfo=datetime.UTC),
            "ts_us": datetime.datetime(2021, 6, 15, 12, tzinfo=datetime.UTC),
            "day": datetime.date(2021, 6, 15),
            "tm": datetime.time(4, 5, 6),
            "amt": decimal.Decimal("99.99"),
            "maybe": None,
        },
    ]
    with open(path, "wb") as fh:
        fastavro.writer(fh, fastavro.parse_schema(avro_schema), records)


def test_avro_schema_matches_decoded_logical_types(tmp_path):
    """`schema()` advertises the Arrow logical types the reader actually decodes.

    Before the fix `schema()` returned ``int64``/``int32`` for the timestamp/date/time
    columns, disagreeing with the decoded batches.
    """
    pytest.importorskip("fastavro")
    from batcher.io.formats.structured.avro import AvroSource

    path = str(tmp_path / "logical.avro")
    _logical_avro(path)

    src = AvroSource(path)
    advertised = src.schema()
    decoded = pa.Table.from_batches(src.read()).schema

    # The advertised schema must equal the schema of the batches the source reads.
    assert advertised.equals(decoded), f"{advertised}\n!=\n{decoded}"
    # And it must carry the logical types, not their int/long backing.
    assert pa.types.is_timestamp(advertised.field("ts_ms").type)
    assert advertised.field("ts_ms").type.unit == "ms"
    assert advertised.field("ts_us").type.unit == "us"
    assert pa.types.is_date(advertised.field("day").type)
    assert pa.types.is_time(advertised.field("tm").type)
    assert pa.types.is_decimal(advertised.field("amt").type)


def test_avro_fastavro_fallback_decodes_logical_types(tmp_path, monkeypatch):
    """The `fastavro` fallback assembles logical-type values instead of crashing.

    Before the fix, coercing the ``datetime``/``date``/``Decimal`` values fastavro yields
    into the (wrongly) int-typed schema raised ``ArrowInvalid``. The fallback (native
    reader stubbed unavailable) must now match the native decode exactly.
    """
    pytest.importorskip("fastavro")
    from batcher.io.formats.structured import avro as avro_mod
    from batcher.io.formats.structured.avro import AvroSource

    path = str(tmp_path / "logical_fb.avro")
    _logical_avro(path)

    native = pa.Table.from_batches(AvroSource(path).read())

    monkeypatch.setattr(avro_mod, "_read_native", lambda data, batch_rows: None)
    fallback = pa.Table.from_batches(AvroSource(path).read())

    assert fallback.schema.equals(native.schema)
    assert fallback.to_pydict() == native.to_pydict()
    assert fallback.column("ts_ms")[0].as_py() == datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
    assert fallback.column("day")[0].as_py() == datetime.date(2020, 1, 1)
