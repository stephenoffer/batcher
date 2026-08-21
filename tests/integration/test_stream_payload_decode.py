"""A decoded broker payload end to end: schema at plan time, expressions, aggregate, sink.

The unit tests hold the codecs themselves. This holds the thing that makes them worth
having: that naming a wire format puts the payload's *type* into the plan, so an expression
over it resolves before the query runs and the optimizer sees real columns rather than an
opaque `binary`.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

fastavro = pytest.importorskip("fastavro")

pytestmark = pytest.mark.integration

ORDER = {
    "type": "record",
    "name": "Order",
    "fields": [
        {"name": "user", "type": "string"},
        {"name": "amount", "type": ["null", "long"]},
    ],
}

ROWS = [
    {"user": "u1", "amount": 10},
    {"user": "u2", "amount": 5},
    {"user": "u1", "amount": 7},
]


def _avro(record: dict) -> bytes:
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, fastavro.parse_schema(ORDER), record)
    return buffer.getvalue()


class _AvroBroker(BrokerSource):
    """A bounded broker publishing one batch of Avro-encoded orders."""

    format_name = "avro_decode_test_broker"
    bounded = True

    def __init__(self, topic: str, **kwargs: object) -> None:
        super().__init__(topic, **kwargs)
        self._served = False

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        if self._served:
            return None
        self._served = True
        return [
            BrokerMessage(value=_avro(row), partition=0, offset=i, timestamp=i, topic=self.topic)
            for i, row in enumerate(ROWS)
        ]


@pytest.fixture(autouse=True)
def registered():
    SOURCES.add(_AvroBroker.format_name, _AvroBroker)
    try:
        yield
    finally:
        SOURCES._items.pop(_AvroBroker.format_name, None)


def _orders() -> bt.Dataset:
    return bt.read.table(_AvroBroker.format_name, "orders", value_format="avro", value_schema=ORDER)


def test_the_plan_sees_the_decoded_payload_type_before_the_query_runs():
    """`Dataset.schema` is answered from the plan's static analysis, on no rows, so this is
    the property that lets an expression over the payload type-check at plan time."""
    assert _orders().schema.field("value").type == pa.struct(
        [pa.field("user", pa.string(), nullable=False), pa.field("amount", pa.int64())]
    )


def test_expressions_reach_into_the_decoded_payload():
    result = (
        _orders()
        .select(
            user=bt.col("value").struct.field("user"),
            amount=bt.col("value").struct.field("amount"),
        )
        .group_by("user")
        .agg(total=bt.col("amount").sum())
        .to_pydict()
    )
    assert sorted(zip(result["user"], result["total"], strict=True)) == [("u1", 17), ("u2", 5)]


def test_the_undecoded_source_is_still_opaque_bytes():
    """The decode is opt-in; nothing changes for a source that did not ask for it."""
    raw = bt.read.table(_AvroBroker.format_name, "orders")
    assert raw.schema.field("value").type == pa.binary()


def test_a_decoded_stream_writes_to_a_sink(tmp_path):
    """The decoded struct has to survive the whole plan, not just the first expression."""
    out = tmp_path / "orders.parquet"
    _orders().select(user=bt.col("value").struct.field("user")).write.parquet(str(out))
    assert sorted(bt.read.parquet(str(out)).to_pydict()["user"]) == ["u1", "u1", "u2"]


def test_a_projection_that_drops_the_payload_still_returns_the_other_columns():
    assert _orders().select("offset", "topic").to_pydict()["offset"] == [0, 1, 2]
