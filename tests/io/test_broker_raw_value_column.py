"""`raw_value_column=` keeps a broker message's raw payload for dead-letter routing (AP-479).

`value_decode_mode="permissive"` nulls a record that will not decode, and the decoded
column replaces `value`, so the bytes were gone: a bad message could not be replayed once
the producer was fixed. The option keeps them beside the decoded value. Tested through a
bounded in-process broker, so no Kafka is needed.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.session._scan import _scan
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

pytestmark = pytest.mark.unit

_PAYLOADS = [b'{"a": 1}', b"{not json", None, b'{"a": 4}']


class _Broker(BrokerSource):
    """A bounded broker that publishes one fixed batch of payloads, then ends."""

    format_name = "raw_value_test_broker"
    bounded = True

    def __init__(self, topic: str, *, payloads=(), **kwargs) -> None:
        super().__init__(topic, **kwargs)
        self._payloads = list(payloads)
        self._served = False

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        if self._served:
            return None
        self._served = True
        return [
            BrokerMessage(value=p, partition=0, offset=i, timestamp=i, topic=self.topic)
            for i, p in enumerate(self._payloads)
        ]


def _source(**overrides):
    opts = {
        "payloads": _PAYLOADS,
        "value_format": "json",
        "value_schema": {"a": "int64"},
        "value_decode_mode": "permissive",
        "raw_value_column": "value_raw",
    }
    return _Broker("events", **{**opts, **overrides})


@pytest.fixture
def registered():
    from batcher.io.formats.base import SOURCES

    SOURCES.add(_Broker.format_name, _Broker)
    try:
        yield
    finally:
        SOURCES._items.pop(_Broker.format_name, None)


def test_the_raw_payload_rides_beside_the_decoded_value():
    batch = next(iter(_source().iter_batches()))
    assert batch.schema == _source().schema()
    assert batch.column("value").to_pylist() == [{"a": 1}, None, None, {"a": 4}]
    assert batch.column("value_raw").to_pylist() == _PAYLOADS


def test_the_schema_declares_the_column_before_any_poll():
    assert _source().schema().field("value_raw").type == pa.binary()


def test_dead_letters_are_plain_dataset_code():
    ds = _scan(_source())
    dead = ds.filter(bt.col("value").is_null() & bt.col("value_raw").is_not_null())
    # The malformed record is a dead letter; the tombstone (a null payload) is not.
    assert dead.select("offset", "value_raw").to_pydict() == {
        "offset": [1],
        "value_raw": [b"{not json"],
    }


def test_projecting_only_the_raw_column_still_reads_it():
    batch = next(iter(_source().iter_batches(projection=["value_raw"])))
    assert batch.column("value_raw").to_pylist() == _PAYLOADS


def test_a_split_rebuilds_with_the_column(registered):
    source = _source()
    assert source.splits()[0].schema() == source.schema()


def test_without_the_option_nothing_changes():
    assert "value_raw" not in _source(raw_value_column=None).schema().names


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"value_format": None, "value_schema": None}, "needs value_format"),
        ({"value_decode_mode": "fail"}, "needs value_decode_mode='permissive'"),
        ({"raw_value_column": "offset"}, "already a broker column"),
    ],
)
def test_a_column_that_could_not_work_is_refused(overrides, match):
    with pytest.raises(PlanError, match=match):
        _source(**overrides)
