"""Codec-specific options reach the codec from a stream source and sink.

`build_payload_codecs` learned to forward a ``"{side}_codec_options"`` dict, but the broker
source and the Kafka sink built their codec config from a fixed list of keys, so the dict
never arrived: Protobuf's ``message_indexes`` and the string codec's ``encoding`` could not
be set from ``bt.read.kafka`` or ``write.kafka`` at all.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.formats.streaming.kafka_sink import KafkaStreamSink

pytestmark = pytest.mark.unit


def test_the_source_forwards_codec_options():
    ds = bt.read.kafka("orders", value_format="string", value_codec_options={"encoding": "latin-1"})
    assert ds._sources[0]._value_codec._encoding == "latin-1"


def test_the_sink_forwards_codec_options():
    sink = KafkaStreamSink(
        topic="orders", key_format="string", key_codec_options={"encoding": "latin-1"}
    )
    assert sink._key_codec._encoding == "latin-1"


def test_a_non_dict_option_value_is_refused():
    with pytest.raises(bt.PlanError, match="value_codec_options must be a dict"):
        bt.read.kafka("orders", value_format="string", value_codec_options="latin-1")
