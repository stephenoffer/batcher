"""A broker split must rebuild the source it came from, not the constructor's defaults.

Every broker pulls its own settings out of ``**options`` into named keyword parameters, so
they never reach the client config as bogus keys. The cost is that `BrokerSource._options`
— which is all a `BrokerSplit` carries — stops holding them, and a worker rebuilding the
source silently gets the defaults.

That is not a degraded distributed read, it is a different query: a Kafka source configured
``starting_offsets="latest"`` replayed the whole topic from the beginning on every worker,
and one configured ``fail_on_data_loss=False`` stopped on exactly the condition the user
chose to tolerate. Nothing raised single-node, where the object keeps its own attributes.

So this file checks fidelity *behaviorally*: build a source with non-default settings, take
its split, rebuild through the split's own path, and compare the attributes that decide what
the query reads. A broker that normalizes an option into one it does forward (Kinesis maps
`starting_position` onto `iterator_type`) passes without needing a `_split_options` entry,
because the rebuilt reader genuinely agrees.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

pytestmark = pytest.mark.unit


def _rebuild(source: BrokerSource):
    """Rebuild `source` the way a worker does: through its own split's reader path.

    The real sources are constructed with an explicit ``partitions=`` so `splits()` can
    enumerate without a live cluster — partition *discovery* is not what is under test
    here, and reaching a broker to check an option-passing contract would make this an
    integration test.
    """
    return source.splits()[0]._reader()


class _Broker(BrokerSource):
    """A broker that consumes one option by name, the shape every real one has."""

    format_name = "split_fidelity_broker"

    __slots__ = ("_consumed",)

    def __init__(self, topic: str, *, consumed: str = "default", **options) -> None:
        super().__init__(topic, **options)
        self._consumed = consumed

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        return [BrokerMessage(value=b"v", partition=0, offset=0, timestamp=0, topic=self.topic)]


class _FaithfulBroker(_Broker):
    """The same broker, declaring what it consumed."""

    format_name = "split_fidelity_faithful"

    def _split_options(self):
        return {"consumed": self._consumed}


@pytest.fixture
def registered():
    """Register both test brokers so a split's registry lookup resolves them."""
    from batcher.io.formats.base import SOURCES

    for cls in (_Broker, _FaithfulBroker):
        SOURCES.add(cls.format_name, cls)
    try:
        yield
    finally:
        for cls in (_Broker, _FaithfulBroker):
            SOURCES._items.pop(cls.format_name, None)


def test_the_base_seam_defaults_to_carrying_nothing_extra():
    """A broker that consumes nothing beyond what it forwards needs no override."""
    assert _Broker("t")._split_options() == {}


def test_a_broker_that_declares_its_consumed_option_rebuilds_faithfully(registered):
    source = _FaithfulBroker("t", consumed="explicit")
    assert _rebuild(source)._consumed == "explicit"


def test_the_seam_is_what_makes_the_difference(registered):
    """Without the override the rebuild silently gets the constructor default.

    Pinned so the seam cannot be quietly removed as redundant: this is the exact failure
    that shipped on four brokers at once.
    """
    assert _rebuild(_Broker("t", consumed="explicit"))._consumed == "default"


# --- the real brokers ------------------------------------------------------


def test_kafka_carries_the_settings_that_decide_what_it_reads():
    from batcher.io.formats.streaming.kafka import KafkaSource

    source = KafkaSource(
        "orders",
        partitions=[0],
        bootstrap_servers="b:9092",
        starting_offsets="latest",
        fail_on_data_loss=False,
        poll_timeout=0.05,
        metadata_timeout=2.0,
    )
    rebuilt = _rebuild(source)
    assert rebuilt._offset_reset == "latest", "a lost starting_offsets replays the topic"
    assert rebuilt._fail_on_data_loss is False, "a lost fail_on_data_loss stops the query"
    assert rebuilt._poll_timeout == 0.05
    assert rebuilt._metadata_timeout == 2.0


def test_kafka_carries_an_explicit_per_partition_starting_offset_map():
    from batcher.io.formats.streaming.kafka import KafkaSource

    source = KafkaSource(
        "orders", partitions=[0], bootstrap_servers="b:9092", starting_offsets={0: 4096}
    )
    assert _rebuild(source)._start_at == {0: 4096}


def test_pulsar_carries_its_declared_partition_count():
    """Pulsar does not tell a client its partition count, so losing it is a wrong read."""
    pytest.importorskip("pulsar", reason="pulsar-client is an optional extra")
    from batcher.io.formats.streaming.pulsar import PulsarSource

    source = PulsarSource(
        "topic",
        partitions=[0],
        num_partitions=8,
        receive_timeout_millis=250,
        starting_position="latest",
    )
    rebuilt = _rebuild(source)
    assert rebuilt._num_partitions == 8
    assert rebuilt._receive_timeout_millis == 250
    assert rebuilt._starting_position == source._starting_position


def test_pubsub_carries_its_pull_deadline():
    from batcher.io.formats.streaming.pubsub import PubSubSource

    assert _rebuild(PubSubSource("sub", pull_timeout=0.25))._pull_timeout == 0.25


def test_kinesis_normalizes_its_start_into_an_option_it_already_forwards():
    """Not every broker needs the seam — Kinesis maps `starting_position` onto
    `iterator_type`, which does travel. The check is fidelity, not the mechanism."""
    from batcher.io.formats.streaming.kinesis import KinesisSource

    source = KinesisSource("stream", partitions=[0], starting_position="latest")
    assert _rebuild(source)._options["iterator_type"] == "LATEST"


def test_every_broker_source_rebuilds_with_the_same_declared_schema():
    """The schema is the one thing an epoch's concat cannot tolerate a difference in."""
    from batcher.io.formats.streaming.kafka import KafkaSource

    source = KafkaSource("orders", partitions=[0], bootstrap_servers="b:9092", include_headers=True)
    assert _rebuild(source).schema() == source.schema()


# --- the distributed epoch read --------------------------------------------


def test_an_epoch_read_decodes_the_payload_the_same_way_the_source_does(registered):
    """`read_epoch` is the distributed path's whole read, and CI never runs a cluster — so
    this is the only place the decode on a worker is checked at all. A split that decoded
    differently from its siblings would fail the epoch's concat on the schema."""
    pytest.importorskip("fastavro")
    import io

    import fastavro

    schema = {"type": "record", "name": "R", "fields": [{"name": "a", "type": "long"}]}

    def encode(value: int) -> bytes:
        buffer = io.BytesIO()
        fastavro.schemaless_writer(buffer, fastavro.parse_schema(schema), {"a": value})
        return buffer.getvalue()

    class _AvroBroker(_Broker):
        format_name = "split_fidelity_avro"

        def _poll(self):
            return [
                BrokerMessage(value=encode(1), partition=0, offset=0, timestamp=0, topic=self.topic)
            ]

    from batcher.io.formats.base import SOURCES

    SOURCES.add(_AvroBroker.format_name, _AvroBroker)
    try:
        source = _AvroBroker("t", value_format="avro", value_schema=schema)
        split = source.splits()[0]
        batches, position = split.read_epoch()
        assert batches[0].schema == source.schema()
        assert batches[0].column("value").to_pylist() == [{"a": 1}]
        assert position == 0
    finally:
        SOURCES._items.pop(_AvroBroker.format_name, None)


def test_an_epoch_read_honours_the_projection_it_is_given(registered):
    """The same build-time projection the poll loop gets, on the worker path."""
    source = _FaithfulBroker("t")
    batches, _ = source.splits()[0].read_epoch(None, ["offset", "topic"])
    assert batches[0].schema.names == ["offset", "topic"]
