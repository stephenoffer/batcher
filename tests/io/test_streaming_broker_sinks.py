"""The broker sink capability contract, and the four sinks built on it, against fake clients.

None of these need a broker. Each fake models only the client surface the sink calls
(``send_async``/``flush`` for Pulsar, ``put_records``/``put_record`` for Kinesis,
``publish`` futures for Pub/Sub, ``create_batch``/``send_batch`` for Event Hubs), which is
where the request shapes, the retry handling and the acknowledgement rule live. The live
smoke tests under ``tests/integration/live/`` exercise the real clients when credentials
are set.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

from batcher._internal.errors import IOError, PlanError
from batcher.io.formats.streaming import STREAM_SINKS
from batcher.io.formats.streaming.broker_sinks import (
    DEDUP_ID_HEADER,
    BrokerStreamSink,
    EventHubsStreamSink,
    KinesisStreamSink,
    PubSubStreamSink,
    PulsarStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.kafka_sink import KafkaStreamSink

ALL_SINKS = {
    "kafka": KafkaStreamSink,
    "pulsar": PulsarStreamSink,
    "kinesis": KinesisStreamSink,
    "pubsub": PubSubStreamSink,
    "eventhubs": EventHubsStreamSink,
}

#: The column vocabulary of the shared contract; a sink reads a subset of it.
_CONTRACT_COLUMNS = {"value", "key", "topic", "partition", "headers"}


# --- the shared contract ------------------------------------------------------
@pytest.mark.parametrize(("name", "cls"), sorted(ALL_SINKS.items()))
def test_every_broker_sink_is_registered_and_states_its_capabilities(name, cls):
    assert STREAM_SINKS.get(name) is cls
    assert issubclass(cls, BrokerStreamSink)
    caps = cls.capabilities
    assert isinstance(caps, SinkCapabilities)
    assert caps.broker == name
    assert "value" in caps.columns
    assert set(caps.columns) <= _CONTRACT_COLUMNS
    for statement in (caps.ordering, caps.batching, caps.retry_identity):
        assert statement.strip(), f"{name} leaves a capability unstated"


def test_only_kafka_publishes_a_null_value():
    assert [n for n, c in sorted(ALL_SINKS.items()) if c.capabilities.null_values] == ["kafka"]


def test_a_broker_without_headers_refuses_dedup_ids_at_construction():
    with pytest.raises(PlanError, match="no per-record metadata"):
        KinesisStreamSink(topic="s", dedup_ids="writer-1")


@pytest.mark.parametrize("cls", [PulsarStreamSink, PubSubStreamSink, EventHubsStreamSink])
def test_a_null_value_is_refused_where_the_broker_has_no_tombstone(cls):
    sink = cls(topic="t")
    with pytest.raises(PlanError, match="cannot publish a null 'value'"):
        sink.write_batch(0, pa.table({"value": pa.array(["a", None])}))


@pytest.mark.parametrize("cls", [PulsarStreamSink, KinesisStreamSink, PubSubStreamSink])
def test_the_value_column_and_its_type_are_checked_before_publishing(cls):
    sink = cls(topic="t")
    with pytest.raises(PlanError, match="needs a 'value' column"):
        sink.write_batch(0, pa.table({"payload": ["a"]}))
    with pytest.raises(PlanError, match="must be binary or string, not int64"):
        sink.write_batch(0, pa.table({"value": [1]}))


def test_event_hubs_has_no_per_row_destination_so_it_needs_a_hub():
    with pytest.raises(PlanError, match=r"needs a destination: pass topic=\.\.\. to write"):
        EventHubsStreamSink().write_batch(0, pa.table({"value": ["a"], "topic": ["x"]}))


def test_a_nonpositive_flush_timeout_is_refused_for_every_sink():
    for cls in ALL_SINKS.values():
        with pytest.raises(PlanError, match="flush_timeout"):
            cls(topic="t", flush_timeout=0)


def test_an_empty_micro_batch_reports_without_connecting():
    sink = PulsarStreamSink(topic="t")
    assert sink.write_batch(3, pa.table({"value": pa.array([], pa.string())})) == "pulsar:t:3:0"


# --- Pulsar ---------------------------------------------------------------------
class _PulsarProducer:
    def __init__(self, topic, kwargs, fail_rows=(), never_ack=False):
        self.topic = topic
        self.kwargs = kwargs
        self.sent: list[tuple[bytes, dict]] = []
        self._callbacks = []
        self._fail_rows = set(fail_rows)
        self._never_ack = never_ack
        self.closed = False
        self.flushes = 0

    def send_async(self, content, callback, **kwargs):
        self.sent.append((content, kwargs))
        self._callbacks.append(callback)

    def flush(self):
        self.flushes += 1
        if self._never_ack:
            return
        for i, callback in enumerate(self._callbacks):
            callback("ServerError" if i in self._fail_rows else "Ok", object())
        self._callbacks = []

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pulsar(monkeypatch):
    state = types.SimpleNamespace(producers={}, client_args=None, fail_rows=(), never_ack=False)

    class Client:
        def __init__(self, url, **kwargs):
            state.client_args = (url, kwargs)
            self.closed = False

        def create_producer(self, topic, **kwargs):
            producer = _PulsarProducer(topic, kwargs, state.fail_rows, state.never_ack)
            state.producers[topic] = producer
            return producer

        def close(self):
            self.closed = True

    module = types.ModuleType("pulsar")
    module.Client = Client
    module.AuthenticationToken = lambda token: ("token", token)
    module.Result = types.SimpleNamespace(Ok="Ok")
    monkeypatch.setitem(sys.modules, "pulsar", module)
    return state


def test_pulsar_routes_by_key_carries_properties_and_reports_after_the_flush(fake_pulsar):
    sink = PulsarStreamSink(topic="events", service_url="pulsar://b:6650")
    sink.open()
    headers = pa.array(
        [[{"key": "trace", "value": b"abc"}], None],
        type=pa.list_(pa.struct([("key", pa.string()), ("value", pa.binary())])),
    )
    table = pa.table({"value": ["a", "b"], "key": ["k1", None], "headers": headers})
    assert sink.write_batch(5, table) == "pulsar:events:5:2"
    producer = fake_pulsar.producers["events"]
    assert producer.sent == [
        (b"a", {"partition_key": "k1", "properties": {"trace": "abc"}}),
        (b"b", {}),
    ]
    assert producer.flushes == 1
    assert producer.kwargs == {"block_if_queue_full": True}
    assert fake_pulsar.client_args == ("pulsar://b:6650", {})


def test_pulsar_producer_name_derives_replay_stable_sequence_ids(fake_pulsar):
    sink = PulsarStreamSink(topic="events", producer_name="etl-1")
    sink.open()
    sink.write_batch(2, pa.table({"value": ["a", "b"]}))
    producer = fake_pulsar.producers["events"]
    assert producer.kwargs["producer_name"] == "etl-1"
    assert [kw["sequence_id"] for _, kw in producer.sent] == [2 << 32, (2 << 32) | 1]


def test_pulsar_opens_one_producer_per_destination_topic(fake_pulsar):
    sink = PulsarStreamSink(topic="default")
    sink.open()
    sink.write_batch(0, pa.table({"value": ["a", "b", "c"], "topic": ["x", None, "x"]}))
    assert sorted(fake_pulsar.producers) == ["default", "x"]
    assert len(fake_pulsar.producers["x"].sent) == 2


def test_pulsar_dedup_ids_become_a_property(fake_pulsar):
    sink = PulsarStreamSink(topic="events", dedup_ids="w")
    sink.open()
    sink.write_batch(9, pa.table({"value": ["a", "b"]}))
    props = [kw["properties"] for _, kw in fake_pulsar.producers["events"].sent]
    assert props == [{DEDUP_ID_HEADER: "w:9:0"}, {DEDUP_ID_HEADER: "w:9:1"}]


def test_pulsar_a_rejected_send_fails_the_micro_batch(fake_pulsar):
    fake_pulsar.fail_rows = (1,)
    sink = PulsarStreamSink(topic="events")
    sink.open()
    with pytest.raises(IOError, match="1 message\\(s\\) of micro-batch 0 were rejected"):
        sink.write_batch(0, pa.table({"value": ["a", "b"]}))


def test_pulsar_unacknowledged_sends_fail_the_epoch(fake_pulsar):
    fake_pulsar.never_ack = True
    sink = PulsarStreamSink(topic="events", flush_timeout=0.05)
    sink.open()
    with pytest.raises(IOError, match="unacknowledged"):
        sink.write_batch(0, pa.table({"value": ["a"]}))


def test_pulsar_auth_token_is_resolved_from_a_secret_reference(fake_pulsar, monkeypatch):
    monkeypatch.setenv("PULSAR_TEST_TOKEN", "jwt-123")
    sink = PulsarStreamSink(topic="events", auth_token="env:PULSAR_TEST_TOKEN")
    sink.open()
    assert fake_pulsar.client_args[1] == {"authentication": ("token", "jwt-123")}


def test_pulsar_text_properties_refuse_non_utf8_header_values(fake_pulsar):
    sink = PulsarStreamSink(topic="events")
    sink.open()
    headers = pa.array(
        [[{"key": "raw", "value": b"\xff"}]],
        type=pa.list_(pa.struct([("key", pa.string()), ("value", pa.binary())])),
    )
    with pytest.raises(PlanError, match="header 'raw' is not UTF-8"):
        sink.write_batch(0, pa.table({"value": ["a"], "headers": headers}))


def test_pulsar_close_flushes_and_closes_everything_once(fake_pulsar):
    sink = PulsarStreamSink(topic="events")
    sink.open()
    sink.write_batch(0, pa.table({"value": ["a"]}))
    producer = fake_pulsar.producers["events"]
    sink.close()
    sink.close()
    assert producer.closed
    assert producer.flushes == 2


# --- Kinesis --------------------------------------------------------------------
class _Throttled(Exception):
    pass


_Throttled.__name__ = "ProvisionedThroughputExceededException"


class _KinesisClient:
    def __init__(self, *, fail_first=(), throttle_calls=0, always_fail=False):
        self.calls: list[dict] = []
        self.single: list[dict] = []
        self._fail_first = set(fail_first)
        self._throttle_calls = throttle_calls
        self._always_fail = always_fail
        self._seq = 0

    def put_records(self, **request):
        if self._throttle_calls:
            self._throttle_calls -= 1
            raise _Throttled("slow down")
        self.calls.append(request)
        results = []
        failed = 0
        for record in request["Records"]:
            data = record["Data"]
            if self._always_fail or data in self._fail_first:
                self._fail_first.discard(data)
                failed += 1
                results.append({"ErrorCode": "ProvisionedThroughputExceededException"})
            else:
                results.append({"SequenceNumber": "1", "ShardId": "shardId-0"})
        return {"FailedRecordCount": failed, "Records": results}

    def put_record(self, **request):
        self._seq += 1
        self.single.append(request)
        return {"SequenceNumber": f"seq-{self._seq}", "ShardId": "shardId-0"}


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    from batcher.io.formats.streaming.broker_sinks import kinesis

    monkeypatch.setattr(kinesis.time, "sleep", lambda _s: None)


def _kinesis(client, **kwargs) -> KinesisStreamSink:
    sink = KinesisStreamSink(topic="stream-a", **kwargs)
    sink._client = client
    return sink


def test_kinesis_puts_records_with_the_key_as_partition_key():
    client = _KinesisClient()
    sink = _kinesis(client)
    assert sink.write_batch(4, pa.table({"value": ["a", "b"], "key": ["k", None]})) == (
        "kinesis:stream-a:4:2"
    )
    (request,) = client.calls
    assert request == {
        "StreamName": "stream-a",
        "Records": [
            {"Data": b"a", "PartitionKey": "k"},
            {"Data": b"b", "PartitionKey": "4-1"},
        ],
    }


def test_kinesis_splits_a_micro_batch_at_the_500_record_limit():
    client = _KinesisClient()
    sink = _kinesis(client)
    sink.write_batch(0, pa.table({"value": [str(i) for i in range(1201)]}))
    assert [len(c["Records"]) for c in client.calls] == [500, 500, 201]


def test_kinesis_splits_a_micro_batch_at_the_5_mib_request_limit():
    client = _KinesisClient()
    sink = _kinesis(client)
    big = b"x" * (2 * 1024 * 1024)
    sink.write_batch(0, pa.table({"value": pa.array([big, big, big], pa.binary())}))
    assert [len(c["Records"]) for c in client.calls] == [2, 1]


def test_kinesis_resends_only_the_records_that_failed():
    client = _KinesisClient(fail_first={b"b"})
    sink = _kinesis(client)
    sink.write_batch(0, pa.table({"value": ["a", "b", "c"]}))
    assert [[r["Data"] for r in c["Records"]] for c in client.calls] == [
        [b"a", b"b", b"c"],
        [b"b"],
    ]


def test_kinesis_retries_a_throttled_request():
    client = _KinesisClient(throttle_calls=2)
    sink = _kinesis(client)
    sink.write_batch(0, pa.table({"value": ["a"]}))
    assert len(client.calls) == 1


def test_kinesis_fails_the_epoch_once_attempts_are_exhausted():
    client = _KinesisClient(always_fail=True)
    sink = _kinesis(client, max_attempts=3)
    with pytest.raises(IOError, match="still failed after 3 attempt"):
        sink.write_batch(0, pa.table({"value": ["a"]}))
    assert len(client.calls) == 3


def test_kinesis_ordered_chains_sequence_numbers_per_partition_key():
    client = _KinesisClient()
    sink = _kinesis(client, ordered=True)
    sink.write_batch(0, pa.table({"value": ["a", "b", "c"], "key": ["k", "j", "k"]}))
    assert client.calls == []
    assert [r.get("SequenceNumberForOrdering") for r in client.single] == [None, None, "seq-1"]


def test_kinesis_refuses_an_option_it_would_otherwise_drop():
    with pytest.raises(PlanError, match="unknown option"):
        KinesisStreamSink(topic="s", compression="zstd")


def test_kinesis_refuses_a_non_utf8_binary_key():
    sink = _kinesis(_KinesisClient())
    with pytest.raises(PlanError, match="key is text"):
        sink.write_batch(0, pa.table({"value": ["a"], "key": pa.array([b"\xff"], pa.binary())}))


# --- Pub/Sub --------------------------------------------------------------------
class _Future:
    def __init__(self, error=None):
        self._error = error

    def result(self, timeout=None):
        if self._error is not None:
            raise self._error
        return "message-id"


@pytest.fixture
def fake_pubsub(monkeypatch):
    state = types.SimpleNamespace(published=[], options=None, resumed=[], fail=False)

    class PublisherClient:
        def __init__(self, publisher_options=None):
            state.options = publisher_options

        def publish(self, topic, data, ordering_key="", **attrs):
            state.published.append((topic, data, ordering_key, attrs))
            return _Future(RuntimeError("deadline") if state.fail else None)

        def resume_publish(self, topic, ordering_key):
            state.resumed.append((topic, ordering_key))

    pubsub_v1 = types.SimpleNamespace(
        PublisherClient=PublisherClient,
        types=types.SimpleNamespace(PublisherOptions=lambda **kw: ("options", kw)),
    )
    cloud = types.ModuleType("google.cloud")
    cloud.pubsub_v1 = pubsub_v1
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    return state


def test_pubsub_publishes_data_with_headers_as_attributes(fake_pubsub):
    sink = PubSubStreamSink(topic="projects/p/topics/t", dedup_ids="w")
    sink.open()
    assert sink.write_batch(1, pa.table({"value": ["a"]})) == "pubsub:projects/p/topics/t:1:1"
    assert fake_pubsub.published == [
        ("projects/p/topics/t", b"a", "", {DEDUP_ID_HEADER: "w:1:0"}),
    ]
    assert fake_pubsub.options is None


def test_pubsub_ordered_uses_the_key_as_the_ordering_key(fake_pubsub):
    sink = PubSubStreamSink(topic="projects/p/topics/t", ordered=True)
    sink.open()
    sink.write_batch(0, pa.table({"value": ["a"], "key": ["acct-1"]}))
    assert fake_pubsub.options == ("options", {"enable_message_ordering": True})
    assert fake_pubsub.published[0][2] == "acct-1"


def test_pubsub_refuses_a_key_it_has_nowhere_to_put(fake_pubsub):
    sink = PubSubStreamSink(topic="projects/p/topics/t")
    with pytest.raises(PlanError, match="no message key"):
        sink.write_batch(0, pa.table({"value": ["a"], "key": ["k"]}))


def test_pubsub_a_failed_future_fails_the_epoch_and_resumes_the_key(fake_pubsub):
    fake_pubsub.fail = True
    sink = PubSubStreamSink(topic="projects/p/topics/t", ordered=True)
    sink.open()
    with pytest.raises(IOError, match="were not published"):
        sink.write_batch(0, pa.table({"value": ["a"], "key": ["k"]}))
    assert fake_pubsub.resumed == [("projects/p/topics/t", "k")]


# --- Event Hubs -----------------------------------------------------------------
class _EventData:
    def __init__(self, body):
        self.body = body
        self.properties = None


class _EventBatch:
    def __init__(self, capacity, kwargs):
        self.kwargs = kwargs
        self.events: list[_EventData] = []
        self._capacity = capacity

    def add(self, event):
        if len(self.events) >= self._capacity:
            raise ValueError("EventDataBatch has reached its size limit")
        self.events.append(event)

    def __len__(self):
        return len(self.events)


@pytest.fixture
def fake_eventhub(monkeypatch):
    state = types.SimpleNamespace(sent=[], capacity=100, conn=None, fail=False)

    class EventHubProducerClient:
        @classmethod
        def from_connection_string(cls, conn_str, eventhub_name):
            state.conn = (conn_str, eventhub_name)
            return cls()

        def create_batch(self, **kwargs):
            return _EventBatch(state.capacity, kwargs)

        def send_batch(self, batch, timeout=None):
            if state.fail:
                raise RuntimeError("service busy")
            state.sent.append(batch)

        def close(self):
            pass

    module = types.ModuleType("azure.eventhub")
    module.EventHubProducerClient = EventHubProducerClient
    module.EventData = _EventData
    monkeypatch.setitem(sys.modules, "azure.eventhub", module)
    return state


def test_eventhubs_groups_rows_by_partition_key_and_partition_id(fake_eventhub, monkeypatch):
    monkeypatch.setenv("EH_TEST_CONN", "Endpoint=sb://x/")
    sink = EventHubsStreamSink(topic="hub", connection_str="env:EH_TEST_CONN")
    sink.open()
    table = pa.table(
        {
            "value": ["a", "b", "c", "d"],
            "key": ["k1", "k2", "k1", None],
            "partition": pa.array([None, None, None, 3], pa.int32()),
        }
    )
    assert sink.write_batch(0, table) == "eventhubs:hub:0:4"
    assert fake_eventhub.conn == ("Endpoint=sb://x/", "hub")
    routed = {
        tuple(sorted(b.kwargs.items())): [e.body for e in b.events] for b in fake_eventhub.sent
    }
    assert routed == {
        (("partition_key", "k1"),): [b"a", b"c"],
        (("partition_key", "k2"),): [b"b"],
        (("partition_id", "3"),): [b"d"],
    }


def test_eventhubs_splits_a_group_at_the_batch_size_limit(fake_eventhub):
    fake_eventhub.capacity = 2
    sink = EventHubsStreamSink(topic="hub", connection_str="Endpoint=sb://x/", dedup_ids="w")
    sink.open()
    sink.write_batch(7, pa.table({"value": ["a", "b", "c", "d", "e"]}))
    assert [len(b) for b in fake_eventhub.sent] == [2, 2, 1]
    first = fake_eventhub.sent[0].events[0]
    assert first.properties == {DEDUP_ID_HEADER: b"w:7:0"}


def test_eventhubs_a_refused_batch_fails_the_epoch(fake_eventhub):
    fake_eventhub.fail = True
    sink = EventHubsStreamSink(topic="hub", connection_str="Endpoint=sb://x/")
    sink.open()
    with pytest.raises(IOError, match="was not accepted"):
        sink.write_batch(0, pa.table({"value": ["a"]}))


def test_eventhubs_needs_a_connection_string(fake_eventhub):
    with pytest.raises(PlanError, match="connection_str"):
        EventHubsStreamSink(topic="hub").open()


# --- end to end through ds.write ------------------------------------------------
def test_write_pulsar_runs_a_stream_into_the_sink(fake_pulsar):
    import batcher as bt

    demo = bt.read.rate_micro_batch(10, num_rows=30).select(value=bt.col("value").cast("string"))
    query = demo.write.pulsar("out", trigger=bt.Trigger.available_now())
    assert query.await_termination()
    sent = fake_pulsar.producers["out"].sent
    assert sorted(int(v) for v, _ in sent) == list(range(30))
