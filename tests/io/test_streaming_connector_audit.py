"""The five-point connector audit, applied to the streaming broker sources.

Each defect below shipped in `io/formats/streaming/` and each one passes a review and a
green suite while being wrong. They are the same classes the SQL connectors were audited
for, but a broker makes several of them worse:

1. **Fake streaming.** `read()` was ``list(self.iter_batches(...))`` on a source whose
   `iter_batches` polls *forever* by contract. That is not a slow read, it is a
   non-terminating one — a `collect()` on a Kafka topic accumulates until the box dies,
   reading as a hang rather than as the misuse it is.
2. **Materializing `schema()`.** Not a defect here, and the reason is worth pinning: the
   broker schema is a fixed constant. Consuming from a broker to learn a schema would
   *advance a consumer offset* and lose the messages it consumed, so the test asserts
   `schema()` opens no client at all.
3. **Identity collision.** `identity()` was ``kafka:<topic>`` — the same topic name on the
   production and staging clusters shared one learned-statistics key, so Kyber planned one
   with the other's cardinalities and nothing errored.
4. **Credential leaks.** Broker options carry `sasl.password`, `sasl_plain_password`, and
   Event Hubs' `connection_str` (a SAS key). They were rendered by `BrokerSplit`'s generated
   `repr`, and — worse, because it is *persisted* — reachable from `identity()`.
5. **Distributed-write data loss.** Brokers here are read-only, so there is no
   `write_partitioned` disposition to get wrong. Pinned as a test rather than assumed.
6. **Offset/checkpoint correctness.** Pulsar and Pub/Sub acknowledged inside `_poll` — at
   *read* time, before the epoch was ever published. A crash in that window made the broker
   believe the messages were handled: at-most-once, i.e. silent data loss. Acks now happen
   in `_commit_delivered`, which `iter_batches` calls only after the consumer asks for the
   next batch, which it does only once the epoch is published.

Plus two cross-cutting hazards: **process-salted `hash()`** in the `offset` column (the same
message got a different offset on every run and every worker, silently breaking the ordering
and de-dup that column exists for), and **abandoned clients** (a generator dropped after its
first yield runs its `finally` only at GC, which for a reference cycle is never).

None of these drivers are installed, so each is driven through a spy modelling the real
client's contract — following `tests/io/test_sql.py`'s ``_SpyCursor`` and
`tests/io/test_connector_streaming.py`'s spies. The spies log their calls, so "did not ack
early" is asserted as *the ack call had not yet been made at that point in the sequence*,
rather than inferred from a result shape.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from typing import Any

import pytest

from batcher._internal.errors import PlanError
from batcher.io.formats.streaming.broker import (
    BrokerMessage,
    BrokerSource,
    BrokerSplit,
    redact_broker_options,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------
class _UnboundedBroker(BrokerSource):
    """A broker that always has messages — i.e. an honest unbounded stream.

    `_poll` never returns None, which is exactly the production contract. Any code path that
    tries to materialize this must refuse rather than loop.
    """

    format_name = "test_unbounded"
    __slots__ = ("closed", "polls")

    def __init__(self, topic: str = "t", **opts: Any) -> None:
        super().__init__(topic, poll_size=2, **opts)
        self.polls = 0
        self.closed = 0

    def _discover_partitions(self) -> list[int]:
        return [0]

    def _poll(self) -> list[BrokerMessage]:
        self.polls += 1
        base = self.polls * 10
        return [
            BrokerMessage(value=b"x", partition=0, offset=base + i, timestamp=0, topic=self.topic)
            for i in range(2)
        ]

    def close(self) -> None:
        self.closed += 1


class _BoundedBroker(_UnboundedBroker):
    """The same broker, but finite — `_poll` returns None after two polls."""

    format_name = "test_bounded"
    bounded = True
    __slots__ = ()

    def _poll(self) -> list[BrokerMessage] | None:
        if self.polls >= 2:
            return None
        return super()._poll()


# --------------------------------------------------------------------------
# 1. Fake streaming — an unbounded read() must refuse, not hang
# --------------------------------------------------------------------------
def test_read_on_an_unbounded_broker_refuses_instead_of_never_returning():
    """`read()` was `list(iter_batches())`, which on an unbounded broker cannot terminate."""
    with pytest.raises(PlanError, match="unbounded"):
        _UnboundedBroker().read()


def test_read_error_names_the_streaming_alternative():
    """The error has to be actionable: a hang taught the user nothing."""
    with pytest.raises(PlanError, match="iter_batches"):
        _UnboundedBroker().read()


def test_read_still_materializes_a_bounded_broker():
    """The refusal is scoped to `bounded = False` — a finite broker still satisfies `Source`."""
    assert sum(b.num_rows for b in _BoundedBroker().read()) == 4


def test_iter_batches_streams_rather_than_materializing():
    """One poll per batch consumed — the generator must not run ahead of its consumer."""
    broker = _UnboundedBroker()
    it = broker.iter_batches()
    next(it)
    assert broker.polls == 1, "iter_batches polled ahead of the consumer"
    next(it)
    assert broker.polls == 2


# --------------------------------------------------------------------------
# 2. schema() must not consume from the broker (it would advance an offset)
# --------------------------------------------------------------------------
def test_schema_polls_nothing():
    """Consuming to learn a schema would advance a consumer offset and lose those messages."""
    broker = _UnboundedBroker()
    assert broker.schema().names == ["key", "value", "partition", "offset", "timestamp", "topic"]
    assert broker.polls == 0, "schema() consumed from the broker"


@pytest.mark.parametrize("name", ["kafka", "kinesis", "pubsub", "pulsar", "eventhubs"])
def test_schema_needs_no_client_and_so_no_credentials(name):
    """`schema()` on a real connector must not dial the broker — no client is installed here.

    If `schema()` needed a live consumer this would raise `BackendError` (driver missing)
    rather than return the fixed schema.
    """
    from batcher.io.formats.base import SOURCES

    source = SOURCES.get(name)("some-topic")
    assert source.schema().names[:2] == ["key", "value"]


# --------------------------------------------------------------------------
# 3. Identity collision — the same topic on two clusters is two relations
# --------------------------------------------------------------------------
def test_identity_distinguishes_the_same_topic_on_different_clusters():
    """`kafka:events` names a topic, not a relation: prod and staging must not share a key."""
    prod = _UnboundedBroker("events", bootstrap_servers="prod-kafka:9092")
    staging = _UnboundedBroker("events", bootstrap_servers="staging-kafka:9092")
    assert prod.identity() != staging.identity()


def test_identity_is_stable_across_processes():
    """A `hash()`-based identity is salted per process, so no statistic is ever reused.

    This is the failure that looks like a working feedback loop while never improving a
    plan, so it is asserted across a real subprocess boundary rather than within one.
    """
    script = (
        "from batcher.io.formats.streaming.kafka import KafkaSource;"
        "print(KafkaSource('events', bootstrap_servers='prod:9092').identity())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, f"identity differs between processes: {runs}"


def test_split_identity_also_carries_the_connection():
    """A split had the same cluster-blindness as the source it came from."""
    common = {"format_name": "kafka", "topic": "events", "partition": 3, "poll_size": 10}
    prod = BrokerSplit(**common, options={"bootstrap_servers": "prod:9092"})
    staging = BrokerSplit(**common, options={"bootstrap_servers": "staging:9092"})
    assert prod.identity() != staging.identity()


def test_identity_survives_a_credential_rotation():
    """Rotating a password must not orphan the topic's accumulated statistics."""
    before = _UnboundedBroker("t", bootstrap_servers="k:9092", sasl_password="old")
    after = _UnboundedBroker("t", bootstrap_servers="k:9092", sasl_password="new")
    assert before.identity() == after.identity()


# --------------------------------------------------------------------------
# 4. Credential leaks — repr (printed) and identity (persisted)
# --------------------------------------------------------------------------
_CREDENTIALS = [
    ("sasl.password", "PW_SECRET"),
    ("sasl_plain_password", "PW_SECRET"),
    ("connection_str", "Endpoint=sb://h/;SharedAccessKey=PW_SECRET"),
    ("sas_token", "PW_SECRET"),
    ("api_key", "PW_SECRET"),
]


@pytest.mark.parametrize(("key", "value"), _CREDENTIALS)
def test_split_repr_does_not_render_broker_credentials(key, value):
    """A split is pickled to every worker and appears verbatim in any traceback naming it."""
    split = BrokerSplit(
        format_name="kafka", topic="t", partition=0, poll_size=10, options={key: value}
    )
    assert "PW_SECRET" not in repr(split)


@pytest.mark.parametrize(("key", "value"), _CREDENTIALS)
def test_identity_does_not_carry_a_credential(key, value):
    """Worse than a `repr` leak: identity is *persisted* as the stats key and outlives us."""
    source = _UnboundedBroker("t", bootstrap_servers="k:9092", **{key.replace(".", "_"): value})
    assert "PW_SECRET" not in source.identity()

    split = BrokerSplit(
        format_name="kafka", topic="t", partition=0, poll_size=10, options={key: value}
    )
    assert "PW_SECRET" not in split.identity()


def test_redaction_masks_rather_than_drops():
    """Masking is what keeps the fingerprint stable across a rotation (see the identity test)."""
    redacted = redact_broker_options({"bootstrap_servers": "k:9092", "sasl.password": "PW_SECRET"})
    assert redacted == {"bootstrap_servers": "k:9092", "sasl.password": "***"}


def test_split_still_carries_credentials_in_its_payload():
    """Redaction is for display and identity only — the worker must still be able to dial.

    A split that redacted the option itself would produce an authentication failure on every
    worker, so the distinction between the payload and its rendering is the whole fix.
    """
    split = BrokerSplit(
        format_name="kafka",
        topic="t",
        partition=0,
        poll_size=10,
        options={"sasl.password": "PW_SECRET"},
    )
    assert pickle.loads(pickle.dumps(split)).options["sasl.password"] == "PW_SECRET"


# --------------------------------------------------------------------------
# 5. Distributed writes — brokers here are read-only; pin it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["kafka", "kinesis", "pubsub", "pulsar", "eventhubs"])
def test_broker_sources_expose_no_destructive_write_path(name):
    """No `write_partitioned` means no disposition to hand destructively to every shard.

    Pinned rather than assumed: adding a broker *sink* later reopens the class-5 defect, and
    this test is what makes that visible at the moment it is added.
    """
    from batcher.io.formats.base import SINKS, SOURCES

    source = SOURCES.get(name)("t")
    assert not hasattr(source, "write_partitioned")
    assert not hasattr(source, "write")
    assert name not in SINKS


# --------------------------------------------------------------------------
# 6. Offset/checkpoint correctness
# --------------------------------------------------------------------------
class _SpyPulsarConsumer:
    """Models `pulsar.Consumer`: receive / acknowledge / seek / close, with a call log."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.acked: list[Any] = []
        self.calls: list[str] = []
        self.closed = 0
        self.sought: list[Any] = []

    def receive(self, timeout_millis: int = 1000) -> Any:
        if not self._messages:
            raise _PulsarTimeout
        self.calls.append("receive")
        return self._messages.pop(0)

    def acknowledge(self, msg: Any) -> None:
        self.calls.append("acknowledge")
        self.acked.append(msg)

    def seek(self, message_id: Any) -> None:
        self.calls.append("seek")
        self.sought.append(message_id)

    def close(self) -> None:
        self.closed += 1


class _PulsarTimeout(Exception):
    """Stands in for `pulsar.Timeout`."""


class _SpyMessageId:
    def __init__(self, ledger: int, entry: int) -> None:
        self._ledger, self._entry = ledger, entry

    def ledger_id(self) -> int:
        return self._ledger

    def entry_id(self) -> int:
        return self._entry

    def serialize(self) -> bytes:
        return f"{self._ledger}:{self._entry}".encode()


class _SpyPulsarMessage:
    def __init__(self, data: bytes, ledger: int, entry: int) -> None:
        self._data, self._mid = data, _SpyMessageId(ledger, entry)

    def data(self) -> bytes:
        return self._data

    def message_id(self) -> _SpyMessageId:
        return self._mid

    def publish_timestamp(self) -> int:
        return 0

    def partition_key(self) -> str:
        return ""


class _SpyPulsarClient:
    """Models `pulsar.Client` — only `close` matters here; `subscribe` is pre-empted."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _fake_message_id_module():
    """A stand-in `pulsar` module exposing the two names the source imports off it."""
    import types

    class _MessageId:
        @staticmethod
        def deserialize(raw: bytes) -> _SpyMessageId:
            ledger, _, entry = raw.decode().partition(":")
            return _SpyMessageId(int(ledger), int(entry))

    return types.SimpleNamespace(Timeout=_PulsarTimeout, MessageId=_MessageId)


def _pulsar_source(monkeypatch, consumer: _SpyPulsarConsumer):
    """A `PulsarSource` wired to `consumer`, with the `pulsar` module faked out.

    The client handles are seeded directly rather than monkeypatched: `BrokerSource` uses
    `__slots__`, so per-instance attribute patching is not possible — and seeding them is
    also what makes `_client()` return the spy without ever importing the real driver.
    """
    from batcher.io.formats.streaming import pulsar as mod

    monkeypatch.setattr(mod, "_import_pulsar", _fake_message_id_module)
    source = mod.PulsarSource("t", poll_size=2)
    source._client_obj = _SpyPulsarClient()
    source._consumer = consumer
    return source


def test_pulsar_does_not_acknowledge_before_the_epoch_is_published(monkeypatch):
    """Acking at poll time is at-most-once: a crash before the publish drops the messages.

    The generator is paused at its `yield` — the batch is assembled but the engine has not
    staged, logged, or published it. Nothing may be acked at that point.
    """
    consumer = _SpyPulsarConsumer([_SpyPulsarMessage(b"a", 1, 1), _SpyPulsarMessage(b"b", 1, 2)])
    source = _pulsar_source(monkeypatch, consumer)

    it = source.iter_batches()
    batch = next(it)  # assembled and yielded, NOT yet published
    assert batch.num_rows == 2
    assert consumer.acked == [], "acknowledged at poll time — a crash here loses the batch"


def test_pulsar_acknowledges_once_the_consumer_asks_for_the_next_batch(monkeypatch):
    """Control returns past the `yield` only after the epoch is published — the ack moment.

    Asking for a *second* batch is what drives the generator past its first `yield`, which
    is the moment `_commit_delivered` runs. The assertion is that epoch one's messages — and
    only those — are acked by then: the ack tracks the publish exactly one epoch behind.
    """
    epoch1 = [_SpyPulsarMessage(b"a", 1, 1), _SpyPulsarMessage(b"b", 1, 2)]
    epoch2 = [_SpyPulsarMessage(b"c", 1, 3), _SpyPulsarMessage(b"d", 1, 4)]
    consumer = _SpyPulsarConsumer([*epoch1, *epoch2])
    source = _pulsar_source(monkeypatch, consumer)

    it = source.iter_batches()
    next(it)
    next(it)  # publishes epoch 1, then polls epoch 2
    assert consumer.acked == epoch1, "ack did not track the publish"


def test_pulsar_seek_repositions_the_consumer(monkeypatch):
    """`_apply_seek` was unimplemented, so `seek` recorded a position nothing ever applied.

    A restart then resumed from wherever the subscription sat, silently ignoring the
    checkpoint — the source claimed `Checkpointable` while not honoring a checkpoint.
    """
    consumer = _SpyPulsarConsumer([])
    source = _pulsar_source(monkeypatch, consumer)
    source.seek({"offsets": {"0": _token(7, 9)}})
    assert consumer.sought, "seek() did not reposition the consumer"


def _token(ledger: int, entry: int) -> str:
    import base64

    return base64.b64encode(_SpyMessageId(ledger, entry).serialize()).decode("ascii")


def test_pulsar_checkpoint_round_trips_a_native_message_id(monkeypatch):
    """The int64 `offset` folds (ledger, entry) lossily, so it cannot be the resume token.

    The snapshot must carry the exact `MessageId`, and it must be JSON-safe — the checkpoint
    log is written as JSON, so raw `serialize()` bytes could not go in.
    """
    import json

    consumer = _SpyPulsarConsumer([_SpyPulsarMessage(b"a", 7, 9)])
    source = _pulsar_source(monkeypatch, consumer)
    next(source.iter_batches())

    position = source.snapshot_position()
    json.dumps(position)  # must not raise — the checkpoint log is JSON
    assert position["offsets"]["0"] == _token(7, 9)


def test_pulsar_closes_its_client_when_the_generator_is_abandoned(monkeypatch):
    """A generator dropped after its first yield runs its `finally` only at GC — or never."""
    consumer = _SpyPulsarConsumer([_SpyPulsarMessage(b"a", 1, 1)])
    source = _pulsar_source(monkeypatch, consumer)

    it = source.iter_batches()
    next(it)
    it.close()  # the consumer walks away, as a trigger or an upstream error does
    assert consumer.closed == 1, "abandoning the generator leaked the broker connection"


class _SpyPubSubClient:
    """Models `SubscriberClient`: pull / acknowledge, with a call log."""

    def __init__(self, batches: list[list[str]]) -> None:
        self._batches = list(batches)
        self.acked: list[str] = []

    def pull(self, request: dict) -> Any:
        import datetime
        import types

        ids = self._batches.pop(0) if self._batches else []
        received = [
            types.SimpleNamespace(
                ack_id=f"ack-{i}",
                message=types.SimpleNamespace(
                    data=i.encode(),
                    message_id=i,
                    publish_time=datetime.datetime(2024, 1, 1),
                    ordering_key="",
                ),
            )
            for i in ids
        ]
        return types.SimpleNamespace(received_messages=received)

    def acknowledge(self, request: dict) -> None:
        self.acked.extend(request["ack_ids"])


def _pubsub_source(client: _SpyPubSubClient):
    """A `PubSubSource` whose client handle is seeded with the spy (see `_pulsar_source`)."""
    from batcher.io.formats.streaming import pubsub as mod

    source = mod.PubSubSource("projects/p/subscriptions/s", poll_size=2)
    source._client_obj = client
    return source


def test_pubsub_does_not_acknowledge_before_the_epoch_is_published():
    """Acking at pull time means no redelivery — a crash before the publish loses the data."""
    client = _SpyPubSubClient([["m1", "m2"]])
    source = _pubsub_source(client)

    it = source.iter_batches()
    next(it)
    assert client.acked == [], "acknowledged at pull time — a crash here loses the batch"


def test_pubsub_acknowledges_after_the_publish():
    """The ack lands once the consumer asks for the next batch, i.e. after the publish.

    Epoch one's ack ids — and only those — are sent by the time epoch two is pulled, so the
    ack trails the publish by exactly one epoch.
    """
    client = _SpyPubSubClient([["m1", "m2"], ["m3", "m4"]])
    source = _pubsub_source(client)

    it = source.iter_batches()
    next(it)
    next(it)  # publishes epoch 1, then pulls epoch 2
    assert client.acked == ["ack-m1", "ack-m2"], "ack did not track the publish"


def test_pubsub_offsets_are_stable_across_processes():
    """`abs(hash(message_id))` is salted per process, so the same message got a new offset.

    That column is what downstream de-dup and ordering key off, so it silently stopped being
    comparable across restart and across workers — the two boundaries that matter.
    """
    script = (
        "from batcher.io.formats.streaming.pubsub import _stable_offset;"
        "print(_stable_offset('projects/p/messages/12345'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, f"offset differs between processes: {runs}"


def test_kinesis_non_numeric_sequence_offset_is_stable_across_processes():
    """The same salted-`hash()` hazard on the Kinesis sequence-number fallback path."""
    script = (
        "from batcher.io.formats.streaming.kinesis import _seq_to_offset;"
        "print(_seq_to_offset('not-a-number'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, f"offset differs between processes: {runs}"


def test_kinesis_numeric_sequence_still_preserves_ordering():
    """The stable-hash fix must not disturb the ordinary numeric path."""
    from batcher.io.formats.streaming.kinesis import _seq_to_offset

    assert _seq_to_offset("100") < _seq_to_offset("200")


# --------------------------------------------------------------------------
# Resource lifetime — the base contract every broker inherits
# --------------------------------------------------------------------------
def test_iter_batches_closes_the_source_when_abandoned():
    """The `finally` must run on `close()`, not merely at garbage collection."""
    broker = _UnboundedBroker()
    it = broker.iter_batches()
    next(it)
    it.close()
    assert broker.closed == 1


def test_iter_batches_closes_the_source_on_an_exception():
    """An upstream error must not leak the connection either."""
    broker = _UnboundedBroker()
    it = broker.iter_batches()
    next(it)
    with pytest.raises(RuntimeError):
        it.throw(RuntimeError("upstream failed"))
    assert broker.closed == 1


def test_bounded_read_closes_the_source():
    """`read()` drains the generator to exhaustion, which must also run the cleanup."""
    broker = _BoundedBroker()
    broker.read()
    assert broker.closed == 1


class _SpyKafkaConsumer:
    """Models the `confluent_kafka.Consumer` surface this source drives."""

    def __init__(self) -> None:
        self.closed = 0
        self.commits = 0

    def commit(self, asynchronous: bool = False) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed += 1


def test_kafka_closes_its_consumer_when_the_generator_is_abandoned():
    """`Consumer.close()` also commits finally and leaves the group cleanly.

    Skipping it leaks more than a socket: the group waits out `session.timeout.ms` before
    rebalancing, stalling the very partitions this consumer held.
    """
    from batcher.io.formats.streaming.kafka import KafkaSource

    source = KafkaSource("t", bootstrap_servers="k:9092")
    consumer = _SpyKafkaConsumer()
    source._consumer = consumer
    source.close()
    assert consumer.closed == 1
    assert source._consumer is None, "close() must be idempotent — a second close would raise"


def test_eventhubs_closes_its_client():
    """The AMQP link stayed open until GC, which for a reference cycle never happens."""
    from batcher.io.formats.streaming.eventhubs import EventHubsSource

    source = EventHubsSource("hub", connection_str="Endpoint=sb://h/;SharedAccessKey=PW_SECRET")
    client = _SpyKafkaConsumer()  # same close()/counter shape
    source._client_obj = client
    source.close()
    assert client.closed == 1
    assert source._client_obj is None


def test_close_is_safe_on_a_source_that_never_opened_a_client():
    """`iter_batches` closes from a `finally`, which runs even if the first poll raised."""
    from batcher.io.formats.streaming.eventhubs import EventHubsSource
    from batcher.io.formats.streaming.kafka import KafkaSource
    from batcher.io.formats.streaming.pulsar import PulsarSource

    for source in (KafkaSource("t"), EventHubsSource("h"), PulsarSource("t")):
        source.close()  # must not raise, and must not dial a broker
