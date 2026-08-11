"""Pub/Sub and Event Hubs: API request limits, consumer reuse, and idle-poll cost.

Each contract here is about what the source *asks the service for* — a request the API
would reject, a link renegotiated per poll, a wait paid once per partition — none of which a
result comparison can see. Driven against fakes that model the client surfaces used.
"""

from __future__ import annotations

import datetime

import pytest

from batcher.io.formats.streaming.eventhubs import (
    _DRAIN_WAIT_SECONDS,
    _FIRST_WAIT_SECONDS,
    EventHubsSource,
)
from batcher.io.formats.streaming.pubsub import _ACK_CHUNK, _PULL_MAX_MESSAGES, PubSubSource


# --------------------------------------------------------------------------
# Pub/Sub: the API's own request limits.
# --------------------------------------------------------------------------
class _FakeSubscriber:
    def __init__(self) -> None:
        self.pulls: list[dict] = []
        self.acks: list[list[str]] = []
        self.messages: list = []

    def pull(self, request, timeout):
        self.pulls.append(request)
        return type("Resp", (), {"received_messages": list(self.messages)})()

    def acknowledge(self, request):
        ids = list(request["ack_ids"])
        if len(ids) > _ACK_CHUNK:
            raise ValueError("request too large")  # what the real API answers
        self.acks.append(ids)


def _received(n: int) -> list:
    class _Msg:
        def __init__(self, i: int) -> None:
            self.data = b"v"
            self.message_id = f"m{i}"
            self.ordering_key = ""
            self.publish_time = type("T", (), {"timestamp": lambda self: 0.0})()

    return [type("RM", (), {"message": _Msg(i), "ack_id": f"ack-{i}"})() for i in range(n)]


def test_a_pull_never_asks_for_more_than_the_api_allows():
    """Pub/Sub caps a synchronous pull at 1,000 and rejects a larger request outright, so
    the engine's 16,384-row default poll size was never a legal request."""
    client = _FakeSubscriber()
    src = PubSubSource("projects/p/subscriptions/s", poll_size=16_384)
    src._client_obj = client
    src._poll()
    assert client.pulls[0]["max_messages"] == _PULL_MAX_MESSAGES


def test_a_smaller_poll_size_is_passed_through_unchanged():
    client = _FakeSubscriber()
    src = PubSubSource("projects/p/subscriptions/s", poll_size=50)
    src._client_obj = client
    src._poll()
    assert client.pulls[0]["max_messages"] == 50


def test_acks_are_chunked_so_the_commit_does_not_exceed_the_request_bound():
    """A whole poll's ack ids in one request exceeds the 512 KiB bound and fails the
    *commit* — so the epoch published and then every message came back on the next poll."""
    client = _FakeSubscriber()
    client.messages = _received(_ACK_CHUNK * 2 + 7)
    src = PubSubSource("projects/p/subscriptions/s", poll_size=_PULL_MAX_MESSAGES)
    src._client_obj = client

    assert len(src._poll()) == _ACK_CHUNK * 2 + 7
    src._commit_delivered()

    assert [len(chunk) for chunk in client.acks] == [_ACK_CHUNK, _ACK_CHUNK, 7]
    flat = [ack for chunk in client.acks for ack in chunk]
    assert flat == [f"ack-{i}" for i in range(_ACK_CHUNK * 2 + 7)]
    assert src._pending_acks == []


def test_a_commit_with_nothing_pending_sends_no_request():
    client = _FakeSubscriber()
    src = PubSubSource("projects/p/subscriptions/s")
    src._client_obj = client
    src._commit_delivered()
    assert client.acks == []


# --------------------------------------------------------------------------
# Event Hubs: one consumer per partition, not one per poll.
# --------------------------------------------------------------------------
class _FakeEvent:
    """One `EventData` as `azure-eventhub` actually shapes it.

    The property names are the load-bearing part of this fake. The source used to read
    ``body_as_bytes()`` and ``enqueued_time_utc_ms``, **neither of which exists** on any
    released `azure-eventhub` — and the old fake defined them, so the suite proved the
    connector agreed with a client that was never there. Modeling the real surface is what
    turns "can't make progress at all" into a failing test.
    """

    def __init__(self, body: object, offset: str = "1", partition_key: bytes | None = None):
        self.body = body
        self.offset = offset
        self.partition_key = partition_key
        self.enqueued_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)


class _FakeConsumer:
    """An `EventHubConsumer` stand-in: `receive` returns None and delivers via the callback."""

    def __init__(self, partition_id: str, position: str, on_event_received) -> None:
        self.partition_id = partition_id
        self.position = position
        self.waits: list[float] = []
        self.closed = 0
        self._deliver = on_event_received
        #: Events the hub is holding for this partition; handed over on the next `receive`.
        self.pending: list[_FakeEvent] = []

    def receive(self, batch, max_batch_size, max_wait_time):
        assert batch is True, "the source must ask for batched delivery"
        self.waits.append(max_wait_time)
        ready, self.pending = self.pending[:max_batch_size], self.pending[max_batch_size:]
        if ready:
            self._deliver(ready)
        return None  # the real client returns nothing; the callback is the only channel

    def close(self):
        self.closed += 1


class _FakeHubClient:
    def __init__(self, partitions: list[str]) -> None:
        self._partitions = partitions
        self.created: list[_FakeConsumer] = []
        self.partition_id_calls = 0
        self.closed = 0

    def get_partition_ids(self):
        self.partition_id_calls += 1
        return list(self._partitions)

    def _create_consumer(self, consumer_group, partition_id, event_position, on_event_received):
        consumer = _FakeConsumer(partition_id, event_position, on_event_received)
        self.created.append(consumer)
        return consumer

    def close(self):
        self.closed += 1


def _hub(partitions: list[str]) -> tuple[EventHubsSource, _FakeHubClient]:
    src = EventHubsSource("hub", connection_str="Endpoint=sb://x")
    client = _FakeHubClient(partitions)
    src._client_obj = client
    return src, client


def test_consumers_are_opened_once_and_reused_across_polls():
    """A consumer per partition per poll is a full AMQP link negotiation before a single
    event is read — eighty a second on a 100ms trigger over eight partitions — and it throws
    away the consumer's own prefetch, so every poll started with an empty local buffer."""
    src, client = _hub(["0", "1", "2"])
    for _ in range(5):
        src._poll()
    assert len(client.created) == 3
    assert all(c.closed == 0 for c in client.created)


def test_the_partition_list_is_fetched_once_not_per_poll():
    src, client = _hub(["0", "1"])
    for _ in range(4):
        src._poll()
    assert client.partition_id_calls == 1


def test_only_the_first_partition_of_a_poll_pays_the_wait():
    """A serial loop waiting `max_wait_time` per partition made the effective trigger
    cadence `partitions x wait` — eight seconds of latency for a hub with nothing to say."""
    src, client = _hub(["0", "1", "2"])
    src._poll()
    waits = [c.waits[0] for c in client.created]
    assert waits == [_FIRST_WAIT_SECONDS, _DRAIN_WAIT_SECONDS, _DRAIN_WAIT_SECONDS]


def test_a_seek_reopens_the_partition_at_the_checkpointed_offset():
    """A consumer already open sits at its old position, so recovery would otherwise keep
    reading from wherever the pre-crash consumer had reached."""
    src, client = _hub(["0", "1"])
    src._poll()
    first = list(client.created)
    assert first[0].position == "-1"  # the configured start

    src.seek({"offsets": {"0": "12345"}})
    assert first[0].closed == 1  # partition 0's consumer was dropped
    assert first[1].closed == 0  # partition 1 was untouched

    src._poll()
    reopened = [c for c in client.created if c not in first]
    assert [c.partition_id for c in reopened] == ["0"]
    assert reopened[0].position == "12345"


def test_close_releases_every_consumer_and_then_the_client():
    src, client = _hub(["0", "1"])
    src._poll()
    src.close()
    assert all(c.closed == 1 for c in client.created)
    assert client.closed == 1


# --------------------------------------------------------------------------
# Event Hubs: the source actually delivers the events the client hands it.
# --------------------------------------------------------------------------
def test_polled_events_reach_the_batch():
    """The whole source was a no-op: `receive_message_batch` exists on no released client,
    and the `on_event_received` callback it registered was ``lambda *_: None``, so every
    event the client fetched was discarded. A hub with data produced nothing, forever."""
    src, client = _hub(["0"])
    src._poll()  # opens the consumer
    client.created[0].pending = [_FakeEvent(b"one", "11"), _FakeEvent(b"two", "12")]

    messages = src._poll()

    assert [m.value for m in messages] == [b"one", b"two"]
    assert [m.partition for m in messages] == [0, 0]
    assert [m.resume_token for m in messages] == ["11", "12"]
    assert [m.offset for m in messages] == [11, 12]
    assert all(m.topic == "hub" for m in messages)


def test_an_events_enqueued_time_becomes_the_timestamp_column():
    """`EventData` carries `enqueued_time` (a datetime), never `enqueued_time_utc_ms`."""
    src, client = _hub(["0"])
    src._poll()
    client.created[0].pending = [_FakeEvent(b"x")]

    (message,) = src._poll()

    expected = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC).timestamp() * 1000
    assert message.timestamp == int(expected)


def test_a_multi_section_data_body_is_joined_not_stringified():
    """A DATA body is `bytes` *or an iterable of bytes*, one element per AMQP data section.
    Assuming the scalar shape turns a chunked payload into the `repr` of a list."""
    src, client = _hub(["0"])
    src._poll()
    client.created[0].pending = [_FakeEvent([b"ab", b"cd"])]

    (message,) = src._poll()

    assert message.value == b"abcd"


def test_a_non_numeric_offset_does_not_kill_the_query():
    """An Event Hubs offset is typed `str` and only conventionally numeric; `int(...)` on a
    token that is not raised `ValueError` out of the poll loop."""
    src, client = _hub(["0"])
    src._poll()
    client.created[0].pending = [_FakeEvent(b"x", offset="abc")]

    (message,) = src._poll()

    assert message.resume_token == "abc"  # the seekable position is preserved verbatim
    assert isinstance(message.offset, int)


def test_an_event_batch_becomes_one_arrow_batch_with_the_broker_schema():
    """End to end: the messages a poll produces assemble into the fixed broker schema."""
    src, client = _hub(["0"])
    src._poll()
    client.created[0].pending = [_FakeEvent(b"one", "11"), _FakeEvent(b"two", "12")]

    batch = src._make_batch(src._poll())

    assert batch.num_rows == 2
    assert batch.column("value").to_pylist() == [b"one", b"two"]
    src.close()  # idempotent
    assert client.closed == 1


def test_close_drops_the_client_handle_even_when_it_raises():
    src, client = _hub(["0"])
    src._poll()

    def angry():
        client.closed += 1
        raise RuntimeError("amqp gone")

    client.close = angry
    with pytest.raises(RuntimeError, match="amqp gone"):
        src.close()
    src.close()  # the handle was dropped before the failing call
    assert client.closed == 1
