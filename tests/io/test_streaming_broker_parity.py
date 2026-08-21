"""Contracts every broker owes, held on the brokers that did not keep them.

Three of the five accepted an option and did nothing with it, which is the failure mode this
file exists for: `poll_bytes` bounded a poll on Kafka and Pulsar and was silently ignored on
Kinesis, Pub/Sub and Event Hubs; `include_headers` produced a column of nulls on every
broker but Kafka, though three of them carry the metadata under another name; and the
Pub/Sub source never closed the subscriber it opened.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource, as_header_pairs

pytestmark = pytest.mark.unit


# --- the shared budget -----------------------------------------------------


class _Broker(BrokerSource):
    format_name = "parity_test_broker"

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        return []


def test_the_budget_starts_at_the_configured_poll_bytes():
    assert _Broker("t", poll_bytes=4096)._poll_budget().remaining == 4096


def test_a_budget_is_spent_when_its_allowance_runs_out():
    budget = _Broker("t", poll_bytes=10)._poll_budget()
    budget.spend(4)
    assert not budget.spent
    budget.spend(6)
    assert budget.spent


def test_the_sweep_order_rotates_so_no_partition_is_starved():
    """Always stopping at the same place starves the tail — and under a per-partition
    watermark a starved partition stalls the whole stream's frontier, not just its own."""
    source = _Broker("t")
    orders = [source._poll_budget().order([0, 1, 2]) for _ in range(4)]
    assert orders == [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]]


def test_a_single_partition_sweep_is_left_alone():
    assert _Broker("t")._poll_budget().order([7]) == [7]


# --- header normalization --------------------------------------------------


def test_metadata_of_any_value_type_becomes_bytes():
    """Four clients, four value types: Pulsar and Pub/Sub hand back str, Event Hubs
    whatever was published. The column's type is binary, so one shape reaches it."""
    assert as_header_pairs({"trace": "abc", "n": 7, "raw": b"x"}) == [
        ("trace", b"abc"),
        ("n", b"7"),
        ("raw", b"x"),
    ]


def test_a_bytes_key_is_decoded_because_the_columns_key_is_a_string():
    assert as_header_pairs({b"k": "v"}) == [("k", b"v")]


def test_no_metadata_is_none_not_an_empty_list():
    """The same null-versus-empty distinction Kafka's headers already draw, so "this
    message had none" stays distinguishable from "this broker does not carry them"."""
    assert as_header_pairs(None) is None
    assert as_header_pairs({}) is None


# --- Pub/Sub ---------------------------------------------------------------


class _FakeMessage:
    def __init__(self, data: bytes, attributes: dict | None = None) -> None:
        self.data = data
        self.attributes = attributes or {}
        self.message_id = "m1"
        self.ordering_key = ""

        class _T:
            @staticmethod
            def timestamp():
                return 0.0

        self.publish_time = _T()


class _FakeReceived:
    def __init__(self, message: _FakeMessage, ack_id: str) -> None:
        self.message = message
        self.ack_id = ack_id


class _FakeResponse:
    def __init__(self, received) -> None:
        self.received_messages = list(received)


class _FakeSubscriber:
    def __init__(self, response) -> None:
        self._response = response
        self.closed = 0

    def pull(self, request, timeout):
        return self._response

    def close(self):
        self.closed += 1


def _pubsub(response, **kwargs):
    from batcher.io.formats.streaming.pubsub import PubSubSource

    source = PubSubSource("sub", **kwargs)
    source._client_obj = _FakeSubscriber(response)
    return source


def test_pubsub_bounds_a_pull_by_payload_bytes():
    """Accepted and ignored, a subscription of large messages built a batch bounded only by
    poll_size times the 10 MiB message limit."""
    response = _FakeResponse([_FakeReceived(_FakeMessage(b"x" * 100), f"a{i}") for i in range(10)])
    source = _pubsub(response, poll_bytes=250)
    assert len(source._poll()) == 3


def test_a_message_larger_than_the_whole_budget_is_still_delivered():
    """Dropping it would re-pull it forever: a stalled subscription, not a bounded one."""
    response = _FakeResponse([_FakeReceived(_FakeMessage(b"x" * 5000), "a0")])
    assert len(_pubsub(response, poll_bytes=100)._poll()) == 1


def test_pubsub_only_holds_ack_ids_for_the_messages_it_kept():
    """What is trimmed must be redelivered, which happens only if it is never acked."""
    response = _FakeResponse([_FakeReceived(_FakeMessage(b"x" * 100), f"a{i}") for i in range(10)])
    source = _pubsub(response, poll_bytes=250)
    source._poll()
    assert source._pending_acks == ["a0", "a1", "a2"]


def test_pubsub_carries_message_attributes_when_asked():
    response = _FakeResponse([_FakeReceived(_FakeMessage(b"v", {"trace": "abc"}), "a0")])
    assert _pubsub(response, include_headers=True)._poll()[0].headers == [("trace", b"abc")]


def test_pubsub_does_not_pay_for_attributes_it_was_not_asked_for():
    response = _FakeResponse([_FakeReceived(_FakeMessage(b"v", {"trace": "abc"}), "a0")])
    assert _pubsub(response)._poll()[0].headers is None


def test_pubsub_closes_the_subscriber_it_opened():
    """`SubscriberClient` owns a gRPC channel and transport threads, and nothing released
    either — so a driver that restarts a query leaked both, every restart."""
    source = _pubsub(_FakeResponse([]))
    client = source._client_obj
    source.close()
    assert client.closed == 1


def test_closing_pubsub_twice_is_safe():
    """`iter_batches` closes from a `finally`, so a second call is guaranteed."""
    source = _pubsub(_FakeResponse([]))
    source.close()
    source.close()


def test_a_pubsub_close_that_raises_still_drops_the_handle():
    """Otherwise the next close — which `iter_batches` guarantees — re-closes a dead client
    and raises again, this time out of a `finally` where it masks the original error."""

    class _Angry(_FakeSubscriber):
        def close(self):
            raise RuntimeError("broker gone")

    source = _pubsub(_FakeResponse([]))
    source._client_obj = _Angry(_FakeResponse([]))
    with pytest.raises(RuntimeError):
        source.close()
    source.close()  # the handle was dropped, so this is a no-op rather than a second raise


def test_an_unopened_pubsub_source_closes_cleanly():
    from batcher.io.formats.streaming.pubsub import PubSubSource

    PubSubSource("sub").close()


# --- Event Hubs ------------------------------------------------------------


def test_event_hubs_carries_application_properties_when_asked():
    from batcher.io.formats.streaming.eventhubs import _event_to_message

    class _Event:
        offset = "42"
        partition_key = None
        properties: ClassVar[dict] = {"trace": "abc"}
        body = b"v"

        @staticmethod
        def body_as_str():  # pragma: no cover - the bytes path is taken first
            return "v"

    message = _event_to_message(_Event(), 0, "hub", include_headers=True)
    assert message.headers == [("trace", b"abc")]


def test_event_hubs_leaves_properties_alone_when_not_asked():
    from batcher.io.formats.streaming.eventhubs import _event_to_message

    class _Event:
        offset = "42"
        partition_key = None
        properties: ClassVar[dict] = {"trace": "abc"}
        body = b"v"

    assert _event_to_message(_Event(), 0, "hub").headers is None


# --- Kinesis ---------------------------------------------------------------


def test_kinesis_stops_decoding_shards_once_the_batch_is_full():
    """The unread shard's iterator is deliberately not advanced, so its records are fetched
    again next epoch rather than lost."""
    from batcher.io.formats.streaming.kinesis import KinesisSource

    class _StubbedKinesis(KinesisSource):
        """Kinesis with the three client-facing steps replaced, so no AWS call is made.

        A subclass rather than attribute assignment because `KinesisSource` uses `__slots__`,
        which is what makes the real source cheap per poll.
        """

        advanced: ClassVar[list[str]] = []

        def _active_shards(self):
            return [(0, "shard-0"), (1, "shard-1")]

        def _get_records(self, shards):
            return [{"Records": []} for _ in shards]

        def _advance(self, shard_id, resp):
            type(self).advanced.append(shard_id)

        def _decode(self, shard_number, resp):
            return [
                BrokerMessage(
                    value=b"x" * 100,
                    partition=shard_number,
                    offset=0,
                    timestamp=0,
                    topic="s",
                )
            ]

    _StubbedKinesis.advanced = []
    messages = _StubbedKinesis("stream", partitions=[0, 1], poll_bytes=10)._poll()

    assert len(messages) == 1, "the second shard is over budget and is left for next epoch"
    assert len(_StubbedKinesis.advanced) == 1, (
        "and its iterator is not advanced, so nothing is skipped"
    )
