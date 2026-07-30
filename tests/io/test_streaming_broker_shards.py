"""Kinesis shard handling and Pulsar drain/seek contracts, against fake clients.

These pin behaviors that only show up against a live service and only as silence: a reader
that goes permanently quiet after a reshard, a query killed by a five-minute iterator
expiry, a poll whose cost scales with the shard count, and a recovery that repositions the
wrong partition.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.io.formats.streaming.kinesis import (
    KinesisSource,
    _is_expired_iterator,
    _is_throttle,
)
from batcher.io.formats.streaming.pulsar import _ENTRY_ID_BITS, _message_id_to_offset


class _Expired(Exception):
    pass


class _Throttled(Exception):
    pass


_Expired.__name__ = "ExpiredIteratorException"
_Throttled.__name__ = "ProvisionedThroughputExceededException"


class _FakeKinesis:
    """Models the four `boto3` kinesis calls the source drives."""

    def __init__(self, shards: list[str]) -> None:
        self.shards = list(shards)
        self.iterator_requests: list[dict] = []
        self.get_records_calls: list[str] = []
        self.list_shards_calls = 0
        self.raise_on: dict[str, Exception] = {}
        self.closed: set[str] = set()

    def list_shards(self, **kwargs):
        self.list_shards_calls += 1
        return {"Shards": [{"ShardId": s} for s in self.shards]}

    def get_shard_iterator(self, **kwargs):
        self.iterator_requests.append(kwargs)
        return {"ShardIterator": f"iter::{kwargs['ShardId']}"}

    def get_records(self, ShardIterator, Limit):
        shard = ShardIterator.split("::", 1)[1]
        self.get_records_calls.append(shard)
        err = self.raise_on.pop(shard, None)
        if err is not None:
            raise err
        record = {
            "Data": b"d",
            "SequenceNumber": "42",
            "ApproximateArrivalTimestamp": None,
            "PartitionKey": "k",
        }
        if shard in self.closed:
            return {"Records": [record], "NextShardIterator": None}
        return {"Records": [record], "NextShardIterator": ShardIterator}


def _kinesis(client: _FakeKinesis, **kwargs) -> KinesisSource:
    src = KinesisSource("stream", **kwargs)
    src._client_obj = client
    return src


# --------------------------------------------------------------------------
# Reshard: retiring a parent must not leave the reader permanently quiet.
# --------------------------------------------------------------------------
def test_a_closed_shard_invalidates_the_cached_shard_list():
    """A shard closes only because a reshard replaced it with children.

    The children are absent from the cached shard list, and that cache lived for the whole
    run — so retiring the parent without dropping the cache made the reader go silent on the
    resharded key range forever, looking exactly like an idle stream.
    """
    client = _FakeKinesis(["shardId-000000000000"])
    src = _kinesis(client)
    assert len(src._poll()) == 1
    assert client.list_shards_calls == 1  # cached after the first look

    client.closed.add("shardId-000000000000")
    src._poll()  # the parent drains and retires
    client.shards = ["shardId-000000000001", "shardId-000000000002"]  # its children

    messages = src._poll()
    assert client.list_shards_calls == 2  # the cache was invalidated by the close
    assert sorted(m.partition for m in messages) == [1, 2]


def test_a_closed_shard_is_never_polled_again():
    client = _FakeKinesis(["shardId-000000000000"])
    src = _kinesis(client)
    client.closed.add("shardId-000000000000")
    src._poll()
    client.get_records_calls.clear()
    assert src._poll() == []
    assert client.get_records_calls == []


# --------------------------------------------------------------------------
# Iterator expiry and throttling are routine, not fatal.
# --------------------------------------------------------------------------
def test_expired_and_throttled_are_recognised_by_name_and_by_client_error_code():
    assert _is_expired_iterator(_Expired()) is True
    assert _is_throttle(_Throttled()) is True

    class _ClientError(Exception):
        def __init__(self) -> None:
            super().__init__()
            self.response = {"Error": {"Code": "ExpiredIteratorException"}}

    assert _is_expired_iterator(_ClientError()) is True
    assert _is_throttle(_ClientError()) is False
    assert _is_expired_iterator(ValueError("real")) is False


def test_an_expired_iterator_is_rebuilt_from_the_last_delivered_sequence():
    """A shard iterator lives five minutes. Any trigger slower than that outlives it, and
    letting the expiry escape killed the query; rebuilding from `TRIM_HORIZON` would have
    replayed the whole shard instead."""
    client = _FakeKinesis(["shardId-000000000000"])
    src = _kinesis(client)
    messages = src._poll()
    src._track_positions(messages)  # what `_poll_loop` does after a real poll

    client.raise_on["shardId-000000000000"] = _Expired()
    assert src._poll() == []  # skipped this pass, not fatal

    assert src._poll()  # the next pass rebuilds the iterator and reads again
    assert client.iterator_requests[-1]["ShardIteratorType"] == "AFTER_SEQUENCE_NUMBER"
    assert client.iterator_requests[-1]["StartingSequenceNumber"] == "42"


def test_a_throttled_shard_keeps_its_iterator_and_does_not_stop_its_siblings():
    client = _FakeKinesis(["shardId-000000000000", "shardId-000000000001"])
    src = _kinesis(client)
    client.raise_on["shardId-000000000000"] = _Throttled()
    messages = src._poll()
    assert [m.partition for m in messages] == [1]  # the sibling still delivered
    assert "shardId-000000000000" in src._iterators  # untouched: read again next poll


# --------------------------------------------------------------------------
# Shard fan-out: a poll must not cost one serialized round-trip per shard.
# --------------------------------------------------------------------------
def test_multi_shard_polls_run_concurrently_and_stay_positionally_aligned():
    import threading

    shards = [f"shardId-{i:012d}" for i in range(8)]
    client = _FakeKinesis(shards)
    barrier = threading.Barrier(8, timeout=10)
    plain_get_records = client.get_records

    def blocking_get_records(ShardIterator, Limit):
        barrier.wait()  # deadlocks unless all eight calls are genuinely in flight at once
        return plain_get_records(ShardIterator, Limit)

    client.get_records = blocking_get_records
    src = _kinesis(client)
    messages = src._poll()
    assert sorted(m.partition for m in messages) == list(range(8))
    src.close()


def test_a_single_shard_reader_never_spins_up_a_pool():
    client = _FakeKinesis(["shardId-000000000000"])
    src = _kinesis(client)
    src._poll()
    assert src._pool_obj is None
    src.close()  # idempotent with no pool


# --------------------------------------------------------------------------
# Pulsar: message-id folding and multi-partition recovery.
# --------------------------------------------------------------------------
class _MessageId:
    def __init__(self, ledger: int, entry: int) -> None:
        self._ledger, self._entry = ledger, entry

    def ledger_id(self) -> int:
        return self._ledger

    def entry_id(self) -> int:
        return self._entry


def test_two_entries_in_one_ledger_never_fold_to_the_same_offset():
    """Twenty bits held 1,048,576 entries. A BookKeeper ledger routinely holds more, and past
    that the entry id wrapped: two different messages got one offset, silently breaking the
    ordering and de-duplication that column exists for."""
    wrapped = 1 << 20
    assert _message_id_to_offset(_MessageId(7, 5)) != _message_id_to_offset(
        _MessageId(7, 5 + wrapped)
    )
    assert _ENTRY_ID_BITS == 32
    # Still monotonic within a ledger, and still inside int64 for a large ledger id.
    assert _message_id_to_offset(_MessageId(7, 5)) < _message_id_to_offset(_MessageId(7, 6))
    assert _message_id_to_offset(_MessageId(2**30, 1)) < (1 << 63)


def test_an_ambiguous_multi_partition_pulsar_resume_is_refused(monkeypatch):
    """A Pulsar seek repositions the whole consumer. Walking a multi-partition checkpoint
    issued one seek per entry and the last silently won, so every other partition resumed at
    a position belonging to a different partition."""
    from batcher.io.formats.streaming import pulsar as pmod

    src = pmod.PulsarSource("t", num_partitions=4)
    with pytest.raises(PlanError, match="one split per partition"):
        src.seek({"offsets": {"0": "a", "1": "b"}})

    # One partition per consumer — the shape `BrokerSplit` builds — resumes normally.
    seen: list[object] = []
    monkeypatch.setattr(pmod, "_deserialize_message_id", lambda token: token)
    scoped = pmod.PulsarSource("t", partitions=[2])
    scoped._consumer = type("C", (), {"seek": lambda self, mid: seen.append(mid)})()
    scoped._client_obj = object()
    scoped.seek({"offsets": {"2": "msgid"}})
    assert seen == ["msgid"]


def test_a_seek_drops_messages_that_were_polled_but_never_published(monkeypatch):
    """Those handles are about to be delivered again from the new position; acking them
    would acknowledge records the engine never published."""
    from batcher.io.formats.streaming import pulsar as pmod

    monkeypatch.setattr(pmod, "_deserialize_message_id", lambda token: None)
    src = pmod.PulsarSource("t", partitions=[0])
    src._unacked = [object(), object()]
    src.seek({"offsets": {"0": "tok"}})
    assert src._unacked == []


def test_pulsar_close_drops_both_handles_even_when_the_consumer_raises():
    from batcher.io.formats.streaming import pulsar as pmod

    closed: list[str] = []

    class _AngryConsumer:
        def close(self):
            closed.append("consumer")
            raise RuntimeError("broker gone")

    class _Client:
        def close(self):
            closed.append("client")

    src = pmod.PulsarSource("t")
    src._consumer = _AngryConsumer()
    src._client_obj = _Client()
    with pytest.raises(RuntimeError, match="broker gone"):
        src.close()
    # The client is still released, so its IO threads are not stranded by the failure.
    assert closed == ["consumer", "client"]
    src.close()  # idempotent: both handles were dropped before the failing call
    assert closed == ["consumer", "client"]


def test_pulsar_drains_with_a_short_timeout_after_the_first_receive():
    """Charging every `receive` the full timeout meant the one that finds the queue empty —
    on every single poll — cost a full second."""
    from batcher.io.formats.streaming import pulsar as pmod

    class _Timeout(Exception):
        pass

    class _FakePulsar:
        Timeout = _Timeout

    class _Msg:
        def data(self):
            return b"v"

        def message_id(self):
            return _MessageId(1, 1)

        def publish_timestamp(self):
            return 0

        def partition_key(self):
            return ""

    class _Consumer:
        def __init__(self, count):
            self.timeouts: list[int] = []
            self.left = count

        def receive(self, timeout_millis):
            self.timeouts.append(timeout_millis)
            if self.left == 0:
                raise _Timeout()
            self.left -= 1
            return _Msg()

    import batcher.io.formats.streaming.pulsar as target

    original = target._import_pulsar
    target._import_pulsar = lambda: _FakePulsar
    try:
        src = pmod.PulsarSource("t", poll_size=10, receive_timeout_millis=5000)
        src._consumer = _Consumer(3)
        src._client_obj = object()
        assert len(src._poll()) == 3
        assert src._consumer.timeouts == [5000, 1, 1, 1]
    finally:
        target._import_pulsar = original


def test_pulsar_drains_a_whole_buffered_batch_in_one_client_call():
    """`batch_receive()` is one crossing of the Python/C++ boundary for the whole buffer.

    The drain used to be one `receive()` per message, so a poll that collected a full
    16,384-message budget made 16,384 calls plus one more to be told the queue was empty —
    per-message overhead on the latency path, for messages the client already held.
    """
    from batcher.io.formats.streaming import pulsar as pmod

    class _Timeout(Exception):
        pass

    class _FakePulsar:
        Timeout = _Timeout

        class ConsumerBatchReceivePolicy:
            def __init__(self, max_num_message, max_num_bytes, timeout_ms):
                self.args = (max_num_message, max_num_bytes, timeout_ms)

    class _Msg:
        def data(self):
            return b"v"

        def message_id(self):
            return _MessageId(1, 1)

        def publish_timestamp(self):
            return 0

        def partition_key(self):
            return ""

    class _Consumer:
        def __init__(self, buffered):
            self.receives: list[int] = []
            self.batch_calls = 0
            self._buffered = buffered

        def receive(self, timeout_millis):
            self.receives.append(timeout_millis)
            return _Msg()

        def batch_receive(self):
            self.batch_calls += 1
            out, self._buffered = [_Msg()] * self._buffered, 0
            return out

    import batcher.io.formats.streaming.pulsar as target

    original = target._import_pulsar
    target._import_pulsar = lambda: _FakePulsar
    try:
        src = pmod.PulsarSource("t", poll_size=100, receive_timeout_millis=5000)
        src._consumer = _Consumer(40)
        src._client_obj = object()
        assert len(src._poll()) == 41
        # One blocking wait for the first message, one call for the other forty.
        assert src._consumer.receives == [5000]
        assert src._consumer.batch_calls == 1
    finally:
        target._import_pulsar = original


def test_pulsar_sizes_its_batch_policy_for_the_drain():
    """The policy is what bounds `batch_receive`: a poll's budget, and a millisecond before
    it gives up — so an empty buffer costs nothing and the blocking wait stays the `receive`."""
    from batcher.io.formats.streaming import pulsar as pmod

    class _Policy:
        def __init__(self, max_num_message, max_num_bytes, timeout_ms):
            self.args = (max_num_message, max_num_bytes, timeout_ms)

    class _FakePulsar:
        Timeout = RuntimeError
        ConsumerBatchReceivePolicy = _Policy

        class ConsumerType:
            Shared = "shared"

    captured: dict = {}

    class _Client:
        def subscribe(self, topics, **kwargs):
            captured.update(kwargs)
            return object()

    import batcher.io.formats.streaming.pulsar as target

    original = target._import_pulsar
    target._import_pulsar = lambda: _FakePulsar
    try:
        src = pmod.PulsarSource("t", poll_size=512)
        src._client_obj = _Client()
        src._client()
    finally:
        target._import_pulsar = original
    assert captured["batch_receive_policy"].args == (
        512,
        src.poll_bytes,
        pmod._DRAIN_TIMEOUT_MILLIS,
    )


def test_pulsar_bounds_its_batch_by_bytes_at_the_client_not_afterwards():
    """An unacked Pulsar message is redelivered only after `ackTimeout`, so trimming an
    over-large received batch would reorder the partition rather than defer it. The bound
    therefore has to be the policy's, so the client stops filling."""
    from batcher.io.formats.streaming import pulsar as pmod

    class _Policy:
        def __init__(self, max_num_message, max_num_bytes, timeout_ms):
            self.args = (max_num_message, max_num_bytes, timeout_ms)

    class _FakePulsar:
        Timeout = RuntimeError
        ConsumerBatchReceivePolicy = _Policy

        class ConsumerType:
            Shared = "shared"

    captured: dict = {}

    class _Client:
        def subscribe(self, topics, **kwargs):
            captured.update(kwargs)
            return object()

    import batcher.io.formats.streaming.pulsar as target

    original = target._import_pulsar
    target._import_pulsar = lambda: _FakePulsar
    try:
        src = pmod.PulsarSource("t", poll_size=16_384, poll_bytes=4 << 20)
        src._client_obj = _Client()
        src._client()
    finally:
        target._import_pulsar = original
    assert captured["batch_receive_policy"].args[1] == 4 << 20
