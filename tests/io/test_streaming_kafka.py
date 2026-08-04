"""Kafka source contracts: poll latency, record errors, metadata, commit, reader reuse.

None of these need a broker. They drive `KafkaSource` against a fake consumer that models
the `confluent_kafka.Consumer` surface the source actually uses, which is enough to pin the
behaviors that a live-broker test would only observe as "slow" or "quiet".
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.streaming.broker import (
    BrokerMessage,
    BrokerSource,
    BrokerSplit,
    redact_broker_options,
)
from batcher.io.formats.streaming.broker.split import _EPOCH_READERS
from batcher.io.formats.streaming.kafka import (
    _NO_OFFSET,
    _PARTITION_EOF,
    KafkaSource,
    _is_benign_record_error,
    _is_no_offset,
)


class _Err:
    """A stand-in for `confluent_kafka.KafkaError`."""

    def __init__(self, code: int, retriable: bool = False, text: str = "boom") -> None:
        self._code = code
        self._retriable = retriable
        self._text = text

    def code(self) -> int:
        return self._code

    def retriable(self) -> bool:
        return self._retriable

    def __str__(self) -> str:
        return self._text


class _Rec:
    """A stand-in for `confluent_kafka.Message`."""

    def __init__(self, offset: int, *, error: _Err | None = None, partition: int = 0) -> None:
        self._offset = offset
        self._error = error
        self._partition = partition

    def error(self):
        return self._error

    def value(self):
        return b"v%d" % self._offset

    def key(self):
        return None

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def timestamp(self):
        return (1, 1000 + self._offset)

    def topic(self):
        return "t"


class _FakeConsumer:
    """Records every `consume` call so the poll *shape* — not just its result — is testable."""

    def __init__(self, batches: list[list[_Rec]]) -> None:
        self.batches = list(batches)
        self.calls: list[tuple[int, float]] = []
        self.closed = 0
        self.commits = 0

    def consume(self, num_messages, timeout):
        self.calls.append((num_messages, timeout))
        return self.batches.pop(0) if self.batches else []

    def commit(self, asynchronous):
        self.commits += 1

    def close(self):
        self.closed += 1


def _source(consumer: _FakeConsumer, **kwargs) -> KafkaSource:
    src = KafkaSource("t", **kwargs)
    src._consumer = consumer  # `_client()` hands this back instead of dialling a broker
    return src


# --------------------------------------------------------------------------
# Poll latency: block for the first record only, then drain without blocking.
# --------------------------------------------------------------------------
def test_poll_blocks_for_one_record_then_drains_with_a_zero_timeout():
    """`consume(N, T)` waits out T for the *N*th message; asking for one inverts that.

    A topic producing a handful of records per trigger used to answer a 16,384-message
    request only when `poll_timeout` expired, so every micro-batch cost a fixed second of
    latency regardless of when the record actually arrived.
    """
    consumer = _FakeConsumer([[_Rec(0)], [_Rec(1), _Rec(2)], []])
    src = _source(consumer, poll_size=100, poll_timeout=0.5)

    messages = src._poll()

    assert [m.offset for m in messages] == [0, 1, 2]
    # Exactly one blocking call, for a single record; every follow-up drain is non-blocking.
    assert consumer.calls[0] == (1, 0.5)
    assert all(timeout == 0 for _, timeout in consumer.calls[1:])


def test_an_empty_poll_costs_exactly_one_blocking_call():
    """Nothing to read must not turn into a drain storm against an idle broker."""
    consumer = _FakeConsumer([])
    src = _source(consumer, poll_size=100, poll_timeout=0.25)
    assert src._poll() == []
    assert consumer.calls == [(1, 0.25)]


def test_the_drain_stops_at_poll_size():
    """A backlogged partition fills the batch and no more, so memory stays bounded."""
    consumer = _FakeConsumer([[_Rec(0)], [_Rec(i) for i in range(1, 50)], [_Rec(99)]])
    src = _source(consumer, poll_size=10, poll_timeout=0.1)
    messages = src._poll()
    assert len(messages) == 50  # a drain pass is never truncated mid-list...
    assert consumer.calls == [(1, 0.1), (9, 0)]  # ...but no further pass is requested


# --------------------------------------------------------------------------
# Record errors: EOF is routine, a real failure must not be swallowed.
# --------------------------------------------------------------------------
def test_partition_eof_and_retriable_errors_are_skipped():
    assert _is_benign_record_error(_Err(_PARTITION_EOF)) is True
    assert _is_benign_record_error(_Err(-195, retriable=True)) is True
    consumer = _FakeConsumer([[_Rec(0, error=_Err(_PARTITION_EOF)), _Rec(7)], []])
    src = _source(consumer, poll_size=10)
    assert [m.offset for m in src._poll()] == [7]


def test_a_real_record_error_is_raised_not_dropped():
    """An unknown topic or a failed handshake arrives per-record on every poll.

    Dropping it left the query "running", reading nothing, reporting nothing — the failure
    was indistinguishable from an idle topic.
    """
    assert _is_benign_record_error(_Err(3, retriable=False)) is False
    consumer = _FakeConsumer([[_Rec(0, error=_Err(3, text="UNKNOWN_TOPIC_OR_PART"))]])
    src = _source(consumer, poll_size=10)
    with pytest.raises(BackendError, match="UNKNOWN_TOPIC_OR_PART"):
        src._poll()


# --------------------------------------------------------------------------
# Commit, close, metadata.
# --------------------------------------------------------------------------
def test_a_no_offset_commit_is_benign_and_anything_else_propagates():
    class _KafkaException(Exception):
        pass

    assert _is_no_offset(_KafkaException(_Err(_NO_OFFSET))) is True
    assert _is_no_offset(_KafkaException(_Err(3))) is False
    assert _is_no_offset(_KafkaException()) is False

    class _Rebalanced(_FakeConsumer):
        def commit(self, asynchronous):
            raise _KafkaException(_Err(_NO_OFFSET))

    _source(_Rebalanced([]))._commit_delivered()  # a revoked assignment is not a failure

    class _Broken(_FakeConsumer):
        def commit(self, asynchronous):
            raise _KafkaException(_Err(3, text="nope"))

    with pytest.raises(Exception, match="nope"):
        _source(_Broken([]))._commit_delivered()


def test_close_drops_the_handle_even_when_the_client_raises():
    """A second `close()` — and `iter_batches` guarantees one — must not re-close a dead
    consumer and raise out of a `finally`, masking the original error."""

    class _AngryClose(_FakeConsumer):
        def close(self):
            self.closed += 1
            raise RuntimeError("broker gone")

    consumer = _AngryClose([])
    src = _source(consumer)
    with pytest.raises(RuntimeError, match="broker gone"):
        src.close()
    src.close()  # idempotent: the handle was dropped before the failing call
    assert consumer.closed == 1


def test_metadata_lookup_is_bounded_and_names_a_missing_topic():
    class _Meta:
        def __init__(self, topics):
            self.topics = topics

    class _TopicMeta:
        def __init__(self, partitions, error=None):
            self.partitions = partitions
            self.error = error

    class _MetaConsumer(_FakeConsumer):
        def __init__(self, meta):
            super().__init__([])
            self.meta = meta
            self.timeouts: list[float] = []

        def list_topics(self, topic, timeout):
            self.timeouts.append(timeout)
            return self.meta

    good = _MetaConsumer(_Meta({"t": _TopicMeta({2: object(), 0: object(), 1: object()})}))
    src = _source(good, metadata_timeout=3.0)
    assert src._discover_partitions() == [0, 1, 2]
    assert good.timeouts == [3.0]  # never an unbounded metadata fetch on the driver
    assert "metadata_timeout" not in src._options  # not leaked into the client config

    missing = _MetaConsumer(_Meta({}))
    with pytest.raises(BackendError, match="was not found"):
        _source(missing)._discover_partitions()

    broken = _MetaConsumer(_Meta({"t": _TopicMeta({}, error="LEADER_NOT_AVAILABLE")}))
    with pytest.raises(BackendError, match="LEADER_NOT_AVAILABLE"):
        _source(broken)._discover_partitions()


# --------------------------------------------------------------------------
# Credential redaction: neither over- nor under-broad.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["sasl.password", "sasl_plain_password", "connection_str", "ssl.key.pem", "sas_key"],
)
def test_secret_shaped_options_are_masked(key):
    assert redact_broker_options({key: "s3cret"}) == {key: "***"}


@pytest.mark.parametrize("key", ["sasl.mechanism", "sasl.username", "bootstrap_servers"])
def test_non_secret_options_survive_so_two_clusters_keep_distinct_identities(key):
    """Over-redaction collapses distinct configs onto one learned-statistics key.

    A bare `"sas"` hint matched every `sasl.*` option, so two clusters differing only in
    their SASL identity fingerprinted identically and shared one stats entry.
    """
    assert redact_broker_options({key: "plain"}) == {key: "plain"}


# --------------------------------------------------------------------------
# Distributed epoch reads reuse their consumer across micro-batches.
# --------------------------------------------------------------------------
class _CountingBroker(BrokerSource):
    """A bounded broker that counts how many instances get built."""

    format_name = "counting_broker"
    built = 0
    __slots__ = ("_cursor",)

    def __init__(self, topic, *, poll_size=2, partitions=None, **options):
        super().__init__(topic, poll_size=poll_size, **options)
        type(self).built += 1
        self._cursor = 0

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        start = self._resume_from.get(0)
        if start is not None and self._cursor <= start:
            self._cursor = start + 1
        out = [
            BrokerMessage(value=b"x", partition=0, offset=self._cursor + i, timestamp=0, topic="t")
            for i in range(self.poll_size)
        ]
        self._cursor += self.poll_size
        return out


@pytest.fixture
def counting_split(monkeypatch):
    from batcher.io.formats.base import SOURCES

    _EPOCH_READERS.clear()
    _CountingBroker.built = 0
    monkeypatch.setitem(SOURCES._items, "counting_broker", _CountingBroker)
    yield BrokerSplit(format_name="counting_broker", topic="t", partition=0, poll_size=2)
    _EPOCH_READERS.clear()


def test_successive_epochs_reuse_one_consumer(counting_split):
    """Rebuilding a client per epoch means a connect, a metadata fetch, and a group join
    ahead of every trigger — and a client that lives one poll can never prefetch."""
    batches, position = counting_split.read_epoch(None)
    assert [b.num_rows for b in batches] == [2]
    for _ in range(5):
        batches, position = counting_split.read_epoch(position)
        assert [b.num_rows for b in batches] == [2]
    assert _CountingBroker.built == 1


def test_a_non_contiguous_resume_rebuilds_rather_than_reading_from_a_stale_cursor(
    counting_split,
):
    """Reuse is conditional: any position other than where the cached consumer stopped
    closes it and seeks a fresh one, so the cache can cost a reconnection, never a wrong row."""
    _, position = counting_split.read_epoch(None)
    assert _CountingBroker.built == 1
    counting_split.read_epoch(position)  # contiguous -> reused
    assert _CountingBroker.built == 1
    counting_split.read_epoch(0)  # a restart from an older checkpoint -> rebuilt and seeked
    assert _CountingBroker.built == 2


def test_an_empty_epoch_reports_the_offset_it_started_from(monkeypatch):
    """Returning `None` does not mean "unchanged": the driver drops a `None` position from
    the epoch's offset map, losing the partition's checkpoint entirely."""
    from batcher.io.formats.base import SOURCES

    class _IdleBroker(_CountingBroker):
        format_name = "idle_broker"
        __slots__ = ()

        def _poll(self):
            return []

    _EPOCH_READERS.clear()
    monkeypatch.setitem(SOURCES._items, "idle_broker", _IdleBroker)
    split = BrokerSplit(format_name="idle_broker", topic="t", partition=0, poll_size=2)
    assert split.read_epoch(41) == ([], 41)
    _EPOCH_READERS.clear()


# --------------------------------------------------------------------------
# A poll is bounded by bytes as well as by count.
# --------------------------------------------------------------------------
class _FatRec(_Rec):
    """A record that reports its payload size the way `confluent_kafka.Message` does."""

    def __init__(self, offset: int, size: int) -> None:
        super().__init__(offset)
        self._size = size
        self.value_calls = 0

    def len(self):
        return self._size

    def value(self):
        self.value_calls += 1
        return b"x" * self._size


def test_a_poll_stops_on_the_byte_budget_even_with_count_left():
    """`poll_size` bounds a batch by count, and a count says nothing about memory when a
    message can be a megabyte: 16,384 of them is a 16 GiB micro-batch, and the Arrow
    `binary` builder's 32-bit offsets fail outright past 2 GiB."""
    mib = 1 << 20
    consumer = _FakeConsumer(
        [[_FatRec(0, mib)], [_FatRec(i, mib) for i in range(1, 5)], [_FatRec(9, mib)]]
    )
    src = _source(consumer, poll_size=16_384, poll_timeout=0.5)
    src.poll_bytes = 4 * mib

    messages = src._poll()

    # First record (1 MiB) plus one drain pass of four; the budget is spent, so the third
    # pass — which the count alone would have taken — never happens.
    assert len(messages) == 5
    assert len(consumer.calls) == 2


def test_the_count_still_bounds_a_poll_of_small_messages():
    """The byte bound is a ceiling, not a replacement: a topic of small messages must still
    fill its `poll_size` exactly as before."""
    consumer = _FakeConsumer([[_Rec(0)], [_Rec(1), _Rec(2)], []])
    src = _source(consumer, poll_size=100, poll_timeout=0.5)
    assert len(src._poll()) == 3
    assert [n for n, _t in consumer.calls] == [1, 99, 97]


def test_measuring_a_poll_does_not_copy_its_payloads():
    """`Message.len()` is a size librdkafka already knows; `value()` copies the payload into
    a Python `bytes`. Measuring with `value()` would copy the whole batch a second time —
    on exactly the megabyte-message topics this bound exists for."""
    recs = [_FatRec(0, 1 << 20)]
    consumer = _FakeConsumer([recs, []])
    src = _source(consumer, poll_size=10, poll_timeout=0.5)
    src._poll()
    # One copy, in `_decode`, which has to keep the bytes. None for the measurement.
    assert recs[0].value_calls == 1


# --- startingOffsets / failOnDataLoss (Spark option parity) --------------------------


def test_starting_offsets_earliest_and_latest_map_to_the_reset_policy():
    from batcher.io.formats.streaming.kafka import _parse_starting_offsets

    assert _parse_starting_offsets("earliest", "t") == ("earliest", {})
    assert _parse_starting_offsets("latest", "t") == ("latest", {})
    assert _parse_starting_offsets(None, "t") == ("earliest", {})


def test_a_partition_map_becomes_explicit_seeks():
    from batcher.io.formats.streaming.kafka import _parse_starting_offsets

    assert _parse_starting_offsets({0: 10, 1: 20}, "t") == ("earliest", {0: 10, 1: 20})


def test_sparks_nested_topic_form_is_unwrapped():
    from batcher.io.formats.streaming.kafka import _parse_starting_offsets

    assert _parse_starting_offsets({"t": {"0": 5}}, "t") == ("earliest", {0: 5})


def test_a_json_string_is_accepted_because_spark_writes_one():
    from batcher.io.formats.streaming.kafka import _parse_starting_offsets

    assert _parse_starting_offsets('{"t": {"0": 7}}', "t") == ("earliest", {0: 7})


def test_sparks_negative_sentinels_mean_earliest_and_latest():
    from batcher.io.formats.streaming.kafka import _parse_starting_offsets

    assert _parse_starting_offsets({0: -2}, "t") == ("earliest", {})
    assert _parse_starting_offsets({0: -1}, "t") == ("latest", {})


def test_an_unknown_starting_position_is_refused_with_a_suggestion():
    from batcher._internal.errors import PlanError
    from batcher.io.formats.streaming.kafka import _parse_starting_offsets

    with pytest.raises(PlanError, match="unknown starting_offsets"):
        _parse_starting_offsets("earlist", "t")


def test_the_spark_options_never_reach_the_client_config():
    src = KafkaSource("t", starting_offsets="latest", fail_on_data_loss=False)
    assert "starting_offsets" not in src._options
    assert "fail_on_data_loss" not in src._options
    assert src._offset_reset == "latest"


def test_an_offset_that_aged_out_of_the_log_fails_by_default():
    """`failOnDataLoss=True` is Spark's default and the safe one: rows were skipped."""
    src = KafkaSource("t")
    with pytest.raises(BackendError, match="Kafka read failed"):
        src._decode([_Rec(0, error=_Err(1))])  # OFFSET_OUT_OF_RANGE


def test_data_loss_can_be_tolerated_explicitly_but_is_logged(caplog):
    src = KafkaSource("t", fail_on_data_loss=False)
    with caplog.at_level("WARNING"):
        assert src._decode([_Rec(0, error=_Err(1))]) == []
    assert any("no longer available" in r.getMessage() for r in caplog.records)


def test_tolerating_data_loss_does_not_tolerate_any_other_error():
    src = KafkaSource("t", fail_on_data_loss=False)
    with pytest.raises(BackendError, match="Kafka read failed"):
        src._decode([_Rec(0, error=_Err(-193, text="sasl handshake failed"))])
