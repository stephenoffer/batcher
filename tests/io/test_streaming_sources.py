"""Streaming connectors: seen-store + incremental file discovery (no deps).

The broker tests are skipped unless their optional client is installed
(``pytest.importorskip``); they only assert the registry wiring and the deferred
``BackendError`` contract — no live broker is required or contacted.

The seen-store and Auto Loader analog tests are *real* and run with no optional
dependency: they exercise the stdlib-SQLite ``SeenStore`` and
``IncrementalFileSource`` over a local temp directory of Parquet files, proving
exactly-once incremental discovery (first pass yields all, second yields none,
a newly added file yields only the new one).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming import IncrementalFileSource  # registers all sources
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource, broker_schema
from batcher.io.formats.streaming.seen_store import SeenStore


# --------------------------------------------------------------------------
# SeenStore — pure stdlib SQLite, no optional dependency.
# --------------------------------------------------------------------------
def test_seen_store_mark_and_seen(tmp_path):
    store = SeenStore(str(tmp_path / "seen.sqlite"))
    assert store.seen("a.parquet") is False
    store.mark("a.parquet", size=10, mtime=1.0)
    assert store.seen("a.parquet") is True
    store.close()


def test_seen_store_unseen_preserves_order_and_dedups(tmp_path):
    store = SeenStore(str(tmp_path / "seen.sqlite"))
    store.mark("b.parquet", size=1, mtime=1.0)
    candidates = ["a.parquet", "b.parquet", "c.parquet"]
    assert store.unseen(candidates) == ["a.parquet", "c.parquet"]
    assert store.unseen([]) == []
    store.close()


def test_seen_store_max_seen_and_persistence(tmp_path):
    path = str(tmp_path / "seen.sqlite")
    store = SeenStore(path)
    assert store.max_seen() is None
    store.mark("a.parquet", size=1, mtime=1.0)
    store.mark("c.parquet", size=1, mtime=1.0)
    assert store.max_seen() == "c.parquet"
    store.close()

    # Reopen: state is durable across "process restarts".
    reopened = SeenStore(path)
    assert reopened.seen("a.parquet") is True
    assert reopened.max_seen() == "c.parquet"
    reopened.close()


def test_seen_store_mark_is_idempotent(tmp_path):
    store = SeenStore(str(tmp_path / "seen.sqlite"))
    store.mark("a.parquet", size=1, mtime=1.0)
    store.mark("a.parquet", size=2, mtime=2.0)  # update, not error
    assert store.unseen(["a.parquet"]) == []
    store.close()


def test_seen_store_mark_many_matches_repeated_mark(tmp_path):
    # `mark_many` is `mark` per record in one transaction: same visibility, one commit.
    store = SeenStore(str(tmp_path / "seen.sqlite"))
    store.mark_many([("a.parquet", 1, 1.0), ("b.parquet", 2, 2.0), ("c.parquet", 3, 3.0)])
    assert store.unseen(["a.parquet", "b.parquet", "c.parquet", "d.parquet"]) == ["d.parquet"]
    # Idempotent + durable across reopen, exactly like `mark`.
    store.mark_many([("a.parquet", 9, 9.0)])  # ON CONFLICT update, not error
    store.close()
    reopened = SeenStore(str(tmp_path / "seen.sqlite"))
    assert reopened.unseen(["a.parquet"]) == []
    reopened.close()


def test_seen_store_mark_many_empty_is_noop(tmp_path):
    store = SeenStore(str(tmp_path / "seen.sqlite"))
    store.mark_many([])  # must not raise or open a transaction
    assert store.max_seen() is None
    store.close()


# --------------------------------------------------------------------------
# IncrementalFileSource — Auto Loader analog over a local temp dir.
# --------------------------------------------------------------------------
def _write_parquet(path, table):
    pq.write_table(table, str(path))


def _rows(source, projection=None):
    batches = list(source.iter_batches(projection))
    if not batches:
        return []
    return pa.Table.from_batches(batches).to_pylist()


def test_incremental_file_source_exactly_once_discovery(tmp_path):
    data_dir = tmp_path / "incoming"
    data_dir.mkdir()
    state_dir = tmp_path / "state"

    _write_parquet(data_dir / "0001.parquet", pa.table({"id": [1, 2]}))
    _write_parquet(data_dir / "0002.parquet", pa.table({"id": [3, 4]}))

    def make_source():
        return IncrementalFileSource(str(data_dir), "parquet", state_dir=str(state_dir))

    # First discovery: both files.
    rows_first = _rows(make_source())
    assert sorted(r["id"] for r in rows_first) == [1, 2, 3, 4]

    # Second discovery: nothing new (dedup via the durable seen store).
    rows_second = _rows(make_source())
    assert rows_second == []

    # Add a third file; discovery yields only the new one.
    _write_parquet(data_dir / "0003.parquet", pa.table({"id": [5, 6]}))
    rows_third = _rows(make_source())
    assert sorted(r["id"] for r in rows_third) == [5, 6]

    # And once more: nothing new again.
    assert _rows(make_source()) == []


def test_incremental_file_source_max_files_per_trigger_backpressures(tmp_path):
    data_dir = tmp_path / "incoming"
    data_dir.mkdir()
    state_dir = tmp_path / "state"
    for i in range(1, 6):
        _write_parquet(data_dir / f"{i:04d}.parquet", pa.table({"id": [i]}))

    def make_source():
        return IncrementalFileSource(
            str(data_dir), "parquet", state_dir=str(state_dir), max_files_per_trigger=2
        )

    # A 5-file backlog drains two-at-a-time (oldest names first), not in one giant epoch.
    assert sorted(r["id"] for r in _rows(make_source())) == [1, 2]
    assert sorted(r["id"] for r in _rows(make_source())) == [3, 4]
    assert sorted(r["id"] for r in _rows(make_source())) == [5]
    assert _rows(make_source()) == []  # backlog fully drained, nothing re-read


def test_incremental_file_source_rejects_bad_max_files(tmp_path):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="max_files_per_trigger must be >= 1"):
        IncrementalFileSource(
            str(tmp_path), "parquet", state_dir=str(tmp_path / "s"), max_files_per_trigger=0
        )


def test_incremental_file_source_schema_and_registry(tmp_path):
    data_dir = tmp_path / "incoming"
    data_dir.mkdir()
    state_dir = tmp_path / "state"
    _write_parquet(data_dir / "0001.parquet", pa.table({"id": [1], "v": [1.5]}))

    src = IncrementalFileSource(str(data_dir), "parquet", state_dir=str(state_dir))
    assert src.schema().names == ["id", "v"]
    assert src.row_count() is None
    assert "files_incremental" in SOURCES


def test_incremental_file_source_splits_are_picklable(tmp_path):
    import pickle

    data_dir = tmp_path / "incoming"
    data_dir.mkdir()
    state_dir = tmp_path / "state"
    _write_parquet(data_dir / "0001.parquet", pa.table({"id": [1]}))

    src = IncrementalFileSource(str(data_dir), "parquet", state_dir=str(state_dir))
    splits = src.splits()
    assert len(splits) == 1
    restored = pickle.loads(pickle.dumps(splits[0]))
    assert restored.read()[0].to_pylist() == [{"id": 1}]


# --------------------------------------------------------------------------
# Broker base — schema + batch assembly, no client needed.
# --------------------------------------------------------------------------
def test_broker_schema_is_fixed():
    schema = broker_schema()
    assert schema.names == ["key", "value", "partition", "offset", "timestamp", "topic"]
    assert schema.field("value").type == pa.binary()
    assert schema.field("offset").type == pa.int64()
    assert schema.field("topic").type == pa.string()


def test_broker_make_batch_assembles_fixed_schema():
    from batcher.io.formats.streaming.broker import BrokerSource

    messages = [
        BrokerMessage(value=b"a", partition=0, offset=10, timestamp=100, topic="t", key=b"k"),
        BrokerMessage(value=b"b", partition=0, offset=11, timestamp=101, topic="t"),
    ]
    batch = BrokerSource._make_batch(messages)
    assert batch.schema == broker_schema()
    assert batch.num_rows == 2
    assert batch.column("value").to_pylist() == [b"a", b"b"]
    assert batch.column("key").to_pylist() == [b"k", None]
    assert batch.column("offset").to_pylist() == [10, 11]


def test_broker_backs_off_on_empty_polls(monkeypatch):
    # A fast-empty broker (Kinesis returns immediately with no records) must not spin the
    # poll loop: empty polls back off with a growing, capped sleep, reset after real data.
    import time

    from batcher.io.formats.streaming.broker import BrokerSource

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    class _EmptyThenData(BrokerSource):
        format_name = "empty_then_data"
        __slots__ = ("_calls",)

        def __init__(self):
            super().__init__("t", poll_size=5)
            self._calls = 0

        def _discover_partitions(self):
            return [0]

        def _poll(self):
            self._calls += 1
            data = {
                4: [BrokerMessage(value=b"a", partition=0, offset=0, timestamp=0, topic="t")],
                7: [BrokerMessage(value=b"b", partition=0, offset=1, timestamp=1, topic="t")],
            }
            if self._calls in data:
                return data[self._calls]
            if self._calls > 8:
                return None  # end of stream
            return []  # an idle poll

    out = list(_EmptyThenData().iter_batches())
    assert len(out) == 2  # the two non-empty polls yielded a batch each
    # First idle streak grows geometrically; after real data the back-off resets to the floor.
    assert sleeps[:3] == [0.01, 0.02, 0.04]
    assert sleeps[3] == 0.01  # reset after the batch at poll 4
    assert max(sleeps) <= 0.25  # capped


class _BoundedTestBroker(BrokerSource):
    """A finite, replayable broker for the checkpoint contract — no client needed.

    Emits ``total`` messages (offsets ``0..total-1``) on partition 0 in
    ``poll_size`` chunks and honors ``_resume_from`` so a ``seek`` resumes strictly
    after a checkpointed offset. Used to exercise the base offset-tracking machinery
    that Kafka/Kinesis/… inherit.
    """

    format_name = "test_broker"
    __slots__ = ("_cursor", "_started", "_total")

    def __init__(self, topic: str = "t", *, total: int = 20, poll_size: int = 5, **opts):
        super().__init__(topic, poll_size=poll_size, **opts)
        self._total = total
        self._cursor = 0
        self._started = False

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        if not self._started:
            self._started = True
            resume = self._resume_from.get(0)
            self._cursor = 0 if resume is None else int(resume) + 1
        if self._cursor >= self._total:
            return None  # bounded: signals end-of-stream
        end = min(self._cursor + self.poll_size, self._total)
        msgs = [
            BrokerMessage(
                value=str(o).encode(), partition=0, offset=o, timestamp=o, topic=self.topic
            )
            for o in range(self._cursor, end)
        ]
        self._cursor = end
        return msgs


def test_broker_source_is_checkpointable():
    from batcher.io.source import is_checkpointable

    assert is_checkpointable(_BoundedTestBroker())


def test_broker_tracks_and_snapshots_offsets():
    broker = _BoundedTestBroker(total=12, poll_size=5)
    batches = list(broker.iter_batches())
    # 12 rows in chunks of 5 -> offsets 0..11; the snapshot is the last offset seen.
    assert sum(b.num_rows for b in batches) == 12
    assert broker.snapshot_position() == {"offsets": {"0": 11}}


def test_broker_seek_resumes_strictly_after_offset():
    broker = _BoundedTestBroker(total=12, poll_size=5)
    broker.seek({"offsets": {"0": 6}})  # resume strictly after offset 6
    offsets = [o for b in broker.iter_batches() for o in b.column("offset").to_pylist()]
    assert offsets == list(range(7, 12))  # 7..11, none replayed or skipped


def test_broker_resume_token_overrides_offset_in_snapshot():
    # A native resume token (e.g. a Kinesis sequence) is snapshotted in place of the
    # lossy int64 offset, so a client can seek to the exact position.
    broker = _BoundedTestBroker()
    broker._track_positions(
        [
            BrokerMessage(
                value=b"x", partition=0, offset=99, timestamp=0, topic="t", resume_token="seq-abc"
            )
        ]
    )
    assert broker.snapshot_position() == {"offsets": {"0": "seq-abc"}}


class _FakeEvent:
    """A stand-in for azure ``EventData`` — no ``azure-eventhub`` install needed."""

    def __init__(self, body, key, offset="7", ts=123):
        self._body = body
        self.partition_key = key
        self.offset = offset
        self.enqueued_time_utc_ms = ts

    def body_as_bytes(self):
        return self._body


def test_eventhubs_event_preserves_binary_body_and_coerces_key():
    from batcher.io.formats.streaming.broker import broker_schema
    from batcher.io.formats.streaming.eventhubs import _event_to_message

    # A binary (non-UTF-8) payload must survive verbatim, and a str partition key must
    # coerce to the bytes the fixed broker schema declares.
    raw = b"\xff\x00protobuf\x80"
    msg = _event_to_message(_FakeEvent(raw, key="tenant-9"), partition_id=2, topic="hub")
    assert msg.value == raw  # not decoded/re-encoded
    assert msg.key == b"tenant-9"
    assert msg.partition == 2 and msg.offset == 7 and msg.timestamp == 123
    # And the message assembles cleanly into the binary broker schema (would raise before).
    from batcher.io.formats.streaming.broker import BrokerSource

    batch = BrokerSource._make_batch([msg])
    assert batch.schema.equals(broker_schema())
    assert batch.column("value")[0].as_py() == raw


def test_eventhubs_event_handles_none_key_and_offset():
    from batcher.io.formats.streaming.eventhubs import _event_to_message

    msg = _event_to_message(_FakeEvent(b"x", key=None, offset=None), partition_id=0, topic="h")
    assert msg.key is None
    assert msg.offset == 0  # a null offset maps to 0, never raises
    assert msg.resume_token is None


class _FakeEHConsumer:
    def __init__(self, events):
        self._events = events

    def receive_message_batch(self, max_batch_size, max_wait_time):
        return self._events

    def close(self):
        pass


class _FakeEHClient:
    def __init__(self, events_by_partition):
        self._events = events_by_partition
        self.created: list[tuple[str, str]] = []

    def get_partition_ids(self):
        return [str(p) for p in self._events]

    def _create_consumer(self, consumer_group, partition_id, event_position, on_event_received):
        self.created.append((partition_id, event_position))
        return _FakeEHConsumer(self._events.get(int(partition_id), []))


def test_eventhubs_resumes_from_checkpointed_offset():
    from batcher.io.formats.streaming.eventhubs import EventHubsSource

    src = EventHubsSource("hub", partitions=[0], connection_str="x")
    fake = _FakeEHClient({0: [_FakeEvent(b"a", key=None, offset="100")]})
    src._client_obj = fake

    # Fresh: no checkpoint, so the configured starting_position ("-1") is used.
    msgs = src._poll()
    assert fake.created == [("0", "-1")]
    assert msgs[0].resume_token == "100"  # the exact offset string, for seeking
    src._track_positions(msgs)
    assert src.snapshot_position() == {"offsets": {"0": "100"}}

    # Recover: seek then poll resumes strictly after the checkpointed offset, not from start.
    src.seek({"offsets": {"0": "100"}})
    fake.created.clear()
    src._poll()
    assert fake.created == [("0", "100")]


def test_broker_split_is_picklable():
    import pickle

    from batcher.io.formats.streaming.broker import BrokerSplit

    split = BrokerSplit(
        format_name="kafka", topic="t", partition=3, poll_size=100, options={"group": "g"}
    )
    restored = pickle.loads(pickle.dumps(split))
    assert restored.partition == 3
    # The identity now carries a connection fingerprint too, so the same topic name on two
    # clusters is two relations rather than one shared statistics key. Asserted as a prefix
    # plus round-trip stability — see `test_streaming_connector_audit.py` for the contract.
    assert restored.identity().startswith("kafka:t:p3:")
    assert restored.identity() == split.identity()


# --------------------------------------------------------------------------
# Broker connectors — registry wiring; deferred-dependency contract.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["kafka", "kinesis", "eventhubs", "pubsub", "pulsar", "files_incremental"],
)
def test_streaming_sources_registered(name):
    assert name in SOURCES


def test_pubsub_idle_pull_timeout_is_an_empty_poll_not_a_failure():
    """A pull that hits its deadline on an idle subscription yields [], not an exception."""
    from batcher.io.formats.streaming.pubsub import PubSubSource, _is_pull_timeout

    class DeadlineExceeded(Exception):  # mimics google.api_core.exceptions by name
        pass

    class _FakeSub:
        def pull(self, request, timeout):
            raise DeadlineExceeded("no messages")

    assert _is_pull_timeout(DeadlineExceeded("x")) is True
    assert _is_pull_timeout(ValueError("real error")) is False

    src = PubSubSource("projects/p/subscriptions/s", pull_timeout=0.1)
    src._client_obj = _FakeSub()
    assert src._poll() == []  # idle, not a crash


def test_pubsub_non_timeout_error_propagates():
    from batcher.io.formats.streaming.pubsub import PubSubSource

    class _AngrySub:
        def pull(self, request, timeout):
            raise RuntimeError("auth failed")

    src = PubSubSource("projects/p/subscriptions/s")
    src._client_obj = _AngrySub()
    with pytest.raises(RuntimeError, match="auth failed"):
        src._poll()


def test_kafka_poll_timeout_is_configurable():
    """`poll_timeout` reaches `consume()` and is kept out of the confluent-kafka config."""
    from batcher.io.formats.streaming.kafka import KafkaSource

    class _FakeConsumer:
        def __init__(self):
            self.timeouts: list[float] = []

        def consume(self, num_messages, timeout):
            self.timeouts.append(timeout)
            return []

    src = KafkaSource("t", poll_timeout=0.25)
    # `poll_timeout` must not have leaked into the broker options (→ a bogus `poll.timeout`).
    assert "poll_timeout" not in src._options
    fake = _FakeConsumer()
    src._consumer = fake  # `_client()` returns this instead of building a real one
    assert src._poll() == []
    assert fake.timeouts == [0.25]


def test_pulsar_receive_timeout_is_configurable(monkeypatch):
    """`receive_timeout_millis` reaches `receive()` and stays out of the client options."""
    from batcher.io.formats.streaming import pulsar as pmod

    class _Timeout(Exception):
        pass

    class _FakePulsar:
        Timeout = _Timeout

    class _FakeConsumer:
        def __init__(self):
            self.timeouts = []

        def receive(self, timeout_millis):
            self.timeouts.append(timeout_millis)
            raise _Timeout()  # idle topic -> empty poll

    src = pmod.PulsarSource("t", receive_timeout_millis=250)
    assert "receive_timeout_millis" not in src._options
    monkeypatch.setattr(pmod, "_import_pulsar", lambda: _FakePulsar)
    src._client_obj = object()  # skip real client build
    src._consumer = _FakeConsumer()  # `_client()` returns this
    assert src._poll() == []
    assert src._consumer.timeouts == [250]


def test_kafka_deferred_dependency_raises_backend_error():
    pytest.importorskip(
        "pytest"
    )  # always available; brokers themselves are tested only if installed.
    from batcher._internal.errors import BackendError
    from batcher.io.formats.streaming.kafka import KafkaSource

    src = KafkaSource("t", bootstrap_servers="localhost:9092")
    try:
        import confluent_kafka  # noqa: F401
    except ImportError:
        with pytest.raises(BackendError, match="\\[kafka\\]"):
            src._client()
