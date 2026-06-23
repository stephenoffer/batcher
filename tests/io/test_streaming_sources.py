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
from batcher.io.formats.streaming.broker import BrokerMessage, broker_schema
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


def test_broker_split_is_picklable():
    import pickle

    from batcher.io.formats.streaming.broker import BrokerSplit

    split = BrokerSplit(
        format_name="kafka", topic="t", partition=3, poll_size=100, options={"group": "g"}
    )
    restored = pickle.loads(pickle.dumps(split))
    assert restored.partition == 3
    assert restored.identity() == "kafka:t:p3"


# --------------------------------------------------------------------------
# Broker connectors — registry wiring; deferred-dependency contract.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["kafka", "kinesis", "eventhubs", "pubsub", "pulsar", "files_incremental"],
)
def test_streaming_sources_registered(name):
    assert name in SOURCES


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
        with pytest.raises(BackendError, match="batcher\\[kafka\\]"):
            src._client()
