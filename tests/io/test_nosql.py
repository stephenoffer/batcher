"""NoSQL / operational-store connector coverage — registration, splits, errors.

The connectors lazily import their optional drivers, so registration, identity,
partition math, and split construction work without any driver installed; tests
that need a real driver are gated with `pytest.importorskip` and skip cleanly.

The load-bearing invariants asserted here, all reachable without a live server:

* every connector registers under its documented name (plus the ``scylla`` alias);
* a missing driver raises a typed `BackendError` with an actionable
  ``pip install 'batcher-engine[<extra>]'`` hint;
* `splits()` returns PICKLABLE, connection-free value objects — verified with a
  `pickle.dumps`/`loads` round-trip — for the connectors whose parallel unit is
  computed locally (token ranges, scan segments, slot ranges);
* those local split enumerations form a disjoint, exhaustive cover of their key
  space (token ring / hash slots), the property the distributed read relies on.

Runs without the native engine — these exercise the Python IO layer only.
"""

from __future__ import annotations

import builtins
import pickle
from collections.abc import Callable
from itertools import pairwise

import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql import (
    CassandraSource,
    CouchbaseSource,
    DynamoDBSource,
    ElasticsearchSource,
    HBaseSource,
    MongoSink,
    MongoSource,
    Neo4jSource,
    PartitionSpec,
    RedisSource,
    ScyllaSource,
)
from batcher.io.formats.nosql.base import _ScanSplit, require_driver, rows_to_batches
from batcher.io.formats.nosql.cassandra import _MAX_TOKEN, _MIN_TOKEN, _token_ranges
from batcher.io.formats.nosql.redis import _NUM_SLOTS, _crc16_slot

# --- registration ------------------------------------------------------------


def test_connectors_registered() -> None:
    expected = {
        "mongo": MongoSource,
        "cassandra": CassandraSource,
        "scylla": ScyllaSource,
        "dynamodb": DynamoDBSource,
        "redis": RedisSource,
        "elasticsearch": ElasticsearchSource,
        "couchbase": CouchbaseSource,
        "neo4j": Neo4jSource,
        "hbase": HBaseSource,
    }
    for name, cls in expected.items():
        assert name in SOURCES
        assert SOURCES.get(name) is cls
    assert SINKS.get("mongo") is MongoSink


# --- identity without a backend (no credentials leak) ------------------------


def test_identity_does_not_require_backend_or_leak_creds() -> None:
    mongo = MongoSource(uri="mongodb://secret@h/db", database="d", collection="c")
    assert mongo.identity() == "mongo:d.c"
    assert "secret" not in mongo.identity()

    cass = CassandraSource(contact_points=["h"], keyspace="ks", table="t", partition_key="id")
    assert cass.identity() == "cassandra:ks.t"
    assert (
        ScyllaSource(contact_points=["h"], keyspace="ks", table="t", partition_key="id").identity()
        == "scylla:ks.t"
    )

    dynamo = DynamoDBSource(table="t", region_name="us-east-1", aws_secret_access_key="zzz")
    assert dynamo.identity() == "dynamodb:us-east-1/t"
    assert "zzz" not in dynamo.identity()

    redis = RedisSource(host="h", port=6379, db=2, password="pw")
    assert redis.identity() == "redis:h:6379/2"
    assert "pw" not in redis.identity()

    assert ElasticsearchSource(hosts="h", index="idx", api_key="k").identity() == (
        "elasticsearch:idx"
    )
    cb = CouchbaseSource(
        connstr="couchbases://h",
        username="u",
        password="p",
        database="d",
        scope="s",
        collection="c",
    )
    assert cb.identity() == "couchbase:d.s.c"
    assert "p" not in cb.identity()

    neo = Neo4jSource(uri="bolt://h", username="u", password="p", cypher="MATCH (n) RETURN n")
    assert neo.identity() == "neo4j:bolt://h/default"
    assert HBaseSource(host="h", table="t").identity() == "hbase:h:9090/t"


# --- missing-driver errors are typed + actionable ----------------------------


def test_require_driver_missing_raises_actionable() -> None:
    with pytest.raises(BackendError, match=r"batcher\[mongo\]"):
        require_driver("definitely_not_a_real_module_xyz", extra="mongo")


@pytest.mark.parametrize(
    ("make", "blocked", "extra"),
    [
        (
            lambda: CassandraSource(
                contact_points=["h"], keyspace="ks", table="t", partition_key="id"
            )._infer_schema(),
            "cassandra",
            "cassandra",
        ),
        (lambda: DynamoDBSource(table="t")._infer_schema(), "boto3", "dynamodb"),
        (lambda: RedisSource(host="h")._client(), "redis", "redis"),
        (
            lambda: ElasticsearchSource(hosts="h", index="i")._client(),
            "elasticsearch",
            "elasticsearch",
        ),
        (
            lambda: CouchbaseSource(
                connstr="x", username="u", password="p", database="d", scope="s", collection="c"
            )._cluster(),
            "couchbase_columnar",
            "couchbase",
        ),
        (
            lambda: Neo4jSource(
                uri="bolt://h", username="u", password="p", cypher="RETURN 1"
            )._driver(),
            "neo4j",
            "neo4j",
        ),
        (lambda: HBaseSource(host="h", table="t")._connection(), "happybase", "hbase"),
        (
            lambda: MongoSource(uri="m://h", database="d", collection="c")._client(),
            "pymongo",
            "mongo",
        ),
    ],
)
def test_missing_driver_raises_for_each(
    make: Callable[[], object], blocked: str, extra: str, monkeypatch
) -> None:
    real_import = builtins.__import__

    def _block(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"no {blocked}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(BackendError, match=rf"batcher\[{extra}\]"):
        make()


# --- splits are PICKLABLE, connection-free value objects ---------------------


def _assert_picklable_splits(source, expected_count: int) -> list[_ScanSplit]:
    """Every split survives a pickle round-trip and is a connection-free locator."""
    splits = source.splits()
    assert len(splits) == expected_count
    for split in splits:
        assert isinstance(split, _ScanSplit)
        restored = pickle.loads(pickle.dumps(split))
        assert restored.partition == split.partition
        assert restored.identity() == split.identity()
        assert restored.conn_kwargs == split.conn_kwargs
    return splits


def test_cassandra_token_range_splits_picklable_and_cover() -> None:
    source = CassandraSource(
        contact_points=["h"],
        keyspace="ks",
        table="t",
        partition_key="id",
        partition_spec=PartitionSpec(segments=8),
    )
    splits = _assert_picklable_splits(source, 8)
    ranges = [s.partition for s in splits]
    # Disjoint, exhaustive cover of the whole Murmur3 token ring.
    assert ranges[0][0] == _MIN_TOKEN
    assert ranges[-1][1] == _MAX_TOKEN + 1
    for (_, end), (nxt, _) in pairwise(ranges):
        assert end == nxt


def test_dynamodb_segment_splits_picklable() -> None:
    source = DynamoDBSource(table="t", partition_spec=PartitionSpec(segments=4))
    splits = _assert_picklable_splits(source, 4)
    # One split per parallel-scan segment, each tagged (segment, total).
    assert [s.partition for s in splits] == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_redis_slot_range_splits_picklable_and_cover() -> None:
    source = RedisSource(host="h", partition_spec=PartitionSpec(segments=4))
    splits = _assert_picklable_splits(source, 4)
    ranges = [s.partition for s in splits]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == _NUM_SLOTS
    for (_, end), (nxt, _) in pairwise(ranges):
        assert end == nxt


def test_scylla_uses_token_range_splits() -> None:
    source = ScyllaSource(
        contact_points=["h"],
        keyspace="ks",
        table="t",
        partition_key="id",
        partition_spec=PartitionSpec(segments=2),
    )
    splits = _assert_picklable_splits(source, 2)
    assert splits[0].source_cls is ScyllaSource


def test_single_segment_sources_yield_one_split() -> None:
    # Default partition spec (segments=1) gives a single whole-store split.
    for source in (
        DynamoDBSource(table="t"),
        RedisSource(host="h"),
        ElasticsearchSource(hosts="h", index="i"),
        CouchbaseSource(
            connstr="x", username="u", password="p", database="d", scope="s", collection="c"
        ),
        Neo4jSource(uri="bolt://h", username="u", password="p", cypher="RETURN 1"),
    ):
        _assert_picklable_splits(source, 1)


def test_elasticsearch_slice_splits_picklable() -> None:
    source = ElasticsearchSource(hosts="h", index="i", partition_spec=PartitionSpec(segments=3))
    splits = _assert_picklable_splits(source, 3)
    assert [s.partition for s in splits] == [(0, 3), (1, 3), (2, 3)]


# --- pure helpers ------------------------------------------------------------


def test_token_ranges_partition_whole_ring() -> None:
    ranges = _token_ranges(16)
    assert len(ranges) == 16
    assert ranges[0][0] == _MIN_TOKEN
    assert ranges[-1][1] == _MAX_TOKEN + 1


def test_crc16_slot_in_range_and_honors_hashtag() -> None:
    assert 0 <= _crc16_slot("some-key") < _NUM_SLOTS
    # Hashtag braces select the hashed substring, so these map to the same slot.
    assert _crc16_slot("{user1}.profile") == _crc16_slot("{user1}.session")


def test_rows_to_batches_chunks_at_batch_size() -> None:
    rows = ({"a": i} for i in range(2500))
    batches = list(rows_to_batches(rows, batch_rows=1000))
    assert [b.num_rows for b in batches] == [1000, 1000, 500]
    assert sum(b.num_rows for b in batches) == 2500


def test_rows_to_batches_empty_yields_nothing() -> None:
    assert list(rows_to_batches(iter([]))) == []


# --- real Mongo round-trip (gated) -------------------------------------------


def test_mongo_roundtrip_with_mongomock() -> None:
    pytest.importorskip("pymongoarrow")
    mongomock = pytest.importorskip("mongomock")
    import pyarrow as pa

    client = mongomock.MongoClient()
    client["db"]["coll"].insert_many([{"_id": i, "v": f"r{i}"} for i in range(3)])

    source = MongoSource(uri="mongodb://localhost", database="db", collection="coll")
    # Inject the in-memory client so no live server is required.
    source._client = lambda: client  # type: ignore[method-assign]
    out = pa.Table.from_batches(source.read())
    assert out.num_rows == 3
    assert set(out.column_names) >= {"_id", "v"}
