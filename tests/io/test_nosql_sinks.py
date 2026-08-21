"""The operational-store write path, against fake drivers.

Batcher could read nine NoSQL and operational stores and write exactly one of them. The
asymmetry mattered more than a missing feature list suggests: these are the stores a
pipeline's *output* lands in — a session cache, a feature store, a search index, a
dimension table an application reads — so "reads it" without "writes it" means the last
step of the pipeline happens somewhere else.

None of these servers runs in CI, so the drivers are faked. That is a real limit and worth
stating: these tests prove the sink issues the right operations in the right shape and
handles the failure modes each API actually has, not that a server accepted them. What they
do cover is the class of bug that a live server would *hide* just as easily, because each
of these APIs reports partial failure inside a success:

* DynamoDB's ``BatchWriteItem`` returns the requests it did not apply under
  ``UnprocessedItems`` with a 200 status. Ignoring that field writes some of the rows and
  reports success.
* Elasticsearch's ``_bulk`` reports per-document failures inside an HTTP 200 response.
* Cassandra's ``execute_concurrent_with_args`` returns a ``(success, result)`` pair per
  statement rather than raising.

Two live defects are pinned here as well. `MongoSink` required a ``collection=`` keyword
the writer never passes, so the documented ``ds.write.mongo("orders", uri=..., database=...)``
raised ``TypeError`` on every invocation. And it read ``self.uri`` raw where the source
resolves it, so an ``env:`` reference that read fine failed to connect on write.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS
from batcher.io.formats.nosql import (
    STORE_WRITE_MODES,
    BulkSink,
    CassandraSink,
    DynamoDBSink,
    ElasticsearchSink,
    HBaseSink,
    MongoSink,
    RedisSink,
)

pytestmark = pytest.mark.io

_ROWS = pa.table({"id": ["a", "b"], "amount": [10, 20]})


# --- registration and the shared vocabulary -------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("mongo", MongoSink),
        ("dynamodb", DynamoDBSink),
        ("cassandra", CassandraSink),
        ("redis", RedisSink),
        ("elasticsearch", ElasticsearchSink),
        ("hbase", HBaseSink),
    ],
)
def test_each_store_sink_is_registered(name, cls) -> None:
    assert SINKS.get(name) is cls


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (MongoSink, {"uri": "mongodb://h", "database": "d"}),
        (DynamoDBSink, {"region_name": "us-east-1"}),
        (CassandraSink, {"contact_points": ["h"], "keyspace": "k"}),
        (RedisSink, {}),
        (ElasticsearchSink, {}),
        (HBaseSink, {"host": "thrift"}),
    ],
)
def test_upsert_is_the_default_because_a_store_is_maintained_not_replaced(cls, kwargs) -> None:
    """`ds.write`'s usual default is `overwrite`; defaulting to it here would empty a store."""
    assert cls(**kwargs).mode == "upsert"


@pytest.mark.parametrize(
    ("cls", "kwargs", "declined"),
    [
        (DynamoDBSink, {"region_name": "us-east-1"}, "append"),
        (DynamoDBSink, {"region_name": "us-east-1"}, "overwrite"),
        (CassandraSink, {"contact_points": ["h"], "keyspace": "k"}, "append"),
        (RedisSink, {}, "overwrite"),
        (HBaseSink, {"host": "thrift"}, "append"),
        (HBaseSink, {"host": "thrift"}, "overwrite"),
    ],
)
def test_a_mode_the_store_cannot_express_is_declined_by_name(cls, kwargs, declined) -> None:
    """Declining is the contract: an approximation would be a wrong answer with a nice name."""
    with pytest.raises(BackendError, match=r"cannot express|implements"):
        cls(**kwargs, mode=declined)


def test_the_declared_modes_match_what_the_sink_implements() -> None:
    """`Writer` reads `dml_modes`; a sink narrowing only `supported_modes` would drift."""
    for cls in (MongoSink, DynamoDBSink, CassandraSink, RedisSink, ElasticsearchSink, HBaseSink):
        assert cls.dml_modes == cls.supported_modes
        assert set(cls.supported_modes) <= set(STORE_WRITE_MODES)


def test_an_empty_write_is_a_no_op_except_when_overwriting() -> None:
    sink = ElasticsearchSink(mode="upsert")
    assert sink.write(_ROWS.slice(0, 0), "docs").rows == 0


def test_a_destructive_mode_is_refused_past_the_first_shard() -> None:
    sink = ElasticsearchSink(mode="overwrite")
    with pytest.raises(BackendError, match="distributed write"):
        sink.write_partitioned(_ROWS, "docs", file_index=1)


def test_a_key_scoped_mode_is_allowed_across_shards(monkeypatch) -> None:
    sink = ElasticsearchSink(mode="upsert", key_field="id")
    client = _FakeElasticsearch()
    monkeypatch.setattr(ElasticsearchSink, "_client", lambda self: client)
    assert sink.write_partitioned(_ROWS, "docs", file_index=3)[0].rows == 2


# --- DynamoDB -------------------------------------------------------------------------


class _FakeDynamo:
    """A ``batch_write_item`` that can refuse part of a batch, as the real one does."""

    def __init__(self, unprocessed_rounds: int = 0) -> None:
        self.calls: list[dict] = []
        self._left = unprocessed_rounds

    def batch_write_item(self, *, RequestItems):
        self.calls.append(RequestItems)
        table, requests = next(iter(RequestItems.items()))
        if self._left > 0:
            self._left -= 1
            return {"UnprocessedItems": {table: requests[:1]}}
        return {"UnprocessedItems": {}}


@pytest.fixture
def dynamo(monkeypatch):
    client = _FakeDynamo()
    monkeypatch.setattr(DynamoDBSink, "_client", lambda self: client)
    return client


def test_dynamodb_upsert_sends_put_requests(dynamo) -> None:
    DynamoDBSink(region_name="us-east-1").write(_ROWS, "orders")
    (batch,) = dynamo.calls
    assert list(batch) == ["orders"]
    assert [next(iter(r)) for r in batch["orders"]] == ["PutRequest", "PutRequest"]
    assert batch["orders"][0]["PutRequest"]["Item"]["id"] == {"S": "a"}


def test_dynamodb_delete_sends_key_only_requests(dynamo) -> None:
    DynamoDBSink(region_name="us-east-1", mode="delete", key_field="id").write(_ROWS, "orders")
    (batch,) = dynamo.calls
    assert batch["orders"][0] == {"DeleteRequest": {"Key": {"id": {"S": "a"}}}}


def test_dynamodb_delete_names_a_missing_key_field(dynamo) -> None:
    sink = DynamoDBSink(region_name="us-east-1", mode="delete", key_field="pk")
    with pytest.raises(BackendError, match="needs every key field"):
        sink.write(_ROWS, "orders")


def test_dynamodb_chunks_at_the_service_limit_of_25(dynamo) -> None:
    wide = pa.table({"id": [str(i) for i in range(60)], "amount": list(range(60))})
    DynamoDBSink(region_name="us-east-1").write(wide, "orders")
    assert [len(call["orders"]) for call in dynamo.calls] == [25, 25, 10]


def test_dynamodb_retries_what_the_service_did_not_process(monkeypatch) -> None:
    """The 200-with-a-remainder response is how a partial write reports success."""
    client = _FakeDynamo(unprocessed_rounds=2)
    monkeypatch.setattr(DynamoDBSink, "_client", lambda self: client)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    DynamoDBSink(region_name="us-east-1").write(_ROWS, "orders")
    assert len(client.calls) == 3, "the unprocessed remainder was not resent"


def test_dynamodb_raises_when_the_remainder_never_clears(monkeypatch) -> None:
    client = _FakeDynamo(unprocessed_rounds=99)
    monkeypatch.setattr(DynamoDBSink, "_client", lambda self: client)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    with pytest.raises(BackendError, match="unprocessed"):
        DynamoDBSink(region_name="us-east-1").write(_ROWS, "orders")


def test_dynamodb_drops_nulls_rather_than_writing_a_null_attribute(dynamo) -> None:
    """DynamoDB has no "absent" value to write; an attribute is present or it is not."""
    DynamoDBSink(region_name="us-east-1").write(
        pa.table({"id": ["a"], "amount": pa.array([None], pa.int64())}), "orders"
    )
    assert list(dynamo.calls[0]["orders"][0]["PutRequest"]["Item"]) == ["id"]


# --- Cassandra ------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, log: list) -> None:
        self.log = log

    def prepare(self, cql: str):
        self.log.append(("prepare", cql))
        return cql


class _FakeCluster:
    def __init__(self, log: list) -> None:
        self.log = log

    def shutdown(self) -> None:
        self.log.append(("shutdown", None))


@pytest.fixture
def cassandra(monkeypatch):
    log: list = []
    monkeypatch.setattr(
        CassandraSink, "_session", lambda self: (_FakeCluster(log), _FakeSession(log))
    )
    module = types.ModuleType("cassandra.concurrent")

    def execute_concurrent_with_args(session, statement, parameters, **kwargs):
        log.append(("execute", statement, parameters, kwargs))
        return [(True, None) for _ in parameters]

    module.execute_concurrent_with_args = execute_concurrent_with_args
    monkeypatch.setitem(sys.modules, "cassandra", types.ModuleType("cassandra"))
    monkeypatch.setitem(sys.modules, "cassandra.concurrent", module)
    return log


def test_cassandra_upsert_prepares_one_insert_and_binds_every_row(cassandra) -> None:
    CassandraSink(contact_points=["h"], keyspace="k").write(_ROWS, "orders")
    prepared = next(cql for kind, cql in cassandra if kind == "prepare")
    assert prepared == "INSERT INTO orders (id, amount) VALUES (?, ?)"
    execute = next(entry for entry in cassandra if entry[0] == "execute")
    assert execute[2] == [("a", 10), ("b", 20)]


def test_cassandra_delete_binds_only_the_primary_key(cassandra) -> None:
    CassandraSink(contact_points=["h"], keyspace="k", mode="delete", key_columns=["id"]).write(
        _ROWS, "orders"
    )
    prepared = next(cql for kind, cql in cassandra if kind == "prepare")
    assert prepared == "DELETE FROM orders WHERE id = ?"
    execute = next(entry for entry in cassandra if entry[0] == "execute")
    assert execute[2] == [("a",), ("b",)]


def test_cassandra_closes_the_cluster_even_when_the_write_fails(cassandra, monkeypatch) -> None:
    module = sys.modules["cassandra.concurrent"]
    monkeypatch.setattr(
        module,
        "execute_concurrent_with_args",
        lambda *a, **k: [(False, RuntimeError("write timeout")), (True, None)],
    )
    with pytest.raises(BackendError, match="1 of 2 statements failed"):
        CassandraSink(contact_points=["h"], keyspace="k").write(_ROWS, "orders")
    assert ("shutdown", None) in cassandra


def test_cassandra_delete_without_keys_is_refused_at_construction() -> None:
    with pytest.raises(BackendError, match="needs key_columns"):
        CassandraSink(contact_points=["h"], keyspace="k", mode="delete")


# --- Redis ----------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, log: list) -> None:
        self.log = log

    def set(self, key, value):
        self.log.append(("set", key, value))

    def hset(self, key, mapping):
        self.log.append(("hset", key, mapping))

    def delete(self, key):
        self.log.append(("delete", key))

    def expire(self, key, ttl):
        self.log.append(("expire", key, ttl))

    def execute(self):
        self.log.append(("execute",))


class _FakeRedis:
    def __init__(self) -> None:
        self.log: list = []

    def pipeline(self, transaction=False):
        return _FakePipeline(self.log)

    def close(self) -> None:
        self.log.append(("close",))


@pytest.fixture
def redis(monkeypatch):
    client = _FakeRedis()
    monkeypatch.setattr(RedisSink, "_client", lambda self: client)
    return client


def test_a_two_column_frame_writes_one_string_per_key(redis) -> None:
    """The shape `bt.read.table("redis", ...)` returns must round-trip through the sink."""
    RedisSink(prefix="s:").write(pa.table({"key": ["u1"], "value": ["active"]}), "session")
    assert ("set", "s:u1", "active") in redis.log


def test_a_wider_frame_writes_one_hash_per_key(redis) -> None:
    RedisSink(prefix="o:", key_field="id").write(_ROWS, "orders")
    assert ("hset", "o:a", {"amount": "10"}) in redis.log


def test_the_destination_names_the_prefix_when_none_is_given(redis) -> None:
    RedisSink(key_field="id").write(_ROWS, "orders")
    assert ("hset", "orders:a", {"amount": "10"}) in redis.log


def test_a_ttl_expires_every_key_written(redis) -> None:
    RedisSink(prefix="s:", ttl_seconds=60).write(
        pa.table({"key": ["u1"], "value": ["active"]}), "session"
    )
    assert ("expire", "s:u1", 60) in redis.log


def test_delete_removes_the_keys_and_writes_nothing(redis) -> None:
    RedisSink(prefix="o:", key_field="id", mode="delete").write(_ROWS, "orders")
    assert [entry for entry in redis.log if entry[0] == "delete"] == [
        ("delete", "o:a"),
        ("delete", "o:b"),
    ]


def test_a_null_becomes_an_empty_string_rather_than_the_word_none(redis) -> None:
    RedisSink(prefix="o:", key_field="id").write(
        pa.table({"id": ["a"], "note": pa.array([None], pa.string())}), "orders"
    )
    assert ("hset", "o:a", {"note": ""}) in redis.log


def test_a_row_with_no_key_column_names_the_keyword_that_fixes_it(redis) -> None:
    with pytest.raises(BackendError, match="key_field"):
        RedisSink(key_field="missing").write(_ROWS, "orders")


def test_the_client_is_closed_even_when_a_row_is_rejected(redis) -> None:
    with pytest.raises(BackendError):
        RedisSink(key_field="missing").write(_ROWS, "orders")
    assert ("close",) in redis.log


# --- Elasticsearch --------------------------------------------------------------------


class _FakeElasticsearch:
    def __init__(self, failures: int = 0) -> None:
        self.operations: list = []
        self.deleted: list = []
        self._failures = failures

    def bulk(self, *, operations, refresh):
        self.operations.append((operations, refresh))
        items = [{"index": {"status": 201}} for _ in operations]
        if self._failures:
            items[0] = {"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}
            return {"errors": True, "items": items}
        return {"errors": False, "items": items}

    def delete_by_query(self, **kwargs):
        self.deleted.append(kwargs)

    def close(self) -> None:
        pass


@pytest.fixture
def es(monkeypatch):
    client = _FakeElasticsearch()
    monkeypatch.setattr(ElasticsearchSink, "_client", lambda self: client)
    return client


def test_upsert_addresses_each_document_by_its_id(es) -> None:
    ElasticsearchSink(key_field="id").write(_ROWS, "docs")
    operations, _refresh = es.operations[0]
    assert operations[0] == {"index": {"_index": "docs", "_id": "a"}}
    assert operations[1] == {"id": "a", "amount": 10}


def test_append_lets_the_cluster_assign_ids(es) -> None:
    ElasticsearchSink(mode="append").write(_ROWS, "docs")
    operations, _refresh = es.operations[0]
    assert operations[0] == {"index": {"_index": "docs"}}


def test_delete_sends_only_the_action_line(es) -> None:
    ElasticsearchSink(mode="delete", key_field="id").write(_ROWS, "docs")
    operations, _refresh = es.operations[0]
    assert operations == [
        {"delete": {"_index": "docs", "_id": "a"}},
        {"delete": {"_index": "docs", "_id": "b"}},
    ]


def test_overwrite_clears_the_index_before_indexing(es) -> None:
    ElasticsearchSink(mode="overwrite", key_field="id").write(_ROWS, "docs")
    assert es.deleted and es.deleted[0]["query"] == {"match_all": {}}
    assert es.operations


def test_a_per_item_failure_inside_a_200_is_surfaced(monkeypatch) -> None:
    """The whole reason the response body is read rather than the status code."""
    client = _FakeElasticsearch(failures=1)
    monkeypatch.setattr(ElasticsearchSink, "_client", lambda self: client)
    with pytest.raises(BackendError, match="operations failed"):
        ElasticsearchSink(key_field="id").write(_ROWS, "docs")


def test_refresh_is_off_unless_asked_for(es) -> None:
    ElasticsearchSink(key_field="id").write(_ROWS, "docs")
    assert es.operations[0][1] == "false"
    ElasticsearchSink(key_field="id", refresh=True).write(_ROWS, "docs")
    assert es.operations[1][1] == "true"


def test_a_missing_key_field_names_the_keyword_that_fixes_it(es) -> None:
    with pytest.raises(BackendError, match="key_field"):
        ElasticsearchSink(key_field="missing").write(_ROWS, "docs")


# --- MongoDB: the two defects the writer path used to carry ---------------------------


def test_the_collection_is_the_write_destination_not_a_required_keyword() -> None:
    """`ds.write.mongo("orders", ...)` raised TypeError on every call before this."""
    sink = MongoSink(uri="mongodb://h", database="shop")
    assert sink.collection is None


def test_write_mongo_reaches_the_sink_rather_than_failing_on_a_signature(monkeypatch) -> None:
    seen: dict = {}

    class _Recorder(MongoSink):
        def _apply(self, rows, path):
            seen["rows"], seen["path"] = rows, path

    monkeypatch.setitem(SINKS._items, "mongo", _Recorder)
    bt.from_pydict({"_id": ["a"], "amount": [1]}).write.mongo(
        "orders", uri="mongodb://h", database="shop"
    )
    assert seen["path"] == "orders"
    assert seen["rows"] == [{"_id": "a", "amount": 1}]


def test_a_credential_reference_is_resolved_where_the_connection_is_dialed(monkeypatch) -> None:
    """`env:` worked for reads and silently did not for writes."""
    monkeypatch.setenv("MONGO_URL", "mongodb://real-host:27017")
    sink = MongoSink(uri="env:MONGO_URL", database="shop")
    assert sink._secret("uri") == "mongodb://real-host:27017"


def test_bulk_sink_is_abstract_so_a_subclass_must_say_how_it_writes() -> None:
    with pytest.raises(TypeError):
        BulkSink()  # type: ignore[abstract]


# --- HBase ----------------------------------------------------------------------------


class _FakeHBaseBatch:
    def __init__(self, log: list) -> None:
        self.log = log

    def put(self, key, cells):
        self.log.append(("put", key, cells))

    def delete(self, key):
        self.log.append(("delete", key))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.log.append(("send",))


class _FakeHBaseTable:
    def __init__(self, log: list) -> None:
        self.log = log

    def batch(self):
        return _FakeHBaseBatch(self.log)


class _FakeHBaseConnection:
    def __init__(self) -> None:
        self.log: list = []
        self.tables: list[str] = []

    def table(self, name):
        self.tables.append(name)
        return _FakeHBaseTable(self.log)

    def close(self) -> None:
        self.log.append(("close",))


@pytest.fixture
def hbase(monkeypatch):
    connection = _FakeHBaseConnection()
    monkeypatch.setattr(HBaseSink, "_connection", lambda self: connection)
    return connection


def test_a_column_without_a_family_gets_the_default_one(hbase) -> None:
    HBaseSink(host="thrift").write(pa.table({"row_key": ["r1"], "amount": [10]}), "events")
    assert ("put", b"r1", {b"cf:amount": b"10"}) in hbase.log


def test_a_qualified_column_keeps_its_family_so_a_read_round_trips(hbase) -> None:
    """`HBaseSource` emits `family:qualifier` names; re-qualifying them would nest them."""
    HBaseSink(host="thrift").write(pa.table({"row_key": ["r1"], "d:amount": [10]}), "events")
    assert ("put", b"r1", {b"d:amount": b"10"}) in hbase.log


def test_a_null_cell_is_left_unwritten_rather_than_stored_as_the_word_none(hbase) -> None:
    HBaseSink(host="thrift").write(
        pa.table({"row_key": ["r1"], "amount": pa.array([None], pa.int64())}), "events"
    )
    assert ("put", b"r1", {}) in hbase.log


def test_delete_removes_whole_rows_by_key(hbase) -> None:
    HBaseSink(host="thrift", mode="delete").write(pa.table({"row_key": ["r1", "r2"]}), "events")
    assert [entry for entry in hbase.log if entry[0] == "delete"] == [
        ("delete", b"r1"),
        ("delete", b"r2"),
    ]


def test_the_destination_names_the_table(hbase) -> None:
    HBaseSink(host="thrift").write(pa.table({"row_key": ["r1"]}), "events")
    assert hbase.tables == ["events"]


def test_an_hbase_row_with_no_key_column_names_the_keyword_that_fixes_it(hbase) -> None:
    with pytest.raises(BackendError, match="key_field"):
        HBaseSink(host="thrift", key_field="missing").write(pa.table({"row_key": ["r1"]}), "events")
