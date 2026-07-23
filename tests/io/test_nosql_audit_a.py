"""NoSQL connector audit — statistics identity, credential containment, connection lifetime.

Each test here pins a defect that shipped in `io/formats/nosql/` and that every existing
gate passed while it was wrong. They fall into three families:

* **Identity collisions.** `identity()` is the key learned statistics are persisted under,
  so two sources with equal identities are *one relation* to Kyber. Every connector keyed
  itself on a store-local name (``db.collection``, ``keyspace.table``, ``region/table``)
  and omitted the server, so the same name on production and staging shared one statistics
  entry. The symptom is not an error — it is Kyber planning one dataset with the other's
  cardinalities, indefinitely, because the key outlives the process.

* **Credentials in a persisted key.** A password in a `repr` is printed; a password in
  `identity()` is *written to the metadata store*. The tests below assert both that the
  secret is absent and that **rotating** it leaves the identity unchanged — the second
  half matters because a key that moves on rotation silently orphans everything the
  optimizer has learned about the table.

* **Connection lifetime.** Connections opened without `try`/`finally` (or held across
  `yield`s that a caller abandons) are released only whenever the garbage collector
  happens to run. Nothing fails; the connection count climbs.

No driver is installed for Mongo, Cassandra or Couchbase, so the connection entry points
(`_client`/`_session`/`_cluster`) are replaced with spies that model the real API surface —
including which operations are *streaming*, since a fake that returns a list where the
driver returns a cursor cannot tell a streaming read from a materializing one. Driver
modules are injected into `sys.modules` and removed again on teardown.

`boto3` *is* installed, and the DynamoDB tests still fake it (see `dynamo_types`): importing
it for real leaves it cached in `sys.modules`, which defeats the `__import__` block that
`test_nosql.py` uses to prove the missing-driver error, and breaks that test whenever this
module happens to run first.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pyarrow as pa
import pytest

from batcher.io.formats.nosql import (
    CassandraSource,
    CouchbaseSource,
    DynamoDBSource,
    MongoSink,
    MongoSource,
)

pytestmark = pytest.mark.unit


# --- spies -------------------------------------------------------------------


class _LifetimeLog:
    """Tracks how many connections are open at once, and how many were never closed."""

    def __init__(self) -> None:
        self.open = 0
        self.peak = 0
        self.closed = 0
        self.events: list[str] = []

    def opened(self) -> None:
        self.open += 1
        self.peak = max(self.peak, self.open)
        self.events.append("open")

    def shut(self) -> None:
        self.open -= 1
        self.closed += 1
        self.events.append("close")


class _FakeRows:
    """A streaming result: rows arrive lazily, one at a time, as a real cursor does."""

    def __init__(self, rows: list[dict[str, Any]], log: _LifetimeLog) -> None:
        self._rows = rows
        self._log = log

    def rows(self):
        for row in self._rows:
            self._log.events.append("row")
            yield row

    def __iter__(self):
        return iter(self.rows())


class _FakeCouchbaseCluster:
    def __init__(self, rows: list[dict[str, Any]], log: _LifetimeLog) -> None:
        self._rows = rows
        self._log = log
        log.opened()

    def execute_query(self, stmt: str) -> _FakeRows:
        self._log.events.append(f"query:{stmt.split()[1]}")
        return _FakeRows(self._rows, self._log)

    def close(self) -> None:
        self._log.shut()


class _FakeCassandraSession:
    def __init__(self, rows: list[Any], log: _LifetimeLog) -> None:
        self._rows = rows
        self._log = log

    def execute(self, stmt: str):
        for row in self._rows:
            self._log.events.append("row")
            yield row


class _FakeCassandraCluster:
    def __init__(self, log: _LifetimeLog) -> None:
        self._log = log
        log.opened()

    def shutdown(self) -> None:
        self._log.shut()


class _Row:
    """Stands in for the driver's namedtuple row, which exposes `_asdict()`."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def _asdict(self) -> dict[str, Any]:
        return dict(self._data)


class _FakeMongoClient:
    def __init__(self, log: _LifetimeLog) -> None:
        self._log = log
        log.opened()

    def __getitem__(self, _name: str) -> _FakeMongoClient:
        return self

    def count_documents(self, _query: dict) -> int:
        return 2

    def close(self) -> None:
        self._log.shut()


@pytest.fixture
def mongo_driver(monkeypatch) -> list[str]:
    """Inject minimal ``pymongo``/``pymongoarrow`` modules; return the call log."""
    calls: list[str] = []

    def _find_arrow_all(_coll, _query, projection=None, limit=None):
        calls.append("find_arrow_all")
        return pa.table({"id": [1, 2, 3, 4]})

    api = types.ModuleType("pymongoarrow.api")
    api.find_arrow_all = _find_arrow_all
    arrow = types.ModuleType("pymongoarrow")
    arrow.api = api

    class _ReplaceOne:
        def __init__(self, flt, doc, upsert=False):
            self.flt = flt

    pymongo = types.ModuleType("pymongo")
    pymongo.ReplaceOne = _ReplaceOne
    pymongo.MongoClient = lambda uri: calls.append(f"connect:{uri}") or _SinkClient()

    class _SinkClient:
        def __getitem__(self, _n):
            return self

        def bulk_write(self, ops, ordered=False):
            calls.append("bulk_write")

        def close(self):
            calls.append("sink_close")

    monkeypatch.setitem(sys.modules, "pymongoarrow", arrow)
    monkeypatch.setitem(sys.modules, "pymongoarrow.api", api)
    monkeypatch.setitem(sys.modules, "pymongo", pymongo)
    return calls


# --- bug class 3: identity collisions ----------------------------------------


def test_mongo_identity_distinguishes_two_servers() -> None:
    """The same ``db.collection`` on two clusters must not share a statistics key."""
    prod = MongoSource(uri="mongodb://prod.internal/app", database="app", collection="orders")
    staging = MongoSource(uri="mongodb://staging.internal/app", database="app", collection="orders")
    assert prod.identity() != staging.identity()
    # Same connection + same collection is still one relation — the key must be stable,
    # or nothing is ever reused and the learned-stats loop silently never converges.
    twin = MongoSource(uri="mongodb://prod.internal/app", database="app", collection="orders")
    assert prod.identity() == twin.identity()


def test_cassandra_identity_distinguishes_two_rings() -> None:
    """``keyspace.table`` repeats across every ring; the contact points disambiguate."""
    a = CassandraSource(
        contact_points=["prod-1", "prod-2"], keyspace="app", table="events", partition_key="id"
    )
    b = CassandraSource(
        contact_points=["stage-1"], keyspace="app", table="events", partition_key="id"
    )
    assert a.identity() != b.identity()
    assert (
        a.identity()
        == CassandraSource(
            contact_points=["prod-1", "prod-2"], keyspace="app", table="events", partition_key="id"
        ).identity()
    )


def test_couchbase_identity_distinguishes_two_clusters() -> None:
    def build(connstr: str) -> CouchbaseSource:
        return CouchbaseSource(
            connstr=connstr,
            username="u",
            password="p",
            database="d",
            scope="s",
            collection="c",
        )

    assert build("couchbases://prod").identity() != build("couchbases://stage").identity()
    assert build("couchbases://prod").identity() == build("couchbases://prod").identity()


def test_dynamodb_identity_distinguishes_endpoints() -> None:
    """DynamoDB Local and the real service are different relations at the same name."""
    local = DynamoDBSource(
        table="orders", region_name="us-east-1", endpoint_url="http://localhost:8000"
    )
    cloud = DynamoDBSource(table="orders", region_name="us-east-1")
    assert local.identity() != cloud.identity()
    assert cloud.identity() == DynamoDBSource(table="orders", region_name="us-east-1").identity()


def test_split_identity_inherits_the_fixed_prefix() -> None:
    """Splits key off the source identity, so the collision reached the distributed path too."""
    prod = DynamoDBSource(table="t", region_name="us-east-1", endpoint_url="http://a")
    stage = DynamoDBSource(table="t", region_name="us-east-1", endpoint_url="http://b")
    assert {s.identity() for s in prod.splits()}.isdisjoint({s.identity() for s in stage.splits()})


# --- bug class 4: credentials in a persisted key -----------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("mongodb://user:hunter2@host/db", "mongodb://user:***@host/db"),
        ("mongodb://host/db", "mongodb://host/db"),
        ("mongodb://user@host/db", "mongodb://user@host/db"),
        ("mongodb+srv://u:p@a.example.com/db", "mongodb+srv://u:***@a.example.com/db"),
        (None, None),
    ],
)
def test_redact_mongo_uri(uri: str | None, expected: str | None) -> None:
    from batcher.io.formats.nosql.mongo import redact_mongo_uri

    assert redact_mongo_uri(uri) == expected


def test_mongo_identity_excludes_the_password_and_survives_rotation() -> None:
    """The password must be absent from the key, and rotating it must not move the key."""
    before = MongoSource(uri="mongodb://u:hunter2@h/db", database="d", collection="c")
    after = MongoSource(uri="mongodb://u:rotated9@h/db", database="d", collection="c")
    assert "hunter2" not in before.identity()
    assert "rotated9" not in after.identity()
    # Identity is persisted: if rotation changed it, every statistic learned about this
    # collection would be orphaned on a schedule.
    assert before.identity() == after.identity()
    # A different *user* is still a different connection.
    other = MongoSource(uri="mongodb://admin:hunter2@h/db", database="d", collection="c")
    assert other.identity() != before.identity()


def test_cassandra_identity_excludes_auth_and_survives_rotation() -> None:
    def build(password: str) -> CassandraSource:
        return CassandraSource(
            contact_points=["h"],
            keyspace="ks",
            table="t",
            partition_key="id",
            auth={"username": "u", "password": password},
        )

    assert "hunter2" not in build("hunter2").identity()
    assert build("hunter2").identity() == build("rotated9").identity()


def test_dynamodb_identity_excludes_aws_credentials_and_survives_rotation() -> None:
    """`connection_fingerprint` misses ``aws_``-prefixed keys, so they must not be passed."""

    def build(secret: str) -> DynamoDBSource:
        return DynamoDBSource(
            table="t",
            region_name="us-east-1",
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key=secret,
        )

    assert "hunter2" not in build("hunter2").identity()
    assert "AKIAEXAMPLE" not in build("hunter2").identity()
    assert build("hunter2").identity() == build("rotated9").identity()


def test_couchbase_identity_excludes_password_and_survives_rotation() -> None:
    def build(password: str) -> CouchbaseSource:
        return CouchbaseSource(
            connstr="couchbases://h",
            username="u",
            password=password,
            database="d",
            scope="s",
            collection="c",
        )

    assert "hunter2" not in build("hunter2").identity()
    assert build("hunter2").identity() == build("rotated9").identity()


# --- connection lifetime -----------------------------------------------------


def test_couchbase_infer_schema_closes_its_cluster(monkeypatch) -> None:
    log = _LifetimeLog()
    monkeypatch.setattr(
        CouchbaseSource, "_cluster", lambda self: _FakeCouchbaseCluster([{"id": 1}], log)
    )
    src = _couchbase()
    assert src.schema().names == ["id"]
    assert log.open == 0 and log.closed == 1


def test_couchbase_total_rows_closes_its_cluster_on_both_paths(monkeypatch) -> None:
    """Including the failure path — `_total_rows` swallows exceptions, and leaked on them."""
    log = _LifetimeLog()
    monkeypatch.setattr(CouchbaseSource, "_cluster", lambda self: _FakeCouchbaseCluster([7], log))
    assert _couchbase()._total_rows() == 7
    assert log.open == 0

    class _Boom(_FakeCouchbaseCluster):
        def execute_query(self, stmt: str):
            raise RuntimeError("analytics service unavailable")

    fail = _LifetimeLog()
    monkeypatch.setattr(CouchbaseSource, "_cluster", lambda self: _Boom([], fail))
    assert _couchbase()._total_rows() is None  # swallowed, as designed
    assert fail.open == 0, "the cluster was dropped on the exception path"


def test_couchbase_read_closes_its_cluster_and_streams(monkeypatch) -> None:
    log = _LifetimeLog()
    monkeypatch.setattr(
        CouchbaseSource,
        "_cluster",
        lambda self: _FakeCouchbaseCluster([{"id": i} for i in range(3)], log),
    )
    src = _couchbase()
    batches = list(src._read_partition((0, 0), ["id"]))
    assert sum(b.num_rows for b in batches) == 3
    assert log.open == 0, "the read cluster was never closed"
    # Rows are pulled lazily from the SDK result rather than listed up front.
    assert "row" in log.events


def test_couchbase_read_does_not_hold_two_clusters_at_once(monkeypatch) -> None:
    """Schema inference opens a cluster of its own; it must not nest inside the read's."""
    log = _LifetimeLog()
    monkeypatch.setattr(
        CouchbaseSource, "_cluster", lambda self: _FakeCouchbaseCluster([{"id": 1}], log)
    )
    list(_couchbase()._read_partition((0, 0), None))
    assert log.peak == 1, f"{log.peak} clusters were open simultaneously"


def test_cassandra_read_does_not_hold_two_clusters_at_once(monkeypatch) -> None:
    log = _LifetimeLog()

    def _fake_session(self):
        cluster = _FakeCassandraCluster(log)
        return cluster, _FakeCassandraSession([_Row({"id": 1}), _Row({"id": 2})], log)

    monkeypatch.setattr(CassandraSource, "_session", _fake_session)
    src = CassandraSource(contact_points=["h"], keyspace="ks", table="t", partition_key="id")
    batches = list(src._read_partition((0, 100), None))
    assert sum(b.num_rows for b in batches) == 2
    assert log.peak == 1, f"{log.peak} clusters were open simultaneously"
    assert log.open == 0


def test_mongo_read_closes_the_client_before_yielding(monkeypatch, mongo_driver) -> None:
    """A caller that abandons the generator after one batch must not leak the client.

    ``pymongoarrow`` has already materialized the whole partition by the time the first
    batch can be yielded, so there is nothing to hold the connection open for. Closing in a
    `finally` *around* the yields deferred it to garbage collection on every early exit.
    """
    log = _LifetimeLog()
    monkeypatch.setattr(MongoSource, "_client", lambda self: _FakeMongoClient(log))
    src = MongoSource(uri="mongodb://h/db", database="d", collection="c")
    gen = src._read_partition((None, None), None)
    next(gen)  # first batch only — then walk away without exhausting or closing
    assert log.open == 0, "the Mongo client was still open while batches were outstanding"
    gen.close()


# --- credential resolution parity between source and sink --------------------


def test_mongo_sink_resolves_a_secret_reference_uri(monkeypatch, mongo_driver) -> None:
    """`MongoSink` dialed the raw attribute, so an ``env:`` URI worked for reads only."""
    monkeypatch.setenv("BC_TEST_MONGO_URI", "mongodb://resolved-host/db")
    sink = MongoSink(uri="env:BC_TEST_MONGO_URI", database="d", collection="c")
    sink.write(pa.table({"_id": [1], "v": ["a"]}), "target")
    assert "connect:mongodb://resolved-host/db" in mongo_driver
    assert not any(c.startswith("connect:env:") for c in mongo_driver)


# --- DynamoDB scan mechanics -------------------------------------------------


@pytest.fixture
def dynamo_types(monkeypatch):
    """A stand-in ``boto3.dynamodb.types``, so these tests never import the real boto3.

    `boto3` *is* installed here, which makes using it tempting and wrong. Importing it
    leaves it in `sys.modules`, and `test_nosql.py::test_missing_driver_raises_for_each`
    proves its missing-driver error by blocking `builtins.__import__` — a block a cached
    module walks straight past. The connector then built a real client and raised
    `NoRegionError` instead of the expected `BackendError`.

    That failure appears only when this file runs *first*, so alphabetical ordering hid it
    and the pollution would have surfaced later as an unrelated, intermittent failure. The
    fake keeps the import from ever happening; `monkeypatch.setitem` removes the entries
    again on teardown, and the memoized decoder is cleared on both sides so neither a fake
    nor a real one survives this module.
    """
    from batcher.io.formats.nosql.dynamodb import _type_deserializer

    class _FakeTypeDeserializer:
        def deserialize(self, value: dict[str, Any]) -> Any:
            ((tag, raw),) = value.items()
            return int(raw) if tag == "N" else raw

    types_mod = types.ModuleType("boto3.dynamodb.types")
    types_mod.TypeDeserializer = _FakeTypeDeserializer
    types_mod.TypeSerializer = object
    dynamodb_mod = types.ModuleType("boto3.dynamodb")
    dynamodb_mod.types = types_mod
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.dynamodb = dynamodb_mod

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "boto3.dynamodb", dynamodb_mod)
    monkeypatch.setitem(sys.modules, "boto3.dynamodb.types", types_mod)
    _type_deserializer.cache_clear()
    yield
    _type_deserializer.cache_clear()


class _PagingDynamoClient:
    """Models `Scan`'s `LastEvaluatedKey` pagination, counting requests."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(dict(kwargs))
        page = len(self.requests)
        if kwargs.get("Limit") == 1:
            return {"Items": [{"id": {"N": "1"}}]}
        if page == 1:
            return {"Items": [{"id": {"N": "1"}}], "LastEvaluatedKey": {"id": {"N": "1"}}}
        return {"Items": [{"id": {"N": "2"}}]}


def test_dynamodb_scan_paginates_and_streams(monkeypatch, dynamo_types) -> None:
    client = _PagingDynamoClient()
    monkeypatch.setattr(DynamoDBSource, "_client", lambda self: client)
    src = DynamoDBSource(table="t", region_name="us-east-1")
    batches = list(src._read_partition((0, 1), ["id"]))
    assert sum(b.num_rows for b in batches) == 2
    assert len(client.requests) == 2  # the second request followed LastEvaluatedKey
    assert client.requests[1]["ExclusiveStartKey"] == {"id": {"N": "1"}}


def test_dynamodb_scan_shares_one_type_deserializer(dynamo_types) -> None:
    """The decoder is stateless; it was rebuilt per item in the connector's hottest loop."""
    from batcher.io.formats.nosql.dynamodb import _type_deserializer

    assert _type_deserializer() is _type_deserializer()


# --- helpers -----------------------------------------------------------------


def _couchbase() -> CouchbaseSource:
    return CouchbaseSource(
        connstr="couchbases://h",
        username="u",
        password="p",
        database="d",
        scope="s",
        collection="c",
    )
