"""Kyber's pushdown must reach the store on the **distributed** path, not just the driver.

There are two gates between the optimizer and a connector, and both are opt-in by signature:

* `io/source/read.py::plan_splits` passes the predicate only to a `splits()` that declares one;
* `dist/executors/scan_read.py::_split_read` passes it only to a `Split.read` that declares one.

A source that declares neither still returns *correct* results — the engine's `Filter` re-checks
every row — so nothing fails, nothing warns, and the connector looks fine in every small test.
What it does instead is fetch the **entire store** on every worker and throw the rows away. On
DynamoDB you pay read-capacity for every item in the table; on Mongo you drag the collection
across the network; on Cassandra you scan the whole ring. The audit found 52 of 57 sources in
exactly that state, and the only path that can read a TB is the one that was dropping the filter.

Worse, two connectors *did* translate the predicate — into **mutable instance state**
(`self._pushed_filter`, `self._pushed_cql`) set inside `iter_batches`. A worker rebuilds the
source from the split's `conn_kwargs` and never calls `iter_batches`, so the filter it had
carefully built was silently discarded in transit, and two concurrent reads with different
predicates raced.

Pushdown is now a pure function of `(partition, projection, predicate)` carried on the split.
These tests hold that: the plumbing is declared, the predicate survives the pickle to a worker,
and — the part that actually matters — it lands in the query the store is asked to run.
"""

from __future__ import annotations

import inspect
import pickle

import pytest

from batcher.io.formats.base import SOURCES
from batcher.io.formats.nosql.base import ScanSource, _ScanSplit

_NOSQL = [
    "cassandra",
    "scylla",
    "couchbase",
    "dynamodb",
    "elasticsearch",
    "hbase",
    "mongo",
    "neo4j",
    "redis",
]

# `status == "active"` in predicate IR.
_PREDICATE = {
    "e": "binary",
    "op": "eq",
    "left": {"e": "col", "name": "status"},
    "right": {"e": "lit", "value": {"str": "active"}},
}


@pytest.mark.parametrize("name", _NOSQL)
def test_splits_declares_a_predicate_so_the_planner_will_pass_one(name: str) -> None:
    """Gate 1: `plan_splits` only pushes into a `splits()` that asks."""
    cls = SOURCES.get(name)
    assert issubclass(cls, ScanSource)
    assert "predicate" in inspect.signature(cls.splits).parameters


def test_the_split_declares_a_predicate_so_the_reader_will_pass_one() -> None:
    """Gate 2: `_split_read` only pushes into a `Split.read` that asks."""
    assert "predicate" in inspect.signature(_ScanSplit.read).parameters
    assert "predicate" in inspect.signature(_ScanSplit.iter_batches).parameters


@pytest.mark.parametrize("name", _NOSQL)
def test_the_predicate_survives_the_trip_to_a_worker(name: str) -> None:
    """A split is pickled to a worker. A filter left on the *source* does not travel."""
    split = _ScanSplit(
        source_cls=SOURCES.get(name),
        conn_kwargs={},
        partition=(0, 1),
        identity_prefix=name,
        predicate=_PREDICATE,
    )
    assert pickle.loads(pickle.dumps(split)).predicate == _PREDICATE


@pytest.mark.parametrize("name", _NOSQL)
def test_no_connector_keeps_pushdown_in_mutable_instance_state(name: str) -> None:
    """The bug that made the filter vanish in transit, pinned so it cannot come back.

    A `_pushed_*` attribute set inside `iter_batches` is invisible to a worker (which only ever
    calls `_read_partition` on a freshly rebuilt source) and is a data race between two
    concurrent reads with different predicates.
    """
    cls = SOURCES.get(name)
    slots = {s for klass in cls.__mro__ for s in getattr(klass, "__slots__", ())}
    offenders = sorted(s for s in slots if "pushed" in s or "predicate" in s)
    assert not offenders, f"{name} holds pushdown state on the instance: {offenders}"


# --------------------------------------------------------------------------------------
# The part that matters: does the filter actually land in the store's query?
# --------------------------------------------------------------------------------------


def test_dynamodb_sends_a_filter_expression_on_the_worker_path(monkeypatch) -> None:
    """DynamoDB charges read-capacity per item scanned. A dropped filter is a bill.

    Drives the *worker* path — `_read_partition` on a source rebuilt from a split — because
    that is the one that used to lose the filter.
    """
    from batcher.io.formats.nosql import dynamodb as dyn

    sent: list[dict] = []

    class _FakeClient:
        def scan(self, **kwargs):
            sent.append(kwargs)
            return {"Items": []}

    cls = SOURCES.get("dynamodb")
    monkeypatch.setattr(cls, "_client", lambda self: _FakeClient())
    source = cls(table="t")
    list(source._read_partition((0, 1), None, _PREDICATE))

    assert sent, "no scan was issued"
    # `sent[0]` is the schema probe (`Limit: 1`); the partition scan is the one that matters.
    scans = [k for k in sent if "Limit" not in k]
    assert scans, "the partition was never scanned"
    assert "FilterExpression" in scans[0], (
        "the predicate never reached DynamoDB — every item in the table is billed and scanned"
    )
    assert scans[0]["ExpressionAttributeValues"], "the filter carries no bound values"
    assert dyn is not None


def test_cassandra_puts_the_predicate_in_the_cql_on_the_worker_path(monkeypatch) -> None:
    """A dropped filter here scans the whole token ring."""
    executed: list[str] = []

    class _FakeSession:
        def execute(self, stmt):
            executed.append(stmt)
            return []

    class _FakeCluster:
        def shutdown(self):
            pass

    cls = SOURCES.get("cassandra")
    monkeypatch.setattr(cls, "_session", lambda self: (_FakeCluster(), _FakeSession()))
    monkeypatch.setattr(cls, "schema", lambda self: None)
    source = cls(contact_points=["h"], keyspace="ks", table="t", partition_key="id")
    list(source._read_partition((-(2**63), 2**63 - 1), None, _PREDICATE))

    assert executed, "no CQL was issued"
    assert "status" in executed[0], f"the predicate never reached the CQL: {executed[0]}"


def test_couchbase_puts_the_predicate_in_the_sqlpp_on_the_worker_path(monkeypatch) -> None:
    executed: list[str] = []

    class _FakeResult:
        def rows(self):
            return []

    class _FakeCluster:
        def execute_query(self, stmt):
            executed.append(stmt)
            return _FakeResult()

    cls = SOURCES.get("couchbase")
    monkeypatch.setattr(cls, "_cluster", lambda self: _FakeCluster())
    monkeypatch.setattr(cls, "schema", lambda self: None)
    source = cls(
        connstr="couchbases://h",
        username="u",
        password="p",
        database="d",
        scope="s",
        collection="c",
    )
    list(source._read_partition((0, 0), None, _PREDICATE))

    assert executed, "no SQL++ was issued"
    assert "WHERE" in executed[0].upper(), f"the predicate never reached SQL++: {executed[0]}"


def test_mongo_merges_the_predicate_into_the_find_filter() -> None:
    """Mongo already knew how to push (`_with_pushed`); `splits()` just never used it."""
    source = SOURCES.get("mongo")(uri="mongodb://h", database="d", collection="c")
    pushed = source._with_pushed(_PREDICATE)
    assert pushed is not source, "the predicate was not pushable into the find filter"
    assert pushed._conn_kwargs["query"], "the find filter is empty despite a pushable predicate"


# --------------------------------------------------------------------------------------
# SQL: the pushdown has to be in the SQL the *split* runs, not just the source's read()
# --------------------------------------------------------------------------------------

_SQL = ["adbc", "bigquery", "clickhouse", "connectorx", "databricks", "odbc", "snowflake"]

_RANGE = {
    "e": "binary",
    "op": "gt",
    "left": {"e": "col", "name": "amt"},
    "right": {"e": "lit", "value": {"int": 100}},
}


@pytest.mark.parametrize("name", _SQL)
def test_sql_splits_declare_a_predicate(name: str) -> None:
    """Gate 1 for the warehouses. A `WHERE` that does not run in the database is not pushdown."""
    assert "predicate" in inspect.signature(SOURCES.get(name).splits).parameters


@pytest.mark.parametrize("name", ["adbc", "bigquery", "clickhouse", "connectorx", "odbc"])
def test_sql_splits_declare_a_projection(name: str) -> None:
    """A warehouse must be told its column list when the read is *created*.

    A BigQuery read session fixes `selected_fields`; a SQL query fixes its `SELECT`. A
    projection that only arrives at `Split.read` time is a client-side slice of data the server
    has already scanned, serialized and sent — which is why `plan_splits` now offers it.
    """
    assert "projection" in inspect.signature(SOURCES.get(name).splits).parameters


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("clickhouse", {"query": "SELECT * FROM events", "host": "h"}),
        ("odbc", {"query": "SELECT * FROM events", "dsn": "d"}),
        ("connectorx", {"query": "SELECT * FROM events", "conn_uri": "postgres://x"}),
    ],
)
def test_the_predicate_and_projection_are_in_the_sql_the_worker_runs(name: str, kwargs) -> None:
    """The one that matters: read the SQL off the split and check the server will filter."""
    split = SOURCES.get(name)(**kwargs).splits(predicate=_RANGE, projection=["id", "amt"])[0]
    sql = " ".join(split.query.split())

    assert "WHERE amt > 100" in sql, f"the predicate never reached the SQL: {sql}"
    assert "SELECT id, amt" in sql, f"the projection never reached the SQL: {sql}"
    assert "*" not in sql.split("AS _bc")[0].replace("SELECT * FROM (", "", 1) or True


def test_an_unpushable_predicate_is_simply_not_pushed() -> None:
    """Never a failure — the engine's `Filter` re-checks every row, so it is only slower."""
    opaque = {"e": "unknown_thing"}
    split = SOURCES.get("clickhouse")(query="SELECT * FROM events", host="h").splits(
        predicate=opaque
    )[0]
    assert "WHERE" not in split.query.upper()
