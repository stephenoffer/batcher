"""Audit of the Elasticsearch / HBase / Neo4j / Redis connectors and their shared spine.

Every test here pins a bug that shipped: a `schema()` that ran the user's whole query, an
`identity()` that collided across servers so Kyber applied one relation's cardinalities to
another, a password persisted into that identity, and connections opened on paths that
never closed them. Each one passed review and the existing suite while being wrong.

The drivers are not installed, so each connector's connection factory (`_client` /
`_connection` / `_driver`) is replaced with a fake that models the parts of the real API
the connector actually calls — including, crucially, that a schema is available *without*
consuming the result, which is the whole point of the fix it guards.
"""

from __future__ import annotations

import io
import subprocess
import sys
from typing import Any

import pyarrow as pa
import pytest

from batcher.io.formats.nosql.elasticsearch import ElasticsearchSource
from batcher.io.formats.nosql.hbase import HBaseSource
from batcher.io.formats.nosql.neo4j import Neo4jSource
from batcher.io.formats.nosql.redis import RedisSource

pytestmark = pytest.mark.unit


# --- Arrow IPC fixtures ------------------------------------------------------


def _ipc_bytes(batches: list[pa.RecordBatch], *, corrupt_tail: bool = False) -> bytes:
    """Serialize `batches` as an Arrow IPC stream, optionally with an unreadable tail.

    `corrupt_tail` is what makes laziness *observable*. The schema message and the first
    batch stay valid, so a reader that streams gets both; the trailing bytes are a bogus
    message header, so a reader that calls `read_all()` — materializing the whole result
    to answer a question about its columns — raises instead. A test that merely compared
    outputs could not tell the two apart.
    """
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, batches[0].schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    raw = buf.getvalue()
    if not corrupt_tail:
        return raw
    return raw[:-8] + b"\xff\xff\xff\xff" + (2**31 - 1).to_bytes(4, "little")


_ESQL_BATCH = pa.RecordBatch.from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])


# --- fakes -------------------------------------------------------------------


class _FakeESQL:
    def __init__(self, owner: _FakeESClient) -> None:
        self._owner = owner

    def query(self, *, query: str, format: str) -> Any:
        self._owner.queries.append(query)
        assert format == "arrow"
        return _FakeResponse(self._owner.payload)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body


class _FakeESClient:
    """Models `elasticsearch.Elasticsearch` for the calls the connector makes."""

    def __init__(self, payload: bytes = b"", hits: list[dict] | None = None) -> None:
        self.payload = payload
        self.queries: list[str] = []
        self.searches: list[dict] = []
        self.closed = 0
        self._hits = hits or []
        self.esql = _FakeESQL(self)

    def search(self, **kwargs: Any) -> dict:
        self.searches.append(kwargs)
        return {"hits": {"hits": [{"_source": h} for h in self._hits]}}

    def close(self) -> None:
        self.closed += 1


class _FakeHBaseTable:
    def __init__(self, rows: list[tuple[bytes, dict]], log: list[str]) -> None:
        self._rows = rows
        self._log = log

    def regions(self) -> list[dict]:
        return [{"start_key": b""}]

    def scan(self, **_kwargs: Any) -> Any:
        log = self._log

        def gen():
            try:
                yield from self._rows
            finally:
                log.append("scanner-closed")

        return gen()


class _FakeHBaseConnection:
    def __init__(self, rows: list[tuple[bytes, dict]], log: list[str]) -> None:
        self._rows = rows
        self._log = log
        log.append("open")

    def table(self, _name: str) -> _FakeHBaseTable:
        return _FakeHBaseTable(self._rows, self._log)

    def close(self) -> None:
        self._log.append("close")


class _FakeNeo4jSession:
    def __init__(self, rows: list[dict], fail: bool) -> None:
        self._rows = rows
        self._fail = fail

    def __enter__(self) -> _FakeNeo4jSession:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def run(self, _cypher: str) -> list[dict]:
        if self._fail:
            raise RuntimeError("no CALL subqueries on this server")
        return self._rows


class _FakeNeo4jDriver:
    def __init__(self, rows: list[dict], log: list[str], *, fail: bool = False) -> None:
        self._rows = rows
        self._fail = fail
        self._log = log
        log.append("open")

    def session(self, **_kwargs: Any) -> _FakeNeo4jSession:
        return _FakeNeo4jSession(self._rows, self._fail)

    def close(self) -> None:
        self._log.append("close")


class _FakeRedisClient:
    """Models `redis.Redis`; deliberately has no ``cluster`` attribute (non-cluster)."""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self.closed = 0
        self.scans = 0

    def scan(self, *, cursor: int, match: str, count: int) -> tuple[int, list[str]]:
        self.scans += 1
        return 0, list(self._keys)

    def get(self, key: str) -> str:
        return f"v:{key}"

    def close(self) -> None:
        self.closed += 1


# =============================================================================
# Bug class 3 — identity collisions across connections
# =============================================================================


def test_elasticsearch_identity_discriminates_the_cluster() -> None:
    """The same index name on two clusters must not share one statistics key.

    `_identity_suffix` returned the bare index, so ``orders`` on production and ``orders``
    on staging were literally the same string. Kyber then plans the small index with the
    large one's cardinalities, and nothing errors.
    """
    prod = ElasticsearchSource(hosts="https://prod:9200", index="orders")
    staging = ElasticsearchSource(hosts="https://staging:9200", index="orders")
    assert prod.identity() != staging.identity()


def test_elasticsearch_identity_discriminates_the_query() -> None:
    """Two different reads of one index are two different relations."""
    everything = ElasticsearchSource(hosts="h", index="orders")
    filtered = ElasticsearchSource(hosts="h", index="orders", query={"term": {"eu": True}})
    esql = ElasticsearchSource(hosts="h", index="orders", esql="FROM orders | STATS n = COUNT(*)")
    assert len({everything.identity(), filtered.identity(), esql.identity()}) == 3


def test_redis_identity_discriminates_the_match_glob() -> None:
    """``user:*`` and ``session:*`` on one database are different relations.

    The suffix was ``host:port/db``, which drops `match` entirely — so a scan of a handful
    of session keys and a scan of every user shared a cardinality estimate.
    """
    users = RedisSource(host="h", match="user:*")
    sessions = RedisSource(host="h", match="session:*")
    assert users.identity() != sessions.identity()


def test_hbase_identity_discriminates_the_cluster() -> None:
    same_table_two_clusters = {
        HBaseSource(host="prod", table="events").identity(),
        HBaseSource(host="staging", table="events").identity(),
    }
    assert len(same_table_two_clusters) == 2


def test_identity_is_unchanged_by_a_password_rotation() -> None:
    """Rotating a credential must not orphan everything the optimizer has learned.

    This is why `connection_fingerprint` excludes credential-ish keys rather than hashing
    the kwargs wholesale: a key that moved on every rotation would silently return Kyber
    to cold estimates on whatever schedule the security team keeps.
    """
    before = RedisSource(host="h", password="hunter2").identity()
    after = RedisSource(host="h", password="hunter3").identity()
    assert before == after


def test_identity_is_stable_across_processes() -> None:
    """The fingerprint must be `sha256`, not `hash()`, which Python salts per process.

    A `hash()`-based identity changes on every run, so no statistic is ever looked up
    again. The feedback loop appears to work — stats are written, no error is raised —
    while never once improving a plan. Two interpreters with different hash seeds are the
    only way to see it.
    """
    script = (
        "from batcher.io.formats.nosql.redis import RedisSource;"
        "print(RedisSource(host='h', db=3, match='k:*').identity())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"identity is not stable across processes: {runs}"


# =============================================================================
# Bug class 4 — credentials in a persisted identity
# =============================================================================


def test_neo4j_identity_masks_a_password_embedded_in_the_bolt_uri() -> None:
    """A password inside the URI was being written to the metadata store.

    `identity()` is *persisted* as the learned-statistics key, so this outlives the
    process — strictly worse than a `repr` leak, which dies with the traceback.
    """
    source = Neo4jSource(
        uri="bolt://alice:hunter2@graph:7687",
        username="alice",
        password="hunter2",
        cypher="MATCH (n) RETURN n",
    )
    assert "hunter2" not in source.identity()
    assert "graph:7687" in source.identity()


def test_neo4j_identity_is_unchanged_by_rotating_the_uri_password() -> None:
    """Masking happens *before* fingerprinting, so a rotation keeps the same key.

    `connection_fingerprint` drops credentials by key name and cannot see one embedded in
    a larger string, so without `_fingerprint_material` the digest moved on every rotation
    and orphaned the relation's statistics.
    """

    def ident(password: str) -> str:
        return Neo4jSource(
            uri=f"bolt://alice:{password}@graph:7687",
            username="alice",
            password=password,
            cypher="MATCH (n) RETURN n",
        ).identity()

    assert ident("hunter2") == ident("hunter3")


# =============================================================================
# Bug class 2 — a schema() that runs the whole query
# =============================================================================


def test_elasticsearch_esql_schema_probes_with_limit_rather_than_running_the_query(
    monkeypatch,
) -> None:
    """`schema()` must ask the cluster for one row, not compute the whole aggregation.

    The plan needs a schema *before* it executes, so the unprobed version submitted the
    user's query twice: once to read its column names and throw every row away, and again
    to actually run it.
    """
    client = _FakeESClient(payload=_ipc_bytes([_ESQL_BATCH]))
    source = ElasticsearchSource(hosts="h", index="i", esql="FROM i | STATS n = COUNT(*)")
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)

    assert source.schema() == _ESQL_BATCH.schema
    assert client.queries == ["FROM i | STATS n = COUNT(*) | LIMIT 1"]


def test_elasticsearch_esql_schema_reads_only_the_ipc_header(monkeypatch) -> None:
    """The schema comes off the stream header; no batch is materialized to get it.

    The payload's first batch is valid and its tail is not, so a `read_all()` — the old
    `_esql_arrow` — raises, while reading the header succeeds. This is what distinguishes
    "cheap schema" from "expensive schema that happens to return the right answer".
    """
    client = _FakeESClient(payload=_ipc_bytes([_ESQL_BATCH], corrupt_tail=True))
    source = ElasticsearchSource(hosts="h", index="i", esql="FROM i")
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)

    assert source.schema() == _ESQL_BATCH.schema


def test_elasticsearch_search_schema_asks_for_one_document(monkeypatch) -> None:
    """The scroll path's schema probe stays a ``size=1`` search."""
    client = _FakeESClient(hits=[{"a": 1, "b": "x"}])
    source = ElasticsearchSource(hosts="h", index="i")
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)

    assert source.schema().names == ["a", "b"]
    assert [s["size"] for s in client.searches] == [1]


# =============================================================================
# Bug class 1 — streaming that materializes first
# =============================================================================


def test_elasticsearch_esql_iter_batches_streams_rather_than_materializing(monkeypatch) -> None:
    """The ES|QL read yields its first batch without reading the rest of the stream.

    ``_esql_arrow`` called `read_all()`, so the "streaming" entry point built the entire
    result as one `pa.Table` before yielding anything — defeating every caller that chose
    `iter_batches` precisely to bound memory. The corrupt tail makes the difference
    observable: streaming reaches the first batch, materializing raises.
    """
    client = _FakeESClient(payload=_ipc_bytes([_ESQL_BATCH], corrupt_tail=True))
    source = ElasticsearchSource(hosts="h", index="i", esql="FROM i")
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)

    batches = source._read_partition((0, 1), projection=None)
    assert next(batches).to_pylist() == _ESQL_BATCH.to_pylist()
    batches.close()


def test_elasticsearch_esql_iter_batches_yields_every_batch(monkeypatch) -> None:
    """Streaming must not lose rows — the whole result still arrives, batch by batch."""
    second = pa.RecordBatch.from_pylist([{"a": 3, "b": "z"}])
    client = _FakeESClient(payload=_ipc_bytes([_ESQL_BATCH, second]))
    source = ElasticsearchSource(hosts="h", index="i", esql="FROM i")
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)

    rows = [r for b in source._read_partition((0, 1), projection=None) for r in b.to_pylist()]
    assert rows == [*_ESQL_BATCH.to_pylist(), *second.to_pylist()]


# =============================================================================
# Resource leaks — connections opened on paths that never closed them
# =============================================================================


def test_redis_read_partition_closes_its_client(monkeypatch) -> None:
    """Redis had no cleanup at all — not even at GC, since nothing closed the client."""
    client = _FakeRedisClient(["a", "b"])
    source = RedisSource(host="h")
    monkeypatch.setattr(RedisSource, "_client", lambda _self: client)

    list(source._read_partition((0, 16_384), projection=None))
    assert client.closed == 1


def test_redis_client_is_closed_when_the_consumer_abandons_the_scan(monkeypatch) -> None:
    """A downstream LIMIT abandons the generator; the connection must still come back.

    This is the case a `try/finally` inside a generator exists for, and the case a missing
    one hides best: the normal path looks fine in every test that drains the iterator.
    """
    client = _FakeRedisClient([f"k{i}" for i in range(10)])
    source = RedisSource(host="h")
    monkeypatch.setattr(RedisSource, "_client", lambda _self: client)

    batches = source._read_partition((0, 16_384), projection=None)
    next(batches)
    batches.close()
    assert client.closed == 1


def test_elasticsearch_closes_its_client_when_the_consumer_abandons_the_read(monkeypatch) -> None:
    clients: list[_FakeESClient] = []

    def factory() -> _FakeESClient:
        client = _FakeESClient(payload=_ipc_bytes([_ESQL_BATCH]))
        clients.append(client)
        return client

    source = ElasticsearchSource(hosts="h", index="i", esql="FROM i")
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: factory())

    batches = source._read_partition((0, 1), projection=None)
    next(batches)
    batches.close()
    assert clients and all(c.closed == 1 for c in clients)


def test_neo4j_total_rows_closes_the_driver_on_the_failure_path(monkeypatch) -> None:
    """`_total_rows` had no `finally` at all, and failing is its *expected* path.

    An older server without CALL subqueries takes the `except` branch on every partition
    enumeration, leaking a Bolt driver and its whole connection pool each time.
    """
    log: list[str] = []
    source = Neo4jSource(uri="bolt://h", username="u", password="p", cypher="MATCH (n) RETURN n")
    monkeypatch.setattr(Neo4jSource, "_driver", lambda _self: _FakeNeo4jDriver([], log, fail=True))

    assert source._total_rows() is None
    assert log == ["open", "close"]


def test_neo4j_read_partition_does_not_strand_a_driver_when_schema_fails(monkeypatch) -> None:
    """`self.schema()` sat between `_driver()` and the `try`, so a raise leaked the driver.

    `schema()` dials Neo4j itself, so it is a *likely* raiser, not a theoretical one — and
    the stranded driver had no reference left that could ever close it.
    """
    log: list[str] = []
    source = Neo4jSource(uri="bolt://h", username="u", password="p", cypher="MATCH (n) RETURN n")
    monkeypatch.setattr(Neo4jSource, "_driver", lambda _self: _FakeNeo4jDriver([], log))
    monkeypatch.setattr(
        Neo4jSource, "schema", lambda _self: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError):
        list(source._read_partition((0, 0), projection=None))
    assert log.count("open") == log.count("close")


def test_hbase_read_partition_does_not_strand_a_connection_when_schema_fails(monkeypatch) -> None:
    """Same shape as Neo4j: `schema()` opened its own connection between the two."""
    log: list[str] = []
    source = HBaseSource(host="h", table="t")
    monkeypatch.setattr(HBaseSource, "_connection", lambda _self: _FakeHBaseConnection([], log))
    monkeypatch.setattr(
        HBaseSource, "schema", lambda _self: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError):
        list(source._read_partition(("", ""), projection=None))
    assert log.count("open") == log.count("close")


def test_hbase_closes_the_scanner_and_connection_when_the_consumer_abandons(monkeypatch) -> None:
    """A server-side HBase scanner is a held resource, not just a Python generator."""
    log: list[str] = []
    rows = [(f"k{i}".encode(), {b"cf:v": b"x"}) for i in range(5)]
    source = HBaseSource(host="h", table="t")
    monkeypatch.setattr(HBaseSource, "_connection", lambda _self: _FakeHBaseConnection(rows, log))
    monkeypatch.setattr(HBaseSource, "schema", lambda _self: pa.schema([("row_key", pa.string())]))

    batches = source._read_partition(("", ""), projection=None)
    next(batches)
    batches.close()
    assert "scanner-closed" in log
    assert log.count("open") == log.count("close")
