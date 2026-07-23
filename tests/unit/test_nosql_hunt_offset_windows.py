"""Partition-cover correctness for the SKIP/LIMIT (offset-window) NoSQL sources.

`offset_windows` builds a disjoint, exhaustive cover whose **last** window is
unbounded (``limit == 0``) and starts at a non-zero offset. Neo4j (Cypher
``SKIP``/``LIMIT``) and Couchbase (SQL++ ``OFFSET``/``LIMIT``) each translate a
window into a query. Both guarded the offset on ``limit`` being truthy, so the
unbounded tail window silently dropped its ``SKIP``/``OFFSET`` and re-read the
*entire* result — every prior window's rows came back a second time (duplicates).

These tests capture the generated query for the tail window ``(offset > 0, 0)``
and assert the offset survives. They pass only with the fix.
"""

from __future__ import annotations

from typing import Any

import pytest

from batcher.io.formats.nosql.couchbase import CouchbaseSource
from batcher.io.formats.nosql.neo4j import Neo4jSource

pytestmark = pytest.mark.unit


class _FakeNeo4jSession:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __enter__(self) -> _FakeNeo4jSession:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def run(self, cypher: str) -> Any:
        self._log.append(cypher)
        return iter([])


class _FakeNeo4jDriver:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def session(self, database: str | None = None) -> _FakeNeo4jSession:
        return _FakeNeo4jSession(self._log)

    def close(self) -> None:
        pass


def test_neo4j_tail_window_keeps_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    src = Neo4jSource(
        uri="bolt://ignored",
        username="u",
        password="p",
        cypher="MATCH (n) RETURN n.id AS id",
        order_by="id",
    )
    monkeypatch.setattr(Neo4jSource, "_driver", lambda self: _FakeNeo4jDriver(log))

    # The unbounded tail window of a multi-segment cover: read from offset 5 to the end.
    list(src._read_partition((5, 0), None))

    partition_stmts = [c for c in log if "MATCH (n)" in c and "LIMIT 1" not in c]
    assert partition_stmts, "no partition query was issued"
    stmt = partition_stmts[-1]
    assert "SKIP 5" in stmt, f"tail window dropped its SKIP: {stmt!r}"
    assert "ORDER BY id" in stmt, f"tail window dropped its ORDER BY: {stmt!r}"
    # Unbounded tail: no LIMIT clause (LIMIT 1 is only the schema probe, filtered above).
    assert " LIMIT " not in stmt, f"tail window must be unbounded: {stmt!r}"


class _FakeCouchbaseResult:
    def rows(self) -> list[Any]:
        return []


class _FakeCluster:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.closed = 0

    def execute_query(self, stmt: str) -> _FakeCouchbaseResult:
        self._log.append(stmt)
        return _FakeCouchbaseResult()

    def close(self) -> None:
        # Modeled because the source now closes its cluster; a fake without `close()`
        # would make the leak fix untestable and quietly assert the old behavior.
        self.closed += 1


def test_couchbase_tail_window_keeps_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    src = CouchbaseSource(
        connstr="couchbases://ignored",
        username="u",
        password="p",
        database="db",
        scope="s",
        collection="c",
    )
    monkeypatch.setattr(CouchbaseSource, "_cluster", lambda self: _FakeCluster(log))

    # The unbounded tail window: read from offset 5 to the end of the collection.
    list(src._read_partition((5, 0), None))

    partition_stmts = [s for s in log if "FROM `db`.`s`.`c`" in s and "COUNT(*)" not in s]
    assert partition_stmts, "no partition query was issued"
    stmt = partition_stmts[-1]
    assert "OFFSET 5" in stmt, f"tail window dropped its OFFSET: {stmt!r}"
    assert "ORDER BY META(c).id" in stmt, f"tail window dropped its ORDER BY: {stmt!r}"
