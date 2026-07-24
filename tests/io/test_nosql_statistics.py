"""Cheap catalog metadata for NoSQL stores: Elasticsearch `_count`, DynamoDB `DescribeTable`.

An operational store has no footer, but most maintain a cheap metadata endpoint that states
a table's cardinality (and sometimes its size) without a scan. Surfacing it lets Kyber size
joins and the worker fan-out from a real number instead of a default guess. Two are wired and
pinned here, with fake clients (no live server), against the invariant that governs every
metadata shortcut:

* **Elasticsearch** answers `row_count()` from the `_count` API, which is *exact* for the
  base DSL query — so it may size cardinality and answer `count()`.
* **DynamoDB** answers `statistics()` from `DescribeTable`'s `ItemCount`/`TableSizeBytes`,
  which refresh only ~every six hours — so it is `exact_rows=False`, advisory only, and must
  never answer an exact `count()`.
"""

from __future__ import annotations

from typing import Any

import pytest

from batcher.io.formats.nosql import DynamoDBSource, ElasticsearchSource

pytestmark = pytest.mark.unit


# --- Elasticsearch _count -------------------------------------------------------


class _FakeESCountClient:
    """A minimal ES client exposing `count()` and recording how it was called."""

    def __init__(self, count: int, *, raise_on_count: bool = False) -> None:
        self._count = count
        self._raise = raise_on_count
        self.count_calls: list[dict[str, Any]] = []
        self.closed = False

    def count(self, *, index: str, query: dict[str, Any]) -> dict[str, Any]:
        if self._raise:
            raise RuntimeError("cluster unavailable")
        self.count_calls.append({"index": index, "query": query})
        return {"count": self._count}

    def close(self) -> None:
        self.closed = True


def test_elasticsearch_row_count_uses_count_api(monkeypatch) -> None:
    """`row_count()` is the exact `_count`, and it closes the client it opened."""
    client = _FakeESCountClient(4200)
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)
    src = ElasticsearchSource(hosts="http://es:9200", index="orders")

    assert src.row_count() == 4200
    assert client.count_calls == [{"index": "orders", "query": {"match_all": {}}}]
    assert client.closed  # the connection is released, not leaked


def test_elasticsearch_statistics_is_exact(monkeypatch) -> None:
    """The exact `_count` surfaces as an exact `SourceStatistics` the estimator trusts."""
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: _FakeESCountClient(7))
    from batcher.io.source import source_statistics

    stats = source_statistics(ElasticsearchSource(hosts="http://es:9200", index="i"))
    assert stats is not None
    assert stats.row_count == 7
    assert stats.exact_rows is True


def test_elasticsearch_esql_has_no_cheap_count(monkeypatch) -> None:
    """An ES|QL result has no cheap count, so `row_count()` declines without a call."""
    client = _FakeESCountClient(1)
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)
    src = ElasticsearchSource(hosts="http://es:9200", index="i", esql="FROM i")

    assert src.row_count() is None
    assert client.count_calls == []  # no query issued at all


def test_elasticsearch_row_count_failure_is_none(monkeypatch) -> None:
    """A `_count` that raises degrades to None rather than failing the plan."""
    client = _FakeESCountClient(1, raise_on_count=True)
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda _self: client)
    src = ElasticsearchSource(hosts="http://es:9200", index="i")
    assert src.row_count() is None


# --- DynamoDB DescribeTable -----------------------------------------------------


class _FakeDynamoClient:
    """A minimal DynamoDB client exposing `describe_table`."""

    def __init__(self, table_desc: dict[str, Any] | None, *, raise_on_describe: bool = False):
        self._desc = table_desc
        self._raise = raise_on_describe
        self.describe_calls: list[str] = []

    def describe_table(self, *, TableName: str) -> dict[str, Any]:  # boto3's capitalized API
        if self._raise:
            raise RuntimeError("AccessDeniedException")
        self.describe_calls.append(TableName)
        return {"Table": self._desc}


def test_dynamodb_statistics_advisory_count_and_size(monkeypatch) -> None:
    """`statistics()` carries ItemCount + TableSizeBytes, marked advisory (never exact)."""
    client = _FakeDynamoClient({"ItemCount": 5000, "TableSizeBytes": 1_048_576})
    monkeypatch.setattr(DynamoDBSource, "_client", lambda _self: client)
    src = DynamoDBSource(table="events", region_name="us-east-1")

    stats = src.statistics()
    assert stats is not None
    assert stats.row_count == 5000
    assert stats.byte_size == 1_048_576
    assert stats.exact_rows is False  # DescribeTable refreshes ~every 6 hours
    assert client.describe_calls == ["events"]


def test_dynamodb_row_count_stays_none(monkeypatch) -> None:
    """`row_count()` is the exact-or-None contract; an estimate must not answer it."""
    client = _FakeDynamoClient({"ItemCount": 5000, "TableSizeBytes": 1})
    monkeypatch.setattr(DynamoDBSource, "_client", lambda _self: client)
    assert DynamoDBSource(table="events", region_name="us-east-1").row_count() is None


def test_dynamodb_statistics_size_only(monkeypatch) -> None:
    """A table with a size but a zero/absent count still contributes its byte size."""
    client = _FakeDynamoClient({"TableSizeBytes": 2048})
    monkeypatch.setattr(DynamoDBSource, "_client", lambda _self: client)
    stats = DynamoDBSource(table="t", region_name="us-east-1").statistics()
    assert stats is not None
    assert stats.byte_size == 2048
    assert stats.row_count is None


def test_dynamodb_statistics_empty_is_none(monkeypatch) -> None:
    """A description with neither counter yields no statistics record."""
    client = _FakeDynamoClient({})
    monkeypatch.setattr(DynamoDBSource, "_client", lambda _self: client)
    assert DynamoDBSource(table="t", region_name="us-east-1").statistics() is None


def test_dynamodb_statistics_failure_is_none(monkeypatch) -> None:
    """A DescribeTable that raises (no permission) degrades to None."""
    client = _FakeDynamoClient(None, raise_on_describe=True)
    monkeypatch.setattr(DynamoDBSource, "_client", lambda _self: client)
    assert DynamoDBSource(table="t", region_name="us-east-1").statistics() is None
