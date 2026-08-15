"""The `SingleResultQuerySource` base holds every backend to one pushdown contract.

ClickHouse, ODBC and ConnectorX each restated the same five `Source` methods, and the one
that matters is `splits`: **the pushdown has to be inside the SQL the split carries.** A
split is what a distributed worker rebuilds its reader from, so a predicate held on the
source and not in the split's own query never reaches the server — the worker issues an
unfiltered read, the whole relation crosses the wire, and the engine's `Filter` discards it
afterwards. Correct results, arbitrarily expensive, and invisible to any test that only
checks the rows.

Three copies meant three chances to forget it, and the copies had already drifted apart in
their comments about it. These tests hold the base — and therefore all three backends — to
the contract at once, so a fourth backend that subclasses it inherits the coverage.

The stubs replace only `_split_for`, the one hook a subclass owns. Everything asserted here
is the base's own behavior, which is the point: none of it is restated per backend any more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from batcher.io.formats.sql._source_base import SingleResultQuerySource

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

_TYPED = pa.schema([("k", pa.int64()), ("v", pa.string())])
#: What a driver returns for a probe it cannot type: columns with no usable type. Every
#: backend has at least one driver version that does this, which is why the probe has a
#: fallback rather than being trusted.
_UNTYPED = pa.schema([("k", pa.null()), ("v", pa.null())])

_PREDICATE = {
    "e": "binary",
    "op": "gt",
    "left": {"e": "col", "name": "k"},
    "right": {"e": "lit", "value": {"int": 3}},
}


@dataclass(frozen=True, slots=True)
class _StubSplit:
    """A split that records the SQL it was built with instead of dialing a server."""

    sql: str
    probe_schema: pa.Schema = field(default=_TYPED)

    def schema(self) -> pa.Schema:
        return self.probe_schema if "1 = 0" in self.sql else _TYPED

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        del projection
        return [pa.record_batch({"k": [1], "v": ["a"]})]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"stub:{self.sql}"


@dataclass(frozen=True, slots=True)
class _StubSource(SingleResultQuerySource):
    """The minimum a backend has to supply: how to build a split, and its identity."""

    query: str
    probe_schema: pa.Schema = field(default=_TYPED)

    def _split_for(self, sql: str) -> _StubSplit:
        return _StubSplit(sql, self.probe_schema)

    def identity(self) -> str:
        return f"stub:{self.query}"


def _sql_of(
    source: SingleResultQuerySource,
    *,
    target_size: int | None = None,
    predicate: dict | None = None,
    projection: list[str] | None = None,
) -> str:
    """The SQL the single split a `splits()` call vends will actually execute.

    Typed against `splits`'s real signature rather than `**kwargs`, so a change to that
    signature fails the checker here instead of needing an `arg-type` suppression to hide it.
    """
    splits = source.splits(target_size, predicate, projection)
    assert len(splits) == 1, "a single-result source is exactly one split"
    only = splits[0]
    assert isinstance(only, _StubSplit), "the stub source must vend the stub split"
    return only.sql


def test_a_pushed_predicate_is_inside_the_split_s_own_sql() -> None:
    """The whole reason the base exists: the worker's query is the filtered one."""
    sql = _sql_of(_StubSource("SELECT * FROM t"), predicate=_PREDICATE)
    assert "WHERE" in sql
    assert "k > 3" in sql


def test_a_pushed_projection_is_inside_the_split_s_own_sql() -> None:
    sql = _sql_of(_StubSource("SELECT * FROM t"), projection=["k"])
    assert sql.startswith("SELECT k FROM")


def test_a_predicate_filters_below_the_projection_that_drops_its_column() -> None:
    """`select("k").filter(col("v") == ...)` must not project `v` away before filtering.

    Projecting first yields ``SELECT * FROM (SELECT k FROM t) WHERE v = 'a'``, where `v` no
    longer exists — a hard "no such column" from the server, not a slow query. The base
    inherits the right order from `push_down`; this pins it at the level a backend sees.
    """
    predicate = {
        "e": "binary",
        "op": "eq",
        "left": {"e": "col", "name": "v"},
        "right": {"e": "lit", "value": {"str": "a"}},
    }
    sql = _sql_of(_StubSource("SELECT * FROM t"), predicate=predicate, projection=["k"])
    assert sql.index("WHERE") > sql.index("SELECT k FROM"), (
        "the WHERE must be nested inside the projection, not layered above it"
    )


def test_read_and_iter_batches_push_the_same_way_splits_does() -> None:
    """Three paths, one pushdown. A source that pushed only on `splits` was the old bug."""
    source = _StubSource("SELECT * FROM t")
    assert source.read(["k"], _PREDICATE) == _StubSplit(
        _sql_of(source, predicate=_PREDICATE, projection=["k"])
    ).read(["k"])
    assert list(source.iter_batches(["k"], _PREDICATE))


def test_target_size_cannot_split_a_single_result_source() -> None:
    """The server decides how it returns one result; no byte budget applies."""
    source = _StubSource("SELECT * FROM t")
    assert len(source.splits(1)) == 1
    assert len(source.splits(1 << 40)) == 1


def test_schema_asks_a_zero_row_probe_and_never_the_real_query() -> None:
    """A schema lookup that runs the query costs the query, and on a warehouse, an invoice."""
    source = _StubSource("SELECT * FROM big_join")
    assert source.schema() == _TYPED


def test_schema_falls_back_to_the_real_query_when_the_probe_is_untyped() -> None:
    """A driver that answers `WHERE 1 = 0` with null-typed columns teaches nothing.

    The fallback is what makes the probe safe to attempt unconditionally: worst case is the
    full read every one of these backends used to do anyway.
    """
    source = _StubSource("SELECT * FROM t", probe_schema=_UNTYPED)
    assert source.schema() == _TYPED


def test_every_registered_single_result_backend_shares_the_base() -> None:
    """The three backends the base was extracted from still route through it.

    Guards the regression where a backend is "fixed" by pasting the five methods back onto
    it, which is how they diverged the first time.
    """
    from batcher.io.formats.sql.clickhouse import ClickHouseSource
    from batcher.io.formats.sql.connectorx import ConnectorXSource
    from batcher.io.formats.sql.odbc import ODBCSource

    for cls in (ClickHouseSource, ODBCSource, ConnectorXSource):
        assert issubclass(cls, SingleResultQuerySource)
        assert cls.supports_predicate is True
        # The spine comes from the base, not from a re-pasted copy.
        for name in ("schema", "read", "iter_batches", "row_count", "splits", "_split"):
            assert name not in vars(cls), f"{cls.__name__} restates {name}"


def test_connectorx_is_the_one_backend_that_probes_unpartitioned() -> None:
    """Its split carries driver-internal parallelism, so a probe would fan into N queries."""
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    source = ConnectorXSource(
        query="SELECT * FROM t", conn_uri="postgres://h/db", partition_on="k", num_partitions=8
    )
    assert source._split().num_partitions == 8
    assert source._probe_split_for("SELECT 1").num_partitions == 1
    assert source._probe_split_for("SELECT 1").partition_on is None
