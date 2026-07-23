"""Source-side predicate pushdown wired into the connector sources.

These run without any backend installed: they exercise the pure IR→backend-filter
translators (`to_mongo_filter`, `to_iceberg_expression`) and assert each opted-in
connector source advertises `supports_predicate` and accepts a `predicate=` kwarg
on `read`. The engine always keeps its `Filter` re-check, so pushdown is a pure
I/O optimization — these tests cover the translation and the opt-in contract, not
a live connection.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher.io.formats.lakehouse.iceberg import IcebergSource
from batcher.io.formats.nosql.mongo import MongoSource
from batcher.io.formats.sql.bigquery import BigQuerySource
from batcher.io.formats.sql.clickhouse import ClickHouseSource
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.databricks import DatabricksSource
from batcher.io.formats.sql.odbc import ODBCSource
from batcher.io.formats.sql.snowflake import SnowflakeSource
from batcher.io.predicate import to_iceberg_expression, to_mongo_filter
from batcher.io.source.read import plan_splits

pytestmark = pytest.mark.unit

# Every source class that opts into source-side predicate pushdown.
_PUSHDOWN_SOURCES = [
    BigQuerySource,
    ClickHouseSource,
    ConnectorXSource,
    DatabricksSource,
    IcebergSource,
    MongoSource,
    ODBCSource,
    SnowflakeSource,
]

# The SQL sources whose splits must carry the pushdown *in their own query*, with a
# kwargs recipe that constructs one without contacting a server. A split is what a
# worker rebuilds its reader from, so a predicate left outside the split's SQL never
# reaches the server at all.
_SQL_SPLIT_SOURCES = [
    (ClickHouseSource, {"query": "SELECT * FROM t", "host": "h", "client_kwargs": {}}),
    (ConnectorXSource, {"query": "SELECT * FROM t", "conn_uri": "postgresql://h/db"}),
    (ODBCSource, {"query": "SELECT * FROM t", "dsn": "d"}),
    (
        DatabricksSource,
        {
            "query": "SELECT * FROM t",
            "server_hostname": "h",
            "http_path": "/sql/1.0/warehouses/abc",
            "access_token": "tok",
        },
    ),
]


# --- to_mongo_filter (pure, no driver dependency) -----------------------------
def test_mongo_comparisons():
    assert to_mongo_filter((bt.col("x") > 5).to_ir()) == {"x": {"$gt": 5}}
    assert to_mongo_filter((bt.col("x") == 3).to_ir()) == {"x": {"$eq": 3}}
    assert to_mongo_filter((bt.col("x") <= 7).to_ir()) == {"x": {"$lte": 7}}


def test_mongo_flipped_comparison():
    # literal-on-left flips the operator so the column stays the document key.
    assert to_mongo_filter((bt.lit(5) < bt.col("x")).to_ir()) == {"x": {"$gt": 5}}


def test_mongo_and_or():
    ir = ((bt.col("x") > 5) & (bt.col("y") == 3)).to_ir()
    assert to_mongo_filter(ir) == {"$and": [{"x": {"$gt": 5}}, {"y": {"$eq": 3}}]}
    ir_or = ((bt.col("x") > 5) | (bt.col("y") == 3)).to_ir()
    assert to_mongo_filter(ir_or) == {"$or": [{"x": {"$gt": 5}}, {"y": {"$eq": 3}}]}


def test_mongo_is_null():
    assert to_mongo_filter(bt.col("x").is_null().to_ir()) == {"x": None}
    assert to_mongo_filter(bt.col("x").is_not_null().to_ir()) == {"x": {"$ne": None}}


def test_mongo_rejects_column_vs_column():
    assert to_mongo_filter((bt.col("x") > bt.col("y")).to_ir()) is None


# --- to_iceberg_expression (requires pyiceberg) -------------------------------
def test_iceberg_comparison_and_and():
    pytest.importorskip("pyiceberg")
    ir = ((bt.col("x") > 5) & (bt.col("y") == 3)).to_ir()
    assert to_iceberg_expression(ir) is not None


def test_iceberg_rejects_column_vs_column():
    pytest.importorskip("pyiceberg")
    assert to_iceberg_expression((bt.col("x") > bt.col("y")).to_ir()) is None


# --- per-source opt-in contract (no backend, no live connection) --------------
@pytest.mark.parametrize("source_cls", _PUSHDOWN_SOURCES)
def test_source_supports_predicate(source_cls):
    assert source_cls.supports_predicate is True


@pytest.mark.parametrize("source_cls", _PUSHDOWN_SOURCES)
def test_source_read_accepts_predicate(source_cls):
    params = inspect.signature(source_cls.read).parameters
    assert "predicate" in params


# --- the pushdown must live IN the split's own SQL -----------------------------
# `plan_splits` inspects the `splits` signature and only forwards `predicate=` /
# `projection=` when the source *declares* them. A source that omits a parameter is
# therefore silently planned without it: the worker rebuilds an unfiltered,
# unprojected read, the whole relation crosses the wire, and the engine's `Filter`
# discards the rows afterwards. Correct results, unbounded cost — so these assert on
# the SQL the split actually carries, not on the source's own `read()`.


@pytest.mark.parametrize(
    ("source_cls", "kwargs"), _SQL_SPLIT_SOURCES, ids=lambda v: getattr(v, "__name__", "")
)
def test_split_sql_carries_pushdown(source_cls, kwargs):
    predicate = (bt.col("x") > 5).to_ir()
    splits = plan_splits(source_cls(**kwargs), predicate=predicate, projection=["x", "y"])
    assert len(splits) == 1
    sql = splits[0].query
    # The predicate reached the split's SQL, so the *server* does the filtering.
    assert "WHERE" in sql
    assert "x > 5" in sql
    # ...and so does the column pruning.
    assert "SELECT x, y" in sql


def test_databricks_warehouse_split_carries_where():
    """A Databricks warehouse split must filter on the warehouse, not after Cloud Fetch.

    Regression: `DatabricksSource.splits` took no `projection` and called
    `_warehouse_split()` with no predicate, so it was the one SQL connector still
    rebuilding an unfiltered read on every distributed worker.
    """
    source = DatabricksSource(
        query="SELECT * FROM sales",
        server_hostname="h",
        http_path="/sql/1.0/warehouses/abc",
        access_token="tok",
    )
    splits = plan_splits(source, predicate=(bt.col("amount") > 100).to_ir())
    assert len(splits) == 1
    assert "WHERE amount > 100" in splits[0].query


def test_databricks_splits_declares_projection():
    """`plan_splits` forwards `projection=` only to a `splits` that declares it."""
    assert "projection" in inspect.signature(DatabricksSource.splits).parameters


def test_databricks_lakehouse_delegates_to_delta(monkeypatch):
    """The lakehouse branch keeps delegating to Delta, which prunes files by predicate.

    Delta splits are data files whose columns are pruned from the footer on the worker,
    so `projection` is deliberately not forwarded — only `predicate` is.
    """
    seen = {}

    class _FakeDelta:
        def splits(self, target_size=None, predicate=None):
            seen["target_size"] = target_size
            seen["predicate"] = predicate
            return ["delta-split"]

    source = DatabricksSource(table="c.s.t", workspace="https://w", token="tok")
    monkeypatch.setattr(DatabricksSource, "_delta_source", lambda self: _FakeDelta())

    predicate = (bt.col("x") > 5).to_ir()
    assert plan_splits(source, predicate=predicate, projection=["x"]) == ["delta-split"]
    assert seen["predicate"] == predicate
