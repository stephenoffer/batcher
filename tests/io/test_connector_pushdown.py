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
from batcher.io.formats.sql.odbc import ODBCSource
from batcher.io.formats.sql.snowflake import SnowflakeSource
from batcher.io.predicate import to_iceberg_expression, to_mongo_filter

pytestmark = pytest.mark.unit

# Every source class that opts into source-side predicate pushdown.
_PUSHDOWN_SOURCES = [
    BigQuerySource,
    ClickHouseSource,
    ConnectorXSource,
    IcebergSource,
    MongoSource,
    ODBCSource,
    SnowflakeSource,
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
