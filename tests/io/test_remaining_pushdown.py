"""Source-side predicate pushdown for the remaining connector sources.

These run without any backend installed: they assert each opted-in connector
(`ElasticsearchSource`, `DynamoDBSource`, `CassandraSource`, `DatabricksSource`)
advertises `supports_predicate` and accepts a `predicate=` kwarg on both `read`
and `iter_batches`, plus a pure-dict test of the local DynamoDB
`FilterExpression` translator. The engine always keeps its `Filter` re-check, so
pushdown is a pure I/O optimization — these cover the translation and the opt-in
contract, not a live connection.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher.io.formats.nosql.cassandra import CassandraSource
from batcher.io.formats.nosql.dynamodb import DynamoDBSource, _to_dynamo_filter
from batcher.io.formats.nosql.elasticsearch import ElasticsearchSource, _to_es_query
from batcher.io.formats.sql.databricks import DatabricksSource

pytestmark = pytest.mark.unit

# Every source class that opts into source-side predicate pushdown here.
_PUSHDOWN_SOURCES = [
    CassandraSource,
    DatabricksSource,
    DynamoDBSource,
    ElasticsearchSource,
]


# --- per-source opt-in contract (no backend, no live connection) --------------
@pytest.mark.parametrize("source_cls", _PUSHDOWN_SOURCES)
def test_source_supports_predicate(source_cls):
    assert source_cls.supports_predicate is True


@pytest.mark.parametrize("source_cls", _PUSHDOWN_SOURCES)
def test_source_read_accepts_predicate(source_cls):
    assert "predicate" in inspect.signature(source_cls.read).parameters


@pytest.mark.parametrize("source_cls", _PUSHDOWN_SOURCES)
def test_source_iter_batches_accepts_predicate(source_cls):
    assert "predicate" in inspect.signature(source_cls.iter_batches).parameters


# --- _to_dynamo_filter (pure, no boto3 dependency) ----------------------------
def test_dynamo_single_comparison():
    flt = _to_dynamo_filter((bt.col("x") > 5).to_ir())
    assert flt is not None
    assert flt.expression == "#n0 > :v0"
    assert flt.names == {"#n0": "x"}
    assert flt.values == {":v0": 5}


def test_dynamo_equality_and_not_equal():
    eq = _to_dynamo_filter((bt.col("x") == 3).to_ir())
    assert eq is not None
    assert eq.expression == "#n0 = :v0"
    ne = _to_dynamo_filter((bt.col("x") != 3).to_ir())
    assert ne is not None
    assert ne.expression == "#n0 <> :v0"


def test_dynamo_flipped_comparison():
    # literal-on-left flips the operator so the column stays the attribute name.
    flt = _to_dynamo_filter((bt.lit(5) < bt.col("x")).to_ir())
    assert flt is not None
    assert flt.expression == "#n0 > :v0"
    assert flt.names == {"#n0": "x"}
    assert flt.values == {":v0": 5}


def test_dynamo_and():
    ir = ((bt.col("x") > 5) & (bt.col("y") == 3)).to_ir()
    flt = _to_dynamo_filter(ir)
    assert flt is not None
    assert flt.expression == "(#n0 > :v0 AND #n1 = :v1)"
    assert flt.names == {"#n0": "x", "#n1": "y"}
    assert flt.values == {":v0": 5, ":v1": 3}


def test_dynamo_or():
    ir = ((bt.col("x") > 5) | (bt.col("y") == 3)).to_ir()
    flt = _to_dynamo_filter(ir)
    assert flt is not None
    assert flt.expression == "(#n0 > :v0 OR #n1 = :v1)"


def test_dynamo_is_null():
    flt = _to_dynamo_filter(bt.col("x").is_null().to_ir())
    assert flt is not None
    assert flt.expression == "attribute_not_exists(#n0)"
    not_null = _to_dynamo_filter(bt.col("x").is_not_null().to_ir())
    assert not_null is not None
    assert not_null.expression == "attribute_exists(#n0)"


def test_dynamo_rejects_column_vs_column():
    assert _to_dynamo_filter((bt.col("x") > bt.col("y")).to_ir()) is None


# --- _to_es_query (pure, no elasticsearch dependency) -------------------------
def test_es_range_and_term():
    assert _to_es_query((bt.col("x") > 5).to_ir()) == {"range": {"x": {"gt": 5}}}
    assert _to_es_query((bt.col("x") == 3).to_ir()) == {"term": {"x": 3}}


def test_es_and_or():
    ir = ((bt.col("x") > 5) & (bt.col("y") == 3)).to_ir()
    assert _to_es_query(ir) == {"bool": {"must": [{"range": {"x": {"gt": 5}}}, {"term": {"y": 3}}]}}
    ir_or = ((bt.col("x") > 5) | (bt.col("y") == 3)).to_ir()
    assert _to_es_query(ir_or) == {
        "bool": {
            "should": [{"range": {"x": {"gt": 5}}}, {"term": {"y": 3}}],
            "minimum_should_match": 1,
        }
    }


def test_es_rejects_column_vs_column():
    assert _to_es_query((bt.col("x") > bt.col("y")).to_ir()) is None
