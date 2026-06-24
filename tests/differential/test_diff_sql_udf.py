"""Execution tests for registered Python functions called from SQL.

Python UDFs have no DuckDB equivalent, so the oracle is the deterministic function
applied directly (asserted on `to_pydict()`), not a `duck.sql` comparison.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt


@pytest.fixture
def docs():
    return pa.table({"id": [1, 2, 3], "x": [10, 20, 30], "text": ["aa", "bbb", "c"]})


@pytest.mark.differential
def test_scalar_vectorized_projection(docs):
    s = bt.Session()
    s.register("docs", docs)
    s.register_function("doubled", lambda a: pc.multiply(a, 2))
    out = s.sql("SELECT id, doubled(x) AS y FROM docs").collect()
    assert out.to_pydict() == {"id": [1, 2, 3], "y": [20, 40, 60]}


@pytest.mark.differential
def test_scalar_per_row(docs):
    s = bt.Session()
    s.register("docs", docs)
    s.register_function("plus1", lambda v: v + 1, vectorized=False, result_type="int64")
    out = s.sql("SELECT plus1(x) AS y FROM docs").collect()
    assert out.to_pydict() == {"y": [11, 21, 31]}


@pytest.mark.differential
def test_scalar_nested(docs):
    s = bt.Session()
    s.register("docs", docs)
    s.register_function("inc", lambda a: pc.add(a, 1))
    s.register_function("twice", lambda a: pc.multiply(a, 2))
    out = s.sql("SELECT twice(inc(x)) AS y FROM docs").collect()
    assert out.to_pydict() == {"y": [22, 42, 62]}


@pytest.mark.differential
def test_scalar_literal_and_expr_args(docs):
    s = bt.Session()
    s.register("docs", docs)
    s.register_function("addk", lambda a, k: pc.add(a, k))
    out = s.sql("SELECT addk(x + 1, 100) AS y FROM docs").collect()
    assert out.to_pydict() == {"y": [111, 121, 131]}


@pytest.mark.differential
def test_scalar_udf_in_where(docs):
    s = bt.Session()
    s.register("docs", docs)
    s.register_function("big", lambda a: pc.greater(a, 15))
    out = s.sql("SELECT id FROM docs WHERE big(x)").collect()
    assert out.to_pydict() == {"id": [2, 3]}


@pytest.mark.differential
def test_scalar_udf_order_by_alias(docs):
    s = bt.Session()
    s.register("docs", docs)
    s.register_function("neg", lambda a: pc.multiply(a, -1))
    out = s.sql("SELECT id, neg(x) AS n FROM docs ORDER BY n").to_arrow()
    assert out.column("id").to_pylist() == [3, 2, 1]


@pytest.mark.differential
def test_table_function(docs):
    s = bt.Session()
    s.register("docs", docs)

    def add_len(batch: pa.RecordBatch) -> pa.RecordBatch:
        lengths = pc.utf8_length(batch.column("text"))
        return batch.append_column("n", lengths)

    s.register_function("with_len", add_len, table=True, output_columns=["id", "x", "text", "n"])
    out = s.sql("SELECT id, n FROM with_len(docs) ORDER BY id").to_arrow()
    assert out.to_pydict() == {"id": [1, 2, 3], "n": [2, 3, 1]}
