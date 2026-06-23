"""Integration: multi-file schema evolution on read, and the range/date_range sources."""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import SchemaError


def test_schema_mode_union_reconciles_files(tmp_path):
    pq.write_table(
        pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]}), f"{tmp_path}/p1.parquet"
    )
    pq.write_table(
        pa.table({"a": pa.array([3.5], pa.float64()), "c": [True]}), f"{tmp_path}/p2.parquet"
    )
    out = bt.read.parquet(f"{tmp_path}/*.parquet", schema_mode="union").collect()
    assert out.schema.names == ["a", "b", "c"]
    assert out.schema.field("a").type == pa.float64()
    d = out.to_pydict()
    assert sorted(d["a"]) == [1.0, 2.0, 3.5]
    assert d["b"].count(None) == 1  # the p2 row has no b
    assert d["c"].count(True) == 1


def test_schema_mode_strict_is_default_and_unchanged(tmp_path):
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"]}), f"{tmp_path}/p.parquet")
    out = bt.read.parquet(f"{tmp_path}/p.parquet").collect()
    assert out.schema.names == ["a", "b"]


def test_schema_mode_union_with_projection(tmp_path):
    pq.write_table(pa.table({"a": [1], "b": ["x"]}), f"{tmp_path}/p1.parquet")
    pq.write_table(pa.table({"a": [2], "c": [9]}), f"{tmp_path}/p2.parquet")
    out = bt.read.parquet(f"{tmp_path}/*.parquet", schema_mode="union").select("a", "c").collect()
    assert out.schema.names == ["a", "c"]
    assert sorted(out.to_pydict()["a"]) == [1, 2]


def test_range_source():
    out = bt.range(0, 5).collect()
    assert out.to_pydict() == {"value": [0, 1, 2, 3, 4]}
    assert bt.range(1, 10, 2, name="n").to_pydict() == {"n": [1, 3, 5, 7, 9]}


def test_date_range_source():
    out = bt.date_range("2024-01-01", "2024-01-05").collect()
    vals = out.to_pydict()["date"]
    assert vals[0] == datetime.date(2024, 1, 1)
    assert vals[-1] == datetime.date(2024, 1, 5)
    assert len(vals) == 5


def test_date_range_interval():
    out = bt.date_range("2024-01-01", "2024-01-10", interval_days=3).to_pydict()["date"]
    assert out == [datetime.date(2024, 1, d) for d in (1, 4, 7, 10)]


def test_unify_incompatible_raises_on_read(tmp_path):
    pq.write_table(pa.table({"a": ["s"]}), f"{tmp_path}/p1.parquet")
    pq.write_table(pa.table({"a": [1]}), f"{tmp_path}/p2.parquet")
    with pytest.raises(SchemaError):
        bt.read.parquet(f"{tmp_path}/*.parquet", schema_mode="union").collect()
