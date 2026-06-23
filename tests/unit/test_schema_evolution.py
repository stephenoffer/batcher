"""Unit tests for the schema-reconciliation utility (`io.schema_evolution`)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import SchemaError
from batcher.io.schema import normalize_batch, schema_drift, unify_schemas


def test_unify_union_promotes_and_unions_columns():
    s1 = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.string())])
    s2 = pa.schema([pa.field("a", pa.float64()), pa.field("c", pa.bool_())])
    out = unify_schemas([s1, s2], "union")
    assert out.names == ["a", "b", "c"]
    assert out.field("a").type == pa.float64()  # int32 + float64 -> float64
    assert out.field("b").type == pa.string()
    assert out.field("c").type == pa.bool_()


def test_unify_int_widening():
    s1 = pa.schema([pa.field("a", pa.int8())])
    s2 = pa.schema([pa.field("a", pa.int64())])
    assert unify_schemas([s1, s2], "union").field("a").type == pa.int64()


def test_unify_null_adopts_other_type():
    s1 = pa.schema([pa.field("a", pa.null())])
    s2 = pa.schema([pa.field("a", pa.string())])
    assert unify_schemas([s1, s2], "union").field("a").type == pa.string()


def test_unify_incompatible_raises():
    s1 = pa.schema([pa.field("a", pa.string())])
    s2 = pa.schema([pa.field("a", pa.int64())])
    with pytest.raises(SchemaError, match="incompatible"):
        unify_schemas([s1, s2], "union")


def test_unify_strict_rejects_difference():
    s1 = pa.schema([pa.field("a", pa.int64())])
    s2 = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
    with pytest.raises(SchemaError):
        unify_schemas([s1, s2], "strict")


def test_unify_latest_wins():
    s1 = pa.schema([pa.field("a", pa.int64())])
    s2 = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
    assert unify_schemas([s1, s2], "latest") == s2


def test_normalize_batch_adds_nulls_and_casts():
    target = pa.schema([pa.field("a", pa.float64()), pa.field("b", pa.string())])
    b = pa.RecordBatch.from_arrays([pa.array([1, 2], pa.int64())], names=["a"])
    out = normalize_batch(b, target)
    assert out.schema == target
    assert out.column("a").to_pylist() == [1.0, 2.0]
    assert out.column("b").to_pylist() == [None, None]


def test_schema_drift_reports_changes():
    expected = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
    inferred = pa.schema([pa.field("a", pa.float64()), pa.field("c", pa.bool_())])
    drift = schema_drift(inferred, expected)
    assert drift.has_drift
    assert drift.added == ("c",)
    assert drift.removed == ("b",)
    assert drift.type_changed == (("a", "int64", "double"),)
