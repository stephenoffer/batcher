"""Actionable plan-build errors: bad cast dtype and unknown-column suggestions."""

from __future__ import annotations

import pytest

from batcher import col
from batcher._internal.errors import PlanError
from batcher.plan.schema import SchemaRef, suggest_columns

pytestmark = pytest.mark.unit


def test_cast_unknown_dtype_raises_with_suggestion():
    with pytest.raises(PlanError, match="unknown cast dtype"):
        col("x").cast("flot64")
    # The message should suggest the near-miss.
    try:
        col("x").cast("flot64")
    except PlanError as e:
        assert "float64" in str(e)


def test_cast_valid_dtype_ok():
    assert col("x").cast("int64") is not None
    assert col("x").cast("string") is not None


def test_suggest_columns_finds_near_miss():
    assert "did you mean 'salary'" in suggest_columns("salar", ["salary", "name", "id"])
    assert suggest_columns("xyz", ["salary", "name"]) == ""


def test_schema_field_error_includes_suggestion():
    import pyarrow as pa

    ref = SchemaRef.from_arrow(pa.schema([("amount", pa.int64()), ("name", pa.string())]))
    with pytest.raises(KeyError, match="did you mean 'amount'"):
        ref.field("amont")
