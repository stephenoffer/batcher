"""Differential coverage for `Dataset.unpivot` (SQL UNPIVOT / melt) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _wide():
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "q1": pa.array([10, 40, 70], type=pa.int64()),
            "q2": pa.array([20, 50, 80], type=pa.int64()),
            "q3": pa.array([30, 60, 90], type=pa.int64()),
        }
    )


def test_unpivot_matches_duckdb(duck):
    out = bt.from_arrow(_wide()).unpivot(index=["id"], on=["q1", "q2", "q3"]).collect()
    duck.register("t", _wide())
    assert_same(
        out,
        duck.sql(
            "SELECT id, variable, value FROM "
            "(UNPIVOT t ON q1, q2, q3 INTO NAME variable VALUE value)"
        ),
    )


def test_unpivot_then_aggregate(duck):
    out = (
        bt.from_arrow(_wide())
        .unpivot(index=["id"], on=["q1", "q2", "q3"])
        .group_by("variable")
        .agg(total=col("value").sum())
        .collect()
    )
    duck.register("t", _wide())
    assert_same(
        out,
        duck.sql(
            "SELECT variable, SUM(value) AS total FROM "
            "(UNPIVOT t ON q1, q2, q3 INTO NAME variable VALUE value) GROUP BY variable"
        ),
    )


def test_unpivot_infers_value_columns(duck):
    # `on` omitted → every non-index column is melted.
    out = bt.from_arrow(_wide()).unpivot(index=["id"]).collect()
    duck.register("t", _wide())
    assert_same(
        out,
        duck.sql(
            "SELECT id, variable, value FROM "
            "(UNPIVOT t ON q1, q2, q3 INTO NAME variable VALUE value)"
        ),
    )


def test_unpivot_custom_names_and_filter(duck):
    out = (
        bt.from_arrow(_wide())
        .unpivot(index=["id"], on=["q1", "q2"], variable_name="quarter", value_name="amount")
        .filter(col("amount") > 25)
        .collect()
    )
    duck.register("t", _wide())
    assert_same(
        out,
        duck.sql(
            "SELECT * FROM (SELECT id, quarter, amount FROM "
            "(UNPIVOT t ON q1, q2 INTO NAME quarter VALUE amount)) WHERE amount > 25"
        ),
    )


def test_unpivot_mixed_numeric_promotes(duck):
    """Melting an Int64 column together with a Float64 column promotes to Float64
    (DuckDB/Polars), rather than erroring on the concat of differing types."""
    wide = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "a": pa.array([10, 20], type=pa.int64()),
            "b": pa.array([1.5, 2.5], type=pa.float64()),
        }
    )
    out = bt.from_arrow(wide).unpivot(index=["id"], on=["a", "b"]).collect()
    # The stacked value column is the promoted supertype (Float64), matching the schema.
    assert pa.types.is_floating(out.schema.field("value").type)
    duck.register("t", wide)
    assert_same(
        out,
        duck.sql(
            "SELECT id, variable, value FROM (UNPIVOT t ON a, b INTO NAME variable VALUE value)"
        ),
    )


def test_unpivot_incompatible_types_raises_clean_plan_error():
    """Melting columns with no common type (e.g. Utf8 + Int64) must raise a clean
    plan-time PlanError, not an opaque native "cannot concatenate arrays of different
    data types" RuntimeError at execution. DuckDB likewise rejects this at bind time
    ("an explicit cast is required"). The natural ``unpivot(index=[...])`` that melts
    every remaining column is exactly where a string + numeric mix shows up.
    """
    from batcher._internal.errors import PlanError

    ds = bt.from_pydict({"id": [1, 2], "k": ["a", "b"], "v": [10, 20]})
    with pytest.raises(PlanError, match="incompatible types"):
        ds.unpivot(index=["id"])
    with pytest.raises(PlanError, match="incompatible types"):
        ds.unpivot(index=["id"], on=["k", "v"])


def test_unpivot_name_collision_raises():
    """A `value_name`/`variable_name` that collides with an index column must raise —
    silently producing two same-named columns dropped one on the way out."""
    from batcher._internal.errors import PlanError

    ds = bt.from_pydict({"id": [1, 2], "a": [10, 20], "b": [30, 40]})
    for kwargs in (
        {"index": ["id"], "value_name": "id"},
        {"index": ["id"], "variable_name": "id"},
        {"index": ["id"], "variable_name": "v", "value_name": "v"},
    ):
        with pytest.raises(PlanError, match="collide"):
            ds.unpivot(**kwargs)
