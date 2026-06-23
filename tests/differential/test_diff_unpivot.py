"""Differential coverage for `Dataset.unpivot` (SQL UNPIVOT / melt) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
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
    from conftest import assert_same

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
    from conftest import assert_same

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
    from conftest import assert_same

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
    from conftest import assert_same

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
