"""Differential coverage for the `nth_value` window function.

Batcher's value window functions read the whole partition (like `first_value`/
`last_value`), so the DuckDB oracle uses the explicit UNBOUNDED ... UNBOUNDED frame.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, nth_value

pytestmark = pytest.mark.differential

_FRAME = "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"


def _data():
    return pa.table(
        {
            "g": ["a", "a", "a", "b", "b"],
            "t": [1, 2, 3, 1, 2],
            "v": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
        }
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_nth_value_matches_duckdb(duck, n):
    from conftest import assert_same

    duck.register("t", _data())
    out = (
        bt.from_arrow(_data())
        .with_columns(r=nth_value(col("v"), n).over(partition_by=["g"], order_by=["t"]))
        .collect()
    )
    # n=4 exceeds group 'a' (3 rows) and 'b' (2 rows) → null everywhere.
    assert_same(
        out,
        duck.sql(
            f"SELECT g, t, v, nth_value(v, {n}) OVER "
            f"(PARTITION BY g ORDER BY t {_FRAME}) AS r FROM t"
        ),
    )
