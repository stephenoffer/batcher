"""The "replace values without dropping rows" recipe against DuckDB's ``CASE WHEN``.

`docs/user-guide/transform/columns/expressions.md` teaches ``with_columns(x=when(p).then(v)
.otherwise(col("x")))`` as the way to change some values and keep every row. Its two claims
are what is checked: every row survives, and a NULL condition falls through to `otherwise`.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_CASES = {
    "nulls": [1.0, None, 25.0, None, 16.0],
    "empty": [],
    "one_row": [30.0],
    "duplicates": [20.0, 20.0, 5.0, 20.0],
    "nan": [math.nan, 18.0, -0.0, math.nan],
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_replace_where_keeps_every_row(duck, case: str) -> None:
    table = pa.table(
        {
            "id": pa.array(range(len(_CASES[case])), pa.int64()),
            "x": pa.array(_CASES[case], pa.float64()),
        }
    )
    ours = (
        bt.from_arrow(table)
        .with_columns(x=bt.when(bt.col("x") > 15).then(15.0).otherwise(bt.col("x")))
        .collect()
    )
    duck.register("t", table)
    theirs = duck.sql("SELECT id, CASE WHEN x > 15 THEN 15.0 ELSE x END AS x FROM t")
    assert ours.num_rows == table.num_rows
    assert_same(ours, theirs)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_replace_over_a_selector(duck, case: str) -> None:
    values = _CASES[case]
    table = pa.table(
        {
            "a": pa.array(values, pa.float64()),
            "b": pa.array([v if v is None else -v for v in values], pa.float64()),
        }
    )
    numbers = bt.numeric()
    ours = bt.from_arrow(table).with_columns(bt.when(numbers > 2).then(0.0).otherwise(numbers))
    duck.register("t", table)
    theirs = duck.sql(
        "SELECT CASE WHEN a > 2 THEN 0.0 ELSE a END AS a, "
        "CASE WHEN b > 2 THEN 0.0 ELSE b END AS b FROM t"
    )
    assert_same(ours.collect(), theirs)
