"""The `window_algebra` rewrite must match DuckDB after the full optimizer runs.

`nth_value(x, 1)` becomes `first_value(x)`, so the oracle has to confirm the two really do
name the same row — over an ordered partition, over an unordered one, and with NULLs in the
value column, where a positional function's behaviour is least obvious.

The SQL spells the frame `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`
explicitly, following `test_diff_nth_value`: Batcher's value window functions read the
whole partition, while SQL's *default* frame is the running one, so the unbounded frame is
what makes DuckDB compute the same thing. That difference is also why the rewrite is safe
at `n = 1` and only there — the first row of a partition and the first row of a running
frame are the same row, which stops being true from the second onwards.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.window_algebra
from _harness import assert_same, assert_same_ordered
from batcher import col
from batcher._internal.errors import PlanError


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "b", "c"], type=pa.string()),
            "v": pa.array([3, None, 2, 5, 4, None], type=pa.int64()),
            "o": pa.array([1, 2, 3, 1, 2, 1], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "g": pa.array([], type=pa.string()),
            "v": pa.array([], type=pa.int64()),
            "o": pa.array([], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_nth_value_at_one_matches_duckdb_over_an_ordered_partition(duck, t):
    out = (
        bt.from_arrow(t)
        .with_columns(r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]))
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, v, o, nth_value(v, 1) OVER "
            "(PARTITION BY g ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND UNBOUNDED FOLLOWING) AS r FROM t"
        ),
    )


def test_nth_value_at_one_matches_duckdb_without_order_keys(duck, t):
    """No `order_by` on the window: the frame's own sort is the order, or it is refused.

    A positional function over an unordered partition names an arrival-order row, which a
    parallel scan does not fix, so the explicit-order policy refuses it. Over a frame the query
    sorted, the window takes that sort as its order, and the rewrite still has to match DuckDB.
    """
    windowed = bt.nth_value(col("v"), 1).over(partition_by=["g"])
    with pytest.raises(PlanError, match="requires order_by"):
        bt.from_arrow(t).with_columns(r=windowed)
    out = bt.from_arrow(t).sort("o").with_columns(r=windowed).sort("g", "o").collect()
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT g, v, o, nth_value(v, 1) OVER (PARTITION BY g ORDER BY o ROWS BETWEEN "
            "UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS r FROM t ORDER BY g, o"
        ),
    )


def test_nth_value_beyond_one_is_untouched_and_still_matches(duck, t):
    out = (
        bt.from_arrow(t)
        # The whole-partition frame is explicit: with an ORDER BY and no frame, `nth_value`
        # takes SQL's running default, which `test_diff_agg_semantic_params.py` pins.
        .with_columns(
            r=bt.nth_value(col("v"), 2).over(partition_by=["g"], order_by=["o"], frame=(None, None))
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, v, o, nth_value(v, 2) OVER "
            "(PARTITION BY g ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND UNBOUNDED FOLLOWING) AS r FROM t"
        ),
    )


def test_the_rewritten_and_hand_written_forms_agree(duck, t):
    # The rewrite's whole claim, checked directly: writing it either way must give the
    # same column, and both must equal what the oracle says.
    dataset = bt.from_arrow(t)
    rewritten = dataset.with_columns(
        r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"])
    ).collect()
    direct = dataset.with_columns(
        r=bt.first_value(col("v")).over(partition_by=["g"], order_by=["o"])
    ).collect()
    assert rewritten.column("r").to_pylist() == direct.column("r").to_pylist()
    assert_same(
        rewritten,
        duck.sql(
            "SELECT g, v, o, first_value(v) OVER (PARTITION BY g ORDER BY o "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS r FROM t"
        ),
    )


def test_empty_input(duck, empty):
    out = (
        bt.from_arrow(empty)
        .with_columns(r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]))
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, v, o, nth_value(v, 1) OVER "
            "(PARTITION BY g ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND UNBOUNDED FOLLOWING) AS r FROM t"
        ),
    )
