"""Window frame / value / ranking semantics vs DuckDB — deep frame-behavior wave 2.

Covers three defects the earlier waves missed:

* ``lag``/``lead`` with a **negative** offset must flip direction
  (``lag(v, -n) == lead(v, n)``), not collapse to the current row.
* ``min``/``max`` with an **explicit ROWS frame** over a **string** column must
  slide (the running/whole-partition paths already supported Utf8; the framed path
  raised ``UnsupportedWindow``).
* ``min``/``max`` over a **boolean** column (any window path) must order
  ``false < true`` (min = AND, max = OR), like the aggregate MIN/MAX, instead of
  raising ``UnsupportedWindow``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, lag, lead

pytestmark = pytest.mark.differential


@pytest.mark.parametrize("n", [-1, -2, -3])
def test_negative_lag_lead_flip_direction(duck, n):
    t = pa.table(
        {
            "id": pa.array(list(range(6)), pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b", "b"], pa.string()),
            "k": pa.array([1, 2, 3, 1, 2, 3], pa.int64()),
            "v": pa.array([10, 20, 30, 40, 50, 60], pa.int64()),
        }
    )
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .with_columns(
            lg=lag(col("v"), n).over(partition_by=["g"], order_by=["k"]),
            ld=lead(col("v"), n).over(partition_by=["g"], order_by=["k"]),
        )
        .select("id", "lg", "ld")
        .collect()
    )
    want = duck.sql(
        f"SELECT id, lag(v, {n}) OVER (PARTITION BY g ORDER BY k) lg, "
        f"lead(v, {n}) OVER (PARTITION BY g ORDER BY k) ld FROM t"
    )
    assert_same(got, want)


@pytest.mark.parametrize("fn", ["min", "max"])
def test_string_min_max_over_explicit_rows_frame(duck, fn):
    t = pa.table(
        {
            "id": pa.array(list(range(6)), pa.int64()),
            "k": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
            "s": pa.array(["d", "a", None, "c", "b", "e"], pa.string()),
        }
    )
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .window(order_by=["k"], functions={"r": (fn, "s")}, frame=(-1, 1))
        .select("id", "r")
        .collect()
    )
    want = duck.sql(
        f"SELECT id, {fn}(s) OVER (ORDER BY k ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) r FROM t"
    )
    assert_same(got, want)


@pytest.mark.parametrize("fn", ["min", "max"])
def test_boolean_min_max_whole_partition(duck, fn):
    t = pa.table(
        {
            "id": pa.array(list(range(6)), pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b", "b"], pa.string()),
            "b": pa.array([True, False, None, True, True, None], pa.bool_()),
        }
    )
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .window(partition_by=["g"], functions={"r": (fn, "b")})
        .select("id", "r")
        .collect()
    )
    want = duck.sql(f"SELECT id, {fn}(b) OVER (PARTITION BY g) r FROM t")
    assert_same(got, want)


@pytest.mark.parametrize("fn", ["min", "max"])
def test_boolean_min_max_running_and_framed(duck, fn):
    t = pa.table(
        {
            "id": pa.array(list(range(6)), pa.int64()),
            "k": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
            "b": pa.array([True, False, True, None, False, True], pa.bool_()),
        }
    )
    duck.register("t", t)
    # Running (default RANGE frame).
    got = (
        bt.from_arrow(t)
        .window(order_by=["k"], functions={"r": (fn, "b")})
        .select("id", "r")
        .collect()
    )
    want = duck.sql(f"SELECT id, {fn}(b) OVER (ORDER BY k) r FROM t")
    assert_same(got, want)
    # Explicit sliding ROWS frame.
    got = (
        bt.from_arrow(t)
        .window(order_by=["k"], functions={"r": (fn, "b")}, frame=(-1, 1))
        .select("id", "r")
        .collect()
    )
    want = duck.sql(
        f"SELECT id, {fn}(b) OVER (ORDER BY k ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) r FROM t"
    )
    assert_same(got, want)
