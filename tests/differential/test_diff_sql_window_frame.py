"""SQL window-frame (`ROWS BETWEEN …`) differential tests vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def frame_table(duck):
    t = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
            "t": [1, 2, 3, 1, 2, 3],
            "x": [10, 20, 30, 40, 50, 60],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    "frame",
    [
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
        "ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING",
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
    ],
)
def test_window_frame_partitioned(duck, frame_table, frame):
    query = f"SELECT id, g, t, x, SUM(x) OVER (PARTITION BY g ORDER BY t {frame}) AS s FROM t"
    assert_same(bt.sql(query, t=frame_table).collect(), duck.sql(query))


@pytest.mark.differential
def test_window_frame_no_partition(duck, frame_table):
    query = (
        "SELECT id, x, AVG(x) OVER (ORDER BY t ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS m "
        "FROM t"
    )
    assert_same(bt.sql(query, t=frame_table).collect(), duck.sql(query))


@pytest.mark.differential
def test_window_two_frames_in_one_select(duck, frame_table):
    # Two aggregates over the same partition/order but different frames must not
    # collapse into one window call (exercises the (part, order, frame) group key).
    win = "OVER (PARTITION BY g ORDER BY t ROWS BETWEEN"
    query = (
        "SELECT id, "
        f"SUM(x) {win} 1 PRECEDING AND CURRENT ROW) AS trailing, "
        f"SUM(x) {win} CURRENT ROW AND 1 FOLLOWING) AS leading "
        "FROM t"
    )
    assert_same(bt.sql(query, t=frame_table).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize(
    "frame",
    [
        # Single-bound frame: `ROWS N PRECEDING` is shorthand for
        # `ROWS BETWEEN N PRECEDING AND CURRENT ROW`, NOT `... AND UNBOUNDED
        # FOLLOWING`. The translator used to leave the end unbounded, summing the
        # whole tail of the partition (a silently wrong answer).
        "ROWS 1 PRECEDING",
        "ROWS 2 PRECEDING",
        "ROWS UNBOUNDED PRECEDING",
        "ROWS CURRENT ROW",
    ],
)
def test_window_single_bound_frame_defaults_to_current_row(duck, frame_table, frame):
    query = f"SELECT id, x, SUM(x) OVER (ORDER BY t {frame}) AS s FROM t"
    assert_same(bt.sql(query, t=frame_table).collect(), duck.sql(query))


@pytest.mark.differential
def test_named_window_carries_its_frame(duck, frame_table):
    # `WINDOW w AS (... ROWS ...)` referenced by `OVER w` must keep the frame; the
    # inliner used to copy PARTITION BY / ORDER BY but drop the frame, so `OVER w`
    # silently ran the default running frame instead of the trailing 2-row window.
    query = (
        "SELECT id, x, SUM(x) OVER w AS s, AVG(x) OVER w AS a FROM t "
        "WINDOW w AS (PARTITION BY g ORDER BY t ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"
    )
    assert_same(bt.sql(query, t=frame_table).collect(), duck.sql(query))


@pytest.mark.differential
def test_named_window_single_bound_frame(duck, frame_table):
    query = "SELECT id, x, SUM(x) OVER w AS s FROM t WINDOW w AS (ORDER BY t ROWS 1 PRECEDING)"
    assert_same(bt.sql(query, t=frame_table).collect(), duck.sql(query))
