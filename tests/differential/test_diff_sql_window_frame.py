"""SQL window-frame (`ROWS BETWEEN …`) differential tests vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same


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
