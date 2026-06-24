"""Differential coverage for GROUPS and peer-RANGE window frames vs DuckDB.

`ROWS` frames count physical rows; `GROUPS`/`RANGE` count peer groups (ties in the
ORDER BY key), so they differ from `ROWS` exactly when the order key has ties.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _data():
    # Order key `t` has ties (1,1 / 2,2), so peer-based frames differ from ROWS.
    return bt.from_pydict(
        {
            "g": ["a", "a", "a", "a", "a", "b", "b"],
            "t": [1, 1, 2, 2, 3, 1, 2],
            "v": [10, 20, 30, 40, 50, 5, 7],
        }
    )


def test_groups_frame_matches_duckdb(duck):
    from conftest import assert_same

    ds = _data()
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(-1, 0, "groups")
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER (PARTITION BY g ORDER BY t "
        "GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)


def test_range_peer_frame_matches_duckdb(duck):
    from conftest import assert_same

    ds = _data()
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(None, 0, "range")
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER (PARTITION BY g ORDER BY t "
        "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)


def test_groups_current_row_is_peer_sum(duck):
    from conftest import assert_same

    ds = _data()
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(0, 0, "groups")
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER (PARTITION BY g ORDER BY t "
        "GROUPS BETWEEN CURRENT ROW AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)
