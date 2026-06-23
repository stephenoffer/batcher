"""`distinct(subset, keep=...)` — keep one row per key, vs DuckDB QUALIFY."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _events():
    return pa.table(
        {
            "k": pa.array([1, 1, 2, 2, 2], pa.int64()),
            "ts": pa.array([10, 20, 5, 30, 15], pa.int64()),
            "v": ["a", "b", "c", "d", "e"],
        }
    )


def test_distinct_keep_first(duck):
    from conftest import assert_same

    t = _events()
    duck.register("e", t)
    out = bt.from_arrow(t).distinct(["k"], keep="first", order_by="ts").collect()
    assert_same(
        out,
        duck.sql(
            "SELECT k, ts, v FROM (SELECT *, row_number() OVER (PARTITION BY k ORDER BY ts) rn "
            "FROM e) WHERE rn = 1"
        ),
    )


def test_distinct_keep_last(duck):
    from conftest import assert_same

    t = _events()
    duck.register("e", t)
    out = bt.from_arrow(t).distinct(["k"], keep="last", order_by="ts").collect()
    assert_same(
        out,
        duck.sql(
            "SELECT k, ts, v FROM (SELECT *, "
            "row_number() OVER (PARTITION BY k ORDER BY ts DESC) rn "
            "FROM e) WHERE rn = 1"
        ),
    )


def test_distinct_subset_any_collapses_keys():
    t = _events()
    out = bt.from_arrow(t).distinct(["k"], keep="any").collect()
    # One row per distinct key, deterministic (ordered by the key itself).
    assert sorted(out.to_pydict()["k"]) == [1, 2]
    assert out.num_rows == 2
