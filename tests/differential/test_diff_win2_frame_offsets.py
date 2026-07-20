"""Differential coverage for extreme `ROWS`-frame offsets vs DuckDB.

Window-frame offsets are `u64` in the IR. A bound near `i64::MAX` — which DuckDB
accepts, as it fits `INT64` — used to be truncated by a raw `k as i64` cast in the
runtime's frame resolver: `pos + k + 1` overflowed (a hard panic), and a huge
`PRECEDING` offset wrapped its sign and collapsed to an empty (all-null) frame. Both
are wrong: a frame wider than the partition simply reaches the partition edge. These
pin the saturating fix in `window_frame::frame_half_open`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_I64_MAX = 9223372036854775807


def test_frame_current_to_i64max_following_is_suffix(duck):
    """CURRENT ROW .. i64::MAX FOLLOWING is the suffix aggregate, and must not panic."""
    t = bt.from_pydict({"id": [1, 2, 3, 4, 5], "x": [10, 20, 30, 40, 50]}).collect()
    duck.register("t", t)
    query = (
        f"SELECT id, SUM(x) OVER (ORDER BY id "
        f"ROWS BETWEEN CURRENT ROW AND {_I64_MAX} FOLLOWING) AS v FROM t"
    )
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_frame_i64max_preceding_to_current_is_prefix(duck):
    """i64::MAX PRECEDING .. CURRENT ROW is the running (prefix) aggregate, not all-null."""
    t = bt.from_pydict({"id": [1, 2, 3, 4, 5], "x": [10, 20, 30, 40, 50]}).collect()
    duck.register("t", t)
    query = (
        f"SELECT id, SUM(x) OVER (ORDER BY id "
        f"ROWS BETWEEN {_I64_MAX} PRECEDING AND CURRENT ROW) AS v FROM t"
    )
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_frame_wide_both_sides_covers_whole_partition(duck):
    """A frame far wider than the partition on both sides equals the whole-partition agg."""
    t = bt.from_pydict(
        {"g": ["a", "a", "a", "b", "b"], "id": [1, 2, 3, 1, 2], "x": [5.0, 1.0, 9.0, 7.0, 2.0]}
    ).collect()
    duck.register("t", t)
    query = (
        f"SELECT g, id, MAX(x) OVER (PARTITION BY g ORDER BY id "
        f"ROWS BETWEEN {_I64_MAX} PRECEDING AND {_I64_MAX} FOLLOWING) AS v FROM t"
    )
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))
