"""Differential coverage for the convenience sugar that lowers to core ops:
`value_counts`, `cross_join`, `top_k`."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "k": pa.array(["a", "a", "b", "c", "c", "c"]),
            "v": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        }
    )


def test_value_counts_matches_duckdb(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).value_counts("k").collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT k, COUNT(*) AS count FROM t GROUP BY k"))


def test_cross_join_matches_duckdb(duck):
    from conftest import assert_same

    left = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    right = pa.table({"b": pa.array([10, 20], type=pa.int64())})
    out = bt.from_arrow(left).cross_join(bt.from_arrow(right)).collect()
    duck.register("l", left)
    duck.register("r", right)
    assert_same(out, duck.sql("SELECT a, b FROM l CROSS JOIN r"))


def test_top_k_ordered(duck):
    from conftest import assert_same_ordered

    out = bt.from_arrow(_t()).top_k(3, "v").collect()
    duck.register("t", _t())
    assert_same_ordered(out, duck.sql("SELECT k, v FROM t ORDER BY v DESC LIMIT 3"))
