"""FULL OUTER JOIN (DataFrame + SQL) vs DuckDB — keys coalesced across sides."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def lr(duck):
    left = pa.table({"k": [1, 2, 3], "lv": [10, 20, 30]})
    right = pa.table({"k": [2, 3, 4], "rv": [200, 300, 400]})
    duck.register("l", left)
    duck.register("r", right)
    return left, right


@pytest.mark.parametrize("how", ["full", "outer"])
def test_full_outer_join_dataframe(duck, lr, how):
    from conftest import assert_same

    left, right = lr
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k", how=how)
        .select("k", "lv", "rv")
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT coalesce(l.k, r.k) k, l.lv, r.rv FROM l FULL OUTER JOIN r ON l.k = r.k"),
    )


def test_full_outer_join_string_keys(duck):
    from conftest import assert_same

    left = pa.table({"k": ["a", "b", "c"], "x": [1, 2, 3]})
    right = pa.table({"k": ["b", "c", "d"], "y": [20, 30, 40]})
    duck.register("ls", left)
    duck.register("rs", right)
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k", how="full")
        .select("k", "x", "y")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT coalesce(ls.k, rs.k) k, ls.x, rs.y FROM ls FULL OUTER JOIN rs ON ls.k = rs.k"
        ),
    )


def test_full_outer_join_multi_key(duck):
    from conftest import assert_same

    left = pa.table({"a": [1, 1, 2], "b": [1, 2, 1], "lv": [10, 20, 30]})
    right = pa.table({"a": [1, 2, 3], "b": [2, 1, 1], "rv": [200, 300, 400]})
    duck.register("lm", left)
    duck.register("rm", right)
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on=["a", "b"], how="full")
        .select("a", "b", "lv", "rv")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT coalesce(lm.a, rm.a) a, coalesce(lm.b, rm.b) b, lm.lv, rm.rv "
            "FROM lm FULL OUTER JOIN rm ON lm.a = rm.a AND lm.b = rm.b"
        ),
    )


def test_full_outer_join_sql(duck, lr):
    from conftest import assert_same

    out = bt.sql("SELECT k, lv, rv FROM l FULL JOIN r ON l.k = r.k", l=lr[0], r=lr[1]).collect()
    assert_same(
        out, duck.sql("SELECT coalesce(l.k, r.k) k, lv, rv FROM l FULL OUTER JOIN r ON l.k = r.k")
    )
