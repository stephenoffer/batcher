"""Per-partition top-N (QUALIFY) vs DuckDB.

`Filter(Window([rank]), rank <= k)` fuses into a window `rank_limit`; the result
must equal DuckDB's `QUALIFY` across the ranking functions, ties, and boundaries.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _t(duck):
    t = pa.table(
        {
            "k": [1, 1, 1, 2, 2, 3, 1, 2],
            "v": [30, 10, 20, 5, 15, 7, 10, 5],
        }
    )
    duck.register("t", t)
    return bt.from_arrow(t)


def _topn(ds, fn, k, op="le"):
    ranked = ds.window(partition_by=["k"], order_by=["v"], functions={"r": fn})
    pred = (col("r") <= k) if op == "le" else (col("r") < k)
    return ranked.filter(pred).select("k", "v")


def test_row_number_top2(duck):
    from conftest import assert_same

    out = _topn(_t(duck), "row_number", 2).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT k, v FROM t "
            "QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) <= 2"
        ),
    )


def test_rank_top2_keeps_ties(duck):
    from conftest import assert_same

    out = _topn(_t(duck), "rank", 2).collect()
    assert_same(
        out,
        duck.sql("SELECT k, v FROM t QUALIFY rank() OVER (PARTITION BY k ORDER BY v) <= 2"),
    )


def test_dense_rank_top2(duck):
    from conftest import assert_same

    out = _topn(_t(duck), "dense_rank", 2).collect()
    assert_same(
        out,
        duck.sql("SELECT k, v FROM t QUALIFY dense_rank() OVER (PARTITION BY k ORDER BY v) <= 2"),
    )


def test_row_number_strict_lt(duck):
    from conftest import assert_same

    out = _topn(_t(duck), "row_number", 3, op="lt").collect()  # rn < 3  →  top 2
    assert_same(
        out,
        duck.sql("SELECT k, v FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) < 3"),
    )


def test_row_number_eq_one(duck):
    from conftest import assert_same

    ranked = _t(duck).window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})
    out = ranked.filter(col("r") == 1).select("k", "v").collect()
    assert_same(
        out,
        duck.sql("SELECT k, v FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"),
    )


def test_topn_global_partition(duck):
    """No PARTITION BY: top-k over the whole relation."""
    from conftest import assert_same

    ranked = _t(duck).window(order_by=["v"], functions={"r": "row_number"})
    out = ranked.filter(col("r") <= 3).select("k", "v").collect()
    assert_same(
        out,
        duck.sql("SELECT k, v FROM t QUALIFY row_number() OVER (ORDER BY v) <= 3"),
    )


def test_topn_with_extra_predicate(duck):
    """A non-rank conjunct stays as a filter above the fused window."""
    from conftest import assert_same

    ranked = _t(duck).window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})
    out = ranked.filter((col("r") <= 2) & (col("v") > 6)).select("k", "v").collect()
    assert_same(
        out,
        duck.sql(
            "SELECT k, v FROM t "
            "QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) <= 2 AND v > 6"
        ),
    )
