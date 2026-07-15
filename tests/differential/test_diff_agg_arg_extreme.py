"""arg_min / arg_max NULL-handling differential coverage.

DuckDB `arg_max(v, k)` / `arg_min(v, k)` ignore a row when **either** argument is
NULL. The engine skipped only null *keys*, so when the row with the extreme key had a
NULL value it returned NULL instead of the value at the next-best key among the
non-null-value rows. These pin the value-null skip against DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential


def _tbl():
    # group a: the absolute-max-key row (k=9) has a NULL value; DuckDB ignores it, so
    #   arg_max is the value at the next key (k=5) → 30, not NULL.
    # group b: the min-key row (k=1) has a NULL value; arg_min → value at k=3 → 2.
    return pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "b", "b"]),
            "v": pa.array([10, None, 30, None, 2, 7], pa.int64()),
            "k": pa.array([1, 9, 5, 1, 3, 8], pa.int64()),
        }
    )


def test_arg_extreme_skips_null_values(duck):
    duck.register("t", _tbl())
    out = (
        bt.from_arrow(_tbl())
        .group_by("g")
        .agg(amx=col("v").arg_max(col("k")), amn=col("v").arg_min(col("k")))
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT g, arg_max(v, k) AS amx, arg_min(v, k) AS amn FROM t GROUP BY g"),
    )


def test_arg_extreme_all_null_values_is_null(duck):
    tbl = pa.table(
        {
            "g": pa.array(["a", "a", "b"]),
            "v": pa.array([None, None, 5], pa.int64()),
            "k": pa.array([1, 2, 3], pa.int64()),
        }
    )
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).group_by("g").agg(amx=col("v").arg_max(col("k"))).collect()
    assert_same(out, duck.sql("SELECT g, arg_max(v, k) AS amx FROM t GROUP BY g"))


def test_arg_extreme_null_value_single_node_equals_distributed():
    # The value-null skip must hold identically across the distributed merge.
    g = {
        "g": ["a"] * 6 + ["b"] * 6,
        "v": [10, None, 30, None, 40, 5, None, 1, None, 9, 2, None],
        "k": [1, 9, 5, 8, 3, 2, 4, 1, 7, 6, 2, 5],
    }
    ds = (
        bt.from_pydict(g)
        .group_by("g")
        .agg(amx=col("v").arg_max(col("k")), amn=col("v").arg_min(col("k")))
    )
    single = {
        k: (a, b)
        for k, a, b in zip(*[ds.collect().to_pydict()[c] for c in ("g", "amx", "amn")], strict=True)
    }
    dd = ds.collect(distributed=True, num_workers=3).to_pydict()
    multi = {k: (a, b) for k, a, b in zip(dd["g"], dd["amx"], dd["amn"], strict=True)}
    assert single == multi
