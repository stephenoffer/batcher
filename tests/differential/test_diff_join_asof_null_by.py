"""Differential coverage: ASOF join `by` (equality) keys honor SQL ``NULL != NULL``.

The ASOF join groups the right side by its `by` columns and matches each left row within
its own `by` group. `by` is an *equality* key, so a null in any `by` column must match
nothing — exactly as an equi-join's null key does, and exactly as DuckDB's ASOF join does.
Arrow's row encoder gives a null a concrete byte string, so before the fix a null-`by`
right row formed a real group that null-`by` left rows then "matched", conflating every
left null with a right null and returning the wrong `bid`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def test_asof_null_by_key_matches_nothing(duck):
    from conftest import assert_same

    trades = pa.table(
        {
            "sym": pa.array(["A", None, None, "B"]),
            "ts": pa.array([10, 20, 30, 10], type=pa.int64()),
            "px": pa.array([1, 2, 3, 4], type=pa.int64()),
        }
    )
    quotes = pa.table(
        {
            "sym": pa.array([None, "A", None]),
            "ts": pa.array([5, 5, 25], type=pa.int64()),
            "bid": pa.array([100, 200, 300], type=pa.int64()),
        }
    )
    duck.register("t", trades)
    duck.register("q", quotes)
    out = bt.from_arrow(trades).join_asof(bt.from_arrow(quotes), on="ts", by="sym").collect()
    assert_same(
        out,
        duck.sql(
            "SELECT t.sym, t.ts, t.px, q.bid FROM t "
            "ASOF LEFT JOIN q ON t.sym = q.sym AND t.ts >= q.ts"
        ),
    )


def test_asof_multi_col_by_partial_null_matches_nothing(duck):
    from conftest import assert_same

    trades = pa.table(
        {
            "sym": pa.array(["A", "A", "A"]),
            "grp": pa.array([1, None, 1], type=pa.int64()),
            "ts": pa.array([10, 10, 20], type=pa.int64()),
            "px": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    quotes = pa.table(
        {
            "sym": pa.array(["A", "A"]),
            "grp": pa.array([1, None], type=pa.int64()),
            "ts": pa.array([5, 5], type=pa.int64()),
            "bid": pa.array([100, 200], type=pa.int64()),
        }
    )
    duck.register("t2", trades)
    duck.register("q2", quotes)
    out = (
        bt.from_arrow(trades)
        .join_asof(bt.from_arrow(quotes), on="ts", by=["sym", "grp"])
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT t2.sym, t2.grp, t2.ts, t2.px, q2.bid FROM t2 "
            "ASOF LEFT JOIN q2 ON t2.sym = q2.sym AND t2.grp = q2.grp AND t2.ts >= q2.ts"
        ),
    )
