"""Differential coverage for `Dataset.join_asof` (ASOF nearest-match join) vs DuckDB.

Batcher's `join_asof` is left-style (every left row kept), matching DuckDB's
``ASOF LEFT JOIN``. The last inequality is the asof key: ``>=`` is backward (largest
right ≤ left), ``<=`` is forward; equalities are the `by` keys.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _trades():
    return pa.table(
        {
            "sym": pa.array(["A", "A", "A", "B", "B", "C"]),
            "ts": pa.array([10, 25, 40, 10, 30, 5], type=pa.int64()),
            "price": pa.array([100, 101, 102, 200, 201, 300], type=pa.int64()),
        }
    )


def _quotes():
    return pa.table(
        {
            "sym": pa.array(["A", "A", "B", "B"]),
            "ts": pa.array([5, 30, 12, 28], type=pa.int64()),
            "bid": pa.array([1, 2, 3, 4], type=pa.int64()),
        }
    )


def test_asof_backward_by_symbol(duck):
    from conftest import assert_same

    out = bt.from_arrow(_trades()).join_asof(bt.from_arrow(_quotes()), on="ts", by="sym").collect()
    duck.register("trades", _trades())
    duck.register("quotes", _quotes())
    assert_same(
        out,
        duck.sql(
            "SELECT t.sym, t.ts, t.price, q.bid FROM trades t "
            "ASOF LEFT JOIN quotes q ON t.sym = q.sym AND t.ts >= q.ts"
        ),
    )


def test_asof_forward_by_symbol(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_trades())
        .join_asof(bt.from_arrow(_quotes()), on="ts", by="sym", direction="forward")
        .collect()
    )
    duck.register("trades", _trades())
    duck.register("quotes", _quotes())
    assert_same(
        out,
        duck.sql(
            "SELECT t.sym, t.ts, t.price, q.bid FROM trades t "
            "ASOF LEFT JOIN quotes q ON t.sym = q.sym AND t.ts <= q.ts"
        ),
    )


def test_asof_no_by(duck):
    from conftest import assert_same

    left = pa.table({"ts": pa.array([1, 5, 10, 30], type=pa.int64())})
    right = pa.table(
        {"ts": pa.array([2, 6, 20], type=pa.int64()), "v": pa.array([20, 60, 200], type=pa.int64())}
    )
    out = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="ts").collect()
    duck.register("l", left)
    duck.register("r", right)
    assert_same(out, duck.sql("SELECT l.ts, r.v FROM l ASOF LEFT JOIN r ON l.ts >= r.ts"))
