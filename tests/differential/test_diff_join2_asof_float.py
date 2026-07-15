"""ASOF join on a float `on` key: `-0.0`/`0.0`/NaN must match DuckDB.

`-0.0` and `0.0` are the same value (IEEE equality, and how DuckDB's ASOF
inequality treats them), so a nearest-match search must find the exact match. The
Rust asof primitive row-encodes the `on` key, whose total order splits `-0.0 < 0.0`
and gives distinct NaN bit patterns distinct bytes; canonicalizing signed zero / NaN
on the `on` column (as every other key path does) is what keeps it agreeing with
DuckDB. Regression for the wave-2 asof `on`-key signed-zero defect.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _left():
    return pa.table(
        {
            "sym": pa.array(["A", "A", "B", "B", "C"]),
            "ts": pa.array([-0.0, 1.5, 0.0, 2.0, 3.0], type=pa.float64()),
            "v": pa.array([10, 11, 12, 13, 14], type=pa.int64()),
        }
    )


def _right():
    return pa.table(
        {
            "sym": pa.array(["A", "A", "B", "B", "C"]),
            "ts": pa.array([0.0, 1.0, -0.0, 1.5, 5.0], type=pa.float64()),
            "bid": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
        }
    )


def test_asof_backward_float_on_signed_zero(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_left())
        .join_asof(bt.from_arrow(_right()), on="ts", by="sym", direction="backward")
        .collect()
    )
    duck.register("l", _left())
    duck.register("r", _right())
    assert_same(
        out,
        duck.sql(
            "SELECT l.sym, l.ts, l.v, r.bid FROM l "
            "ASOF LEFT JOIN r ON l.sym = r.sym AND l.ts >= r.ts"
        ),
    )


def test_asof_forward_float_on_signed_zero(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_left())
        .join_asof(bt.from_arrow(_right()), on="ts", by="sym", direction="forward")
        .collect()
    )
    duck.register("l", _left())
    duck.register("r", _right())
    assert_same(
        out,
        duck.sql(
            "SELECT l.sym, l.ts, l.v, r.bid FROM l "
            "ASOF LEFT JOIN r ON l.sym = r.sym AND l.ts <= r.ts"
        ),
    )
