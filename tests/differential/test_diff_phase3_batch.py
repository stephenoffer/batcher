"""Differential coverage for the Phase-3 batch: strftime, bitwise ops, unnest.

`.dt.strftime` vs DuckDB ``strftime``; bitwise integer ops vs DuckDB ``&``/``|``/
``#``/``<<``/``>>``; struct ``unnest`` is structural (struct → top-level columns).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def test_dt_strftime(duck):
    from conftest import assert_same

    d = pa.array(["2024-02-15", "2023-12-31", None]).cast(pa.date32())
    t = pa.table({"d": d})
    duck.register("t", t)
    out = bt.from_arrow(t).select(
        ymd=col("d").dt.strftime("%Y-%m-%d"),
        pretty=col("d").dt.strftime("%d/%m/%Y"),
    )
    assert_same(
        out.collect(),
        duck.sql("SELECT strftime(d, '%Y-%m-%d') ymd, strftime(d, '%d/%m/%Y') pretty FROM t"),
    )


def test_bitwise_ops(duck):
    from conftest import assert_same

    t = pa.table({"a": pa.array([12, 7, 255], pa.int64()), "b": pa.array([10, 3, 1], pa.int64())})
    duck.register("t", t)
    out = bt.from_arrow(t).select(
        band=col("a").bitwise_and(col("b")),
        bor=col("a").bitwise_or(col("b")),
        bxor=col("a").bitwise_xor(col("b")),
        shl=col("a").bitwise_left_shift(col("b")),
        shr=col("a").bitwise_right_shift(col("b")),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT (a & b) band, (a | b) bor, xor(a, b) bxor, (a << b) shl, (a >> b) shr FROM t"
        ),
    )


def test_unnest_struct_to_columns():
    s = pa.StructArray.from_arrays(
        [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])], names=["n", "t"]
    )
    tbl = pa.table({"id": [10, 20, 30], "s": s})
    out = bt.from_arrow(tbl).unnest("s")
    assert out.columns == ["id", "n", "t"]
    d = out.to_pydict()
    assert d == {"id": [10, 20, 30], "n": [1, 2, 3], "t": ["a", "b", "c"]}


def test_unnest_non_struct_raises():
    from batcher._internal.errors import PlanError

    tbl = pa.table({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="not a struct"):
        bt.from_arrow(tbl).unnest("x")
