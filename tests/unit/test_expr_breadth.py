"""cast / CASE / null-check expressions and their IR."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col, lit, when

pytest.importorskip("batcher._native", reason="native engine not built")


def test_cast_ir():
    assert col("s").cast("int64").to_ir() == {
        "e": "cast",
        "input": {"e": "col", "name": "s"},
        "dtype": "int64",
        "try_cast": False,
    }


def test_try_cast_ir():
    assert col("s").try_cast("int64").to_ir() == {
        "e": "cast",
        "input": {"e": "col", "name": "s"},
        "dtype": "int64",
        "try_cast": True,
    }


def test_case_ir_shape():
    e = when(col("x") < 2).then(lit("a")).otherwise(lit("b"))
    ir = e.to_ir()
    assert ir["e"] == "case"
    assert len(ir["branches"]) == 1
    assert ir["otherwise"] == {"e": "lit", "value": {"str": "b"}}


def test_cast_roundtrip():
    out = bt.from_pydict({"s": ["1", "2", "3"]}).select(n=col("s").cast("int64")).collect()
    assert out.to_pydict() == {"n": [1, 2, 3]}


def test_is_null_and_not_null():
    out = (
        bt.from_pydict({"y": [10, None, 30]})
        .select(a=col("y").is_null(), b=col("y").is_not_null())
        .collect()
        .to_pydict()
    )
    assert out == {"a": [False, True, False], "b": [True, False, True]}


def test_case_multi_branch():
    out = (
        bt.from_pydict({"x": [1, 2, 3, 4, 5]})
        .select(
            tier=when(col("x") < 2)
            .then(lit("low"))
            .when(col("x") < 4)
            .then(lit("mid"))
            .otherwise(lit("hi"))
        )
        .collect()
        .to_pydict()
    )
    assert out["tier"] == ["low", "mid", "mid", "hi", "hi"]


def test_case_columns_tracked_for_pushdown():
    # referenced_columns must see through CASE so projection pushdown is correct.
    from batcher.plan.expr_ir import referenced_columns

    e = when(col("a") > 0).then(col("b")).otherwise(col("c"))
    assert referenced_columns(e) == {"a", "b", "c"}
