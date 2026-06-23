"""Expression IR builds the exact JSON the engine deserializes."""

from __future__ import annotations

import pytest

from batcher.plan.expr_ir import col, lit


def test_col_and_lit_ir():
    assert col("x").to_ir() == {"e": "col", "name": "x"}
    assert lit(5).to_ir() == {"e": "lit", "value": {"int": 5}}
    assert lit(2.5).to_ir() == {"e": "lit", "value": {"float": 2.5}}
    assert lit("hi").to_ir() == {"e": "lit", "value": {"str": "hi"}}


def test_bool_literal_not_confused_with_int():
    # bool is a subclass of int; the IR must tag it as bool.
    assert lit(True).to_ir() == {"e": "lit", "value": {"bool": True}}


def test_comparison_builds_binary():
    e = col("x") > 2
    assert e.to_ir() == {
        "e": "binary",
        "op": "gt",
        "left": {"e": "col", "name": "x"},
        "right": {"e": "lit", "value": {"int": 2}},
    }


def test_reflected_arithmetic():
    assert (2 * col("x")).to_ir()["op"] == "mul"
    assert (2 * col("x")).to_ir()["left"] == {"e": "lit", "value": {"int": 2}}


def test_boolean_combinators_and_invert():
    e = (col("a") >= 1) & (col("b") == 0)
    assert e.to_ir()["op"] == "and"
    assert (~col("flag")).to_ir() == {"e": "not", "input": {"e": "col", "name": "flag"}}


@pytest.mark.parametrize(
    "op,expected",
    [(col("x") < 1, "lt"), (col("x") <= 1, "le"), (col("x") >= 1, "ge"), (col("x") != 1, "ne")],
)
def test_all_comparisons(op, expected):
    assert op.to_ir()["op"] == expected
