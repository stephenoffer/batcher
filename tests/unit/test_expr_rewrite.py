"""Constant folding and expression simplification: structure + result-preserving."""

from __future__ import annotations

import batcher as bt
from batcher import col, lit
from batcher.kyber.rules.normalize import fold_constants, simplify_expressions


def _filter_plan(predicate):
    ds = bt.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    return ds.filter(predicate)._plan


# --- constant folding ------------------------------------------------------


def test_fold_arithmetic_in_predicate():
    plan = _filter_plan(col("x") > (lit(2) + lit(3)))
    folded = fold_constants(plan)
    assert folded.predicate.to_ir()["right"] == {"e": "lit", "value": {"int": 5}}


def test_fold_comparison_to_bool():
    plan = _filter_plan((lit(1) == lit(1)) & (col("x") > lit(0)))
    folded = fold_constants(plan)
    # left conjunct 1==1 folds to literal true
    assert folded.predicate.to_ir()["left"] == {"e": "lit", "value": {"bool": True}}


def test_int_division_not_folded():
    # Arrow truncates integer division; Python `/` is float — leaving it alone
    # keeps the engine the source of truth.
    plan = _filter_plan(col("x") > (lit(7) / lit(2)))
    folded = fold_constants(plan)
    assert folded.predicate.to_ir()["right"]["e"] == "binary"


def test_float_arithmetic_folds():
    plan = _filter_plan(col("y") > (lit(1.5) * lit(2.0)))
    folded = fold_constants(plan)
    assert folded.predicate.to_ir()["right"] == {"e": "lit", "value": {"float": 3.0}}


# --- simplification --------------------------------------------------------


def test_and_true_collapses():
    plan = _filter_plan((col("x") > lit(2)) & lit(True))
    simplified = simplify_expressions(plan)
    assert simplified.predicate.to_ir()["op"] == "gt"  # the `& true` is gone


def test_double_negation_collapses():
    plan = _filter_plan(~~(col("x") > lit(2)))
    simplified = simplify_expressions(plan)
    assert simplified.predicate.to_ir()["op"] == "gt"


def test_add_zero_identity():
    plan = bt.from_pydict({"x": [1, 2]}).select(z=col("x") + lit(0))._plan
    simplified = simplify_expressions(plan)
    assert simplified.items[0].expr.to_ir() == {"e": "col", "name": "x"}


def test_mul_by_float_one_not_simplified():
    # `x * 1.0` widens an int column to float — dropping it would change the type.
    plan = bt.from_pydict({"x": [1, 2]}).select(z=col("x") * lit(1.0))._plan
    simplified = simplify_expressions(plan)
    assert simplified.items[0].expr.to_ir()["e"] == "binary"


# --- end-to-end results unchanged (through the full optimizer) -------------


def test_fold_and_simplify_preserve_results():
    ds = bt.from_pydict({"v": [1, 2, 3, 4, 5]})
    # 2+1 folds to 3; `& true` simplifies away — result must be rows where v > 3.
    out = ds.filter((col("v") > (lit(2) + lit(1))) & lit(True)).collect().to_pydict()
    assert out == {"v": [4, 5]}


def test_folding_matches_unfolded_equivalent():
    ds = bt.from_pydict({"v": [10, 20, 30, 40]})
    folded = ds.filter(col("v") >= (lit(10) * lit(2))).collect().to_pydict()
    direct = ds.filter(col("v") >= lit(20)).collect().to_pydict()
    assert folded == direct
