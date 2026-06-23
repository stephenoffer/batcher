"""Plan-shape unit tests for the `outer_to_inner_join` rewrite."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.joins import (
    _null_rejecting_cols,
    _strengthened,
    outer_to_inner_join,
)
from batcher.plan.expr_ir import Coalesce, Col, Lit
from batcher.plan.logical import Filter, Join, JoinOutputCol, Scan


def _joined(how: str):
    left = bt.from_pydict({"dept_id": [10, 20], "name": ["a", "b"]})
    right = bt.from_pydict({"dept_id": [10, 30], "dept": ["eng", "ops"]})
    return left.join(right, on="dept_id", how=how)


def test_rule_registered():
    assert "outer_to_inner_join" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_left_join_rejecting_right_becomes_inner():
    plan = _joined("left").filter(col("dept") == "eng")._plan
    assert isinstance(plan, Filter) and isinstance(plan.input, Join)
    out = outer_to_inner_join(plan, None)
    assert isinstance(out, Filter)
    assert out.input.join_type == "inner"


def test_left_join_predicate_on_left_is_noop():
    plan = _joined("left").filter(col("name") == "a")._plan
    assert outer_to_inner_join(plan, None) is None


def test_left_join_is_null_is_noop():
    # IS NULL keeps the null-extended rows — never strengthen.
    plan = _joined("left").filter(col("dept").is_null())._plan
    assert outer_to_inner_join(plan, None) is None


def test_right_join_rejecting_left_becomes_inner():
    plan = _joined("right").filter(col("name") == "a")._plan
    out = outer_to_inner_join(plan, None)
    assert out is not None and out.input.join_type == "inner"


def test_inner_join_is_noop():
    plan = _joined("inner").filter(col("dept") == "eng")._plan
    assert outer_to_inner_join(plan, None) is None


def test_idempotent_no_refire():
    plan = _joined("left").filter(col("dept") == "eng")._plan
    once = outer_to_inner_join(plan, None)
    assert outer_to_inner_join(once, None) is None  # already inner → no change


# --- _strengthened truth table ------------------------------------------------


def test_strengthened_left():
    assert _strengthened("left", rejects_left=False, rejects_right=True) == "inner"
    assert _strengthened("left", rejects_left=True, rejects_right=False) == "left"


def test_strengthened_right():
    assert _strengthened("right", rejects_left=True, rejects_right=False) == "inner"
    assert _strengthened("right", rejects_left=False, rejects_right=True) == "right"


def test_strengthened_full():
    # Rejecting nulls on the right drops left-only rows → a right join remains.
    assert _strengthened("full", rejects_left=False, rejects_right=True) == "right"
    # Rejecting nulls on the left drops right-only rows → a left join remains.
    assert _strengthened("full", rejects_left=True, rejects_right=False) == "left"
    assert _strengthened("full", rejects_left=True, rejects_right=True) == "inner"
    assert _strengthened("full", rejects_left=False, rejects_right=False) == "full"


# --- full join via a constructed node (the API inserts a coalescing project) --


def _full_join_node() -> Join:
    left = Scan(0, _Schema(["dept_id", "name"]))
    right = Scan(1, _Schema(["dept_id", "dept"]))
    output = (
        JoinOutputCol("left", "dept_id", "__fk_l_0"),
        JoinOutputCol("right", "dept_id", "__fk_r_0"),
        JoinOutputCol("left", "name", "name"),
        JoinOutputCol("right", "dept", "dept"),
    )
    return Join(left, right, ("dept_id",), ("dept_id",), "full", output)


class _Schema:
    def __init__(self, names: list[str]) -> None:
        self.names = names


def test_full_join_rejecting_right_data_col_becomes_right():
    join = _full_join_node()
    plan = Filter(join, Col("dept") == Lit("eng"))
    out = outer_to_inner_join(plan, None)
    assert out is not None and out.input.join_type == "right"


def test_full_join_rejecting_left_data_col_becomes_left():
    join = _full_join_node()
    plan = Filter(join, Col("name") == Lit("a"))
    out = outer_to_inner_join(plan, None)
    assert out is not None and out.input.join_type == "left"


# --- _null_rejecting_cols semantics -------------------------------------------


def test_rejecting_comparison():
    assert _null_rejecting_cols(Col("x") > Lit(5)) == {"x"}


def test_rejecting_and_is_union():
    assert _null_rejecting_cols((Col("x") > Lit(5)) & (Col("y") < Lit(3))) == {"x", "y"}


def test_rejecting_or_is_intersection():
    # OR rejects a column only if both sides reject it.
    expr = (Col("x") > Lit(5)) | (Col("x") < Lit(0))
    assert _null_rejecting_cols(expr) == {"x"}
    expr2 = (Col("x") > Lit(5)) | (Col("y") < Lit(0))
    assert _null_rejecting_cols(expr2) == set()


def test_coalesce_does_not_propagate_null():
    # coalesce(x, 0) > 5 is never null-from-x, so x is not rejected.
    expr = Coalesce([Col("x"), Lit(0)]) > Lit(5)
    assert _null_rejecting_cols(expr) == set()


def test_is_null_rejects_nothing():
    assert _null_rejecting_cols(Col("x").is_null()) == set()
