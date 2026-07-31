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


# --- full join as the API actually builds it ---------------------------------
#
# The hand-built nodes above are `Filter(Join(..., full))`, which is a shape no query produces:
# `Dataset.join` always wraps a full join in a projection that coalesces the two sides' key
# columns into the single output key. So those tests passed while the entire `full` branch was
# unreachable on a real plan. These drive the API shape instead, which is what proves the rule
# can see it.


def _full(pred):
    return _joined("full").filter(pred)._plan


def test_the_api_puts_a_coalescing_projection_above_a_full_join():
    # Pinned explicitly: if this ever stops being true, the look-through below is dead weight
    # and the tests that depend on it are testing nothing.
    from batcher.plan.logical import Project

    plan = _full(col("dept") == "eng")
    assert isinstance(plan, Filter)
    assert isinstance(plan.input, Project)
    assert isinstance(plan.input.input, Join)
    assert any(isinstance(item.expr, Coalesce) for item in plan.input.items)


def test_api_full_join_rejecting_right_becomes_right():
    out = outer_to_inner_join(_full(col("dept") == "eng"), None)
    assert out is not None
    assert out.input.input.join_type == "right"


def test_api_full_join_rejecting_left_becomes_left():
    out = outer_to_inner_join(_full(col("name") == "a"), None)
    assert out is not None
    assert out.input.input.join_type == "left"


def test_api_full_join_rejecting_both_becomes_inner():
    out = outer_to_inner_join(_full((col("dept") == "eng") & (col("name") == "a")), None)
    assert out is not None
    assert out.input.input.join_type == "inner"


def test_api_full_join_on_the_coalesced_key_is_a_noop():
    # The output key is `coalesce(left_key, right_key)`, which is non-null on a left-only row
    # AND on a right-only row — so rejecting nulls on it rejects neither side. Treating the
    # coalesced key as a left (or right) reference would drop rows the query must keep, which
    # makes this the most important case in the file.
    assert outer_to_inner_join(_full(col("dept_id") > 5), None) is None


def test_api_full_join_is_null_is_a_noop():
    assert outer_to_inner_join(_full(col("dept").is_null()), None) is None


def test_api_full_join_rewrite_preserves_the_projection():
    before = _full(col("dept") == "eng")
    out = outer_to_inner_join(before, None)
    assert out is not None
    assert [i.alias for i in out.input.items] == [i.alias for i in before.input.items]
    assert [i.expr.to_ir() for i in out.input.items] == [i.expr.to_ir() for i in before.input.items]


def test_api_full_join_rewrite_is_idempotent():
    once = outer_to_inner_join(_full(col("dept") == "eng"), None)
    assert outer_to_inner_join(once, None) is None


def test_the_optimizer_reaches_it_end_to_end():
    from batcher.kyber.optimizer import optimize_logical
    from batcher.plan.visitor import walk

    out = optimize_logical(_full(col("dept") == "eng"))
    assert [n.join_type for n in walk(out) if isinstance(n, Join)] == ["right"]


# --- the shared pass-through helper ------------------------------------------
#
# `passthrough_renames` lives in `plan.logical.transforms` because two rules need the same
# question answered: predicate pushdown, to move a conjunct below a `Project`, and this rewrite,
# to see past the projection a full outer join carries. Both depend on *computed* items being
# excluded, so that exclusion is pinned here rather than left to each caller's confidence.


def test_passthrough_renames_maps_bare_references():
    from batcher.plan.logical import Projection, passthrough_renames

    items = (Projection("out", Col("src")), Projection("same", Col("same")))
    assert passthrough_renames(items) == {"out": "src", "same": "same"}


def test_passthrough_renames_excludes_a_computed_item():
    from batcher.plan.logical import Projection, passthrough_renames

    # `coalesce(a, b)` is the case that makes the exclusion load-bearing: it is non-null
    # wherever *either* argument is, so a fact about the output implies nothing about either
    # input. A literal and an arithmetic item are excluded for the same reason.
    items = (
        Projection("k", Coalesce([Col("a"), Col("b")])),
        Projection("one", Lit(1)),
        Projection("sum", Col("a") + Lit(1)),
        Projection("plain", Col("c")),
    )
    assert passthrough_renames(items) == {"plain": "c"}
