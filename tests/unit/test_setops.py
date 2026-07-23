"""Plan-shape + idempotence unit tests for the `setops` rules.

Each rule gets a "fires and yields the intended shape" case, an idempotence case
(applying it twice equals once), and a "does not fire when unsafe" case (chiefly the
bag-vs-set guard: a rewrite valid for a distinct union must be refused for UNION ALL).
"""

from __future__ import annotations

import batcher as bt
import batcher.kyber.rules.extra.setops as _setops  # noqa: F401  (registers rules)
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Binary, Col, Lit
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    Project,
    Projection,
    SortKeySpec,
    Union,
)

_RULES = {r.name: r for r in DEFAULT_REGISTRY.rules()}


def _apply(name: str, plan):
    """Run one registered rule (bottom-up, once) over `plan`; rules ignore the ctx."""
    return _RULES[name].apply(plan, None)


def _assert_idempotent(name: str, plan) -> None:
    once = _apply(name, plan)
    twice = _apply(name, once)
    assert twice.to_ir() == once.to_ir()


def _p(data: dict):
    return bt.from_pydict(data)._plan


def _abc():
    return _p({"x": [1, 2, 3]}), _p({"x": [3, 4, 5]}), _p({"x": [6, 7]})


def _items():
    return (Projection("x", Col("x")),)


def _keys():
    return (SortKeySpec(Col("x")),)


def _pred():
    return Binary("gt", Col("x"), Lit(1))


def _diff(plan):
    """A structurally-distinct branch (isolated scans all share source_id 0, so a
    plain second scan is IR-identical to the first — wrap it to make it differ)."""
    return Filter(plan, Binary("gt", Col("x"), Lit(0)))


# --- registration -------------------------------------------------------------


def test_all_rules_registered():
    expected = {
        "flatten_nested_union",
        "simplify_singleton_union",
        "prune_empty_union_branch",
        "dedup_distinct_union_branches",
        "drop_distinct_in_distinct_union",
        "push_project_through_union",
        "fold_distinct_union_all",
        "push_filter_through_distinct",
        "prune_distinct_of_empty",
    }
    assert expected <= set(_RULES)


# --- flatten_nested_union -----------------------------------------------------


def test_flatten_union_all():
    a, b, c = _abc()
    plan = Union((Union((a, b), distinct=False), c), distinct=False)
    out = _apply("flatten_nested_union", plan)
    assert isinstance(out, Union) and out.distinct is False
    assert len(out.inputs) == 3
    assert not any(isinstance(i, Union) for i in out.inputs)
    _assert_idempotent("flatten_nested_union", plan)


def test_flatten_distinct_absorbs_any_child():
    a, b, c = _abc()
    # A distinct parent absorbs a UNION ALL child (the outer dedup dominates).
    plan = Union((Union((a, b), distinct=False), c), distinct=True)
    out = _apply("flatten_nested_union", plan)
    assert isinstance(out, Union) and out.distinct is True
    assert len(out.inputs) == 3


def test_flatten_refuses_distinct_child_under_union_all():
    a, b, c = _abc()
    # UNION ALL parent must NOT absorb a distinct child — it would drop duplicates.
    plan = Union((Union((a, b), distinct=True), c), distinct=False)
    out = _apply("flatten_nested_union", plan)
    assert out.to_ir() == plan.to_ir()  # unchanged


# --- simplify_singleton_union -------------------------------------------------


def test_singleton_union_all_becomes_branch():
    a, _b, _c = _abc()
    out = _apply("simplify_singleton_union", Union((a,), distinct=False))
    assert out.to_ir() == a.to_ir()


def test_singleton_union_distinct_becomes_distinct():
    a, _b, _c = _abc()
    out = _apply("simplify_singleton_union", Union((a,), distinct=True))
    assert isinstance(out, Distinct)
    assert out.input.to_ir() == a.to_ir()


def test_singleton_union_noop_when_multiple():
    a, b, _c = _abc()
    plan = Union((a, b), distinct=False)
    assert _apply("simplify_singleton_union", plan).to_ir() == plan.to_ir()
    _assert_idempotent("simplify_singleton_union", Union((a,), distinct=True))


# --- prune_empty_union_branch -------------------------------------------------


def test_prune_empty_leaves_single_branch():
    a, b, _c = _abc()
    out = _apply("prune_empty_union_branch", Union((a, Limit(b, 0)), distinct=False))
    assert out.to_ir() == a.to_ir()


def test_prune_empty_keeps_remaining_branches():
    a, b, c = _abc()
    out = _apply("prune_empty_union_branch", Union((a, Limit(b, 0), c), distinct=False))
    assert isinstance(out, Union) and len(out.inputs) == 2


def test_prune_empty_distinct_single_wraps_distinct():
    a, b, _c = _abc()
    out = _apply("prune_empty_union_branch", Union((a, Limit(b, 0)), distinct=True))
    assert isinstance(out, Distinct)


def test_prune_empty_noop_and_idempotent():
    a, b, _c = _abc()
    plan = Union((a, b), distinct=False)
    assert _apply("prune_empty_union_branch", plan).to_ir() == plan.to_ir()
    _assert_idempotent("prune_empty_union_branch", Union((a, Limit(b, 0), a), distinct=False))


# --- dedup_distinct_union_branches --------------------------------------------


def test_dedup_drops_identical_branch():
    a, _b, _c = _abc()
    other = _diff(a)
    out = _apply("dedup_distinct_union_branches", Union((a, other, a), distinct=True))
    assert isinstance(out, Union) and len(out.inputs) == 2


def test_dedup_refused_for_union_all():
    a, _b, _c = _abc()
    plan = Union((a, _diff(a), a), distinct=False)  # bag semantics: the repeat matters
    assert _apply("dedup_distinct_union_branches", plan).to_ir() == plan.to_ir()


def test_dedup_single_distinct():
    a, _b, _c = _abc()
    out = _apply("dedup_distinct_union_branches", Union((a, a), distinct=True))
    assert isinstance(out, Distinct)
    _assert_idempotent("dedup_distinct_union_branches", Union((a, _diff(a), a), distinct=True))


# --- drop_distinct_in_distinct_union ------------------------------------------


def test_drop_branch_distinct_under_distinct_union():
    a, b, _c = _abc()
    out = _apply("drop_distinct_in_distinct_union", Union((Distinct(a), b), distinct=True))
    assert isinstance(out, Union) and out.distinct is True
    assert not any(isinstance(i, Distinct) for i in out.inputs)


def test_drop_branch_distinct_refused_for_union_all():
    a, b, _c = _abc()
    plan = Union((Distinct(a), b), distinct=False)
    assert _apply("drop_distinct_in_distinct_union", plan).to_ir() == plan.to_ir()


def test_drop_branch_distinct_idempotent_nested():
    a, b, _c = _abc()
    _assert_idempotent(
        "drop_distinct_in_distinct_union", Union((Distinct(Distinct(a)), b), distinct=True)
    )


# `eliminate_sort_in_distinct_union_branch` and `eliminate_sort_before_distinct` were
# removed, and their tests with them, because both rewrites were unsound: `Distinct` is
# order-preserving here, so stripping the sort below it changed the observable row order
# (and therefore which rows a downstream `limit` returned). The tests deleted alongside
# them asserted the sort *was* stripped — they pinned the defect, so keeping them would
# have blocked the fix. `tests/differential/test_diff_setops.py` now asserts the opposite,
# order-sensitively.


# --- push_project_through_union -----------------------------------------------


def test_push_project_through_union_all():
    a, b, _c = _abc()
    out = _apply("push_project_through_union", Project(Union((a, b), distinct=False), _items()))
    assert isinstance(out, Union) and out.distinct is False
    assert all(isinstance(i, Project) for i in out.inputs)


def test_push_project_refused_for_distinct_union():
    a, b, _c = _abc()
    plan = Project(Union((a, b), distinct=True), _items())
    assert _apply("push_project_through_union", plan).to_ir() == plan.to_ir()
    all_proj = Project(Union((a, b), distinct=False), _items())
    _assert_idempotent("push_project_through_union", all_proj)


# --- fold_distinct_union_all --------------------------------------------------


def test_fold_distinct_union_all():
    a, b, _c = _abc()
    out = _apply("fold_distinct_union_all", Distinct(Union((a, b), distinct=False)))
    assert isinstance(out, Union) and out.distinct is True


def test_fold_noop_for_already_distinct_union():
    a, b, _c = _abc()
    plan = Distinct(Union((a, b), distinct=True))
    assert _apply("fold_distinct_union_all", plan).to_ir() == plan.to_ir()


def test_fold_noop_for_non_union():
    a, _b, _c = _abc()
    plan = Distinct(a)
    b = _abc()[1]
    assert _apply("fold_distinct_union_all", plan).to_ir() == plan.to_ir()
    _assert_idempotent("fold_distinct_union_all", Distinct(Union((a, b), distinct=False)))


# --- push_filter_through_distinct ---------------------------------------------


def test_push_filter_through_distinct():
    a, _b, _c = _abc()
    out = _apply("push_filter_through_distinct", Filter(Distinct(a), _pred()))
    assert isinstance(out, Distinct) and isinstance(out.input, Filter)


def test_push_filter_noop_without_distinct():
    a, _b, _c = _abc()
    plan = Filter(a, _pred())
    assert _apply("push_filter_through_distinct", plan).to_ir() == plan.to_ir()
    _assert_idempotent("push_filter_through_distinct", Filter(Distinct(a), _pred()))


# --- prune_distinct_of_empty --------------------------------------------------


def test_prune_distinct_of_empty():
    a, _b, _c = _abc()
    out = _apply("prune_distinct_of_empty", Distinct(Limit(a, 0)))
    assert isinstance(out, Limit) and out.n == 0


def test_prune_distinct_of_empty_noop():
    a, _b, _c = _abc()
    plan = Distinct(a)
    assert _apply("prune_distinct_of_empty", plan).to_ir() == plan.to_ir()
    _assert_idempotent("prune_distinct_of_empty", Distinct(Limit(a, 0)))
