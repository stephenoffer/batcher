"""Plan-shape, idempotence, and safety unit tests for the `predicate_infer` rules.

Each rule gets three checks: it fires and yields the intended shape, applying it twice
equals applying it once (idempotence — the fixpoint driver requires it), and it does
*not* fire on a case where the rewrite would be unsafe or vacuous.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import predicate_infer as m
from batcher.plan.expr_ir import InList, Lit
from batcher.plan.logical import Filter, LogicalPlan


def _plan(pred) -> Filter:
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    return ds.filter(pred)._plan


def _ir(node: LogicalPlan | None):
    return None if node is None else node.predicate.to_ir()


def _is_false(node: LogicalPlan | None) -> bool:
    return isinstance(node, Filter) and node.predicate.to_ir() == Lit(False).to_ir()


def _idempotent(rule, pred) -> None:
    once = rule(_plan(pred), None)
    assert once is not None, "rule expected to fire"
    again = rule(once, None)
    assert again is None or again.to_ir() == once.to_ir()


# --- registration -----------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert set(m.__all__) <= names
    assert len(m.__all__) == 12


# --- remove_duplicate_conjuncts ---------------------------------------------


def test_dedup_fires():
    out = m.remove_duplicate_conjuncts(_plan((col("a") > 5) & (col("a") > 5)), None)
    assert _ir(out) == (col("a") > 5).to_ir()
    _idempotent(m.remove_duplicate_conjuncts, (col("a") > 5) & (col("a") > 5))


def test_dedup_does_not_fire_on_distinct():
    assert m.remove_duplicate_conjuncts(_plan((col("a") > 5) & (col("b") > 5)), None) is None


# --- tighten_comparison_bounds ----------------------------------------------


def test_tighten_keeps_tighter_lower():
    out = m.tighten_comparison_bounds(_plan((col("a") > 5) & (col("a") > 3)), None)
    assert _ir(out) == (col("a") > 5).to_ir()


def test_tighten_strict_beats_nonstrict():
    out = m.tighten_comparison_bounds(_plan((col("a") >= 5) & (col("a") > 5)), None)
    assert _ir(out) == (col("a") > 5).to_ir()
    _idempotent(m.tighten_comparison_bounds, (col("a") >= 5) & (col("a") > 5))


def test_tighten_does_not_fire_across_columns():
    assert m.tighten_comparison_bounds(_plan((col("a") > 5) & (col("b") > 3)), None) is None


# --- filter_range_contradiction ---------------------------------------------


def test_range_contradiction():
    assert _is_false(m.filter_range_contradiction(_plan((col("a") > 5) & (col("a") < 3)), None))


def test_equality_contradiction():
    assert _is_false(m.filter_range_contradiction(_plan((col("a") == 1) & (col("a") == 2)), None))


def test_eq_vs_range_contradiction():
    assert _is_false(m.filter_range_contradiction(_plan((col("a") == 1) & (col("a") > 5)), None))
    _idempotent(m.filter_range_contradiction, (col("a") > 5) & (col("a") < 3))


def test_range_contradiction_does_not_fire_on_satisfiable():
    assert m.filter_range_contradiction(_plan((col("a") > 5) & (col("a") < 10)), None) is None
    # touching bounds `a >= 5 AND a <= 5` is satisfiable (a == 5)
    assert m.filter_range_contradiction(_plan((col("a") >= 5) & (col("a") <= 5)), None) is None


# --- filter_eq_neq_contradiction --------------------------------------------


def test_eq_neq_contradiction():
    assert _is_false(m.filter_eq_neq_contradiction(_plan((col("a") == 1) & (col("a") != 1)), None))
    _idempotent(m.filter_eq_neq_contradiction, (col("a") == 1) & (col("a") != 1))


def test_eq_neq_does_not_fire_on_distinct_values():
    assert m.filter_eq_neq_contradiction(_plan((col("a") == 1) & (col("a") != 2)), None) is None


# --- drop_bound_dominated_neq -----------------------------------------------


def test_drop_neq_outside_range():
    out = m.drop_bound_dominated_neq(_plan((col("a") > 5) & (col("a") != 3)), None)
    assert _ir(out) == (col("a") > 5).to_ir()
    _idempotent(m.drop_bound_dominated_neq, (col("a") > 5) & (col("a") != 3))


def test_keep_neq_inside_range():
    assert m.drop_bound_dominated_neq(_plan((col("a") > 5) & (col("a") != 7)), None) is None


# --- drop_redundant_is_not_null ---------------------------------------------


def test_drop_redundant_is_not_null():
    out = m.drop_redundant_is_not_null(_plan((col("a") > 5) & col("a").is_not_null()), None)
    assert _ir(out) == (col("a") > 5).to_ir()
    _idempotent(m.drop_redundant_is_not_null, (col("a") > 5) & col("a").is_not_null())


def test_keep_is_not_null_without_null_rejecting_sibling():
    # An OR conjunct could still admit a null `a`, so the IS NOT NULL is NOT redundant.
    pred = ((col("a") > 5) | (col("b") < 3)) & col("a").is_not_null()
    assert m.drop_redundant_is_not_null(_plan(pred), None) is None


# --- IN-list refinement -----------------------------------------------------


def test_refine_in_list_by_comparison():
    out = m.refine_in_list_by_comparison(_plan(InList(col("a"), (1, 2, 3)) & (col("a") > 1)), None)
    assert _ir(out) == InList(col("a"), (2, 3)).to_ir()
    _idempotent(m.refine_in_list_by_comparison, InList(col("a"), (1, 2, 3)) & (col("a") > 1))


def test_refine_in_list_by_comparison_empty_is_false():
    out = m.refine_in_list_by_comparison(_plan(InList(col("a"), (1, 2, 3)) & (col("a") > 9)), None)
    assert _is_false(out)


def test_refine_in_list_by_comparison_no_narrowing():
    assert (
        m.refine_in_list_by_comparison(_plan(InList(col("a"), (1, 2, 3)) & (col("a") > 0)), None)
        is None
    )


def test_refine_in_list_by_equality_member():
    out = m.refine_in_list_by_equality(_plan(InList(col("a"), (1, 2, 3)) & (col("a") == 2)), None)
    assert _ir(out) == (col("a") == 2).to_ir()
    _idempotent(m.refine_in_list_by_equality, InList(col("a"), (1, 2, 3)) & (col("a") == 2))


def test_refine_in_list_by_equality_absent_is_false():
    out = m.refine_in_list_by_equality(_plan(InList(col("a"), (1, 2, 3)) & (col("a") == 5)), None)
    assert _is_false(out)


def test_refine_in_list_by_equality_other_column():
    assert (
        m.refine_in_list_by_equality(_plan(InList(col("a"), (1, 2, 3)) & (col("b") == 2)), None)
        is None
    )


def test_refine_in_list_by_neq():
    out = m.refine_in_list_by_neq(_plan(InList(col("a"), (1, 2, 3)) & (col("a") != 2)), None)
    assert _ir(out) == InList(col("a"), (1, 3)).to_ir()
    _idempotent(m.refine_in_list_by_neq, InList(col("a"), (1, 2, 3)) & (col("a") != 2))


def test_refine_in_list_by_neq_nonmember_no_change():
    assert (
        m.refine_in_list_by_neq(_plan(InList(col("a"), (1, 2, 3)) & (col("a") != 9)), None) is None
    )


def test_intersect_in_lists():
    out = m.intersect_in_lists(
        _plan(InList(col("a"), (1, 2, 3)) & InList(col("a"), (2, 3, 4))), None
    )
    assert _ir(out) == InList(col("a"), (2, 3)).to_ir()
    _idempotent(m.intersect_in_lists, InList(col("a"), (1, 2, 3)) & InList(col("a"), (2, 3, 4)))


def test_intersect_in_lists_empty_is_false():
    out = m.intersect_in_lists(_plan(InList(col("a"), (1, 2)) & InList(col("a"), (3, 4))), None)
    assert _is_false(out)


def test_intersect_in_lists_other_column():
    assert (
        m.intersect_in_lists(_plan(InList(col("a"), (1, 2)) & InList(col("b"), (2, 3))), None)
        is None
    )


def test_singleton_in_list_to_eq():
    out = m.singleton_in_list_to_eq(_plan(InList(col("a"), (2,))), None)
    assert _ir(out) == (col("a") == 2).to_ir()
    _idempotent(m.singleton_in_list_to_eq, InList(col("a"), (2,)))


def test_singleton_does_not_fire_on_multi():
    assert m.singleton_in_list_to_eq(_plan(InList(col("a"), (1, 2))), None) is None


# --- infer_transitive_comparisons -------------------------------------------


def test_transitive_adds_edge():
    from batcher.plan.expr_rewrite import split_conjuncts

    out = m.infer_transitive_comparisons(_plan((col("a") < col("b")) & (col("b") < col("c"))), None)
    conjuncts = [c.to_ir() for c in split_conjuncts(out.predicate)]
    assert (col("a") < col("c")).to_ir() in conjuncts
    _idempotent(m.infer_transitive_comparisons, (col("a") < col("b")) & (col("b") < col("c")))


def test_transitive_cycle_is_false():
    out = m.infer_transitive_comparisons(_plan((col("a") < col("b")) & (col("b") < col("a"))), None)
    assert _is_false(out)


def test_transitive_does_not_fire_without_chain():
    assert (
        m.infer_transitive_comparisons(_plan((col("a") < col("b")) & (col("c") < col("b"))), None)
        is None
    )
