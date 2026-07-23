"""Plan-shape, idempotence, and does-not-fire tests for the `conditional` rule family.

Each rule gets three kinds of test: it *fires* on the shape it targets, it is *idempotent*
(re-applying returns None), and — the part that matters — it *does not fire* where firing would
be unsound: a NULL-ish condition that is not a literal, a dropped arm whose type is not carried
by a surviving arm (which would silently change the CASE's output type), a NaN/signed-zero
literal pair, and a non-adjacent COALESCE duplicate.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, lit
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import conditional as cd
from batcher.plan.expr_ir import Case, Coalesce, Greatest, Least, NullIf
from batcher.plan.expr_ir.constructors import coalesce, greatest, least, nullif, when

_RULE_NAMES = [
    "case_all_branches_same_result",
    "case_drop_duplicate_conditions",
    "case_drop_unreachable_branches",
    "case_first_true_branch_wins",
    "case_no_branches_to_else",
    "case_to_coalesce",
    "coalesce_dedup_args",
    "coalesce_drop_nulls_after_first_non_null",
    "coalesce_flatten_nested",
    "coalesce_single_arg",
    "greatest_least_dedup_args",
    "greatest_least_flatten_nested",
    "greatest_least_fold_literals",
    "greatest_least_single_arg",
    "nullif_distinct_literals",
]

#: A typed NULL — the engine has no NULL literal, so `NULLIF(v, v)` is the idiom (see the module).
_NULL_INT = nullif(lit(1), lit(1))


def _proj(expr):
    """A `Project` over a fixed source, carrying `expr` as its single item."""
    return bt.from_pydict({"a": [1, 2, None], "b": [4, 5, 6]}).select(r=expr)._plan


def _flt(pred):
    return bt.from_pydict({"a": [1, 2, None], "b": [4, 5, 6]}).filter(pred)._plan


def _expr_ir(node):
    """The IR of the rewritten projection item."""
    return node.items[0].expr.to_ir()


def _ir(expr):
    return expr.to_ir()


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert len(_RULE_NAMES) == 15
    for n in _RULE_NAMES:
        assert n in names


# --- CASE: unreachable / constant branches ----------------------------------


def test_drop_unreachable_false_branch_fires():
    case = when(lit(False)).then(0).when(col("a") > 1).then(1).otherwise(2)
    out = cd.case_drop_unreachable_branches(_proj(case), None)
    assert _expr_ir(out) == _ir(when(col("a") > 1).then(1).otherwise(2))


def test_drop_unreachable_is_idempotent():
    case = when(lit(False)).then(0).when(col("a") > 1).then(1).otherwise(2)
    once = cd.case_drop_unreachable_branches(_proj(case), None)
    assert cd.case_drop_unreachable_branches(once, None) is None


def test_drop_unreachable_does_not_fire_on_a_runtime_condition():
    case = when(col("a") > 99).then(0).otherwise(2)
    assert cd.case_drop_unreachable_branches(_proj(case), None) is None


def test_drop_unreachable_refuses_a_type_changing_drop():
    # The dead branch is the ONLY float arm: dropping it would narrow DOUBLE → INT.
    case = when(lit(False)).then(1.5).when(col("a") > 1).then(1).otherwise(2)
    assert cd.case_drop_unreachable_branches(_proj(case), None) is None


def test_drop_unreachable_all_branches_leaves_husk_for_the_else_rule():
    case = when(lit(False)).then(0).otherwise(2)
    out = cd.case_drop_unreachable_branches(_proj(case), None)
    assert out.items[0].expr.branches == []
    collapsed = cd.case_no_branches_to_else(out, None)
    assert _expr_ir(collapsed) == _ir(lit(2))


def test_first_true_branch_wins_fires():
    case = when(col("a") > 1).then(1).when(lit(True)).then(2).otherwise(3)
    out = cd.case_first_true_branch_wins(_proj(case), None)
    assert _expr_ir(out) == _ir(when(col("a") > 1).then(1).otherwise(2))


def test_first_true_branch_wins_is_idempotent():
    case = when(col("a") > 1).then(1).when(lit(True)).then(2).otherwise(3)
    once = cd.case_first_true_branch_wins(_proj(case), None)
    assert cd.case_first_true_branch_wins(once, None) is None


def test_first_true_does_not_fire_on_false_or_a_runtime_condition():
    assert (
        cd.case_first_true_branch_wins(_proj(when(lit(False)).then(1).otherwise(2)), None) is None
    )
    assert (
        cd.case_first_true_branch_wins(_proj(when(col("a") > 1).then(1).otherwise(2)), None) is None
    )


def test_first_true_refuses_a_type_changing_drop():
    # The ELSE is the only float arm; collapsing onto the TRUE branch would narrow DOUBLE → INT.
    case = when(lit(True)).then(1).otherwise(2.5)
    assert cd.case_first_true_branch_wins(_proj(case), None) is None


def test_all_branches_same_result_fires():
    case = when(col("a") > 1).then(7).when(col("a") < 0).then(7).otherwise(7)
    out = cd.case_all_branches_same_result(_proj(case), None)
    assert _expr_ir(out) == _ir(lit(7))


def test_all_branches_same_result_does_not_fire_when_one_differs():
    case = when(col("a") > 1).then(7).when(col("a") < 0).then(7).otherwise(8)
    assert cd.case_all_branches_same_result(_proj(case), None) is None


def test_no_branches_to_else_fires():
    out = cd.case_no_branches_to_else(_proj(Case([], lit(5))), None)
    assert _expr_ir(out) == _ir(lit(5))


def test_no_branches_does_not_fire_with_a_branch():
    assert cd.case_no_branches_to_else(_proj(when(col("a") > 1).then(1).otherwise(2)), None) is None


def test_drop_duplicate_conditions_fires():
    case = when(col("a") > 1).then(1).when(col("a") > 1).then(2).otherwise(3)
    out = cd.case_drop_duplicate_conditions(_proj(case), None)
    assert _expr_ir(out) == _ir(when(col("a") > 1).then(1).otherwise(3))


def test_drop_duplicate_conditions_does_not_fire_on_distinct_conditions():
    case = when(col("a") > 1).then(1).when(col("a") > 2).then(2).otherwise(3)
    assert cd.case_drop_duplicate_conditions(_proj(case), None) is None


def test_drop_duplicate_conditions_refuses_a_type_changing_drop():
    case = when(col("a") > 1).then(1).when(col("a") > 1).then(2.5).otherwise(3)
    assert cd.case_drop_duplicate_conditions(_proj(case), None) is None


# --- CASE → COALESCE --------------------------------------------------------


def test_case_to_coalesce_is_not_null_shape():
    case = when(col("a").is_not_null()).then(col("a")).otherwise(lit(0))
    out = cd.case_to_coalesce(_proj(case), None)
    assert _expr_ir(out) == _ir(coalesce(col("a"), lit(0)))


def test_case_to_coalesce_is_null_shape():
    case = when(col("a").is_null()).then(lit(0)).otherwise(col("a"))
    out = cd.case_to_coalesce(_proj(case), None)
    assert _expr_ir(out) == _ir(coalesce(col("a"), lit(0)))


def test_case_to_coalesce_does_not_fire_when_the_then_is_a_different_column():
    case = when(col("a").is_not_null()).then(col("b")).otherwise(lit(0))
    assert cd.case_to_coalesce(_proj(case), None) is None


def test_case_to_coalesce_does_not_fire_on_a_plain_condition():
    case = when(col("a") > 1).then(col("a")).otherwise(lit(0))
    assert cd.case_to_coalesce(_proj(case), None) is None


# --- NULLIF -----------------------------------------------------------------


def test_nullif_distinct_literals_fires():
    out = cd.nullif_distinct_literals(_proj(nullif(lit(1), lit(2))), None)
    assert _expr_ir(out) == _ir(lit(1))


def test_nullif_same_literals_is_left_alone():
    """`NULLIF(v, v)` is the engine's typed NULL — it MUST survive untouched."""
    assert cd.nullif_distinct_literals(_proj(_NULL_INT), None) is None


def test_nullif_does_not_fire_across_type_classes():
    # NULLIF(1, 2.5) takes the joined DOUBLE type; folding to the INT literal would narrow it.
    assert cd.nullif_distinct_literals(_proj(nullif(lit(1), lit(2.5))), None) is None


def test_nullif_does_not_fire_on_nan_or_signed_zero():
    nan = float("nan")
    assert cd.nullif_distinct_literals(_proj(nullif(lit(nan), lit(nan))), None) is None
    assert cd.nullif_distinct_literals(_proj(nullif(lit(0.0), lit(-0.0))), None) is None


def test_nullif_does_not_fire_on_a_column():
    assert cd.nullif_distinct_literals(_proj(nullif(col("a"), lit(2))), None) is None


# --- COALESCE ---------------------------------------------------------------


def test_coalesce_flatten_nested_fires():
    out = cd.coalesce_flatten_nested(_proj(coalesce(col("a"), coalesce(col("b"), lit(0)))), None)
    assert _expr_ir(out) == _ir(coalesce(col("a"), col("b"), lit(0)))


def test_coalesce_flatten_is_idempotent():
    once = cd.coalesce_flatten_nested(_proj(coalesce(col("a"), coalesce(col("b"), lit(0)))), None)
    assert cd.coalesce_flatten_nested(once, None) is None


def test_coalesce_drops_a_typed_null_argument():
    out = cd.coalesce_drop_nulls_after_first_non_null(_proj(coalesce(_NULL_INT, lit(5))), None)
    assert _expr_ir(out) == _ir(Coalesce([lit(5)]))


def test_coalesce_truncates_after_the_first_literal():
    out = cd.coalesce_drop_nulls_after_first_non_null(
        _proj(coalesce(col("a"), lit(0), lit(9))), None
    )
    assert _expr_ir(out) == _ir(coalesce(col("a"), lit(0)))


def test_coalesce_refuses_to_drop_a_null_that_carries_the_type():
    # coalesce(int_col, NULL::double) is DOUBLE; dropping the null would narrow it to INT.
    expr = coalesce(col("a"), nullif(lit(2.5), lit(2.5)))
    assert cd.coalesce_drop_nulls_after_first_non_null(_proj(expr), None) is None


def test_coalesce_refuses_to_truncate_an_untypeable_tail():
    expr = coalesce(col("a"), lit(0), col("b") + lit(1))
    assert cd.coalesce_drop_nulls_after_first_non_null(_proj(expr), None) is None


def test_coalesce_single_arg_fires():
    out = cd.coalesce_single_arg(_proj(Coalesce([col("a")])), None)
    assert _expr_ir(out) == _ir(col("a"))


def test_coalesce_single_arg_does_not_fire_on_two():
    assert cd.coalesce_single_arg(_proj(coalesce(col("a"), col("b"))), None) is None


def test_coalesce_dedup_adjacent_args_fires():
    out = cd.coalesce_dedup_args(_proj(coalesce(col("a"), col("a"), col("b"))), None)
    assert _expr_ir(out) == _ir(coalesce(col("a"), col("b")))


def test_coalesce_dedup_does_not_touch_non_adjacent_args():
    assert cd.coalesce_dedup_args(_proj(coalesce(col("a"), col("b"), col("a"))), None) is None


# --- GREATEST / LEAST -------------------------------------------------------


def test_greatest_least_single_arg_fires():
    assert _expr_ir(cd.greatest_least_single_arg(_proj(Greatest([col("a")])), None)) == _ir(
        col("a")
    )
    assert _expr_ir(cd.greatest_least_single_arg(_proj(Least([col("a")])), None)) == _ir(col("a"))


def test_greatest_least_single_arg_does_not_fire_on_two():
    assert cd.greatest_least_single_arg(_proj(greatest(col("a"), col("b"))), None) is None


def test_greatest_least_flatten_nested_fires():
    expr = greatest(col("a"), greatest(col("b"), lit(3)))
    out = cd.greatest_least_flatten_nested(_proj(expr), None)
    assert _expr_ir(out) == _ir(greatest(col("a"), col("b"), lit(3)))


def test_greatest_least_flatten_does_not_cross_kinds():
    expr = greatest(col("a"), least(col("b"), lit(3)))
    assert cd.greatest_least_flatten_nested(_proj(expr), None) is None


def test_greatest_least_dedup_args_fires():
    out = cd.greatest_least_dedup_args(_proj(greatest(col("a"), col("b"), col("a"))), None)
    assert _expr_ir(out) == _ir(greatest(col("a"), col("b")))


def test_greatest_least_dedup_does_not_fire_without_a_duplicate():
    assert cd.greatest_least_dedup_args(_proj(least(col("a"), col("b"))), None) is None


def test_greatest_least_fold_literals_fires():
    assert _expr_ir(cd.greatest_least_fold_literals(_proj(greatest(2, 5, 3)), None)) == _ir(lit(5))
    assert _expr_ir(cd.greatest_least_fold_literals(_proj(least(2, 5, 3)), None)) == _ir(lit(2))
    strs = greatest(lit("a"), lit("c"), lit("b"))
    assert _expr_ir(cd.greatest_least_fold_literals(_proj(strs), None)) == _ir(lit("c"))


def test_greatest_least_fold_refuses_floats():
    """NaN's ordering and `-0.0 == 0.0` make a float fold unsound; it must not fire."""
    assert cd.greatest_least_fold_literals(_proj(greatest(lit(1.0), lit(2.0))), None) is None
    assert cd.greatest_least_fold_literals(_proj(least(lit(0.0), lit(-0.0))), None) is None


def test_greatest_least_fold_refuses_mixed_type_classes():
    # GREATEST(1, 2.5) is DOUBLE; folding to an INT literal would narrow the output type.
    assert cd.greatest_least_fold_literals(_proj(greatest(lit(1), lit(2.5))), None) is None


def test_greatest_least_fold_does_not_fire_with_a_column():
    assert cd.greatest_least_fold_literals(_proj(greatest(col("a"), lit(2))), None) is None


# --- the rules reach into a Filter predicate too -----------------------------


def test_rules_rewrite_inside_a_filter_predicate():
    pred = when(lit(False)).then(lit(True)).when(col("a") > 1).then(lit(True)).otherwise(lit(False))
    out = cd.case_drop_unreachable_branches(_flt(pred), None)
    assert out.predicate.to_ir() == _ir(when(col("a") > 1).then(lit(True)).otherwise(lit(False)))


def test_rules_rewrite_a_nested_conditional():
    """The leaf rewrite runs bottom-up, so a CASE buried inside a COALESCE is reached."""
    expr = coalesce(col("a"), Case([], lit(9)))
    out = cd.case_no_branches_to_else(_proj(expr), None)
    assert _expr_ir(out) == _ir(coalesce(col("a"), lit(9)))


def test_a_rule_leaves_an_unrelated_plan_alone():
    assert cd.case_to_coalesce(_proj(col("a") + lit(1)), None) is None
    assert cd.coalesce_flatten_nested(_flt(col("a") > 1), None) is None
    assert cd.greatest_least_fold_literals(_proj(NullIf(col("a"), lit(2))), None) is None
