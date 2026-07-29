"""Plan-shape, idempotence, and does-not-fire tests for the boolean_algebra rules."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import boolean_algebra as ba
from batcher.kyber.rules.extra import membership_simplify as ms
from batcher.plan.expr_ir import Col, InList
from batcher.plan.expr_ir.constructors import coalesce
from batcher.plan.logical import Project

_RULE_NAMES = [
    "and_false_annihilator",
    "or_true_annihilator",
    "and_idempotent",
    "or_idempotent",
    "and_absorption",
    "or_absorption",
    "complement_total_bool",
    "fold_not_comparison",
    "bool_eq_literal",
    "single_in_list",
    "dedup_in_list",
    "coalesce_simplify",
]


def _flt(pred):
    return (
        bt.from_pydict({"a": [1, 2, None], "x": [10, 20, 30], "f": [True, False, None]})
        .filter(pred)
        ._plan
    )


def _proj(expr):
    return bt.from_pydict({"a": [1, 2, None], "x": [10, 20, 30]}).select(r=expr)._plan


def _pred_ir(node):
    return node.predicate.to_ir()


# --- registration -----------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    for n in _RULE_NAMES:
        assert n in names


# --- annihilators -----------------------------------------------------------


def test_and_false_fires():
    out = ba.and_false_annihilator(_flt((col("a") > 1) & bt.lit(False)), None)
    assert _pred_ir(out) == {"e": "lit", "value": {"bool": False}}


def test_and_false_idempotent():
    once = ba.and_false_annihilator(_flt((col("a") > 1) & bt.lit(False)), None)
    assert ba.and_false_annihilator(once, None) is None


def test_and_false_does_not_fire_without_false():
    assert ba.and_false_annihilator(_flt((col("a") > 1) & (col("x") > 5)), None) is None


def test_or_true_fires():
    out = ba.or_true_annihilator(_flt((col("a") > 1) | bt.lit(True)), None)
    assert _pred_ir(out) == {"e": "lit", "value": {"bool": True}}


def test_or_true_idempotent():
    once = ba.or_true_annihilator(_flt((col("a") > 1) | bt.lit(True)), None)
    assert ba.or_true_annihilator(once, None) is None


def test_annihilator_skips_unsafe_operand():
    # A division can error (zero divisor), so it is not `_safe`: the operand must not
    # be dropped even against a FALSE.
    assert ba.and_false_annihilator(_flt(((col("x") / col("a")) > 1) & bt.lit(False)), None) is None


# --- idempotence ------------------------------------------------------------


def test_and_idempotent_fires():
    out = ba.and_idempotent(_flt((col("a") > 1) & (col("a") > 1)), None)
    assert _pred_ir(out)["op"] == "gt"


def test_and_idempotent_is_idempotent():
    once = ba.and_idempotent(_flt((col("a") > 1) & (col("a") > 1)), None)
    assert ba.and_idempotent(once, None) is None


def test_or_idempotent_fires():
    out = ba.or_idempotent(_flt((col("a") > 1) | (col("a") > 1)), None)
    assert _pred_ir(out)["op"] == "gt"


def test_idempotent_does_not_fire_on_distinct_operands():
    assert ba.and_idempotent(_flt((col("a") > 1) & (col("a") > 2)), None) is None


# --- absorption -------------------------------------------------------------


def test_and_absorption_fires_both_orders():
    x, y = col("a") > 1, col("x") < 5
    assert ba.and_absorption(_flt(x & (x | y)), None) is not None
    assert ba.and_absorption(_flt((x | y) & x), None) is not None


def test_and_absorption_idempotent():
    x, y = col("a") > 1, col("x") < 5
    once = ba.and_absorption(_flt(x & (x | y)), None)
    assert ba.and_absorption(once, None) is None


def test_or_absorption_fires():
    x, y = col("a") > 1, col("x") < 5
    out = ba.or_absorption(_flt(x | (x & y)), None)
    assert _pred_ir(out)["op"] == "gt"


def test_absorption_does_not_fire_without_shared_term():
    x, y, z = col("a") > 1, col("x") < 5, col("a") > 9
    assert ba.and_absorption(_flt(z & (x | y)), None) is None


# --- complementation --------------------------------------------------------


def test_complement_and_total_bool_fires():
    out = ba.complement_total_bool(_flt(col("a").is_null() & ~col("a").is_null()), None)
    assert _pred_ir(out) == {"e": "lit", "value": {"bool": False}}


def test_complement_or_total_bool_fires():
    out = ba.complement_total_bool(_flt(col("a").is_not_null() | ~col("a").is_not_null()), None)
    assert _pred_ir(out) == {"e": "lit", "value": {"bool": True}}


def test_complement_idempotent():
    once = ba.complement_total_bool(_flt(col("a").is_null() & ~col("a").is_null()), None)
    assert ba.complement_total_bool(once, None) is None


def test_complement_does_not_fire_on_nullable_predicate():
    # `(a > 1)` can be null (a is null), so `x AND NOT x` is null, not FALSE.
    x = col("a") > 1
    assert ba.complement_total_bool(_flt(x & ~x), None) is None


# --- NOT over comparison ----------------------------------------------------


def test_fold_not_comparison_fires():
    out = ba.fold_not_comparison(_flt(~(col("a") < 2)), None)
    assert _pred_ir(out)["op"] == "ge"


def test_fold_not_comparison_idempotent():
    once = ba.fold_not_comparison(_flt(~(col("a") < 2)), None)
    assert ba.fold_not_comparison(once, None) is None


def test_fold_not_comparison_does_not_fire_on_non_comparison():
    assert ba.fold_not_comparison(_flt(~(col("a").is_null())), None) is None


# --- boolean equality against a literal -------------------------------------


def test_bool_eq_true_drops_literal():
    out = ba.bool_eq_literal(_flt((col("a") > 1) == True), None)  # noqa: E712
    assert _pred_ir(out)["op"] == "gt"


def test_bool_eq_false_becomes_not():
    out = ba.bool_eq_literal(_flt((col("a") > 1) == False), None)  # noqa: E712
    assert _pred_ir(out)["e"] == "not"


def test_bool_eq_idempotent():
    once = ba.bool_eq_literal(_flt((col("a") > 1) == True), None)  # noqa: E712
    assert ba.bool_eq_literal(once, None) is None


def test_bool_eq_does_not_fire_on_bare_column():
    # `f == True`: `f` is a bare Col of unknown type — the rule must not fire.
    assert ba.bool_eq_literal(_flt(col("f") == True), None) is None  # noqa: E712


# --- IN-list cleanup --------------------------------------------------------


def test_single_in_list_becomes_eq():
    out = ms.single_in_list(_flt(InList(Col("a"), (5,))), None)
    assert _pred_ir(out)["op"] == "eq"


def test_single_in_list_idempotent():
    once = ms.single_in_list(_flt(InList(Col("a"), (5,))), None)
    assert ms.single_in_list(once, None) is None


def test_dedup_in_list_fires():
    out = ms.dedup_in_list(_flt(InList(Col("a"), (5, 5, 6, 5))), None)
    assert [v["int"] for v in _pred_ir(out)["set"]] == [5, 6]


def test_dedup_in_list_does_not_fire_without_duplicates():
    assert ms.dedup_in_list(_flt(InList(Col("a"), (5, 6, 7))), None) is None


# --- COALESCE flattening ----------------------------------------------------


def test_coalesce_flattens_nested():
    node = _proj(coalesce(col("a"), coalesce(col("x"), bt.lit(0))))
    out = ms.coalesce_simplify(node, None)
    assert isinstance(out, Project)
    inputs = out.items[0].expr.to_ir()["inputs"]
    assert len(inputs) == 3 and all(i["e"] != "coalesce" for i in inputs)


def test_coalesce_dedups_repeated_argument():
    node = _proj(coalesce(col("a"), col("x"), col("a")))
    out = ms.coalesce_simplify(node, None)
    inputs = out.items[0].expr.to_ir()["inputs"]
    # The repeated `col('a')` is only ever reached when the first one was null, so
    # dropping it moves neither the value nor the type.
    assert inputs == [{"e": "col", "name": "a"}, {"e": "col", "name": "x"}]


def test_coalesce_does_not_truncate_after_literal():
    """`coalesce_simplify` must NOT truncate the tail after a non-null literal.

    A COALESCE's type is the join of its arms', so dropping the tail can *narrow* it
    (`coalesce(5, CAST(-1 AS DOUBLE))` is a DOUBLE `5.0`, not an INT `5`). That
    truncation is sound only under a type guard, and so belongs to
    `coalesce_drop_nulls_after_first_non_null`. See
    tests/differential/test_diff_kyber3_coalesce_type.py.
    """
    assert ms.coalesce_simplify(_proj(coalesce(col("a"), bt.lit(9), col("x"))), None) is None


def test_coalesce_unwraps_single():
    # Dedup collapses the arms to one, which then unwraps to the bare expression.
    node = _proj(coalesce(col("a"), col("a")))
    out = ms.coalesce_simplify(node, None)
    assert out.items[0].expr.to_ir() == {"e": "col", "name": "a"}


def test_coalesce_idempotent():
    node = _proj(coalesce(col("a"), coalesce(col("x"), bt.lit(0))))
    once = ms.coalesce_simplify(node, None)
    assert ms.coalesce_simplify(once, None) is None


def test_coalesce_does_not_fire_when_clean():
    assert ms.coalesce_simplify(_proj(coalesce(col("a"), col("x"))), None) is None


# --- empty-input sanity -----------------------------------------------------


def test_rules_apply_to_project_nodes():
    # The rules match Project as well as Filter.
    node = bt.from_pydict({"a": [1]}).select(r=(col("a") > 1) == True)._plan  # noqa: E712
    out = ba.bool_eq_literal(node, None)
    assert isinstance(out, Project) and out.items[0].expr.to_ir()["op"] == "gt"
