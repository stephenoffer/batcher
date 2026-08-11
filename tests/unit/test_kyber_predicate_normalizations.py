"""Plan-shape tests for the four `normalize/predicates.py` rewrites.

Correctness lives in `tests/differential/test_diff_kyber_predicate_normalizations.py`.
What these assert is the *reason* each rule exists, which the result alone cannot show:
a shape the rest of Kyber can act on. A predicate that returns the right rows through a
`CASE`, a self-comparison or a redundant `IN` pair is exactly the failure mode here —
correct and unprunable.

Three of the four are null-sensitive and are therefore restricted to a `Filter`, where
NULL and FALSE are indistinguishable. The negative tests below are the load-bearing
ones: they pin that the rules decline the projection context, because a rule that fired
there would still pass every filter test while returning a wrong value.
"""

from __future__ import annotations

import json

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import Optimizer

DS = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6], "g": ["p", "q", "p"]})


def _plan(ds) -> str:
    return json.dumps(Optimizer().optimize(ds._plan).ir)


# --- self-comparison --------------------------------------------------------


def test_self_equality_becomes_a_null_check():
    plan = _plan(DS.filter(col("a") == col("a")))
    assert '"is_not_null"' in plan
    assert '"op": "eq"' not in plan


def test_the_non_strict_orderings_reduce_the_same_way():
    for build in (lambda: col("a") <= col("a"), lambda: col("a") >= col("a")):
        assert '"is_not_null"' in _plan(DS.filter(build()))


def test_a_comparison_of_two_different_columns_is_left_alone():
    plan = _plan(DS.filter(col("a") == col("b")))
    assert '"op": "eq"' in plan
    assert '"is_not_null"' not in plan


def test_a_self_comparison_in_a_projection_is_NOT_rewritten():
    """`a = a` on a null row is NULL; `a IS NOT NULL` is FALSE. A filter erases that
    difference and a projection does not, so the rule must decline here."""
    plan = _plan(DS.select(r=col("a") == col("a")))
    assert '"op": "eq"' in plan
    assert '"is_not_null"' not in plan


# --- boolean CASE -----------------------------------------------------------


def test_a_boolean_case_under_a_filter_becomes_its_condition():
    plan = _plan(DS.filter(bt.when(col("a") > lit(1)).then(lit(True)).otherwise(lit(False))))
    assert '"case"' not in plan
    assert '"op": "gt"' in plan


def test_the_swapped_boolean_case_is_NOT_rewritten():
    """`CASE WHEN c THEN false ELSE true END` is `c IS NOT TRUE`, which **keeps** a null
    row; `NOT c` is NULL and drops it. The rewrite would be wrong even under a filter,
    which is the asymmetry that makes this rule one-directional."""
    plan = _plan(DS.filter(bt.when(col("a") > lit(1)).then(lit(False)).otherwise(lit(True))))
    assert '"case"' in plan


def test_a_boolean_case_in_a_projection_is_NOT_rewritten():
    """The load-bearing negative test. A `CASE` sends a NULL condition to `ELSE`, so it
    answers `false` where the bare condition answers NULL. Under a filter that is
    invisible; as a projected value it is a wrong answer."""
    plan = _plan(DS.select(r=bt.when(col("a") > lit(1)).then(lit(True)).otherwise(lit(False))))
    assert '"case"' in plan


def test_a_case_whose_branches_are_the_same_literal_is_left_to_folding():
    plan = _plan(DS.filter(bt.when(col("a") > lit(1)).then(lit(True)).otherwise(lit(True))))
    assert '"op": "gt"' not in plan  # constant-folded away, not turned into the condition


def test_a_case_with_non_boolean_branches_is_untouched():
    plan = _plan(DS.select(r=bt.when(col("a") > lit(1)).then(lit(10)).otherwise(lit(20))))
    assert '"case"' in plan


# --- IN-list intersection ---------------------------------------------------


def test_two_in_lists_on_one_column_are_intersected():
    """`a IN (1,2,3) AND a IN (2,3,4)` keeps only `{2, 3}`. The range rules then narrow
    the two-element list further, so the assertion is that the excluded values are gone
    rather than that an `in_list` node survives."""
    plan = _plan(DS.filter(col("a").is_in([1, 2, 3]) & col("a").is_in([2, 3, 4])))
    assert '{"int": 1}' not in plan and '{"int": 4}' not in plan


def test_in_lists_on_different_columns_are_left_alone():
    plan = _plan(DS.filter(col("a").is_in([1, 2]) & col("b").is_in([2, 3])))
    assert plan.count('"in_list"') == 2 or '{"int": 1}' in plan


def test_a_disjoint_pair_becomes_the_empty_relation():
    """An empty intersection under a filter keeps no row, so the scan is dropped entirely.

    This asserted the *opposite* until the duplicate `intersect_in_lists` was removed. Two
    rules shared that name; `RuleRegistry.add` kept whichever registered first, and this
    test reached past the registry into the loser's private helper — so it described a
    decline that the shipped optimizer never made. Both halves of the mistake are now
    closed: the dead copy is gone, and `test_no_two_rules_share_a_name` fails on a repeat.

    The fold is sound because a `Filter` cannot tell NULL from FALSE. `a IN (1,2) AND a IN
    (3,4)` is FALSE for a present `a` and NULL for a null one, and both drop the row.
    """
    # Two values per list, because a one-value `is_in` is built as a plain equality and
    # never becomes an `InList` for the rule to see.
    plan = _plan(DS.filter(col("a").is_in([1, 2]) & col("a").is_in([3, 4])))
    assert '"in_list"' not in plan
    assert '"limit"' in plan and '"n": 0' in plan


# --- constant group key -----------------------------------------------------


def test_a_constant_group_key_is_removed_from_the_aggregate():
    plan = _plan(DS.with_columns(k=lit(1)).group_by("g", "k").agg(n=col("a").sum()))
    keys = plan[plan.index('"group_keys"') :]
    assert '"name": "g"' in keys[:120]
    assert '"name": "k"' not in keys[:120], "the constant key must not reach the hash table"


def test_the_removed_key_still_appears_in_the_output():
    """Dropping a column would be a wrong answer, not a faster plan."""
    out = DS.with_columns(k=lit(1)).group_by("g", "k").agg(n=col("a").sum())
    assert out.collect().schema.names == ["g", "k", "n"]


def test_an_all_constant_grouping_is_left_alone():
    """`GROUP BY 1` over an empty input yields no rows; a global aggregate yields one.
    The difference is visible, so the rule declines rather than guessing."""
    plan = _plan(DS.with_columns(k=lit(1)).group_by("k").agg(n=col("a").sum()))
    keys = plan[plan.index('"group_keys"') :]
    assert '"name": "k"' in keys[:120]
