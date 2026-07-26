"""Plan-shape tests for `or_equalities_to_in_list` (DuckDB's `contains_to_in_clause`).

Correctness against DuckDB lives in
`tests/differential/test_diff_kyber_or_to_in_list.py`; here we assert the rule fires on
the shape it claims, stays a no-op on the shapes it must not touch, leaves the optimizer
idempotent — and, the reason it was written, that the eight existing `InList` rules can
now see the predicate.
"""

from __future__ import annotations

import json

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer

DS = bt.from_pydict(
    {"a": [1, 2, 3, None, 5], "b": [1, 1, 1, 1, 1], "s": ["x", "y", "z", None, "x"]}
)


def _ir(ds) -> dict:
    return Optimizer().optimize(ds._plan).ir


def _has_in_list(ds) -> bool:
    return '"in_list"' in json.dumps(_ir(ds))


def test_an_or_chain_of_equalities_becomes_an_in_list():
    assert _has_in_list(DS.filter((col("a") == 1) | (col("a") == 2) | (col("a") == 3)))
    assert _has_in_list(DS.filter((col("s") == "x") | (col("s") == "y")))


def test_a_single_distinct_value_stays_an_equality():
    """One value is an equality, not a membership test — wrapping it in a hash set
    would be slower and would hide it from the equality rules."""
    assert not _has_in_list(DS.filter((col("a") == 1) | (col("a") == 1)))


def test_two_different_columns_are_not_folded():
    """`a = 1 OR s = 'y'` is not a membership test on either column."""
    assert not _has_in_list(DS.filter((col("a") == 1) | (col("s") == "y")))


def test_a_non_equality_disjunct_blocks_the_fold():
    assert not _has_in_list(DS.filter((col("a") == 1) | (col("a") > 2)))


def test_the_optimizer_is_idempotent_on_the_result():
    """Re-applying the rule to its own output must report no change.

    A rule that keeps returning a new-but-equal node never lets the driver's fixpoint
    terminate. The dedup keeps first-occurrence order for the same reason: a reordering
    rewrite would produce a different-but-equal list on every pass.
    """
    from batcher.kyber.rules.normalize.ranges import or_equalities_to_in_list

    node = DS.filter((col("a") == 1) | (col("a") == 2) | (col("a") == 3))._plan
    once = or_equalities_to_in_list(node, None)
    assert once is not None, "the rule must fire on this shape"
    # `None` means "nothing changed", which is what terminates the driver's fixpoint.
    assert or_equalities_to_in_list(once, None) is None


def test_the_range_bounds_rule_still_fires_alongside():
    """`or_to_in_and_range` derives `min <= c <= max` from the same shape and is not
    superseded — zone maps use the bounds directly, and both survive."""
    plan = json.dumps(_ir(DS.filter((col("a") == 1) | (col("a") == 2) | (col("a") == 3))))
    assert '"in_list"' in plan
    assert '"ge"' in plan and '"le"' in plan


def test_the_fold_unlocks_the_in_list_pruning_family():
    """The point of the rule: predicates the `InList` rules can now match.

    `refine_in_list_by_comparison` narrows an `IN` list against a range predicate on the
    same column. On the `OR` form it never fired, because there was no `InList` to
    narrow. Here the `a < 3` conjunct must remove `5` from the membership set.
    """
    ds = DS.filter(((col("a") == 1) | (col("a") == 2) | (col("a") == 5)) & (col("a") < 3))
    plan = json.dumps(_ir(ds))
    assert '"in_list"' in plan
    # 5 cannot satisfy `a < 3`, so a refinement rule that can see the list drops it.
    assert '{"int": 5}' not in plan
