"""Plan-shape, idempotence, and negative tests for the `join_extra` rules."""

from __future__ import annotations

import dataclasses

import batcher as bt
import batcher.kyber.rules.extra.join_extra as je
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.logical import Join, Limit, Project

RULE_NAMES = {
    "semi_anti_join_empty_left",
    "semi_join_empty_right",
    "anti_join_empty_right",
    "dedup_join_keys",
    "inner_join_empty_to_empty",
    "preserving_join_empty_null_side",
}


def _a():
    return bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30]})


def _b():
    return bt.from_pydict({"k": [1, 2], "w": [1, 2]})


def _single_side(join: Join, side: str) -> Join:
    """`join` with its output pruned to one side (as column pruning would produce)."""
    return dataclasses.replace(join, output=tuple(o for o in join.output if o.side == side))


def _apply(name: str, plan):
    """Run the named registered rule over `plan` via its whole-plan wrapper."""
    rule = {r.name: r for r in DEFAULT_REGISTRY.rules()}[name]
    return rule.fn(plan, None)


def _idempotent(name: str, plan) -> None:
    once = _apply(name, plan)
    assert once.to_ir() != plan.to_ir(), f"{name} did not fire"
    twice = _apply(name, once)
    assert twice.to_ir() == once.to_ir(), f"{name} is not idempotent"


# --- registration ------------------------------------------------------------


def test_rules_registered():
    registered = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert registered >= RULE_NAMES


# --- semi_anti_join_empty_left ----------------------------------------------


def test_semi_empty_left_fires():
    for how in ("semi", "anti"):
        plan = _a().limit(0).join(_b(), on="k", how=how)._plan
        out = je.semi_anti_join_empty_left(plan, None)
        assert isinstance(out, Project)
        assert [i.alias for i in out.items] == ["k", "v"]
    _idempotent("semi_anti_join_empty_left", _a().limit(0).join(_b(), on="k", how="semi")._plan)


def test_semi_empty_left_negative():
    # Non-empty left → no fire.
    assert je.semi_anti_join_empty_left(_a().join(_b(), on="k", how="semi")._plan, None) is None
    # Inner join (two-sided output) is not a semi/anti join → no fire.
    assert (
        je.semi_anti_join_empty_left(_a().limit(0).join(_b(), on="k", how="inner")._plan, None)
        is None
    )


# --- semi_join_empty_right ---------------------------------------------------


def test_semi_empty_right_fires():
    plan = _a().join(_b().limit(0), on="k", how="semi")._plan
    out = je.semi_join_empty_right(plan, None)
    assert isinstance(out, Limit) and out.n == 0
    assert isinstance(out.input, Project)
    _idempotent("semi_join_empty_right", plan)


def test_semi_empty_right_negative():
    # Non-empty right → no fire; anti (not semi) → no fire here.
    assert je.semi_join_empty_right(_a().join(_b(), on="k", how="semi")._plan, None) is None
    assert (
        je.semi_join_empty_right(_a().join(_b().limit(0), on="k", how="anti")._plan, None) is None
    )


# --- anti_join_empty_right ---------------------------------------------------


def test_anti_empty_right_fires():
    plan = _a().join(_b().limit(0), on="k", how="anti")._plan
    out = je.anti_join_empty_right(plan, None)
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["k", "v"]
    _idempotent("anti_join_empty_right", plan)


def test_anti_empty_right_negative():
    # Non-empty right → no fire; semi (not anti) → no fire here.
    assert je.anti_join_empty_right(_a().join(_b(), on="k", how="anti")._plan, None) is None
    assert (
        je.anti_join_empty_right(_a().join(_b().limit(0), on="k", how="semi")._plan, None) is None
    )


# --- dedup_join_keys ---------------------------------------------------------


def _dup_key_join() -> Join:
    j = _a().join(_b(), on="k", how="inner")._plan
    return dataclasses.replace(
        j, left_keys=j.left_keys + j.left_keys, right_keys=j.right_keys + j.right_keys
    )


def test_dedup_keys_fires():
    out = je.dedup_join_keys(_dup_key_join(), None)
    assert isinstance(out, Join)
    assert out.left_keys == ("k",) and out.right_keys == ("k",)
    # idempotent: a second pass finds no duplicate.
    assert je.dedup_join_keys(out, None) is None


def test_dedup_keys_negative():
    # A single-key join has nothing to dedup.
    assert je.dedup_join_keys(_a().join(_b(), on="k", how="inner")._plan, None) is None
    # Distinct pairs (a,c) and (b,c) express different constraints → keep both.
    left = bt.from_pydict({"a": [1], "b": [1]})
    right = bt.from_pydict({"c": [1], "d": [1]})
    j = left.join(right, left_on=["a", "b"], right_on=["c", "d"], how="inner")._plan
    two_key = dataclasses.replace(j, left_keys=("a", "b"), right_keys=("c", "c"))
    assert je.dedup_join_keys(two_key, None) is None


# --- inner_join_empty_to_empty ----------------------------------------------


def test_inner_empty_fires_single_sided():
    j = _single_side(_a().join(_b().limit(0), on="k", how="inner")._plan, "left")
    out = je.inner_join_empty_to_empty(j, None)
    assert isinstance(out, Limit) and out.n == 0
    assert isinstance(out.input, Project)
    _idempotent("inner_join_empty_to_empty", j)


def test_inner_empty_negative():
    # No empty side → no fire.
    j_full = _single_side(_a().join(_b(), on="k", how="inner")._plan, "left")
    assert je.inner_join_empty_to_empty(j_full, None) is None
    # Empty but two-sided output → cannot fabricate the schema → no fire.
    mixed = _a().join(_b().limit(0), on="k", how="inner")._plan
    assert je.inner_join_empty_to_empty(mixed, None) is None
    # Not an inner join → no fire.
    left = _single_side(_a().join(_b().limit(0), on="k", how="left")._plan, "left")
    assert je.inner_join_empty_to_empty(left, None) is None


# --- preserving_join_empty_null_side ----------------------------------------


def test_preserving_empty_null_side_fires():
    j = _single_side(_a().join(_b().limit(0), on="k", how="left")._plan, "left")
    out = je.preserving_join_empty_null_side(j, None)
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["k", "v"]
    _idempotent("preserving_join_empty_null_side", j)


def test_preserving_empty_null_side_negative():
    # The PRESERVED side is the empty one → result is empty, not a passthrough → no fire.
    j = _single_side(_a().limit(0).join(_b(), on="k", how="left")._plan, "left")
    assert je.preserving_join_empty_null_side(j, None) is None
    # Non-empty null side → no fire.
    j2 = _single_side(_a().join(_b(), on="k", how="left")._plan, "left")
    assert je.preserving_join_empty_null_side(j2, None) is None
    # Inner join is not preserving → handled by the other rule, not this one.
    j3 = _single_side(_a().join(_b().limit(0), on="k", how="inner")._plan, "left")
    assert je.preserving_join_empty_null_side(j3, None) is None
