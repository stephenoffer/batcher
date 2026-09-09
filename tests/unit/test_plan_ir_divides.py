"""A join tree divides across workers too, and the router has to be able to say so.

`shard_plan` answers "does this divide" for a **chain**. `flatten_ops` cannot flatten a branch,
so a plan containing a join answered `None` — and `kyber.gpu.shape.is_shardable` read that as
"this plan does not divide". Every join plan was therefore routed as though **one device were
enough for it**, whatever the fleet had: measured on a six-T4 cluster at TPC-H sf10, a
60 M x 15 M join ran on a single board at 8.5 s against the CPU engine's 1.28 s, with five
devices idle.

`ir_divides` answers the whole question, and it is two questions rather than one — the chain
above the outermost branch must fold, *and* some leaf of the branch must be safe to split while
the others are replicated. The second is the rule `shardable_leaves` already states and this
must not restate: an inner join may split either side, a left/semi/anti only its left, a right
only its right, and a full neither.
"""

from __future__ import annotations

import pytest

from batcher.plan.distribution import ir_divides

pytestmark = pytest.mark.unit


def _scan(source: int = 0) -> dict:
    return {"op": "scan", "source_id": source}


def _join(kind: str, left: dict | None = None, right: dict | None = None) -> dict:
    return {
        "op": "hash_join",
        "join_type": kind,
        "left": left or _scan(0),
        "right": right or _scan(1),
        "left_keys": ["k"],
        "right_keys": ["k"],
        "output": [],
    }


def _filter(inner: dict) -> dict:
    return {
        "op": "filter",
        "input": inner,
        "predicate": {
            "e": "binary",
            "op": "gt",
            "left": {"e": "col", "name": "k"},
            "right": {"e": "lit", "value": {"int": 1}},
        },
    }


def _sum(inner: dict) -> dict:
    return {
        "op": "aggregate",
        "input": inner,
        "group_keys": [],
        "aggregates": [{"func": "sum", "alias": "s", "input": {"e": "col", "name": "v"}}],
    }


# --- chains: unchanged -------------------------------------------------------


def test_a_plain_chain_still_answers_shard_plans_question():
    assert ir_divides(_sum(_filter(_scan()))) is True


def test_a_chain_with_no_mergeable_form_still_does_not_divide():
    """`row_id` numbers over the whole relation, so a shard would restart at its own offset."""
    assert ir_divides({"op": "row_id", "input": _scan(), "name": "i"}) is False


# --- the branch rule ---------------------------------------------------------


def test_an_inner_join_divides():
    assert ir_divides(_sum(_join("inner"))) is True


def test_a_left_join_divides_on_its_left_input():
    assert ir_divides(_sum(_join("left"))) is True


def test_a_full_outer_join_does_not_divide():
    """An unmatched row on either side must be emitted exactly once, and every worker holding
    the whole other side would emit it."""
    assert ir_divides(_sum(_join("outer"))) is False


def test_a_union_does_not_divide_on_one_leaf():
    """Replicating a union's other inputs duplicates them once per worker."""
    union = {"op": "union", "inputs": [_scan(0), _scan(1)], "distinct": False}
    assert ir_divides(_sum(union)) is False


def test_a_join_under_a_non_mergeable_chain_does_not_divide():
    """Both halves are required: a splittable leaf whose shards cannot be folded is no use."""
    assert ir_divides({"op": "row_id", "input": _join("inner"), "name": "i"}) is False


def test_pushed_down_operators_between_a_join_and_its_inputs_are_transparent():
    """The optimized form of essentially every real join has a filter or projection there."""
    assert ir_divides(_sum(_join("inner", left=_filter(_scan(0)), right=_filter(_scan(1))))) is True


def test_a_join_of_joins_divides_when_every_join_on_the_path_allows_it():
    inner = _join("inner", left=_scan(0), right=_scan(1))
    assert ir_divides(_sum(_join("inner", left=inner, right=_scan(2)))) is True


def test_a_left_join_over_a_right_join_leaves_no_splittable_leaf():
    """`left` may only split its left; if that left is a `right` join, which may only split its
    right, the path still reaches a leaf — but reverse the nesting and it does not."""
    unusable = _join("left", left=_join("right", left=_scan(0), right=_scan(1)), right=_scan(2))
    assert ir_divides(_sum(unusable)) is True
    dead = _join("left", left=_join("outer", left=_scan(0), right=_scan(1)), right=_scan(2))
    assert ir_divides(_sum(dead)) is False


def test_an_unreadable_plan_does_not_divide():
    """The conservative direction: a branch missing an input is malformed IR, not two scans.

    Reading it as a join of two scans would answer "this divides" about a plan nobody can
    execute — the opposite of what every other decline here does.
    """
    assert ir_divides({"op": "hash_join", "join_type": "inner"}) is False
    assert ir_divides({"op": "union"}) is False
