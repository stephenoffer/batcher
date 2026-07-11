"""The shared plan-tree traversal (`plan.visitor`).

`transform_up` is the hot path of every Kyber rewrite, so it is fused into a single
pass over each node's child fields (recurse + rebuild in one traversal) rather than a
`children()` scan plus a `with_children()` re-scan. These pin the contract that fusion
must preserve exactly: post-order rewrite, structural sharing (an unchanged subtree
keeps its identity), and correct handling of both single-plan and tuple-of-plan child
fields.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.plan.visitor import children, transform_up, walk, with_children

pytestmark = pytest.mark.unit


def _union_plan():
    """A Union (tuple children) over two filtered scans (single child + a Scan leaf)."""
    a = bt.from_pydict({"x": [1, 2]})
    b = bt.from_pydict({"x": [3, 4]})
    return a.filter(col("x") > 0).union(b.filter(col("x") < 9))._plan


def test_noop_transform_preserves_identity():
    plan = _union_plan()
    assert transform_up(plan, lambda n: n) is plan  # structural sharing → O(1) fixpoint


def test_transform_up_is_post_order():
    plan = _union_plan()
    order: list[str] = []

    def visit(n):
        order.append(type(n).__name__)
        return n

    transform_up(plan, visit)
    # Every child appears before its parent (children finalized before the node).
    assert order[-1] == "Union"
    assert order.index("Scan") < order.index("Filter") < order.index("Union")


def test_transform_up_matches_children_plus_with_children():
    # The fused transform_up must be observationally identical to the old
    # children()/with_children() composition for any rewrite.
    plan = _union_plan()

    def reference(node, fn):
        rebuilt = with_children(node, [reference(c, fn) for c in children(node)])
        return fn(rebuilt)

    seen = {"n": 0}

    def rule(n):
        seen["n"] += 1
        return n

    fused = transform_up(plan, rule)
    n_fused = seen["n"]
    seen["n"] = 0
    ref = reference(plan, rule)
    # Same nodes visited, and both share structure back to the original (no-op rule).
    assert seen["n"] == n_fused
    assert fused is plan and ref is plan


def test_tuple_children_replaced_positionally():
    plan = _union_plan()  # Union.inputs is a tuple of two Filters
    kids = children(plan)
    assert len(kids) == 2

    # Rewrite that swaps a Filter's whole branch to a sentinel scan on the RIGHT only.
    right = kids[1]
    replacement = bt.from_pydict({"x": [99]})._plan

    def rule(n):
        return replacement if n is right else n

    out = transform_up(plan, rule)
    new_kids = children(out)
    assert new_kids[0] is kids[0]  # left branch identity preserved
    assert new_kids[1] is replacement  # right tuple slot replaced in place
    assert [type(n).__name__ for n in walk(out)].count("Scan") >= 1
