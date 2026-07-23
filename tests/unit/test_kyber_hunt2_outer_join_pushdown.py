"""Plan-shape guard: predicate pushdown must respect a join's null-producing side.

`kyber.rules.pushdown._push_into_join` may sink a filter conjunct below a join only
onto a side that is never null-extended: the *preserved* side. Pushing a predicate onto
the null-producing side of an outer join would drop the null-extended rows the join is
defined to keep — a silent row loss no example-based test that only checks the *value* of
surviving rows can see. These tests assert the structural outcome directly (no engine
needed), so a regression in the `can_push_left` / `can_push_right` logic fails fast.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, kyber
from batcher.plan.logical import Filter, Join
from batcher.plan.visitor import walk


def _optimized(ds: bt.Dataset):
    # The *logical* rewrite keeps the operator tree (a physical plan fuses filters into
    # scans, hiding the structure this test inspects).
    return kyber.optimize_logical(ds._plan, sources=ds._sources)


def _filters_below_joins(plan) -> list[tuple[str, str]]:
    """(join_type, 'left'|'right') for every Filter sitting directly on a join input."""
    found: list[tuple[str, str]] = []
    for node in walk(plan):
        if isinstance(node, Join):
            if isinstance(node.left, Filter):
                found.append((node.join_type, "left"))
            if isinstance(node.right, Filter):
                found.append((node.join_type, "right"))
    return found


def _has_filter_on_join(plan) -> bool:
    return any(isinstance(n, Filter) and isinstance(n.input, Join) for n in walk(plan))


def _mk(how: str, *, side: str):
    left = bt.from_pydict({"k": [1, 2, 3], "a": [10, 20, 30]})
    right = bt.from_pydict({"k": [1, 2], "b": [100, 200]})
    ds = left.join(right, on="k", how=how, suffix="_r")
    return ds.filter(col("a" if side == "left" else "b") > 0)


def test_left_join_filter_on_preserved_side_pushes_down():
    # LEFT join: the left side is preserved, so `a > 0` is safe to push below the join.
    opt = _optimized(_mk("left", side="left"))
    assert ("left", "left") in _filters_below_joins(opt)


def test_left_join_filter_on_null_side_is_not_pushed():
    # LEFT join: the right side is null-extended. `b > 0` must NOT sink onto it (that would
    # convert the LEFT join into an inner join and drop unmatched left rows).
    opt = _optimized(_mk("left", side="right"))
    assert ("left", "right") not in _filters_below_joins(opt)


def test_right_join_filter_on_preserved_side_pushes_down():
    # RIGHT join: the right side is preserved, so `b > 0` is safe to push below.
    opt = _optimized(_mk("right", side="right"))
    assert ("right", "right") in _filters_below_joins(opt)


def test_right_join_filter_on_null_side_is_not_pushed():
    # RIGHT join: the left side is null-extended. `a > 0` must NOT sink onto it.
    opt = _optimized(_mk("right", side="left"))
    assert ("right", "left") not in _filters_below_joins(opt)


def test_full_outer_join_filter_never_pushes_to_a_side():
    # FULL join: BOTH sides are null-producing, so neither may receive a pushed predicate.
    for side in ("left", "right"):
        opt = _optimized(_mk("full", side=side))
        pushed = _filters_below_joins(opt)
        assert ("full", "left") not in pushed and ("full", "right") not in pushed


def test_inner_join_filter_pushes_to_the_referenced_side():
    # Sanity: on an inner join (no null extension) the predicate is free to push down.
    opt_left = _optimized(_mk("inner", side="left"))
    opt_right = _optimized(_mk("inner", side="right"))
    assert ("inner", "left") in _filters_below_joins(opt_left)
    assert ("inner", "right") in _filters_below_joins(opt_right)
