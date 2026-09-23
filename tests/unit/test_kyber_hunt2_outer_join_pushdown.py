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


def _pushed_predicate_columns(plan) -> list[tuple[str, frozenset[str]]]:
    """(join_type, columns the predicate reads) for every Filter sitting on a join input.

    Side-label-free, and that is the point. An **inner** join is commutative, so Kyber is
    free to swap its inputs when it picks a build side -- and it does, choosing `broadcast`
    with the smaller relation. A filter that correctly sank onto the relation holding `a`
    then reports as being on the `right`, and an assertion phrased as ``("inner", "left")``
    fails on a plan that is right. The outer-join assertions below keep using the labels,
    because there the side is not a label: a LEFT join cannot commute without becoming a
    RIGHT join, so "preserved side" and "left" are the same claim.
    """
    from batcher.plan.expr_ir.walk import referenced_columns

    found: list[tuple[str, frozenset[str]]] = []
    for node in walk(plan):
        if not isinstance(node, Join):
            continue
        for child in (node.left, node.right):
            if isinstance(child, Filter):
                found.append((node.join_type, referenced_columns(child.predicate)))
    return found


def _join_types(plan) -> list[str]:
    """The `join_type` of every Join left in the plan, after optimization.

    Kyber may *change* a join's type -- a FULL join under a null-rejecting predicate is a
    LEFT join -- so a plan-shape assertion keyed on the type the query was written with can
    stop matching anything at all. Reading the types back is what keeps such an assertion
    honest.
    """
    return [n.join_type for n in walk(plan) if isinstance(n, Join)]


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
    """FULL join: both sides are null-producing, so neither may receive a pushed predicate.

    The predicate has to be one that keeps the join FULL, and `a > 0` is not: it is
    null-rejecting, so a row present only on the right (where `a` is NULL) cannot survive
    it, and the join narrows to LEFT before pushdown is even asked. `_join_types` below
    pins that, and `test_a_null_rejecting_predicate_narrows_the_full_join` covers the
    other half.

    That is not a detail. The assertion here used to be
    ``("full", "left") not in pushed``, against a plan whose join had already become LEFT
    -- so the tuple's first element could never be `"full"` and the test could not fail.
    It read as the strictest check in the file and was the only one asserting nothing.
    """
    for column in ("a", "b"):
        # `x IS NULL OR x > 0` admits a null-extended row, so the join stays FULL.
        left = bt.from_pydict({"k": [1, 2, 3], "a": [10, 20, 30]})
        right = bt.from_pydict({"k": [1, 2], "b": [100, 200]})
        joined = left.join(right, on="k", how="full", suffix="_r")
        opt = _optimized(joined.filter(col(column).is_null() | (col(column) > 0)))

        assert _join_types(opt) == ["full"], f"the join did not stay FULL: {_join_types(opt)}"
        assert _filters_below_joins(opt) == [], _filters_below_joins(opt)
        assert _has_filter_on_join(opt), "the predicate vanished rather than staying above"


def test_a_null_rejecting_predicate_narrows_the_full_join():
    """The rewrite that made the assertion above vacuous, asserted on purpose instead.

    `full join ... where a > 0` drops every row that exists only on the right, because `a`
    is NULL there and `NULL > 0` is NULL. So the FULL join is a LEFT join, and `a > 0` may
    then sink onto the left -- which is now the *preserved* side, so this is the same rule
    the tests above check rather than an exception to it.
    """
    for side, column, narrowed in (("left", "a", "left"), ("right", "b", "right")):
        opt = _optimized(_mk("full", side=side))
        assert _join_types(opt) == [narrowed], _join_types(opt)
        assert (narrowed, narrowed) in _filters_below_joins(opt), _filters_below_joins(opt)
        assert any(column in columns for _t, columns in _pushed_predicate_columns(opt))


def test_inner_join_filter_pushes_to_the_referenced_side():
    # Sanity: on an inner join (no null extension) the predicate is free to push down.
    #
    # Asserted by the column the pushed predicate reads, not by which input it landed on.
    # `a` lives on the relation written first and `b` on the second, and an inner join may
    # exchange its inputs freely when Kyber picks a build side -- which it does here,
    # choosing a broadcast. What must hold is that the predicate sank onto the relation
    # that *has* the column, and that is what this says.
    for side, column in (("left", "a"), ("right", "b")):
        pushed = _pushed_predicate_columns(_optimized(_mk("inner", side=side)))
        assert any(join_type == "inner" and column in columns for join_type, columns in pushed), (
            f"filter on {column!r} did not sink below the inner join: {pushed}"
        )


def test_the_inner_join_assertion_still_distinguishes_a_predicate_that_stayed_up():
    """Positive control for the test above, which asserts a *presence*.

    Reading the pushed predicate's columns rather than its side makes the assertion easier
    to satisfy, so this shows it is not satisfied by everything. The case that holds the
    predicate above the join is a FULL join the predicate cannot narrow -- see
    `test_full_outer_join_filter_never_pushes_to_a_side` for why `a > 0` is not that case.
    """
    for column in ("a", "b"):
        left = bt.from_pydict({"k": [1, 2, 3], "a": [10, 20, 30]})
        right = bt.from_pydict({"k": [1, 2], "b": [100, 200]})
        joined = left.join(right, on="k", how="full", suffix="_r")
        opt = _optimized(joined.filter(col(column).is_null() | (col(column) > 0)))
        pushed = _pushed_predicate_columns(opt)
        assert not any(column in columns for _join_type, columns in pushed), pushed
