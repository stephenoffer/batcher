"""Ordering rewrites — drop work that the input's known order already provides.

`sort_elimination_from_ordering` removes a `Sort` whose ordering the input already
satisfies. The estimator propagates a `sorted_by` ordering through order-preserving
operators (a source that declares its sort, a `Sort` below, then `Filter`/`Limit`/
`Window` on top all carry it), and this rule consumes it: if the requested sort
keys are a prefix of that known ordering, the sort is redundant and the input flows
through unchanged. The classic win is re-sorting an already-sorted stream
(time-series / pre-sorted lakehouse data) or sorting again by a coarser key.

`RelStats.sorted_by` carries each key's direction and null placement, so the rule
fires for a descending ordering exactly as readily as an ascending one — which is
the common case in practice, since ``ORDER BY ts DESC`` is how nearly every
recent-first query is spelled. Null placement is compared exactly unless the column
is *proven* free of nulls, where the two spellings describe the same row order.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Col
from batcher.plan.logical import LogicalPlan, Sort
from batcher.plan.stats import SortOrder, orderings_satisfy

__all__ = ["requested_ordering", "sort_elimination_from_ordering"]

# NOTE: a `Sort(Sort(x, k1), k2) → Sort(x, k2)` "drop the redundant inner sort" rule looks
# safe but is NOT: Batcher's single-key sort over common types takes a **stable** specialized
# path (docs/architecture/deep-dives/operators/sort-internals.md; test_diff_exec_sort_stability.py),
# so `k1` is a
# real tiebreaker within `k2`'s ties. Dropping the inner sort changes that tie order — a wrong
# result on the payload columns. The "arrow lexsort is unstable" note only describes the
# multi-key/uncommon-type *fallback*, not the common path. Do not add that rule.


def requested_ordering(node: Sort) -> tuple[SortOrder, ...] | None:
    """The ordering `node` asks for, as `SortOrder` keys, or None if it is not nameable.

    A sort over a *computed* key orders the relation by a value no column holds, so no
    delivered ordering could match it and there is nothing to compare against. Every key
    must be a bare column for the answer to mean anything, so one that is not makes the
    whole request unanswerable rather than truncating it — a truncated prefix would
    describe a weaker sort than the one actually requested.

    Args:
        node: The sort whose requested ordering is wanted.

    Returns:
        The requested ordering, or None when any key is not a bare column.
    """
    keys: list[SortOrder] = []
    for key in node.keys:
        if not isinstance(key.expr, Col):
            return None
        keys.append(SortOrder(key.expr.name, bool(key.descending), bool(key.nulls_first)))
    return tuple(keys) or None


@rule(name="sort_elimination_from_ordering", phase=Phase.REWRITE, matches=(Sort,))
def sort_elimination_from_ordering(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sort(x, keys)` → `x` when `x` is already ordered by `keys`.

    Fires for a plain (no `limit`) sort over bare columns whose keys are a prefix of the
    input's known ordering, direction included. Returns None otherwise (a top-N sort, an
    expression key, or an input whose order is unknown or insufficient).
    """
    if node.limit is not None:
        return None
    if not _input_already_ordered_as_requested(node, ctx):
        return None
    return node.input


# REMOVED: `topn_over_sorted_input_to_limit`, which rewrote `Sort(x, keys, limit=n)` ->
# `Limit(x, n)` when `x` already delivered `keys`. The rewrite is sound; the rule was dead.
#
# Nothing ever handed it a `Sort` carrying a `limit`. `Sort.limit` is set only by `fuse_topn`
# (FUSION, phase 5) and the `limit_extra` rules, every one of which runs *after* REWRITE
# (phase 2), so within one optimize pass a phase-2 rule cannot see a top-N that phase 5 has
# not built yet. `ds.sort(...).limit(n)` and `ORDER BY ... LIMIT n` both reach REWRITE as
# `Limit(Sort(...))` -- a plain sort with a `Limit` above it -- which
# `sort_elimination_from_ordering` below already removes, leaving exactly the `limit | scan`
# plan the top-N rule was written to produce. Instrumenting both rules on both paths measured
# it: `sort_elimination_from_ordering` fired once, the top-N rule zero times.
#
# It looked tested because its unit tests built `Sort(limit=n)` by hand and called the
# function directly -- a shape the optimizer never produces at that phase, and the exact
# failure `registry.py` describes, where a rule that never runs keeps a green test.
#
# If a future phase does produce a top-N before REWRITE, the rewrite is in git history and is
# correct; it also survives distribution, because `dist` routes a limit over a splittable
# source through `_distributed_map(..., preserve_order=True)` and so returns the source's own
# first `n` rows.


def _input_already_ordered_as_requested(node: Sort, ctx: OptimizerContext) -> bool:
    """Whether `node`'s input already delivers the ordering `node` asks for.

    Shared by the two rules above so they cannot drift apart on what "already ordered"
    means — the plain-sort and top-N cases differ only in what they build afterwards.
    """
    requested = requested_ordering(node)
    if requested is None:
        return False
    stats = ctx.estimator.estimate(node.input)
    return orderings_satisfy(stats.sorted_by, requested, non_nullable=stats.non_null_columns())
