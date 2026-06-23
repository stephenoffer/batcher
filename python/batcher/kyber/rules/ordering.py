"""Ordering rewrites — drop work that the input's known order already provides.

`sort_elimination_from_ordering` removes a `Sort` whose ordering the input already
satisfies. The estimator propagates a `sorted_by` ordering through order-preserving
operators (a source that declares its sort, a `Sort` below, then `Filter`/`Limit`/
`Window` on top all carry it), and this rule consumes it: if the requested sort
keys are a prefix of that known ordering, the sort is redundant and the input flows
through unchanged. The classic win is re-sorting an already-sorted stream
(time-series / pre-sorted lakehouse data) or sorting again by a coarser key.

`RelStats.sorted_by` is, by contract, a *canonical* ascending/nulls-last column
ordering, so the rule fires only for a matching ascending/nulls-last `Sort` — a
descending or nulls-first sort, or one with a top-N `limit`, is left untouched.
Correctness over an extra rewrite.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Col
from batcher.plan.logical import LogicalPlan, Sort

__all__ = ["sort_elimination_from_ordering"]


@rule(name="sort_elimination_from_ordering", phase=Phase.REWRITE, matches=(Sort,))
def sort_elimination_from_ordering(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Sort(x, keys)` → `x` when `x` is already ordered by `keys`.

    Fires only when the sort is a plain (no `limit`) ascending, nulls-last ordering
    over columns, and those columns are a prefix of the input's known `sorted_by`
    ordering — exactly the canonical form the estimator records, so the comparison
    is sound. Returns None otherwise (a top-N sort, a descending/nulls-first sort,
    an expression key, or an input whose order is unknown or insufficient).
    """
    if node.limit is not None:
        return None
    requested: list[str] = []
    for key in node.keys:
        if not isinstance(key.expr, Col) or key.descending or key.nulls_first:
            return None
        requested.append(key.expr.name)
    if not requested:
        return None
    have = ctx.estimator.estimate(node.input).sorted_by
    # The input must already be ordered by (at least) the requested key prefix.
    if tuple(requested) == tuple(have[: len(requested)]):
        return node.input
    return None
