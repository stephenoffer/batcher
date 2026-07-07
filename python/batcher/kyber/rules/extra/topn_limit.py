"""LIMIT / OFFSET rewrites that the base limit rules don't already cover.

The common `LIMIT`/`OFFSET` shapes are handled elsewhere: `combine_limits` collapses
stacked limits, `push_limit_through_project` sinks a limit under a projection,
`push_limit_into_union` caps a zero-offset limit's `UNION ALL` branches, and
`fuse_topn` turns `Limit(Sort(...))` into a partial-sort top-N. This module adds the
remaining *provably correct* pieces:

- `drop_redundant_limit` / `empty_limit_past_cardinality` — cardinality-driven
  simplifications: a limit that keeps every row is dropped, and one whose offset skips
  past a relation's whole (exact) size is folded to the empty marker `Limit(x, 0)`.
- `push_limit_through_row_index` — a limit sinks below a `with_row_index`, since the
  index numbers rows in arrival order and the first `offset..offset+n` indices are
  identical whether the row-count cut happens before or after the numbering.
- `push_offset_limit_into_union` — the `offset > 0` analogue of `push_limit_into_union`:
  a `UNION ALL` prefix of length `offset + n` never needs more than that many rows from
  any single branch, so each branch is capped at `offset + n`.

Every rule is order-*insensitive*-safe: none reorders rows or changes *which* rows a
limit would return; they only move or remove work while preserving the exact result.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.logical import Limit, LogicalPlan, RowId, Union

__all__ = [
    "drop_redundant_limit",
    "empty_limit_past_cardinality",
    "push_limit_through_row_index",
    "push_offset_limit_into_union",
]


@rule(
    name="drop_redundant_limit",
    phase=Phase.REWRITE,
    matches=(Limit,),
    category=RuleCategory.REWRITE,
)
def drop_redundant_limit(node: Limit, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(x, n)` → `x` when `x` provably has at most `n` rows and there is no offset.

    A limit that would keep every row of its input is pure overhead. This fires only
    on a zero-offset limit whose input's row count is known *exactly* (an in-memory or
    footer-counted scan, a keyless aggregate — never an estimate) and is `<= n`, so the
    limit can never drop a row. The `n > 0` guard leaves the canonical empty marker
    `Limit(x, 0)` untouched. Returns None otherwise, so the rule is idempotent.
    """
    if node.offset != 0 or node.n <= 0 or ctx is None:
        return None
    stats = ctx.estimator.estimate(node.input)
    if stats.provenance.is_exact and stats.rows <= node.n:
        return node.input
    return None


@rule(
    name="empty_limit_past_cardinality",
    phase=Phase.REWRITE,
    matches=(Limit,),
    category=RuleCategory.REWRITE,
)
def empty_limit_past_cardinality(node: Limit, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(x, n, offset)` → `Limit(x, 0)` when `offset` skips past all of `x`'s rows.

    If the input's row count is known *exactly* and is `<= offset`, every row is skipped
    and the result is empty — so the limit folds to the canonical empty marker
    `Limit(x, 0)` (which `propagate_empty_relation` can then hoist). Gated to a positive
    offset (the zero-offset case is `drop_redundant_limit`'s) and an already-positive `n`
    so it fires once and then leaves the `n == 0` marker at a fixpoint.
    """
    if node.offset <= 0 or node.n <= 0 or ctx is None:
        return None
    stats = ctx.estimator.estimate(node.input)
    if stats.provenance.is_exact and stats.rows <= node.offset:
        return Limit(node.input, 0)
    return None


@rule(name="push_limit_through_row_index", phase=Phase.PUSHDOWN, matches=(Limit,))
def push_limit_through_row_index(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(RowId(x, name, r))` → `RowId(Limit(x), name, r + offset)`.

    `with_row_index` numbers rows `r, r+1, …` in arrival order and is otherwise a
    1:1, order-preserving pass-through. Taking rows `[offset, offset+n)` of the numbered
    stream yields exactly `x`'s rows `[offset, offset+n)` carrying indices
    `r+offset … r+offset+n-1` — identical to numbering the *already-limited* rows from a
    base of `r + offset`. So the row-count cut moves below the numbering (fewer indices
    to materialize) with byte-identical output. Returns None when the child isn't a
    `RowId`, keeping the rule idempotent.
    """
    child = node.input
    if not isinstance(child, RowId):
        return None
    limited = Limit(child.input, node.n, node.offset)
    return RowId(limited, child.alias, child.offset + node.offset)


@rule(name="push_offset_limit_into_union", phase=Phase.PUSHDOWN, matches=(Limit,))
def push_offset_limit_into_union(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(UNION ALL(a, b, …), n, offset)` → cap each branch at `offset + n`.

    The `offset > 0` companion to `push_limit_into_union` (which handles offset 0). A
    `UNION ALL` concatenates its inputs, so the outer limit reads at most the first
    `offset + n` rows of the concatenation — and no single branch can contribute more
    than those to that prefix. Capping each branch at `offset + n` therefore preserves
    the exact rows the outer limit selects, while shrinking each branch. The outer
    `Limit(n, offset)` is kept to take the true window.

    Restricted to non-distinct unions (dedup changes counts) with a positive offset; the
    guard against already-capped inputs makes it fire once and then rest at a fixpoint.
    """
    inner = node.input
    if (
        node.offset > 0
        and isinstance(inner, Union)
        and not inner.distinct
        and not any(isinstance(i, Limit) for i in inner.inputs)
    ):
        cap = node.offset + node.n
        capped = tuple(Limit(i, cap, 0) for i in inner.inputs)
        return Limit(Union(capped, distinct=False), node.n, node.offset)
    return None
