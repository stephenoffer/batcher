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
from batcher.plan.logical import Distinct, Limit, LogicalPlan, RowId, Union
from batcher.plan.stats import Provenance

__all__ = [
    "drop_redundant_limit",
    "empty_limit_past_cardinality",
    "fuse_limit_into_distinct",
    "push_limit_through_row_index",
    "push_offset_limit_into_union",
]

#: How much of the input the early exit should be expected to read before it is worth taking.
#:
#: The exit pays when `k` distinct rows turn up in a short prefix, which needs `k` to be small
#: against the key's distinct count — roughly `rows_read ≈ rows x k / ndv`. At a tenth, a
#: `LIMIT 5` fires on a key with 50+ distinct values and is expected to read a tenth of the
#: input or less.
#:
#: The floor matters more than the ceiling. On a *low*-cardinality key the whole-column dedup
#: is already 4.9x DuckDB, because `agg::group::assign`'s dense direct-map deduplicates the
#: column faster than an early exit can be reached; firing there would hand that back.
_DISTINCT_LIMIT_NDV_RATIO = 10.0


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


@rule(
    name="fuse_limit_into_distinct",
    phase=Phase.REWRITE,
    matches=(Limit,),
    category=RuleCategory.REWRITE,
)
def fuse_limit_into_distinct(node: Limit, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(Distinct(x), n, offset)` → the same, with the dedup capped at `offset + n`.

    A `DISTINCT` under a `LIMIT` otherwise deduplicates its whole input before the limit
    throws nearly all of it away. That is asymptotic rather than a constant factor: on a
    high-cardinality key the dedup does work proportional to the *input* to answer a question
    about `k` rows, measured at 0.15x DuckDB on 16M rows and widening with scale. Capping the
    operator lets it stop as soon as `offset + n` distinct rows exist. DuckDB fuses the same
    pair (`PhysicalLimitedDistinct`).

    The cap is `offset + n`, not `n`: the outer limit still takes its window from the capped
    prefix, exactly as the distributed limit path splits the pair.

    **This changes which rows come back, deliberately.** `SELECT DISTINCT ... LIMIT k` with no
    `ORDER BY` leaves the choice of `k` to the engine, and today Batcher's answer is whichever
    `k` the parallel dedup's bucket order happens to put first — so it already varies with the
    shard count. The capped operator keeps the first `k` in *input* order, which is stable
    across single-node, parallel and distributed. Both are valid answers to an under-determined
    query; only one of them is the same answer on one node and on many, which invariant #7
    requires. So this rule makes the result *more* determined, not less.

    Fires only on a whole-column `DISTINCT` (a keyed one's survivor can be replaced by a later
    row, so no prefix settles it) whose key is estimated high-cardinality enough for the exit
    to pay — see `_DISTINCT_LIMIT_NDV_RATIO`. Idempotent: the rewritten `Distinct` carries a
    limit, which the guard below rejects.
    """
    inner = node.input
    if (
        node.n <= 0
        or ctx is None
        or not isinstance(inner, Distinct)
        or inner.keys
        or inner.limit is not None
    ):
        return None
    cap = node.offset + node.n
    # `estimate(inner)` is the dedup's *output* count — the key's distinct combinations — which
    # is the quantity that decides whether a short prefix can hold `cap` of them.
    stats = ctx.estimator.estimate(inner)
    # Only a *measured* low count declines here. The estimator's fallback for an unmeasured
    # column is half the input row count, which reads as high-cardinality for any relation
    # bigger than `2 x cap x ratio` — so treating the fallback as evidence would fire the rule
    # on precisely the low-cardinality key this gate exists to protect. Treating it as a
    # refusal is no better: an in-memory source carries no per-column ndv at all, so the rule
    # would never fire on one.
    #
    # Neither, then. `DEFAULT` provenance is documented as "an unconstrained guess", so it is
    # not evidence in either direction, and the decision moves to where the evidence is: the
    # operator reads a bounded prefix and abandons the early exit if `k` distinct rows have not
    # appeared (`PREFIX_PROBE_MORSELS` in `stream/breaker.rs`). That makes this gate a way to
    # skip a probe that is already known to be pointless, not the thing keeping the exit safe.
    if stats.provenance is not Provenance.DEFAULT and stats.rows < cap * _DISTINCT_LIMIT_NDV_RATIO:
        return None
    capped = Distinct(inner.input, inner.keys, inner.order, limit=cap)
    return Limit(capped, node.n, node.offset)
