"""Sideways information passing into a decorrelated aggregate's input.

A correlated subquery decorrelates to a `GROUP BY <correlation key>` whose result is
joined back to the outer query on that key. The aggregate is computed over the
*whole* inner relation, but only the groups whose key survives the outer query's
filters are ever read — every other group is built, finalized, and then discarded by
the join. On TPC-H Q21 that is a 6M-row `GROUP BY l_orderkey` producing 1.5M groups
of which ~76k are consumed, and it is the single most expensive node in the query.

This rule restricts the aggregate's *input* to the keys the other join side can
actually produce, by inserting a **semi-join** below the `Aggregate`:

    Join(L, Aggregate(A, group_keys=k), L.key = k)
      → Join(L, Aggregate(SemiJoin(A, Project(L, key)), group_keys=k), L.key = k)

Why a semi-join and not a bloom filter or an `IN` list. A bloom probe is not
expressible in the JSON IR — there is no `RelOp` or `Expr` for it, and the existing
`prune_join_side_in_list_by_other_side_bloom` works only on *precomputed* statistics,
which cannot see a key set that is itself computed at runtime. An `IN` list would
require Kyber to read data, which is Core's lane. A semi-join needs no new IR at all:
`"semi"` is already a first-class `join_type` end to end (`bc_ir::JoinType::Semi`), it
is mergeable so it composes across partitions exactly like every other join, and the
Rust hash join already applies its own probe-side bloom pre-filter internally — so the
semi-join *is* the bloom pushdown, expressed as algebra the whole stack understands.

**Why restricting the right side is semantics-preserving.** A row of the aggregate's
output whose key is absent from `L` cannot join with any `L` row. Deleting it changes
the join's result only if the join is required to emit right-side rows that found no
match — which is exactly what `FILTERABLE_SIDES` (owned by `joins.rewrites`, the one
table the runtime-filter families read) already encodes. `inner`, `left`, `semi`, and
`anti` all discard unmatched *right* rows, so the restriction is invisible to them;
`right` and `full` preserve them, so they are refused. Note this makes the rule safe
over Q21's own `LEFT JOIN` and over the `NOT EXISTS` above it: a left join
null-extends the *left* row when the group is missing, and a group we dropped is one
that had no left row to extend in the first place.

The rule runs in `ENFORCE`, which runs **once** — the same phase and for the same
reason as `runtime_join_filter`. The rewrite reintroduces a `Join` above the very
`Aggregate` it matched, so a fixpoint phase would re-fire on its own output forever.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.rules.joins.rewrites import _FILTERABLE_SIDES
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    Filter,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Project,
    Projection,
)

__all__ = ["push_semijoin_into_decorrelated_aggregate"]

# How many times more groups the aggregate must build than the restricting side has
# rows, before the semi-join is worth adding.
#
# The quantity that matters is **how many groups the semi-join actually deletes**, not
# how much bigger the aggregate's input is. `left_rows` bounds the distinct keys the
# other side can contribute (it cannot offer more keys than it has rows), and the
# aggregate's *output* estimate is its group count, so their ratio is the reduction.
#
# Comparing input sizes instead is the trap, and it is not hypothetical: on an input
# ratio TPC-H Q13 scores 9.5x (1,425,000 orders against 150,000 customers) and looks
# like a better candidate than it is — but `customer` contains *every* custkey, so the
# semi-join deletes nothing and costs a second pass over orders. Measured, that gate
# made Q13 43.8ms → 79.7ms. On the reduction the two separate cleanly: Q13 scores 0.67
# (100,075 groups against 150,000 rows — no reduction available) and Q21 scores 18.5
# (1,489,722 groups against 80,469 rows), so the threshold sits in a wide gap.
_MIN_GROUP_REDUCTION = 4.0

# Never pay the extra pass for a small aggregate — the win is bounded by the rows
# skipped, and a scan this size is already cheaper than a second evaluation of the
# probe side.
_MIN_AGG_INPUT_ROWS = 100_000.0


@rule(
    name="push_semijoin_into_decorrelated_aggregate",
    phase=Phase.ENFORCE,
    matches=(Join,),
    category=RuleCategory.ENFORCE,
)
def push_semijoin_into_decorrelated_aggregate(
    node: Join, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Restrict a joined-to aggregate's input to the keys the other side can produce.

    Fires on `Join(L, Aggregate(A))` whose join keys are the aggregate's group keys and
    whose type discards unmatched right rows, inserting `SemiJoin(A, Project(L, keys))`
    below the aggregate. Returns None when the shape does not match, when the join type
    preserves unmatched right rows (`right`/`full`), when a group key is not a plain
    column of `A` (so there is nothing to semi-join on), or when the estimated sizes do
    not justify a second evaluation of `L`.
    """
    if "right" not in _FILTERABLE_SIDES.get(node.join_type, ()):
        return None
    agg, rewrap = _unwrap_filters(node.right)
    if not isinstance(agg, Aggregate) or agg.watermark is not None:
        return None
    if not node.right_keys or len(node.left_keys) != len(node.right_keys):
        return None

    inner_keys = _group_key_source_columns(agg, node.right_keys)
    if inner_keys is None:
        return None
    if not _worth_it(node, agg, ctx):
        return None

    build = _key_only_projection(node.left, node.left_keys)
    if build is None:
        return None
    restricted = Join(
        agg.input,
        build,
        inner_keys,
        node.left_keys,
        "semi",
        tuple(JoinOutputCol("left", c, c) for c in agg.input.available_columns()),
    )
    ctx.notes.setdefault("aggregate_semijoin_pushdown", []).append(node.join_type)
    return Join(
        node.left,
        rewrap(dataclasses.replace(agg, input=restricted)),
        node.left_keys,
        node.right_keys,
        node.join_type,
        node.output,
        node.strategy,
    )


def _unwrap_filters(node: LogicalPlan) -> tuple[LogicalPlan, Callable[[LogicalPlan], LogicalPlan]]:
    """Peel `Filter`s off `node`, returning the inner plan and a function to restore them.

    `runtime_join_filter` shares this phase and runs first, so by the time this rule sees
    the join its right input may already be `Filter(Aggregate(...))` carrying a pushed
    key-range predicate. Those filters constrain the aggregate's *output*; the semi-join
    goes below the aggregate, so the two compose — peel, rewrite, and put them back.
    """
    peeled: list[Filter] = []
    while isinstance(node, Filter):
        peeled.append(node)
        node = node.input

    def rewrap(inner: LogicalPlan) -> LogicalPlan:
        for f in reversed(peeled):
            inner = Filter(inner, f.predicate)
        return inner

    return node, rewrap


def _group_key_source_columns(
    agg: Aggregate, right_keys: tuple[str, ...]
) -> tuple[str, ...] | None:
    """The `agg.input` column each join key groups on, or None if any is not a plain column.

    A join key must name a `group_keys` alias whose expression is a bare `Col`: only then
    does "this key value survives" translate to a predicate on an input column that the
    semi-join can test. A computed group key (`GROUP BY lower(x)`) has no such column, and
    inverting the expression is not something a rewrite may assume it can do.
    """
    by_alias = {k.alias: k.expr for k in agg.group_keys}
    out: list[str] = []
    for key in right_keys:
        expr = by_alias.get(key)
        if not isinstance(expr, Col):
            return None
        out.append(expr.name)
    return tuple(out)


def _key_only_projection(left: LogicalPlan, left_keys: tuple[str, ...]) -> Project | None:
    """`left` narrowed to just the join-key columns — the semi-join's build side.

    The semi-join reads nothing but the keys, and this projection is what stops the
    second evaluation of `left` from materializing every column it carried for the
    original join. Returns None if a key is repeated, since the projection's aliases
    must be unique (a repeated key is degenerate and not worth a rename).
    """
    if len(set(left_keys)) != len(left_keys):
        return None
    return Project(left, tuple(Projection(k, Col(k)) for k in left_keys))


def _worth_it(node: Join, agg: Aggregate, ctx: OptimizerContext) -> bool:
    """Whether the semi-join would delete enough groups to pay for a second pass.

    The cost is one extra evaluation of the restricting side plus a hash build over its
    keys; the saving is the groups the aggregate no longer builds. `left_rows` bounds
    the distinct keys that side can contribute and the aggregate's estimated *output*
    rows are its group count, so their ratio is the reduction on offer. Estimates come
    from the shared `CardinalityEstimator` — the rule decides, it never measures.
    """
    if ctx.estimator.estimate(agg.input).rows < _MIN_AGG_INPUT_ROWS:
        return False
    left_rows = ctx.estimator.estimate(node.left).rows
    if left_rows <= 0:
        return False
    groups = ctx.estimator.estimate(agg).rows
    if groups / left_rows < _MIN_GROUP_REDUCTION:
        return False
    # The rewrite recomputes the restricting side (`node.left`) a second time — once for the
    # semi-join's build, once for the outer join it already fed. That is free when the side is a
    # filtered dimension but ruinous when it is *itself* the expensive relation: on Q21 the
    # "restricting side" is the whole `supplier ⋈ lineitem ⋈ orders ⋈ nation` spine (a 6M-row
    # `lineitem` scan + three joins), whose second evaluation costs *more than the aggregate the
    # semi-join is shrinking*, so the fewer-groups win it scored on cardinality alone is a net
    # ~2x loss in wall time (measured 201 ms → 94 ms with the rule removed). The group-reduction
    # ratio above cannot see this: it prices only how many groups drop, never what recomputing
    # `node.left` costs to drop them. A semi-join can save at most the aggregate's own work, so
    # once the recompute meets or exceeds `cost(agg)` the extra pass alone eats the entire
    # possible saving — refuse. A cheap restricting side (the intended case) stays far below this
    # and still fires. Cost-based and semantics-neutral: refusing only ever falls back to the
    # already-correct un-pushed plan.
    costs = ctx.costs()
    return costs.cost(node.left).total() < costs.cost(agg).total()
