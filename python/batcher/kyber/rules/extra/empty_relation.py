"""Empty-relation folding — turn a provably-empty subtree into the canonical marker.

Batcher's canonical empty relation is `Limit(x, 0)` (what `zonemap_prune_filter` emits
for an always-false predicate). The existing `propagate_empty_relation` folds that marker
up through the *schema-preserving* unary operators (Filter/Sort/Distinct/Sample) and Union.
This module adds the two missing pieces:

- recognize a constant-`FALSE` filter as empty (the shape `predicate_infer`'s contradiction
  rules and `boolean_algebra`'s annihilators produce), so those dead filters become the
  empty marker instead of executing a predicate that keeps no row;
- propagate emptiness through the *schema-changing* operators `Project`, grouped
  `Aggregate`, and `Window` — over zero input rows each produces zero output rows, so the
  empty marker can move above them and `propagate_empty_relation` can fold the rest.

Every rule is unconditionally semantics-preserving (result multiset unchanged) and
idempotent (it strips the inner empty rather than re-wrapping, so it never re-matches its
own output). A global (keyless) aggregate is deliberately excluded — it emits one row
(COUNT 0, others NULL) over empty input, so it is not empty.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir import Lit
from batcher.plan.logical import (
    Aggregate,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Window,
)

__all__ = [
    "aggregate_over_empty",
    "filter_false_to_empty",
    "project_over_empty",
    "window_over_empty",
]


def _empty_input(node: LogicalPlan) -> LogicalPlan | None:
    """The relation under a canonical empty marker `Limit(_, 0)`, else None."""
    inner = node.input
    if isinstance(inner, Limit) and inner.n == 0:
        return inner.input
    return None


@rule(
    name="filter_false_to_empty",
    phase=Phase.SELECTION,
    matches=(Filter,),
    category=RuleCategory.REWRITE,
)
def filter_false_to_empty(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(x, FALSE)` → `Limit(x, 0)`. A constant-`FALSE` predicate keeps no row
    (`WHERE p` keeps a row only when `p` is TRUE), so the filter is the empty relation.
    Folding it to the canonical marker lets `propagate_empty_relation` collapse the
    operators above — the payoff for `predicate_infer`/`boolean_algebra` reducing a
    contradiction to `FALSE`. Returns None for any non-literal-false predicate."""
    pred = node.predicate
    if isinstance(pred, Lit) and pred.value is False:
        return Limit(node.input, 0)
    return None


@rule(
    name="project_over_empty",
    phase=Phase.SELECTION,
    matches=(Project,),
    category=RuleCategory.REWRITE,
)
def project_over_empty(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(Limit(x, 0))` → `Limit(Project(x), 0)`. A projection is row-count
    preserving, so over an empty input it yields no rows; hoisting the empty marker above
    it (keeping the projection for its output schema) exposes the emptiness to any parent
    `propagate_empty_relation` can fold. Idempotent — the rebuilt inner `Project` no longer
    sits over a `Limit(_, 0)`."""
    base = _empty_input(node)
    if base is None:
        return None
    return Limit(Project(base, node.items), 0)


@rule(
    name="aggregate_over_empty",
    phase=Phase.SELECTION,
    matches=(Aggregate,),
    category=RuleCategory.REWRITE,
)
def aggregate_over_empty(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A *grouped* `Aggregate(Limit(x, 0))` → `Limit(Aggregate(x), 0)`. A grouped
    aggregate emits one row per present group, so over empty input it emits none. A global
    (keyless) aggregate is excluded — it emits exactly one row (COUNT 0, others NULL) over
    empty input, so it is not empty. Idempotent (the rebuilt inner aggregate no longer sits
    over the marker)."""
    if not node.group_keys:
        return None
    base = _empty_input(node)
    if base is None:
        return None
    inner = Aggregate(base, node.group_keys, node.aggregates, node.watermark)
    return Limit(inner, 0)


@rule(
    name="window_over_empty",
    phase=Phase.SELECTION,
    matches=(Window,),
    category=RuleCategory.REWRITE,
)
def window_over_empty(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Window(Limit(x, 0))` → `Limit(Window(x), 0)`. A window operator is row-count
    preserving (it adds columns), so over empty input it yields no rows; hoisting the empty
    marker exposes it upward. Idempotent (the rebuilt inner window no longer sits over the
    marker)."""
    base = _empty_input(node)
    if base is None:
        return None
    inner = Window(base, node.partition_keys, node.order_keys, node.functions, node.rank_limit)
    return Limit(inner, 0)
