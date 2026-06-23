"""The Kyber rule abstraction — one small, pure unit of optimization.

Kyber is an *ordered list of phases*; each phase is a set of `Rule`s. A `Rule` is
a pure function from plan to plan, tagged with the `Phase` it runs in, the
`RuleCategory` it belongs to (for introspection), and the set of node types it
`matches` (so the driver can skip rules that can't possibly fire on a given plan —
the indexing that keeps optimization sub-linear as the rule set grows to
thousands). Rules never execute or collect runtime metadata — Core does that. They
*consume* shared analysis (cardinality, cost, learned metadata via
`OptimizerContext`) and *decide*. This is the layering that makes the feedback loop
work: Core measures, Kyber decides.

Two ways to author a rule:

- `node_rule(...)` wraps a *node-local* function `f(node, ctx) -> node | None`
  (None = "no change"); the driver supplies bottom-up traversal and fixpoint
  iteration. This is the default shape — a new rule is a tiny local rewrite plus a
  declared set of matched node types. This is how the rule set scales.
- `plan_rule(...)` wraps a *whole-plan* function `f(plan, ctx) -> plan` for the few
  holistic rewrites (column pruning) and cost-based transforms (join reordering,
  build-side selection) that reason over the whole tree at once.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field

from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.logical import LogicalPlan
from batcher.plan.visitor import transform_up

__all__ = [
    "Phase",
    "Rule",
    "RuleCategory",
    "node_rule",
    "plan_rule",
]


class Phase(enum.IntEnum):
    """The ordered optimization phases. Rules run phase by phase, in this order.

    The integer values define the order; `IntEnum` sorts naturally. Rewrite phases
    (NORMALIZE/REWRITE/PUSHDOWN/FUSION) iterate to a fixpoint because their rules are
    confluent; the cost-based and physical phases (JOIN_REORDER/SELECTION/ENFORCE)
    run once — they make a decision, they don't converge.
    """

    NORMALIZE = 1  # constant folding, expression simplification, canonicalization, CSE
    REWRITE = 2  # subquery decorrelation, set-op rewrites, CTE handling
    PUSHDOWN = 3  # predicate / projection / limit pushdown, partition pruning
    JOIN_REORDER = 4  # cost-based multi-table join ordering (the memo plugs in here)
    FUSION = 5  # operator fusion, top-N fusion, late materialization
    SELECTION = 6  # physical algorithm choice: join build-side, agg strategy, …
    ENFORCE = 7  # distribution/exchange enforcement, validation


class RuleCategory(enum.Enum):
    """What kind of decision a rule makes — for explain/telemetry, not control flow."""

    REWRITE = "rewrite"  # deterministic, semantics-preserving plan transformation
    SELECTION = "selection"  # cost-based physical choice
    ESTIMATION = "estimation"  # annotates the plan with estimates
    VALIDATION = "validation"  # checks an invariant, never rewrites
    ENFORCE = "enforce"  # inserts a required operator (exchange, sort)


@dataclass(frozen=True, slots=True)
class Rule:
    """One optimization step.

    `name` identifies it in explain/telemetry. `apply` returns a new (or unchanged)
    plan and may record decisions on `ctx.notes`. `matches` is the set of plan node
    types the rule can act on — `None` means "any plan" (always attempted). The
    driver uses `matches` to skip rules whose node types are absent from the plan,
    which is what keeps per-plan optimization cost proportional to the *applicable*
    rules rather than the *total* number of rules.
    """

    name: str
    phase: Phase
    fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan]
    matches: frozenset[type] | None = None
    category: RuleCategory = RuleCategory.REWRITE
    idempotent: bool = True
    metadata: dict = field(default_factory=dict, compare=False)
    # For a node-local rule, the underlying `f(node, ctx) -> node | None`. The driver
    # uses this to fuse consecutive node-local rules into a *single* bottom-up
    # traversal (instead of one traversal per rule); `fn` remains the equivalent
    # whole-plan wrapper for running the rule standalone. `None` for whole-plan rules.
    node_fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None] | None = field(
        default=None, compare=False
    )

    def apply(self, plan: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
        return self.fn(plan, ctx)


def plan_rule(
    name: str,
    phase: Phase,
    fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan],
    *,
    matches: tuple[type, ...] | None = None,
    category: RuleCategory = RuleCategory.REWRITE,
    idempotent: bool = True,
) -> Rule:
    """Wrap a whole-plan function `fn(plan, ctx) -> plan` as a `Rule`.

    Use for holistic rewrites and cost-based transforms that reason over the whole
    tree. `matches` (if given) lets the driver skip the rule when none of those node
    types are present.
    """
    return Rule(
        name=name,
        phase=phase,
        fn=fn,
        matches=frozenset(matches) if matches is not None else None,
        category=category,
        idempotent=idempotent,
    )


def node_rule(
    name: str,
    phase: Phase,
    fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None],
    *,
    matches: tuple[type, ...],
    category: RuleCategory = RuleCategory.REWRITE,
    idempotent: bool = True,
) -> Rule:
    """Wrap a node-local function `fn(node, ctx) -> node | None` as a `Rule`.

    The driver supplies bottom-up traversal: `fn` is called on each node, and a
    return of `None` means "leave this node unchanged". `matches` is required — it
    is both the indexing key and the per-node guard (the wrapper only calls `fn` on
    matching node types). This is the default rule shape and the one that scales:
    each new rule is a small local transformation plus the node types it fires on.
    """
    match_set = frozenset(matches)

    def whole_plan(plan: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
        def visit(node: LogicalPlan) -> LogicalPlan:
            if type(node) not in match_set:
                return node
            out = fn(node, ctx)
            return node if out is None else out

        return transform_up(plan, visit)

    return Rule(
        name=name,
        phase=phase,
        fn=whole_plan,
        matches=match_set,
        category=category,
        idempotent=idempotent,
        node_fn=fn,
    )
