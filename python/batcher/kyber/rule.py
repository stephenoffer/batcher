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
from batcher.plan.expr_ir import Expr
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
    """What kind of decision a rule makes — introspection metadata, never control flow.

    Read this as a *claim the rule makes about itself*, not as something the optimizer acts
    on: nothing branches on it, and it is not currently rendered anywhere either. The
    `category` shown in explain output belongs to `DecisionRecord`, which is a different
    type recording a decision a rule already made. Two members (`ESTIMATION`, `VALIDATION`)
    used to sit here describing rule kinds the rule set does not contain; they were removed
    rather than left as vocabulary nobody could use, since an enum arm with no instance is
    indistinguishable from one whose rules were accidentally deleted.

    What it still earns its place for is the distinction the three surviving members draw.
    A `SELECTION` rule makes a cost-based choice among equivalent physical algorithms and an
    `ENFORCE` rule inserts an operator the plan requires; neither is a semantics-preserving
    rewrite, and telling them apart at a glance across 700-odd rules is worth one field.
    """

    REWRITE = "rewrite"  # deterministic, semantics-preserving plan transformation
    SELECTION = "selection"  # cost-based physical choice
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
    # For a node-local rule, the underlying `f(node, ctx) -> node | None`. The driver
    # uses this to fuse consecutive node-local rules into a *single* bottom-up
    # traversal (instead of one traversal per rule); `fn` remains the equivalent
    # whole-plan wrapper for running the rule standalone. `None` for whole-plan rules.
    node_fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None] | None = field(
        default=None, compare=False
    )
    # For a rule whose whole body is a leaf `Expr -> Expr` rewrite applied to every
    # expression the node carries, the leaf itself. The driver uses it to run *every* such
    # rule in a phase in a **single** expression traversal per node, instead of one full
    # traversal per rule — which is what keeps a hundred-odd expression rules affordable
    # (profiled: `transform_expr_up` was two thirds of planning time, one walk per rule per
    # node). `fn`/`node_fn` remain the equivalent standalone forms, so a rule is still
    # unit-testable on its own and a phase that cannot fuse runs it exactly as before.
    expr_fn: Callable[[Expr], Expr] | None = field(default=None, compare=False)
    # The schema-dependent twin of `expr_fn`, for a rule whose body is a leaf
    # `(Expr, SchemaRef) -> Expr` lifted by `guards.schema_rule`. Without it each such rule
    # resolves the node's schema and walks every expression *itself*: with several dozen of
    # them that was a third of planning time, all of it re-walking the same trees. Declaring
    # the leaf lets the driver resolve the schema once per node and run every schema leaf in
    # the same single traversal the plain leaves already share.
    #
    # A schema leaf can only fire where `guards.node_schema` has an answer, so a rule whose
    # `matches` is wider than that silently stops firing on the difference. It is a *safe*
    # failure -- the rule declines rather than guessing -- which is exactly what makes it
    # easy to miss. `node_schema` covers every expression-carrying node type
    # (Filter/Project/Aggregate/Sort/Window), so match no wider than that.
    expr_schema_fn: Callable[[Expr, object], Expr] | None = field(default=None, compare=False)
    # The expression shapes this rule *needs present in the plan* to have any chance of
    # firing. It does two jobs, and both read it the same way:
    #
    #   * the driver drops the rule entirely for a plan whose expressions contain none of
    #     them (`_applicable`), so a string rule never runs on a numeric query;
    #   * when the rule also supplies `expr_fn`/`expr_schema_fn`, it is the fused chain's
    #     dispatch key — the expression is offered only to leaves declaring its type.
    #
    # The two jobs impose slightly different obligations, and the stricter one governs. A
    # rule with a leaf MUST name every type the leaf can *rewrite*, because a missing type
    # means the leaf is never offered that expression. A rule without one need only name a
    # type whose presence is *necessary* — `date_trunc_lt_to_range` rewrites a `Binary` but
    # cannot fire without a `DateTrunc` somewhere, so declaring `DateTrunc` is both correct
    # and a much sharper filter than `Binary`, which nearly every plan has.
    #
    # With ~500 leaves in NORMALIZE, dispatching on this is the difference between one dict
    # lookup and 500 function calls per expression node. `None` means "any expression type"
    # and is the safe default: an undeclared rule keeps running everywhere, so omitting it
    # costs speed and never correctness. Declaring it *wrongly* is the hazard — the rule
    # silently stops firing, results stay correct, and only plan quality degrades — which is
    # why `BATCHER_VERIFY_EXPR_MATCHES=1` re-runs both the chain and the whole phase
    # undeclared and fails on any difference.
    expr_matches: frozenset[type] | None = field(default=None, compare=False)
    # A second-level discriminator *within* a declared type, naming the `op`/`fn` strings the
    # leaf can rewrite. `expr_matches` alone bottoms out on the types the engine leans on
    # hardest: `Binary` is one type carrying arithmetic, comparison, and the boolean
    # connectives, so a predicate made of `and`/`or` was still being offered to every leaf
    # that handles *any* binary — profiled at ~50,000 calls into a single comparison guard for
    # one plan. Keying the index on `(type, op)` splits that one bucket into the dozen the
    # rules actually distinguish. `None` means "any operator of the declared types" and is the
    # safe default; the same "name everything you can act on" obligation as `expr_matches`
    # applies, and `BATCHER_VERIFY_EXPR_MATCHES=1` checks both together.
    expr_ops: frozenset[str] | None = field(default=None, compare=False)

    def apply(self, plan: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
        return self.fn(plan, ctx)


def plan_rule(
    name: str,
    phase: Phase,
    fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan],
    *,
    matches: tuple[type, ...] | None = None,
    category: RuleCategory = RuleCategory.REWRITE,
    expr_matches: tuple[type, ...] | None = None,
    expr_ops: tuple[str, ...] | None = None,
) -> Rule:
    """Wrap a whole-plan function `fn(plan, ctx) -> plan` as a `Rule`.

    Use for holistic rewrites and cost-based transforms that reason over the whole
    tree. `matches` (if given) lets the driver skip the rule when none of those node
    types are present, and `expr_matches`/`expr_ops` do the same for the expression
    shapes it needs — which is the sharper filter for a whole-plan rule, since it has
    no node type to be indexed on and would otherwise run against every plan.
    """
    return Rule(
        name=name,
        phase=phase,
        fn=fn,
        matches=frozenset(matches) if matches is not None else None,
        category=category,
        expr_matches=frozenset(expr_matches) if expr_matches is not None else None,
        expr_ops=frozenset(expr_ops) if expr_ops is not None else None,
    )


def node_rule(
    name: str,
    phase: Phase,
    fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None],
    *,
    matches: tuple[type, ...],
    category: RuleCategory = RuleCategory.REWRITE,
    expr_fn: Callable[[Expr], Expr] | None = None,
    expr_schema_fn: Callable[[Expr, object], Expr] | None = None,
    expr_matches: tuple[type, ...] | None = None,
    expr_ops: tuple[str, ...] | None = None,
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
        node_fn=fn,
        expr_fn=expr_fn,
        expr_schema_fn=expr_schema_fn,
        expr_matches=frozenset(expr_matches) if expr_matches is not None else None,
        expr_ops=frozenset(expr_ops) if expr_ops is not None else None,
    )
