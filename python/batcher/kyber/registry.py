"""The Kyber rule registry — where rules are discovered and assembled.

A `RuleRegistry` is a flat catalogue of `Rule`s. The optimizer asks it for the
rules in each phase and runs them. Two ways to populate it:

- `@rule(...)` — a decorator over a node-local function, for the common case of a
  small local rewrite. Drop a function in a module, decorate it, import the module,
  and it is registered. Nothing else edits.
- `registry.add(rule_obj)` — register a pre-built `Rule` (used for the holistic and
  cost-based rules built via `plan_rule`).

`DEFAULT_REGISTRY` holds the built-in rule set (`register_builtin_rules`), which the
`Optimizer` uses unless given an explicit rule list. Tests build their own
`RuleRegistry` for isolation. The registry stays a plain list — the *scaling* trick
is not a clever data structure here but the per-plan pattern indexing the driver
does from each rule's `matches` set (see `kyber.optimizer`).
"""

from __future__ import annotations

from collections.abc import Callable

from batcher._internal.errors import PlanError
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rule import Phase, Rule, RuleCategory, node_rule
from batcher.plan.logical import LogicalPlan

__all__ = ["DEFAULT_REGISTRY", "RuleRegistry", "register_builtin_rules", "rule"]


class RuleRegistry:
    """An ordered, deduplicated catalogue of optimization rules."""

    def __init__(self) -> None:
        self._rules: list[Rule] = []
        self._names: set[str] = set()
        self._by_name: dict[str, Rule] = {}
        # The phase partition, built on demand and dropped by `add`. Held here rather than
        # rebuilt per `Optimizer` because the *identity* of each phase's list is load-bearing
        # downstream: `optimizer.expr_dispatch.expr_type_index` memoizes the expression-type
        # inversion of a rule list on `id(rules)`, so a fresh list per query missed that memo
        # every time and re-inverted ~700 rules across 7 phases on every query. Measured on a
        # point-lookup shape: 7 of 8 lookups missed, and the 64-entry memo was thrashing.
        self._phase_cache: dict[Phase, list[Rule]] | None = None

    def add(self, rule_obj: Rule) -> Rule:
        """Register a rule under its unique name.

        Re-adding the *same* `Rule` object is a no-op, so importing a rule module twice is
        safe. Registering a **different** rule under a name already taken raises, because
        silently keeping the first one is indistinguishable from the second rule not
        existing: it never runs, its tests still pass (they call the module function
        directly, never the registry), and nothing reports it.

        That is not hypothetical. Two unrelated implementations of `intersect_in_lists` --
        one handling a single `AND` pair, one handling an n-ary conjunction and folding a
        disjoint pair to the empty relation -- both claimed the name. Import order decided
        which one shipped; the loser was dead code carrying a unit test that asserted the
        *opposite* of the optimizer's actual behaviour and could never fail.

        Args:
            rule_obj: The rule to register.

        Returns:
            The rule, so a decorator or module-level `add(...)` can bind its result.

        Raises:
            PlanError: A different rule is already registered under this name.
        """
        existing = self._by_name.get(rule_obj.name)
        if existing is not None:
            if existing is rule_obj:
                return rule_obj
            raise PlanError(
                f"two different rules are registered as {rule_obj.name!r}; a rule name is its "
                "identity in explain, telemetry and the run-order snapshot, so the second "
                "registration would be silently dropped and never run. Rename one, or delete "
                "the redundant implementation."
            )
        self._rules.append(rule_obj)
        self._names.add(rule_obj.name)
        self._by_name[rule_obj.name] = rule_obj
        # Registration order is run order, so a late `add` must be able to change a phase's
        # sequence. Dropping the partition is what keeps `by_phase` honest about that.
        self._phase_cache = None
        return rule_obj

    def rule(
        self,
        *,
        name: str,
        phase: Phase,
        matches: tuple[type, ...],
        category: RuleCategory = RuleCategory.REWRITE,
        expr: Callable | None = None,
        expr_schema: Callable | None = None,
        expr_matches: tuple[type, ...] | None = None,
        expr_ops: tuple[str, ...] | None = None,
    ) -> Callable[
        [Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None]],
        Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None],
    ]:
        """Decorator: register a node-local function as a rule. Returns the original
        function unchanged so it stays unit-testable in isolation."""

        def decorate(
            fn: Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None],
        ) -> Callable[[LogicalPlan, OptimizerContext], LogicalPlan | None]:
            self.add(
                node_rule(
                    name,
                    phase,
                    fn,
                    matches=matches,
                    category=category,
                    expr_fn=expr,
                    expr_schema_fn=expr_schema,
                    expr_matches=expr_matches,
                    expr_ops=expr_ops,
                )
            )
            return fn

        return decorate

    def rules(self) -> list[Rule]:
        """The registered rules, in registration order."""
        return list(self._rules)

    def by_phase(self) -> dict[Phase, list[Rule]]:
        """The registered rules partitioned by phase, in registration order within each.

        Memoized, and the memo is dropped by `add`. The **identity** of each returned list
        is part of the contract, not an implementation detail: the optimizer's expression
        dispatch memoizes a rule list's type inversion on `id(rules)`, so handing out a
        fresh list per query silently disables that memo. Every phase is present, mapping to
        an empty list where no rule declared it, so a caller may index any `Phase` directly.

        Returns:
            A phase-to-rules mapping the caller must treat as read-only — it is shared with
            every other caller and with the next query.
        """
        cached = self._phase_cache
        if cached is not None:
            return cached
        by_phase: dict[Phase, list[Rule]] = {p: [] for p in Phase}
        for rule_obj in self._rules:
            by_phase[rule_obj.phase].append(rule_obj)
        self._phase_cache = by_phase
        return by_phase


DEFAULT_REGISTRY = RuleRegistry()


def rule(
    *,
    name: str,
    phase: Phase,
    matches: tuple[type, ...],
    category: RuleCategory = RuleCategory.REWRITE,
    expr: Callable | None = None,
    expr_schema: Callable | None = None,
    expr_matches: tuple[type, ...] | None = None,
    expr_ops: tuple[str, ...] | None = None,
):
    """Register a node-local rule into the default registry (see `RuleRegistry.rule`).

    `expr` declares the rule's body as a leaf `Expr -> Expr` rewrite, letting the driver run
    it in one shared expression traversal with every other expression rule in the phase.
    `expr_schema` is its schema-dependent twin, a leaf `(Expr, SchemaRef) -> Expr`: the
    driver resolves the node's schema once and runs every such leaf in that same traversal,
    instead of each rule resolving the schema and walking the tree itself."""
    return DEFAULT_REGISTRY.rule(
        name=name,
        phase=phase,
        matches=matches,
        category=category,
        expr=expr,
        expr_schema=expr_schema,
        expr_matches=expr_matches,
        expr_ops=expr_ops,
    )


def register_builtin_rules(registry: RuleRegistry) -> None:
    """Populate `registry` with Kyber's built-in rules.

    These are the optimizations migrated from the original ordered-pass pipeline,
    now expressed as phased rules. They double as the reference examples for the
    rule model:

      - NORMALIZE:    constant folding, expression simplification (whole-tree, confluent)
      - PUSHDOWN:     predicate pushdown (Filter), projection/column pruning
      - JOIN_REORDER: cost-based join ordering (DPccp/DPhyp with a greedy fallback,
                      registered as `join_reorder` from `rules.joins.order`)
      - FUSION:       top-N fusion (Sort+Limit → partial sort)
      - SELECTION:    adaptive join build-side (cost-based, records its decision)

    The join-ordering memo shares the same cardinality estimator and cost model as
    every other cost-based rule; a genetic enumerator for very wide join graphs is a
    future extension of that same seam.
    """
    # Imported lazily so the registry module has no import cycle with the rule
    # bodies, and so importing `registry` is cheap.
    from batcher.kyber.rule import plan_rule
    from batcher.kyber.rules.fusion import fuse_topn
    from batcher.kyber.rules.normalize import ConstantFolding, ExprSimplification
    from batcher.kyber.rules.projections import rewrite_projection
    from batcher.kyber.rules.pushdown import rewrite_predicate
    from batcher.kyber.rules.selection import build_side_rule
    from batcher.plan.logical import Filter, Join, Limit

    _const_fold = ConstantFolding()
    _simplify = ExprSimplification()

    # `plan_rule` defaults to category=REWRITE; only the cost-based selection rule
    # is tagged otherwise. Registration order = within-phase run order.
    builtins = [
        plan_rule("constant_folding", Phase.NORMALIZE, _const_fold.apply),
        plan_rule("expr_simplification", Phase.NORMALIZE, _simplify.apply),
        plan_rule(
            "predicate_pushdown",
            Phase.PUSHDOWN,
            lambda plan, _ctx: rewrite_predicate(plan),
            matches=(Filter,),
        ),
        plan_rule(
            "projection_rewrite", Phase.PUSHDOWN, lambda plan, _ctx: rewrite_projection(plan)
        ),
        plan_rule(
            "topn_fusion", Phase.FUSION, lambda plan, _ctx: fuse_topn(plan), matches=(Limit,)
        ),
        plan_rule(
            "adaptive_build_side",
            Phase.SELECTION,
            build_side_rule,
            matches=(Join,),
            category=RuleCategory.SELECTION,
        ),
    ]
    for builtin in builtins:
        registry.add(builtin)


register_builtin_rules(DEFAULT_REGISTRY)

# Importing the rules package runs the `@rule` decorators, which register the
# algebraic (and future) rules into DEFAULT_REGISTRY. Done last so `rule` and
# DEFAULT_REGISTRY already exist when the rule modules import them back.
from batcher.kyber import rules as _rules  # noqa: E402,F401
