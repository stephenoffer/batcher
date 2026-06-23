"""The Kyber optimizer entry point.

Kyber turns a `LogicalPlan` into a `PhysicalPlan` by running its rules **phase by
phase** (`rule.Phase`). Each phase holds a set of `Rule`s; rewrite phases iterate
to a fixpoint (confluent rules), the cost-based/physical phases run once. Adding an
optimization means registering a `Rule` (drop a decorated function, or
`registry.add(...)`) — never editing this driver.

The driver stays fast as the rule set grows because it **pattern-indexes**: before
running a phase it computes the set of node types present in the plan and skips
every rule whose `matches` set is disjoint from it. So a plan with no `Join` never
pays for the hundred join rules. This is the property that lets the rule set scale
to thousands without each query touching all of them.

Cardinality and cost estimates feeding the cost-based phases sharpen across
executions via the MetadataHub (learned selectivities / join sizes), so the plan a
query gets *improves the more it runs* — Core collects the metadata, Kyber decides
with it.
"""

from __future__ import annotations

import math

from batcher.config import Config, active_config
from batcher.kyber.calibration import calibrate
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, Rule
from batcher.kyber.rules.projections import (
    required_columns_per_source,
    required_predicates_per_source,
)
from batcher.kyber.rules.selection import BuildSideDecision
from batcher.metadata import MetadataHub
from batcher.plan.ids import OpId
from batcher.plan.logical import LogicalPlan
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds
from batcher.plan.stats import RelStats
from batcher.plan.visitor import children, transform_up, walk

__all__ = ["Optimizer", "optimize", "optimize_traced"]

# Memory-budgeting model (consumed by Carbonite admission). Materializing
# operators ("breakers") hold ~all their rows; streaming operators hold ~one morsel.
# The tunables (row footprint, morsel size, unknown-size threshold) live in `Config`.
_BREAKER_KINDS = frozenset({"Aggregate", "Sort", "Distinct", "Join"})


def _annotate_ops(
    plan: LogicalPlan, estimator: CardinalityEstimator, config: Config
) -> tuple[PhysicalOp, ...]:
    """Tag each operator with its estimated rows + memory envelope for Carbonite.

    Kyber measures; Carbonite protects: these per-operator `ResourceBounds` are what
    the admission policy checks a plan's feasibility against, without either layer
    importing the other (the bounds travel on the `PhysicalPlan`).
    """
    row_bytes = config.optimizer.row_bytes
    morsel_rows = config.execution.morsel_rows
    morsel_bytes = max(1, config.execution.morsel_bytes)
    target_rows = max(1, config.optimizer.target_rows_per_task)
    fc = config.flow_control
    credit_ceiling = max(1, fc.default_credits * fc.credit_ceiling_factor)
    # At/above this, a cardinality is a placeholder (unknown source size), not a real
    # estimate — such operators are left unbudgeted so a guess never fails a real query.
    unknown_rows = config.optimizer.cardinality.unknown_rows
    ops: list[PhysicalOp] = []
    try:
        nodes = list(walk(plan))
        for i, node in enumerate(nodes):
            rows = estimator.estimate(node).rows
            kind = type(node).__name__
            known = 0.0 <= rows < unknown_rows
            # Byte-true width: learned per-column widths when measured, else the flat
            # `row_bytes` default (so a cold-start envelope is unchanged). A column of
            # wide payloads (blobs, embeddings) now inflates the envelope correctly.
            width = estimator.row_width(node, row_bytes)
            if not known:
                mem = 0  # unknown size — don't budget (never fail a real query on a guess)
            elif kind in _BREAKER_KINDS:
                mem = int(rows * width)  # materialized state
            else:
                # streaming: ~one morsel in flight, byte-bounded.
                mem = min(int(morsel_rows * width), morsel_bytes)
            # Desired parallelism: a breaker wants enough tasks that each handles
            # ~`target_rows` of the data it *shuffles* — its input volume, not its
            # (possibly tiny) grouped output. Streaming ops inherit the pipeline's
            # width (0 = unset). Carbonite clamps the request to the cpu budget.
            if known and kind in _BREAKER_KINDS:
                in_rows = sum(estimator.estimate(c).rows for c in children(node)) or rows
                n_par = max(1, math.ceil(in_rows / target_rows))
            else:
                n_par = 0
            # Desired credit window: enough in-flight batch slots to cover one task's
            # partition of the materialized state, clamped to the configured ceiling.
            if n_par > 0 and mem > 0:
                partition_bytes = mem / n_par
                c_max = max(1, min(credit_ceiling, math.ceil(partition_bytes / morsel_bytes)))
            else:
                c_max = 0  # no estimate → Carbonite supplies the default window
            ops.append(
                PhysicalOp(
                    op_id=OpId(i),
                    kind=kind,
                    backend="native",
                    algorithm="",
                    bounds=ResourceBounds(
                        m_max_bytes=mem, c_max_credits=c_max, n_max_parallelism=n_par
                    ),
                    inputs=(),
                    properties=PlanProperties(est_rows=rows),
                )
            )
    except Exception:
        return ()  # estimation unavailable (e.g. unbound sources) → Carbonite abstains
    return tuple(ops)


# Confluent rewrite phases iterate to a fixpoint (bounded by
# `OptimizerConfig.fixpoint_iterations`, which caps pathological non-convergence);
# every other phase makes a single decision and runs once.
_FIXPOINT_PHASES = frozenset({Phase.NORMALIZE, Phase.REWRITE, Phase.PUSHDOWN, Phase.FUSION})


def _applicable(rules: list[Rule], present: frozenset[type]) -> list[Rule]:
    """Rules that could fire on a plan containing exactly `present` node types.

    A rule with `matches is None` always applies; otherwise it applies only if its
    matched node types intersect the plan. This is the indexing that keeps per-plan
    cost proportional to the applicable rules, not the total rule count.
    """
    return [r for r in rules if r.matches is None or (r.matches & present)]


def _run_phase(
    plan: LogicalPlan, rules: list[Rule], ctx: OptimizerContext, max_iterations: int
) -> LogicalPlan:
    """Run one phase's rules, up to `max_iterations` (1 = once, >1 = to fixpoint).

    Fixpoint is detected by **object identity first**: `transform_up` shares structure
    (an untouched subtree keeps its identity), and node rules return their input on a
    no-op, so a phase that changed nothing returns the *same* plan object — an O(1)
    check. Only when identity says "changed" do we fall back to comparing lowered IR
    (`to_ir()`), because a whole-plan rule may rebuild an equal-but-new tree
    unconditionally; the IR comparison (not Python `==`, which `Expr.__eq__` overloads
    to build a comparison expression) confirms a *real* change. So semantics are
    exactly as before, just without serializing the plan every iteration.

    `_present` (the node-type set for rule indexing) is likewise computed once and
    refreshed only after an iteration that actually changed the plan.
    """
    if not rules:
        return plan
    present = _present(plan)
    current_ir = None  # lazily computed, only on the identity-says-changed path
    for _ in range(max_iterations):
        updated = _apply_rules(plan, _applicable(rules, present), ctx)
        if updated is plan:  # structural sharing → confirmed fixpoint, O(1)
            break
        if current_ir is None:
            current_ir = plan.to_ir()
        updated_ir = updated.to_ir()
        if updated_ir == current_ir:  # equal-but-new tree (an unconditional rebuilder)
            break
        plan, current_ir, present = updated, updated_ir, _present(updated)
    return plan


def _present(plan: LogicalPlan) -> frozenset[type]:
    """The set of node types in `plan`, for the per-plan rule pattern-index."""
    return frozenset(type(n) for n in walk(plan))


def _apply_rules(plan: LogicalPlan, rules: list[Rule], ctx: OptimizerContext) -> LogicalPlan:
    """Apply a phase's rules in registered order, fusing each maximal run of
    consecutive node-local rules into a *single* bottom-up traversal.

    Previously every node-local rule did its own `transform_up`, so N rules meant N
    full tree walks per fixpoint iteration; here a run of node rules is applied in one
    walk. Whole-plan rules (join reorder, projection pruning, build-side selection)
    still run individually, and the registered order is preserved exactly — so the
    fused pass is observationally identical, just cheaper."""
    out = plan
    i, n = 0, len(rules)
    while i < n:
        if rules[i].node_fn is None:
            out = rules[i].apply(out, ctx)
            i += 1
            continue
        j = i
        while j < n and rules[j].node_fn is not None:
            j += 1
        out = _apply_node_rules(out, rules[i:j], ctx)
        i = j
    return out


def _apply_node_rules(
    plan: LogicalPlan, node_rules: list[Rule], ctx: OptimizerContext
) -> LogicalPlan:
    """One bottom-up pass applying every node-local rule at each node, in order.

    A node-local rule inspects only a node and its already-rewritten subtree (never
    its ancestors), so applying `[r1, r2, …]` at each node in a single `transform_up`
    yields the same tree as running each rule's own `transform_up` in sequence — the
    phase's fixpoint loop still handles rewrites that must propagate up across levels.
    """

    def visit(node: LogicalPlan) -> LogicalPlan:
        for r in node_rules:
            if r.matches is None or type(node) in r.matches:
                rewritten = r.node_fn(node, ctx)
                if rewritten is not None:
                    node = rewritten
        return node

    return transform_up(plan, visit)


class Optimizer:
    """Optimizes logical plans into physical plans by running phased rules."""

    def __init__(
        self,
        config: Config | None = None,
        sources: list | None = None,
        hub: MetadataHub | None = None,
        rules: list[Rule] | None = None,
        source_stats: list | None = None,
    ) -> None:
        self._config = config or active_config()
        self._sources = sources or []
        self._hub = hub
        # Per-source `SourceStatistics` the conductor collected at plan-build time
        # (footer/manifest/catalog metadata). Kyber never reads `io` itself — the
        # stats are handed in, keeping the layer boundary intact.
        self._source_stats = source_stats
        all_rules = rules if rules is not None else DEFAULT_REGISTRY.rules()
        self._by_phase: dict[Phase, list[Rule]] = {p: [] for p in Phase}
        for r in all_rules:
            self._by_phase[r.phase].append(r)

    def _context(self) -> OptimizerContext:
        learned = load_learned_stats(self._hub) if self._hub is not None else {}
        estimator = CardinalityEstimator(
            self._sources,
            learned,
            self._config.optimizer.cardinality,
            source_stats=self._source_stats,
        )
        # Coefficients calibrated from measured op_stats (defaults until a workload
        # has run): this is what lets the cost model reflect the real engine.
        coeffs = calibrate(self._hub, self._config)
        cost_model = CostModel(estimator, coeffs)
        return OptimizerContext(
            config=self._config,
            sources=self._sources,
            hub=self._hub,
            estimator=estimator,
            cost_model=cost_model,
        )

    def _run(self, logical: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
        plan = logical
        fixpoint = self._config.optimizer.fixpoint_iterations
        for phase in Phase:  # IntEnum iterates in declared (ascending) order
            max_iter = fixpoint if phase in _FIXPOINT_PHASES else 1
            plan = _run_phase(plan, self._by_phase[phase], ctx, max_iter)
        return plan

    def optimize(self, logical: LogicalPlan) -> PhysicalPlan:
        return self.optimize_traced(logical)[0]

    def optimize_traced(self, logical: LogicalPlan) -> tuple[PhysicalPlan, list[BuildSideDecision]]:
        """Optimize, also returning the per-join build-side decisions for telemetry.

        Identical to `optimize` but surfaces the `BuildSideDecision`s the SELECTION
        phase recorded on `ctx.notes` — what the adaptive executor reports per stage.
        """
        ctx = self._context()
        plan = self._run(logical, ctx)
        phys = PhysicalPlan(
            ir=plan.to_ir(),
            output_schema=None,
            ops=_annotate_ops(plan, ctx.estimator, ctx.config),
            source_projections=required_columns_per_source(plan),
            source_predicates=required_predicates_per_source(plan),
        )
        return phys, ctx.notes.get("build_side_decisions", [])

    def logical_rewrite(self, logical: LogicalPlan) -> LogicalPlan:
        """Run only the logical rewrite phases, returning the rewritten plan.

        The seam the metadata-answer layer uses to simplify a plan (combine
        limits, drop redundant distincts, zone-map pruning) before estimating it
        with an exact-first estimator of its own.
        """
        return self._run(logical, self._context())

    def logical_stats(self, logical: LogicalPlan) -> tuple[LogicalPlan, RelStats]:
        """Run the logical rewrite phases and estimate the root's `RelStats`.

        Returns the rewritten logical plan and its root statistics. The rewrites
        run first so algebraic simplifications and zone-map pruning have sharpened
        the plan before estimation.
        """
        ctx = self._context()
        plan = self._run(logical, ctx)
        return plan, ctx.estimator.estimate(plan)

    def explain(self, logical: LogicalPlan) -> str:
        """A human-readable view of the optimized plan and its cardinality decisions."""
        ctx = self._context()
        plan = self._run(logical, ctx)
        decisions: list[BuildSideDecision] = ctx.notes.get("build_side_decisions", [])
        lines = _format_plan(plan, ctx.estimator)
        if decisions:
            lines.append("")
            lines.append("join build-side decisions:")
            for d in decisions:
                action = "SWAP (build smaller=left)" if d.swapped else "keep"
                lines.append(
                    f"  - left≈{d.left_rows:,.0f} right≈{d.right_rows:,.0f} "
                    f"[{d.provenance}] → {action}"
                )
        return "\n".join(lines)


def _format_plan(node: LogicalPlan, est: CardinalityEstimator, depth: int = 0) -> list[str]:
    indent = "  " * depth
    rows = est.estimate(node)
    label = type(node).__name__
    out = [f"{indent}{label}  (≈{rows.rows:,.0f} rows, {rows.provenance})"]
    for child in children(node):
        out += _format_plan(child, est, depth + 1)
    return out


def optimize(
    logical: LogicalPlan,
    config: Config | None = None,
    sources: list | None = None,
    hub: MetadataHub | None = None,
    source_stats: list | None = None,
) -> PhysicalPlan:
    """Convenience wrapper around `Optimizer.optimize`."""
    return Optimizer(config, sources, hub, source_stats=source_stats).optimize(logical)


def optimize_traced(
    logical: LogicalPlan,
    config: Config | None = None,
    sources: list | None = None,
    hub: MetadataHub | None = None,
    source_stats: list | None = None,
) -> tuple[PhysicalPlan, list[BuildSideDecision]]:
    """Convenience wrapper around `Optimizer.optimize_traced`."""
    return Optimizer(config, sources, hub, source_stats=source_stats).optimize_traced(logical)
