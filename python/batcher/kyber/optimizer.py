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

from batcher._internal.logging import get_logger
from batcher.config import Config, active_config
from batcher.kyber import plan_cache
from batcher.kyber.annotate import annotate_ops
from batcher.kyber.calibration import calibrate
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.cpu_shares import load_cpu_utilization
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
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan
from batcher.plan.physical import PhysicalPlan
from batcher.plan.stats import RelStats
from batcher.plan.visitor import children, transform_up, walk

__all__ = ["Optimizer", "optimize", "optimize_traced"]

# Confluent rewrite phases iterate to a fixpoint (bounded by `_fixpoint_bound`, which caps
# pathological non-convergence); every other phase makes a single decision and runs once.
_FIXPOINT_PHASES = frozenset({Phase.NORMALIZE, Phase.REWRITE, Phase.PUSHDOWN, Phase.FUSION})

# Headroom above the plan's depth for rules whose convergence isn't purely depth-linear
# (a rewrite that deepens the plan, then pushes through the new level).
_FIXPOINT_DEPTH_SLACK = 4


def _depth(plan: LogicalPlan) -> int:
    """Longest root-to-leaf path in `plan`."""
    stack: list[tuple[LogicalPlan, int]] = [(plan, 1)]
    deepest = 1
    while stack:
        node, d = stack.pop()
        deepest = max(deepest, d)
        stack.extend((c, d + 1) for c in children(node))
    return deepest


def _fixpoint_bound(plan: LogicalPlan, configured: int) -> int:
    """How many iterations a fixpoint phase may take, given the plan it is rewriting.

    The pushdown rules are node-local over a bottom-up traversal, so a predicate descends
    exactly **one level per iteration** — the distance to a fixpoint is the plan's depth, not a
    constant. A fixed cap therefore doesn't bound pathological non-convergence (its purpose);
    it silently truncates *healthy, linear* convergence on any plan deeper than the cap, leaving
    predicates un-pushed and the engine scanning rows it should never have read. TPC-H q8 needs
    9 iterations and the shipped cap is 8, so this fires on the benchmark's own plans.

    Scaling the bound with depth makes the cap do its actual job: a plan that converges early
    still exits early (the loop breaks on structural identity), so this costs nothing for the
    plans that were already fine, while deep plans get the fixpoint they were promised.
    """
    return max(configured, _depth(plan) + _FIXPOINT_DEPTH_SLACK)


def _applicable(rules: list[Rule], present: frozenset[type]) -> list[Rule]:
    """Rules that could fire on a plan containing exactly `present` node types.

    A rule with `matches is None` always applies; otherwise it applies only if its
    matched node types intersect the plan. This is the indexing that keeps per-plan
    cost proportional to the applicable rules, not the total rule count.
    """
    return [r for r in rules if r.matches is None or (r.matches & present)]


def _run_phase(
    plan: LogicalPlan,
    rules: list[Rule],
    ctx: OptimizerContext,
    max_iterations: int,
    present: frozenset[type] | None = None,
) -> tuple[LogicalPlan, dict | None]:
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

    Also returns the lowered IR of the returned plan **when this phase computed it**
    (during fixpoint change-detection), else `None` meaning "the plan is unchanged —
    reuse the caller's last IR". This lets the final lowering reuse the IR the
    fixpoint loop already built instead of serializing the plan a second time.
    """
    if not rules:
        return plan, None
    if present is None:  # standalone callers (tests) don't precompute it
        present = _present(plan)
    # A phase that iterates to a fixpoint may fuse its node rules into one traversal: a
    # rewrite the fused pass skipped (a rule that built a new subtree) is picked up on the
    # next iteration. A once-run phase has no next iteration, so it must not fuse.
    fuse = max_iterations > 1
    # Survives across this phase's fixpoint iterations: a rule that was a no-op on a node
    # object stays a no-op on it, and structural sharing keeps untouched nodes identical.
    # See `_apply_node_rules` — this is what stops the fixpoint from re-deriving "nothing to
    # do" for every rule on every unchanged node, on every iteration.
    noop: dict[tuple[str, int], LogicalPlan] = {}
    current_ir = None  # lazily computed, only on the identity-says-changed path
    changed = False
    converged = False
    for _ in range(max_iterations):
        updated = _apply_rules(plan, _applicable(rules, present), ctx, fuse=fuse, noop=noop)
        if updated is plan:  # structural sharing → confirmed fixpoint, O(1)
            converged = True
            break
        if current_ir is None:
            current_ir = plan.to_ir()
        updated_ir = updated.to_ir()
        if updated_ir == current_ir:  # equal-but-new tree (an unconditional rebuilder)
            converged = True
            break
        plan, current_ir, present = updated, updated_ir, _present(updated)
        changed = True
    if fuse and not converged:
        # The iteration cap was hit while the plan was still changing. The plan a query
        # gets then depends on `fixpoint_iterations`, which means some rule in this phase
        # is non-confluent or oscillating. Results stay correct (every rule is
        # semantics-preserving) but plan quality is silently non-reproducible, so say so.
        get_logger("kyber").warning(
            "phase did not reach a fixpoint in %d iterations; plan quality may depend on "
            "`OptimizerConfig.fixpoint_iterations` (a non-confluent rule?)",
            max_iterations,
        )
    # `current_ir` tracks the latest plan's IR, so when the phase changed the plan it
    # is exactly the returned plan's IR; on a no-op phase we computed nothing new.
    return plan, (current_ir if changed else None)


def _present(plan: LogicalPlan) -> frozenset[type]:
    """The set of node types in `plan`, for the per-plan rule pattern-index."""
    return frozenset(type(n) for n in walk(plan))


def _apply_rules(
    plan: LogicalPlan,
    rules: list[Rule],
    ctx: OptimizerContext,
    *,
    fuse: bool,
    noop: dict[tuple[str, int], LogicalPlan] | None = None,
) -> LogicalPlan:
    """Apply a phase's rules in registered order.

    With `fuse`, each maximal run of consecutive node-local rules shares a *single*
    bottom-up traversal instead of one walk per rule. That is cheaper but **not**
    observationally identical, contrary to what this once claimed: `transform_up` has
    already visited a node's children by the time a rule fires on it, so when a rule
    rewrites a node into a *new subtree* the later rules in that run never see the new
    children. A fixpoint phase recovers them on its next iteration, which is why fusing is
    sound there — and only there.

    The once-run phases (SELECTION, ENFORCE) have no next iteration, so a rewrite the fused
    pass skipped would be lost outright. They run unfused: one `transform_up` per rule,
    exactly the sequential semantics. That costs a few extra walks over the handful of
    rules those phases hold, and nothing at all in the hot fixpoint phases.

    Whole-plan rules (join reorder, projection pruning, build-side selection) always run
    individually; the registered order is preserved in every case.
    """
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
        run = rules[i:j]
        if fuse:
            out = _apply_node_rules(out, run, ctx, noop, fuse_exprs=True)
        else:
            for r in run:
                out = _apply_node_rules(out, [r], ctx, noop)
        i = j
    return out


def _apply_node_rules(
    plan: LogicalPlan,
    node_rules: list[Rule],
    ctx: OptimizerContext,
    noop: dict[tuple[str, int], LogicalPlan] | None = None,
    fuse_exprs: bool = False,
) -> LogicalPlan:
    """One bottom-up pass applying every node-local rule at each node, in order.

    **A rule that did nothing to a node will do nothing to it again.** A rule is a pure
    function of `(node, ctx)`; `ctx` is fixed for the run and plan nodes are immutable, so a
    node's *identity* determines its content. `noop` therefore records "rule `i` was a no-op
    on this exact node object" and skips it on the next fixpoint iteration.

    That is where the cost was. A fixpoint phase re-runs every applicable rule over every
    node until the plan stops changing, and a rewrite typically touches one node — so on each
    later iteration the other ~140 expression rules re-walked the whole expression tree of
    every untouched node to re-derive "nothing to do" (profiled: `transform_expr_up` was 66%
    of planning, 180k calls for one query). Structural sharing means an untouched node comes
    back as the *same object*, so the memo hits, and only the subtree that actually changed —
    whose nodes are new objects — is re-examined.

    The memo holds a strong reference to the node it keyed on, so a freed node's recycled
    `id()` can never produce a stale hit."""

    def visit(node: LogicalPlan) -> LogicalPlan:
        # Every rule whose body is a leaf `Expr -> Expr` rewrite shares ONE traversal of the
        # node's expressions, instead of each walking the whole tree to find its own shape.
        # Fusing is only sound where the driver already fuses node rules — a fixpoint phase,
        # which recovers on its next iteration anything a fused pass stepped over.
        if fuse_exprs:
            node_type = type(node)
            leaves = [
                r.expr_fn
                for r in node_rules
                if r.expr_fn is not None and (r.matches is None or node_type in r.matches)
            ]
            if leaves:
                node = _apply_expr_leaves(node, leaves)
        for r in node_rules:
            if fuse_exprs and r.expr_fn is not None:
                continue  # already applied, in the shared traversal above
            # Re-check the type against the *current* node, not the one this visit started
            # with: an earlier rule may have replaced it with a node of a different type
            # (`eliminate_identity_project` returns its input), and a rule chosen for the old
            # type would then be handed a node it never matched.
            if r.matches is not None and type(node) not in r.matches:
                continue
            key = (r.name, id(node))  # a rule's *name* is its unique, position-independent id
            if noop is not None:
                seen = noop.get(key)
                if seen is not None and seen is node:
                    continue  # this rule already proved itself a no-op on this exact node
            rewritten = r.node_fn(node, ctx)
            if rewritten is None:
                if noop is not None:
                    noop[key] = node  # strong ref: pins the id against reuse
                continue
            node = rewritten
        return node

    return transform_up(plan, visit)


def _apply_expr_leaves(node: LogicalPlan, leaves: list) -> LogicalPlan:
    """Apply every leaf `Expr -> Expr` rewrite to `node`'s expressions in ONE traversal.

    Each leaf is offered every expression node, bottom-up, in registered order; a leaf that
    rewrites an expression hands the rewritten form to the next leaf at that same node. That
    is the expression-level analogue of the node-rule fusion the driver already does, and it
    is what turns "one full expression walk per rule" — the two-thirds of planning time the
    profiler found — into one walk for all of them.
    """

    def combined(expr):
        for leaf in leaves:
            expr = leaf(expr)
        return expr

    return map_node_expressions(node, lambda e: transform_expr_up(e, combined))


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

    def _run(self, logical: LogicalPlan, ctx: OptimizerContext) -> tuple[LogicalPlan, dict | None]:
        """Run every phase; return the optimized plan and its IR if a phase computed it.

        The IR is `None` only when *no* phase changed the plan (every phase was a
        no-op), in which case the caller lowers once with `plan.to_ir()`. Otherwise
        the last phase that changed the plan already built the final plan's IR.
        """
        plan = logical
        last_ir: dict | None = None
        fixpoint = _fixpoint_bound(plan, self._config.optimizer.fixpoint_iterations)
        # The node-type set drives each phase's rule pattern-index. It only changes when
        # a phase rewrites the plan, so compute it once and refresh after a real change
        # rather than re-walking the whole tree at the start of every phase (7 walks → ~1
        # per actual rewrite). Threaded into `_run_phase`, which still refreshes it across
        # its own fixpoint iterations.
        present = _present(plan)
        for phase in Phase:  # IntEnum iterates in declared (ascending) order
            max_iter = fixpoint if phase in _FIXPOINT_PHASES else 1
            plan, ir = _run_phase(plan, self._by_phase[phase], ctx, max_iter, present)
            if ir is not None:  # a no-op phase leaves the plan (and its IR) unchanged
                last_ir = ir
                present = _present(plan)  # refresh once for the next phase
        return plan, last_ir

    def optimize(self, logical: LogicalPlan) -> PhysicalPlan:
        return self.optimize_traced(logical)[0]

    def optimize_traced(self, logical: LogicalPlan) -> tuple[PhysicalPlan, list[BuildSideDecision]]:
        """Optimize, also returning the per-join build-side decisions for telemetry.

        Identical to `optimize` but surfaces the `BuildSideDecision`s the SELECTION
        phase recorded on `ctx.notes` — what the adaptive executor reports per stage.
        """
        phys, _logical, decisions = self.optimize_full(logical)
        return phys, decisions

    def optimize_full(
        self, logical: LogicalPlan
    ) -> tuple[PhysicalPlan, LogicalPlan, list[BuildSideDecision]]:
        """Optimize once, returning the physical plan, the optimized **logical** plan,
        and the per-join build-side decisions — from a single pipeline run.

        The distributed and out-of-core executors read the optimized *logical* structure
        (derived join keys, pushed predicates) while admission/costing read the physical
        plan. Both fall out of one `_run`, so a caller that needs both no longer runs the
        whole optimizer twice (the old `optimize_traced` + `optimize_logical` pair).
        """
        ctx = self._context()
        plan, ir = self._run(logical, ctx)
        phys = PhysicalPlan(
            ir=ir if ir is not None else plan.to_ir(),
            output_schema=None,
            ops=annotate_ops(
                plan,
                ctx.estimator,
                ctx.config,
                ctx.costs(),
                load_cpu_utilization(self._hub, self._config),
            ),
            source_projections=required_columns_per_source(plan),
            source_predicates=_source_predicates(logical, plan),
        )
        return phys, plan, ctx.notes.get("build_side_decisions", [])

    def logical_rewrite(self, logical: LogicalPlan) -> LogicalPlan:
        """Run only the logical rewrite phases, returning the rewritten plan.

        The seam the metadata-answer layer uses to simplify a plan (combine
        limits, drop redundant distincts, zone-map pruning) before estimating it
        with an exact-first estimator of its own.
        """
        return self._run(logical, self._context())[0]

    def logical_stats(self, logical: LogicalPlan) -> tuple[LogicalPlan, RelStats]:
        """Run the logical rewrite phases and estimate the root's `RelStats`.

        Returns the rewritten logical plan and its root statistics. The rewrites
        run first so algebraic simplifications and zone-map pruning have sharpened
        the plan before estimation.
        """
        ctx = self._context()
        plan, _ir = self._run(logical, ctx)
        return plan, ctx.estimator.estimate(plan)

    def explain(self, logical: LogicalPlan) -> str:
        """A human-readable view of the optimized plan and its cardinality decisions."""
        ctx = self._context()
        plan, _ir = self._run(logical, ctx)
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


def _source_predicates(logical: LogicalPlan, optimized: LogicalPlan) -> dict[int, dict]:
    """The predicate to push to each scan, recovered even when a rule consumed the `Filter`.

    Predicates are normally read off the *optimized* plan, where pushdown has parked a
    residual `Filter` just above each `Scan`. But a rule may legitimately absorb that
    `Filter` into the operator above it — the aggregate fusion rewrites
    ``COUNT(*)`` over ``Filter(p)`` into a single ``count_if(CASE WHEN p ...)`` pass over
    the `Scan`, which is strictly faster *and* deletes the only node this extraction knows
    how to read. The predicate then reached the source nowhere, so
    ``SELECT count(*) WHERE day = 42`` — the most ordinary lakehouse query there is —
    silently scanned every data file in the table instead of the one the log says can
    match.

    So a scan the optimized plan has no predicate for falls back to the one the *user's*
    plan put directly above it. That is always sound: a `Filter` sitting on a `Scan`
    constrains every row that scan can contribute to the query, so pre-filtering the
    source removes only rows the plan above was going to discard — whatever
    semantics-preserving shape the optimizer later rewrote it into. Where the optimized
    plan does carry a predicate it wins, since pushdown may have made it tighter.
    """
    predicates = required_predicates_per_source(optimized)
    for source_id, predicate in required_predicates_per_source(logical).items():
        predicates.setdefault(source_id, predicate)
    return predicates


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


def optimize_full(
    logical: LogicalPlan,
    config: Config | None = None,
    sources: list | None = None,
    hub: MetadataHub | None = None,
    source_stats: list | None = None,
) -> tuple[PhysicalPlan, LogicalPlan, list[BuildSideDecision]]:
    """Optimize once (physical + logical + decisions), reusing a cached plan when one exists.

    Optimization is pure in `(logical, sources, config, learned stats)`, so an identical
    query need not be re-planned — see `kyber.plan_cache` for what the key captures and why
    an in-memory source is keyed by object identity. `optimizer.plan_cache_entries = 0`
    disables the memo; a cold plan is computed exactly as before.
    """
    cfg = config if config is not None else active_config()
    max_entries = cfg.optimizer.plan_cache_entries
    if max_entries <= 0:
        return Optimizer(cfg, sources, hub, source_stats=source_stats).optimize_full(logical)

    key = plan_cache.cache_key(logical.content_key(), sources, cfg, hub)
    cached = plan_cache.lookup(key)
    if cached is not None:
        phys, plan, decisions = cached
        return phys, plan, list(decisions)  # decisions are telemetry; hand out a copy

    result = Optimizer(cfg, sources, hub, source_stats=source_stats).optimize_full(logical)
    plan_cache.store(key, result, sources, max_entries)
    phys, plan, decisions = result
    return phys, plan, list(decisions)


def optimize_logical(
    logical: LogicalPlan,
    config: Config | None = None,
    sources: list | None = None,
    hub: MetadataHub | None = None,
    source_stats: list | None = None,
) -> LogicalPlan:
    """Run every optimizer phase but return the optimized **logical** plan, not its IR.

    The adaptive executor splits a plan at its pipeline breakers and re-optimizes each
    stage with measured cardinalities. It must start from the optimized logical
    structure — join conditions derived from `WHERE` equalities, predicates pushed,
    joins reordered — or a stage subtree taken from the *raw* plan can omit the filter
    that constrains a cross join and execute a cartesian product. This is that
    structure (the same `_run` `optimize`/`optimize_traced` use, stopping before the
    PhysicalPlan wrapping so the loop can still splice `Scan`s into it).

    Memoized on the same key as `optimize_full` (see `kyber.plan_cache`) — the adaptive
    executor runs this once per collect over the query's base sources, which is the case
    the memo exists for. `LogicalPlan` nodes are frozen, so a hit hands out a value the
    caller rewrites by transformation, never in place.
    """
    cfg = config if config is not None else active_config()
    max_entries = cfg.optimizer.plan_cache_entries
    if max_entries <= 0:
        return Optimizer(cfg, sources, hub, source_stats=source_stats).logical_rewrite(logical)

    key = plan_cache.cache_key(logical.content_key(), sources, cfg, hub, kind="logical")
    cached = plan_cache.lookup(key)
    if cached is not None:
        return cached

    result = Optimizer(cfg, sources, hub, source_stats=source_stats).logical_rewrite(logical)
    plan_cache.store(key, result, sources, max_entries)
    return result
