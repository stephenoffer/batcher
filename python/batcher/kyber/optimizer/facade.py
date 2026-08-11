"""The `Optimizer` façade and the module-level entry points."""

from __future__ import annotations

from batcher.config import Config, active_config
from batcher.kyber import plan_cache
from batcher.kyber.annotate import annotate_ops
from batcher.kyber.calibration import calibrate
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.cpu_shares import load_cpu_utilization
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.optimizer.driver import (
    _FIXPOINT_PHASES,
    _fixpoint_bound,
    _present,
    _run_phase,
)
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, Rule
from batcher.kyber.rules.projections import (
    required_columns_per_source,
    required_predicates_per_source,
)
from batcher.kyber.rules.selection import BuildSideDecision
from batcher.kyber.spill_rates import learned_spill_factor
from batcher.metadata import MetadataHub
from batcher.metadata.io_stats import relative_read_cost
from batcher.plan.logical import LogicalPlan
from batcher.plan.physical import PhysicalPlan
from batcher.plan.resource import HardwareProfile
from batcher.plan.source_stats import source_identity
from batcher.plan.stats import RelStats
from batcher.plan.visitor import children, walk

__all__ = ["Optimizer", "optimize", "optimize_full", "optimize_logical", "optimize_traced"]


class Optimizer:
    """Optimizes logical plans into physical plans by running phased rules."""

    def __init__(
        self,
        config: Config | None = None,
        sources: list | None = None,
        hub: MetadataHub | None = None,
        rules: list[Rule] | None = None,
        source_stats: list | None = None,
        hardware: HardwareProfile | None = None,
    ) -> None:
        self._config = config or active_config()
        self._sources = sources or []
        self._hub = hub
        # The hardware the plan targets. `None` → detect this machine (the single-node and
        # driver case); the conductor supplies a cluster-derived profile for a distributed run
        # so cache/memory/VRAM-sized thresholds track the workers, not the driver.
        self._hardware = hardware or HardwareProfile.local()
        # Per-source `SourceStatistics` the conductor collected at plan-build time
        # (footer/manifest/catalog metadata). Kyber never reads `io` itself — the
        # stats are handed in, keeping the layer boundary intact.
        self._source_stats = source_stats
        # The default registry partitions itself once and hands back the *same* lists every
        # time. That identity matters: `expr_dispatch.expr_type_index` memoizes a rule list's
        # expression-type inversion on `id(rules)`, and an `Optimizer` is constructed per
        # query — so partitioning here re-created seven lists per query, missed that memo on
        # all but one of its eight lookups, and re-inverted the whole rule set each time.
        # An explicit `rules` list (tests, a caller driving a subset) still partitions here,
        # because it has no registry to memoize against and is not on the per-query path.
        if rules is None:
            self._by_phase: dict[Phase, list[Rule]] = DEFAULT_REGISTRY.by_phase()
        else:
            by_phase: dict[Phase, list[Rule]] = {p: [] for p in Phase}
            for r in rules:
                by_phase[r.phase].append(r)
            self._by_phase = by_phase

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
        # Fitted from what the *target* machine class measured, not what this process did:
        # planning runs on the driver, which on a cluster executes none of the work.
        coeffs = calibrate(self._hub, self._config, self._hardware.fingerprint or None)
        # Each source's measured read throughput, as a multiplier on what its bytes cost
        # relative to the plan's median source. Cold, single-source, or unidentifiable
        # sources all yield 1.0, which is the ranking this had before there was a measurement.
        io_factors = relative_read_cost(self._hub, [source_identity(s) for s in self._sources])
        # The read side of that measurement has a write-side twin. A spilled byte is priced
        # from the device *class*, which is a structural reading that errs toward pessimism by
        # construction, and the fleet's own spill history can falsify it. `None` — a cold
        # store, a device already at the floor, or too few spills — keeps the class factor.
        spill_factor = learned_spill_factor(
            self._hub,
            self._hardware.storage_class or "",
            self._config,
            self._hardware.fingerprint or None,
        )
        # The fleet the plan will run across drives the `net` axis. `1` single-node, which
        # makes that axis identically zero — so a single-node plan is ranked exactly as it
        # was before shuffle volume was costed at all.
        cost_model = CostModel(
            estimator,
            coeffs,
            workers=self._hardware.worker_count,
            source_io_factors=io_factors,
            spill_device_factor=spill_factor,
            # The whole profile, not just its worker count: the `net` axis is priced against
            # the fleet's interconnect *tiers*, and how a shuffle splits over them is a
            # question about node density and fabric width that a worker count cannot answer.
            hardware=self._hardware,
        )
        return OptimizerContext(
            config=self._config,
            sources=self._sources,
            hub=self._hub,
            estimator=estimator,
            cost_model=cost_model,
            hardware=self._hardware,
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
                load_cpu_utilization(self._hub, self._config, self._hardware.fingerprint or None),
                # The fleet shape, so the PACK/SPREAD preference can be a comparison rather
                # than a constant: on dense nodes, concentrating a gang moves a large exchange
                # off the network entirely, which an absolute byte threshold cannot express.
                self._hardware,
            ),
            source_projections=required_columns_per_source(plan),
            source_predicates=_source_predicates(logical, plan),
            prefer_materializing_aggregate=_prefers_materializing_aggregate(plan, ctx),
        )
        return phys, plan, ctx.notes.get("build_side_decisions", [])

    def logical_rewrite(self, logical: LogicalPlan) -> LogicalPlan:
        """Run every optimizer phase, returning the optimized **logical** plan.

        Named for what the caller wants (a rewritten `LogicalPlan`, not a `PhysicalPlan`),
        not for a subset of phases: `_run` iterates *all* of `Phase`, so JOIN_REORDER and
        SELECTION execute here too. The previous "only the logical rewrite phases" wording
        was wrong and contradicted `optimize_logical`, which memoizes this exact call —
        worth knowing, because the metadata-answer layer calls this per `.count()` and so
        pays for join-order search, not just the pruning it is after.

        The seam the metadata-answer layer uses to simplify a plan (combine limits, drop
        redundant distincts, zone-map pruning) before estimating it with an exact-first
        estimator of its own.
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
        # The other decision notes rules record. `OptimizerContext.notes` is documented as
        # the bag rules write to "for explain/telemetry", but only `build_side_decisions` was
        # ever read back — `gpu_resource_sizing` and `runtime_join_filters` were written on
        # every applicable plan and surfaced nowhere, so the reasoning behind a GPU sizing
        # choice or a runtime join filter was unobservable.
        for key, heading in (
            ("gpu_resource_sizing", "gpu resource sizing:"),
            ("runtime_join_filters", "runtime join filters:"),
        ):
            notes = ctx.notes.get(key) or []
            if notes:
                lines.append("")
                lines.append(heading)
                lines.extend(f"  - {n}" for n in notes)
        return "\n".join(lines)


def _format_plan(node: LogicalPlan, est: CardinalityEstimator, depth: int = 0) -> list[str]:
    indent = "  " * depth
    rows = est.estimate(node)
    label = type(node).__name__
    out = [f"{indent}{label}  (≈{rows.rows:,.0f} rows, {rows.provenance})"]
    for child in children(node):
        out += _format_plan(child, est, depth + 1)
    return out


#: Estimated groups at or above which a join-free grouped aggregate is cheaper on the
#: materializing executor than on the streaming one.
#:
#: Measured on the H2O `groupby` suite at its own 1e7-row tier, streaming against
#: materializing: 100 groups 31.8 vs 36.3 ms and 10,000 groups 141 vs 196 ms (streaming
#: wins), against 100,000 groups 185 vs 94 ms and 1e7 groups 1,569 vs 715 ms (materializing
#: wins, by 2.0x and 2.2x). Both executors parallelize and both return the identical answer;
#: what changes is CPU spent — streaming burned roughly twice as much at the high end, and
#: below ~1e4 groups its per-morsel pre-aggregation collapses the relation early enough that
#: holding the whole input buys nothing.
#:
#: The crossover is therefore somewhere in 1e4..1e5 and this sits at its midpoint. Stated
#: plainly because it is an interpolation, not a measurement: the two sides of the bracket
#: are measured and the point between them is not.
MATERIALIZE_AGG_MIN_GROUPS = 50_000


def _prefers_materializing_aggregate(plan: LogicalPlan, ctx: OptimizerContext) -> bool:
    """Whether this plan's grouped aggregate is worth materializing rather than streaming.

    Only for the shape that was measured: a **join-free** plan whose root is a grouped
    `Aggregate`. A grouped aggregate under a join is a different plan with different
    intermediates and none of it was measured, so it keeps the existing routing. The engine
    re-checks the shape itself and ANDs in its own memory-affordability test, so this is a
    hint the data plane may decline, never an instruction.

    A row-wise `Project` **above** the aggregate is peeled first, and must be: most grouped
    aggregates do not have the aggregate at the root, and the projection above it says nothing
    about which executor should run them. `max(v1) - min(v2)` leaves one for the subtraction
    and `stddev` leaves one for its `sqrt` — H2O `groupby` q7 and q6 — so both were rejected
    here and ran ~1.8x slower than the path they qualify for on every other count. The
    projection is over one row per group, so peeling it cannot change the verdict. The Rust
    guard (`materializing_aggregate_is_faster`) peels the same way; the two must agree or the
    hint is discarded by the engine's own shape check.
    """
    from batcher.plan.logical import Aggregate, Join, Project

    node = plan
    while isinstance(node, Project):
        node = node.input
    if not isinstance(node, Aggregate) or not node.group_keys:
        return False
    if any(isinstance(n, Join) for n in walk(plan)):
        return False
    try:
        return ctx.estimator.estimate(plan).rows >= MATERIALIZE_AGG_MIN_GROUPS
    except Exception:
        return False  # no estimate is not evidence for changing the route


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
    """The optimized physical plan, reusing a cached plan when one exists.

    The physical-plan-only projection of `optimize_full`, and for the same reason
    `optimize_traced` is: this is what the streaming terminals reach for
    (`iter_batches`, the micro-batch dispatcher, the watermark rewrites), and it used to
    build a fresh `Optimizer` and re-run every phase on every call. A `for batch in
    ds.iter_batches()` loop therefore re-planned the identical query on each pass while
    `collect()` on the same `Dataset` answered from the memo — the streaming path, the one
    that exists to be incremental, was the one paying full planning cost repeatedly.
    """
    phys, _plan, _decisions = optimize_full(logical, config, sources, hub, source_stats)
    return phys


def optimize_traced(
    logical: LogicalPlan,
    config: Config | None = None,
    sources: list | None = None,
    hub: MetadataHub | None = None,
    source_stats: list | None = None,
) -> tuple[PhysicalPlan, list[BuildSideDecision]]:
    """The physical plan and its build-side decisions, reusing a cached plan when one exists.

    A thin projection of `optimize_full`, and deliberately so: this is the entry point
    `explain()` uses, and it used to construct a fresh `Optimizer` and re-run every phase
    on every call. That made explaining a query cost a full optimization even though
    `collect()` on the same `Dataset` had already planned it and put the result in the
    memo — so the diagnostic tool was the slowest way to look at a plan, and an
    `explain()` / `collect()` pair planned the same query twice. Sharing one memo also
    means what `explain()` shows is the plan `collect()` will actually run.
    """
    phys, _plan, decisions = optimize_full(logical, config, sources, hub, source_stats)
    return phys, decisions


def optimize_full(
    logical: LogicalPlan,
    config: Config | None = None,
    sources: list | None = None,
    hub: MetadataHub | None = None,
    source_stats: list | None = None,
    hardware: HardwareProfile | None = None,
) -> tuple[PhysicalPlan, LogicalPlan, list[BuildSideDecision]]:
    """Optimize once (physical + logical + decisions), reusing a cached plan when one exists.

    Optimization is pure in `(logical, sources, config, learned stats, hardware)`, so an
    identical query on the same hardware need not be re-planned — see `kyber.plan_cache` for
    what the key captures and why an in-memory source is keyed by object identity.
    `optimizer.plan_cache_entries = 0` disables the memo; a cold plan is computed exactly as
    before. `hardware` is `None` for the single-node path (the Optimizer detects this machine);
    the conductor supplies a cluster-derived profile for a distributed run so the plan — and
    the cache entry it is stored under — reflects the workers, not the driver.
    """
    cfg = config if config is not None else active_config()
    max_entries = cfg.optimizer.plan_cache_entries
    if max_entries <= 0:
        return Optimizer(
            cfg, sources, hub, source_stats=source_stats, hardware=hardware
        ).optimize_full(logical)

    key = plan_cache.cache_key(
        logical.content_key(), sources, cfg, hub, source_stats=source_stats, hardware=hardware
    )
    cached = plan_cache.lookup(key)
    if cached is not None:
        phys, plan, decisions = cached
        return phys, plan, list(decisions)  # decisions are telemetry; hand out a copy

    result = Optimizer(
        cfg, sources, hub, source_stats=source_stats, hardware=hardware
    ).optimize_full(logical)
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

    key = plan_cache.cache_key(
        logical.content_key(), sources, cfg, hub, kind="logical", source_stats=source_stats
    )
    cached = plan_cache.lookup(key)
    if cached is not None:
        return cached

    result = Optimizer(cfg, sources, hub, source_stats=source_stats).logical_rewrite(logical)
    plan_cache.store(key, result, sources, max_entries)
    return result
