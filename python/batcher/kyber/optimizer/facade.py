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
from batcher.kyber.rules.source_limits import (
    required_limits_per_source,
    required_orderings_per_source,
)
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
        # The cleanup round's rules come from the registry for the same list-identity reason
        # the phase partition does — see `_run_cleanup`.
        if rules is None:
            self._by_phase: dict[Phase, list[Rule]] = DEFAULT_REGISTRY.by_phase()
            self._cleanup: list[Rule] = DEFAULT_REGISTRY.recanonicalize_rules()
        else:
            by_phase: dict[Phase, list[Rule]] = {p: [] for p in Phase}
            for r in rules:
                by_phase[r.phase].append(r)
            self._by_phase = by_phase
            self._cleanup = [r for r in rules if r.recanonicalize]

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

        A **canonicalization round** runs after FUSION — see `_run_cleanup`.
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
            if phase is Phase.FUSION:
                plan, ir, present = self._run_cleanup(plan, ctx, fixpoint, present)
                if ir is not None:
                    last_ir = ir
        return plan, last_ir

    def _run_cleanup(
        self,
        plan: LogicalPlan,
        ctx: OptimizerContext,
        fixpoint: int,
        present: frozenset[type],
    ) -> tuple[LogicalPlan, dict | None, frozenset[type]]:
        """Re-run the canonicalization rules once the plan's shape has settled.

        The phase list is a **single forward pass**, so every rule sees the plan exactly as it
        stands when its own phase runs — and the phases after it put back the shapes a
        canonicalizer exists to collapse. Projection pushdown stacks `Project` on `Project`;
        join reordering re-parents subtrees so operators that were separated become adjacent;
        fusion re-parents them again. `merge_projections` (NORMALIZE) and
        `merge_adjacent_filters` (PUSHDOWN) have long since run by then, nothing runs them
        again, and the redundant operator ships to the engine.

        Measured over the 99 TPC-DS queries: **46 of them optimized to a plan a second
        `optimize_logical` would still shrink**. This round takes the total plan size from
        4,561 operator nodes to 4,491 and the non-idempotent count from 46 to 21, while
        making planning *faster* — 57.9 ms/query to 54.9 — because every phase after a
        smaller plan has less to walk.

        **Do not read those plan-size figures as a throughput claim.** An interleaved A/B on
        execution (12 TPC-DS queries, 7 alternating rounds, minimum of each) reads **-1.5%
        end to end**, with per-query noise of ±5% on a shared box — so the honest statement is
        that the round is free and slightly positive, not that it is a speed-up. A
        *sequential* A/B on the same queries read -5.4% with one query at -21%; that was
        page-cache ordering bias and did not survive interleaving. The reason to keep the
        round is the plan being *canonical*, which the numbers above do establish: the
        adaptive executor re-optimizes each stage subtree mid-query
        (`api/adaptive/`), and a plan that still shrinks under a second pass makes that
        re-optimization change the plan shape for reasons that have nothing to do with the
        measured cardinalities it is supposed to be reacting to.

        **Why here, and not one phase earlier or later.** Three placements were measured.
        After JOIN_REORDER only: 4,521 nodes, because FUSION then re-creates adjacency the
        round has already passed. After FUSION (this one): 4,491. Both: also 4,491, so the
        earlier round earns nothing once this one exists. And it must not move *later*:
        SELECTION's `split_expensive_filter` deliberately emits `Filter(Filter(...))` so an
        expensive predicate sees only the survivors of a cheap one, and
        `merge_adjacent_filters` would fuse it straight back — the ping-pong that rule's
        module docstring says its phase placement exists to prevent. After FUSION and before
        SELECTION is the one point where the plan's shape has settled and no cost-based
        decision has yet been written into it.

        **Why a declared subset and not a re-run of the rewrite phases.** Re-running
        NORMALIZE + REWRITE + PUSHDOWN wholesale is both slower *and worse*: on the same 99
        queries it cost **+18.6%** planning time and removed **41** nodes, against **-5%**
        and **70** here. Worse because those phases also hold rules that legitimately *grow*
        a plan, and re-running them partially undoes the first pass. Only a *contracting*
        rule is safe to run twice, which is what `Rule.recanonicalize` declares — on the rule
        itself, so the set cannot rot the way a name list in this module would.

        Args:
            plan: The plan as FUSION left it.
            ctx: The optimizer context every phase shares.
            fixpoint: The iteration bound, from `_fixpoint_bound`.
            present: The plan's node-type set, for the rule pattern index.

        Returns:
            The canonicalized plan, the IR this round computed (`None` when it changed
            nothing, meaning the caller keeps the IR it already had), and the refreshed
            node-type set.
        """
        if not self._cleanup:
            return plan, None, present
        plan, ir = _run_phase(plan, self._cleanup, ctx, fixpoint, present)
        return plan, ir, _present(plan) if ir is not None else present

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
            source_limits=required_limits_per_source(plan),
            source_orderings=required_orderings_per_source(plan),
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
#: Both executors parallelize this shape and both return the identical answer; what changes
#: is the CPU spent. Swept directly — one `Int64` key with `g` distinct values over 10 M rows
#: and one `SUM`, arms alternated so box drift cannot be read as a result (milliseconds,
#: best of four):
#:
#: | groups | streaming | materializing |
#: |---|---:|---:|
#: | 500 | **9.8** | 11.2 |
#: | 1,000 | **9.2** | 16.2 |
#: | 2,000 | **10.1** | 16.3 |
#: | 3,000 | **11.8** | 19.3 |
#: | **4,000** | 15.7 | **13.9** |
#: | 8,000 | 20.1 | **17.8** |
#: | 25,000 | 35.4 | **24.4** |
#: | 50,000 | 43.3 | **24.2** |
#: | 400,000 | 26.7 | **23.8** |
#: | 2,000,000 | 30.9 | **29.0** |
#:
#: The crossover sits between 3,000 and 4,000, and this is the first swept point on the far
#: side of it — the conservative choice, since it leaves every measured near-tie where it was.
#:
#: It used to be 50,000, chosen as the midpoint of a 1e4..1e5 bracket whose interior had not
#: been measured. What moved the crossover down an order of magnitude is
#: `bc_interp::agg_par::chunked_partials`: the materializing aggregate used to build one hash
#: table per 16,384-row morsel and hand the merge every one of them, which is ruinous in
#: exactly the band where the group count fills a morsel's table without filling a worker's
#: share. Building one table per worker instead took the 10,000-group case from 63.3 ms to
#: 41.1, and with that the executor that was losing this band now wins it.
MATERIALIZE_AGG_MIN_GROUPS = 4_000


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

    A `Sort` and a `Limit` are peeled with it, and that is the shape it matters most for:
    `GROUP BY k ORDER BY count(*) DESC LIMIT n` is how an analytics leaderboard query is
    written and the `Project`-only peel could not see any of them. Both operators read the
    aggregate's output — one row per group — so, exactly like the projection, they are small
    beside the aggregation whichever executor runs them.

    A `Filter` (`HAVING`) and a **global** `Aggregate` are peeled for the same reason and were
    the two remaining gaps. Both read one row per group, and leaving them out cost the whole
    difference between the two executors on plans that are entirely ordinary: measured on a
    24 M-row group-by producing 10 M groups, `select k, s` above the aggregate ran in **245 ms**
    and `filter(n > 0)` above the same aggregate in **2,116 ms** — same rows in, same
    10,072,152 rows out, and nothing between them but this predicate. `group_by(...).agg(...)`
    ending in a scalar roll-up is the other one, and it is how nearly every benchmark and
    `SELECT count(*) FROM (... GROUP BY k)` is written.

    The global aggregate is peeled *only* when it is global. A second **grouped** aggregate is
    its own aggregation over its own cardinality, so peeling through it would answer this
    question about the wrong node.

    **The group count is read off the `Aggregate`, not off the plan root**, which is what the
    added peels force: a `LIMIT 10` above the aggregate makes the root's estimate 10, so
    estimating the root would refuse every one of the queries this peel exists to admit —
    silently, and while looking like it had asked the right question.
    """
    from batcher.plan.logical import Aggregate, Filter, Join, Limit, Project, Sort

    node = plan
    while True:
        if isinstance(node, (Project, Sort, Limit, Filter)):
            node = node.input
        elif isinstance(node, Aggregate) and not node.group_keys:
            node = node.input  # a scalar roll-up over one row per group
        else:
            break
    if not isinstance(node, Aggregate) or not node.group_keys:
        return False
    if any(isinstance(n, Join) for n in walk(plan)):
        return False
    try:
        return ctx.estimator.estimate(node).rows >= MATERIALIZE_AGG_MIN_GROUPS
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
