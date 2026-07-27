"""Cost model — what will this plan *cost* to run?

Cardinality answers "how many rows"; cost turns rows into a comparable estimate
of work along four axes: **cpu** (compute), **mem** (peak working set, the spill
signal), **io** (bytes scanned/written), **net** (bytes shuffled). Passes that
choose between alternatives — join order, join strategy, whether to spill — pick
the lower-cost plan; WS9 SLA targets reweight the axes into one objective.

The model is deliberately lean: per-operator closed forms over estimated row
counts, with coefficients that are plain constants now and get *calibrated* from
measured `op_stats` later (Core collects, the learning loop corrects). No ML, no
per-rule cost subsystem — one `CostModel.cost(node)` folding the estimator over
the tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# `CostCoefficients` / `CostWeights` are defined once in `config` (the single source
# of truth for tunables) and re-exported here so the cost model's public surface is
# unchanged.
from batcher._internal.hardware import l3_cache_bytes
from batcher.config import CostCoefficients, CostWeights, active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.expr_cost import expr_cost, expr_cost_factor
from batcher.kyber.stats.estimator import combine_ndv
from batcher.kyber.storage_cost import spill_device_factor
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    Scan,
    Sort,
    Union,
    Unnest,
    Window,
)
from batcher.plan.visitor import children

__all__ = ["Cost", "CostCoefficients", "CostModel", "CostWeights"]

# A GPU `map_batches` (model forward pass) is ~100x the per-row cost of a trivial
# column map; the factor is scaled by `(1 + model_memory_gb)` so larger models cost
# more. This makes Kyber treat inference as the pipeline bottleneck and minimize the
# rows reaching it (pushing filters/sampling below the stage).
_GPU_INFERENCE_FACTOR = 100.0

# How much more a random hash-table access costs once the table no longer fits in the last
# level of cache, per octave of overflow.
#
# A cost model that charges one `hash_probe_row` per probe regardless of the table's size
# says a 1,000-row build side and a 100-million-row build side cost the same to probe. They
# do not: the first lives in L1 and the second takes a DRAM round trip on essentially every
# probe, which is one to two orders of magnitude slower. That difference is precisely what
# join *ordering* is choosing between — whether to build the small side or the large one, and
# whether an intermediate should be materialized at all — so leaving it out of the model
# leaves the enumerator ranking plans by a quantity that ignores the dominant term.
#
# The penalty is charged per doubling of `build_bytes / cache_bytes` and capped, which
# reproduces the measured shape: flat while resident, a steep knee at the cache boundary, then
# a plateau once every access already misses and there is nothing left to lose.
_CACHE_MISS_PENALTY_PER_OCTAVE = 0.35
_CACHE_MISS_MAX_FACTOR = 8.0

# Sequential bytes moved per unit of `io` cost relative to a row of CPU work. Spilling is
# charged in the same units as everything else so the axes stay comparable; the constant only
# has to place a spilled byte on the same scale as a row of compute.
_SPILL_WRITE_READ_PASSES = 2.0

# Buffer reserved per input run during an external merge. The merge fan-in is the memory
# budget divided by this, so it is what decides whether an over-budget sort needs one merge
# pass or several.
_EXTERNAL_MERGE_RUN_BUFFER_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class Cost:
    """A four-axis cost estimate. Axes are kept separate so an SLA objective can
    weight them (latency-bound vs cost-bound) instead of collapsing too early."""

    cpu: float = 0.0
    mem: float = 0.0
    io: float = 0.0
    net: float = 0.0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            self.cpu + other.cpu,
            self.mem + other.mem,
            self.io + other.io,
            self.net + other.net,
        )

    def total(self, weights: CostWeights | None = None) -> float:
        """Collapse to a single comparable scalar. `mem` is a peak (a max along the
        tree), so it is *not* summed into the scalar here — it gates feasibility,
        not throughput. Default weights treat cpu/io/net as comparable units."""
        w = weights or active_config().optimizer.cost_weights
        return w.cpu * self.cpu + w.io * self.io + w.net * self.net


class CostModel:
    """Estimates the cost of a plan, consuming a `CardinalityEstimator` for sizes."""

    def __init__(
        self,
        estimator: CardinalityEstimator,
        coeffs: CostCoefficients | None = None,
    ) -> None:
        self._est = estimator
        self._c = coeffs or active_config().optimizer.cost_coeffs
        self._cost_cache: dict[int, tuple[LogicalPlan, Cost]] = {}

    def _rows(self, node: LogicalPlan) -> float:
        return self._est.estimate(node).rows

    def _memory_budget(self) -> float:
        """The working-set budget an operator has before it must spill, in bytes.

        The same static envelope the data plane is given, so the cost model's idea of "this
        will spill" is the engine's. `0` means the user opted out of bounded memory entirely,
        in which case nothing spills and the spill terms vanish.
        """
        return float(active_config().spill_budget_bytes())

    def _cache_factor(self, state_bytes: float) -> float:
        """The per-access slowdown of a hash table of `state_bytes`, from cache residency.

        `1.0` while the table fits in the last-level cache, then growing by
        `_CACHE_MISS_PENALTY_PER_OCTAVE` per doubling beyond it and flattening at
        `_CACHE_MISS_MAX_FACTOR` — the shape a random-access probe actually has. With no
        detectable cache size the factor is 1.0, which is exactly the previous model.
        """
        cache = float(l3_cache_bytes())
        if cache <= 0.0 or state_bytes <= cache:
            return 1.0
        octaves = math.log2(state_bytes / cache)
        return min(_CACHE_MISS_MAX_FACTOR, 1.0 + _CACHE_MISS_PENALTY_PER_OCTAVE * octaves)

    def _spill_io(self, state_bytes: float) -> float:
        """Bytes of spill IO an operator whose state is `state_bytes` will move.

        Zero while the state fits the memory budget. Past it, everything that does not fit is
        written once and read back once. Charging nothing for it (the previous model) makes a
        plan that spills look exactly as cheap as one that does not, which is the single
        largest cost error a plan can contain — a spilled operator is disk-bound, and the
        optimizer's whole reason to prefer a smaller build side is to avoid it.

        Scaled by the measured class of the spill device, so the same overflow is costed as
        the cheap thing it is on local flash and the expensive thing it is on a network volume.
        """
        budget = self._memory_budget()
        if budget <= 0.0 or state_bytes <= budget:
            return 0.0
        return _SPILL_WRITE_READ_PASSES * (state_bytes - budget) * spill_device_factor()

    def _merge_passes(self, state_bytes: float) -> float:
        """External-merge passes an out-of-core sort of `state_bytes` needs.

        A sort that does not fit runs `ceil(log_F(state/budget))` merge passes, each of which
        rewrites the whole run — so its IO grows with the *logarithm* of the overflow, not
        linearly. `F` (the merge fan-in) is the budget divided by one run buffer, and is large
        enough in practice that a single pass covers almost everything; the formula is here so
        that the one case where it does not (a sort many times the budget) is not costed as if
        it were.
        """
        budget = self._memory_budget()
        if budget <= 0.0 or state_bytes <= budget:
            return 0.0
        runs = state_bytes / budget
        fan_in = max(2.0, budget / _EXTERNAL_MERGE_RUN_BUFFER_BYTES)
        return max(1.0, math.ceil(math.log(runs, fan_in)))

    def expr_factor(self, node: LogicalPlan) -> float:
        """The per-row expression-cost multiplier for `node`'s own work.

        A `Filter` pays for its predicate; a `Project` pays for every column it computes
        (ten regexes cost ten regexes). Every other operator's per-row work is structural,
        not expression-driven, so it carries no multiplier.

        Priced at the *measured* JIT speedup, so an expression the Cranelift tier compiles
        is charged what that tier actually costs. Public because Core stamps this factor
        onto each operator's feedback: `calibration` divides it back out, which is what
        keeps the fitted `filter_row` / `project_row` coefficients a property of the
        engine rather than of whichever expressions a workload happened to contain.

        Args:
            node: The plan node whose own per-row work is being priced.

        Returns:
            A multiplier, 1.0 for operators with no expression cost.
        """
        speedup = self._c.jit_speedup
        if isinstance(node, Filter):
            return expr_cost_factor(node.predicate, speedup)
        if isinstance(node, Project):
            return sum(expr_cost_factor(item.expr, speedup) for item in node.items)
        return 1.0

    def expr_cost(self, expr) -> float:
        """Per-row cost of a scalar expression, at the measured JIT speedup.

        The seam cost-based rules use so they price expressions exactly as the cost model
        does, instead of importing the module-level default speedup.

        Args:
            expr: The scalar expression to price.

        Returns:
            Cost in work-units, where an interpreted numeric comparison is 1.0.
        """
        return expr_cost(expr, self._c.jit_speedup)

    def row_bytes(self, node: LogicalPlan) -> float:
        """Estimated average bytes per output row of `node` — the byte-true width
        the memory/IO axes need. Uses learned per-column widths, falling back to
        the flat `bytes_per_row` coefficient when nothing is measured yet (so
        cold-start cost is unchanged). Public so the SELECTION rule can size
        broadcast eligibility in bytes."""
        return self._est.row_width(node, self._c.bytes_per_row)

    def join_op_cost(self, node: Join) -> Cost:
        """A join's own cost at the orientation the plan will *actually* run.

        `op_cost` prices a join as written — build on the right — because that is exactly
        what the build-side rule needs: it compares the two orientations against each
        other. But that rule runs in SELECTION, *after* JOIN_REORDER, so the reorder was
        ranking orders by the cost of an orientation the physical plan would then flip.
        With `hash_build_row` (2.0) twice `hash_probe_row` (1.0), that is not a rounding
        error: it penalizes every order that happens to put the large table on the right,
        even though SELECTION would have swapped it.

        An inner join is commutative, so its cost is the cheaper of the two build sides —
        which is the one SELECTION will pick. A non-inner join is not commutative and its
        build side is fixed, so it is priced as written.

        Args:
            node: The join to price.

        Returns:
            The join's own cost, excluding its inputs.
        """
        base = self.op_cost(node)
        if node.join_type != "inner":
            return base
        c = self._c
        left, right = self._rows(node.left), self._rows(node.right)
        swapped_cpu = c.hash_build_row * left + c.hash_probe_row * right
        as_written_cpu = c.hash_build_row * right + c.hash_probe_row * left
        if swapped_cpu >= as_written_cpu:
            return base
        return replace(
            base,
            cpu=base.cpu - as_written_cpu + swapped_cpu,
            mem=self.row_bytes(node.left) * left,
        )

    def op_cost(self, node: LogicalPlan) -> Cost:
        """Cost of `node` itself, excluding its inputs.

        A `Join` is priced **as written** (build on the right). Join *ordering* should use
        `join_op_cost`, which prices the orientation SELECTION will choose; the build-side
        rule needs this one, since it compares the orientations against each other.
        """
        c = self._c
        out_rows = self._rows(node)

        if isinstance(node, Scan):
            return Cost(cpu=c.scan_row * out_rows, io=self.row_bytes(node) * out_rows)

        if isinstance(node, Filter):
            in_rows = self._rows(node.input)
            # A predicate is not a unit of work: `x > 5` compiles to a vector compare,
            # `regexp_matches(s, ...)` runs an automaton per row. Scaling by the
            # expression's relative cost is what lets the optimizer prefer evaluating
            # the cheap, selective conjunct first (see `split_expensive_filter`).
            return Cost(cpu=c.filter_row * in_rows * self.expr_factor(node))

        if isinstance(node, Project):
            # A projection's cost is the sum over the columns it computes: ten regexes
            # cost ten regexes, not one `project_row`.
            return Cost(cpu=c.project_row * out_rows * self.expr_factor(node))

        if isinstance(node, MapBatches):
            # A GPU model forward pass is orders of magnitude costlier per row than a
            # trivial column map, and scales with model size. Costing it as the
            # bottleneck it is makes Kyber prefer to filter/sample *before* inference
            # (predicate pushdown below a map stage) — the key win for AI pipelines.
            # Any accelerator, not just a GPU. Ray reports NVIDIA/AMD/Intel/MetaX as the `GPU`
            # resource; a TPU, Trainium, or Gaudi stage carries `num_gpus == 0` plus a custom
            # resource instead. Gating on `num_gpus` alone therefore costed those stages as a
            # *trivial column map* — the cheapest node in the plan — so Kyber had no reason to
            # push a filter below them, losing exactly the optimization this factor exists to
            # produce on precisely the hardware whose forward pass is most expensive.
            factor = 1.0
            if node.num_gpus > 0 or getattr(node, "resources", ()):
                factor = _GPU_INFERENCE_FACTOR * (1.0 + node.model_memory_gb)
            return Cost(cpu=c.map_row * out_rows * factor)

        if isinstance(node, Aggregate):
            in_rows = self._rows(node.input)
            # Hash-aggregate: build over the input; state size ~ number of groups. Every input
            # row performs one random access into that state, so the table's cache residency
            # multiplies the whole build term — a two-group aggregate and a 200-million-group
            # aggregate are not the same operator, and only the second is memory-bound.
            state_bytes = self.row_bytes(node) * out_rows
            return Cost(
                cpu=c.hash_build_row * in_rows * self._cache_factor(state_bytes)
                + c.output_row * out_rows,
                mem=state_bytes,
                io=self._spill_io(state_bytes),
            )

        if isinstance(node, Sort):
            n = max(1.0, self._rows(node.input))
            heap = min(node.limit, n) if node.limit else n
            state_bytes = self.row_bytes(node) * heap
            return Cost(
                cpu=c.sort_row * _sort_comparisons(n, heap),
                mem=state_bytes,
                # An out-of-core sort rewrites its runs once per merge pass, and the pass count
                # grows only logarithmically in how far over budget it is.
                io=(
                    _SPILL_WRITE_READ_PASSES
                    * state_bytes
                    * self._merge_passes(state_bytes)
                    * spill_device_factor()
                ),
            )

        if isinstance(node, Join):
            build = self._rows(node.right)  # right is the build side by convention
            probe = self._rows(node.left)
            # NOTE: the output term is deliberately **row**-based, not `rows x width`.
            #
            # The gather really does cost `rows x width` (measured: one join at fixed
            # cardinality takes 12.1 ms carrying two payload columns and 19.8 ms carrying
            # seven), so charging for width looks obviously right — and it was tried. It made
            # TPC-H *worse*: 1.47x -> 1.58x against DuckDB in the same run, because
            # `row_bytes` of an intermediate is itself an estimate, and feeding that
            # uncertainty into the join enumerator's ranking moved more plans the wrong way
            # than the right way. The width signal is real but the width *estimate* is not
            # yet good enough to rank on; it belongs here once intermediate widths are
            # measured rather than inferred. Recorded so it is not "fixed" again blind.
            build_bytes = self.row_bytes(node.right) * build
            # Every probe row makes one random access into the build-side hash table, so the
            # table's cache residency multiplies the probe term. This is the term that
            # distinguishes the two orientations of a join between a tiny dimension and a huge
            # fact table by more than their row counts, and it is what makes the enumerator
            # prefer keeping intermediates small rather than merely keeping row counts down.
            probe_factor = self._cache_factor(build_bytes)
            return Cost(
                cpu=c.hash_build_row * build
                + c.hash_probe_row * probe * probe_factor
                + c.output_row * out_rows,
                # Hash table is built over the right side, so its byte width drives mem.
                mem=build_bytes,
                # A build side past the memory budget partitions to disk and reads both sides
                # back — the grace-hash fallback. Costing it at zero made a plan that spills
                # look identical to one that does not.
                io=self._spill_io(build_bytes),
            )

        if isinstance(node, Distinct):
            in_rows = self._rows(node.input)
            state_bytes = self.row_bytes(node) * out_rows
            return Cost(
                cpu=c.distinct_row * in_rows * self._cache_factor(state_bytes),
                mem=state_bytes,
                io=self._spill_io(state_bytes),
            )

        if isinstance(node, Window):
            in_rows = self._rows(node.input)
            # Partition + order ≈ a sort, but only *within* each partition: `p` partitions of
            # `n/p` rows cost `n·log2(n/p)` comparisons, not `n·log2(n)`. The difference is the
            # whole point of partitioning — a `PARTITION BY user_id` over a million users sorts
            # runs of a handful of rows each, and costing it as one global sort made the
            # optimizer treat the cheapest window in the workload as its most expensive
            # operator.
            per_partition = max(1.0, in_rows / self._window_partitions(node, in_rows))
            return Cost(
                cpu=c.sort_row * in_rows * max(1.0, math.log2(max(2.0, per_partition))),
                mem=self.row_bytes(node) * in_rows,
            )

        if isinstance(node, Unnest):
            # Explode is stateless and row-wise, but it *emits* the fanned-out rows: the
            # cost is per output row, not per input row. Costing it at zero would let
            # Kyber move an explode below a filter or ahead of an inference stage, where
            # it multiplies the rows that stage must pay for.
            return Cost(cpu=c.project_row * out_rows)

        if isinstance(node, Union):
            if not node.distinct:
                return Cost(cpu=c.union_row * out_rows)
            # `UNION` (as opposed to `UNION ALL`) deduplicates, which is a hash build over
            # every concatenated row — the same work a `Distinct` does, and nothing like the
            # streaming concatenation it was priced as.
            in_rows = sum(self._rows(i) for i in node.inputs)
            state_bytes = self.row_bytes(node) * out_rows
            return Cost(
                cpu=c.union_row * in_rows
                + c.distinct_row * in_rows * self._cache_factor(state_bytes),
                mem=state_bytes,
                io=self._spill_io(state_bytes),
            )

        if isinstance(node, Limit):
            return Cost(cpu=c.project_row * out_rows)

        return Cost()

    def _window_partitions(self, node: Window, in_rows: float) -> float:
        """How many partitions a window's `PARTITION BY` cuts its input into.

        Read from the estimator's propagated distinct counts, through the same damped
        combiner every other distinct-combination question uses, so the cost model and the
        cardinality estimator cannot disagree about how many partitions there are. Falls back
        to a single partition when the keys' distinct counts are unmeasured — the conservative
        direction, since that reproduces the previous global-sort cost.
        """
        if not node.partition_keys:
            return 1.0
        stats = self._est.estimate(node.input)
        per_key = []
        for key in node.partition_keys:
            stat = stats.columns.get(key.name) if isinstance(key, Col) else None
            if stat is None or not stat.ndv or stat.ndv <= 0:
                return 1.0
            per_key.append(float(stat.ndv))
        return max(1.0, min(combine_ndv(per_key, in_rows), in_rows))

    def cost(self, node: LogicalPlan) -> Cost:
        """Total cost of the subtree rooted at `node` (this op + all inputs).

        `mem` accumulates as the max single-operator peak rather than the sum:
        breakers run at different times, so peak memory is the tallest, not the
        total. cpu/io/net sum (they're throughput work).

        Memoized by node identity for this model's lifetime, exactly as the estimator memoizes
        its own answers and for the same reason: a plan is immutable while it is being
        optimized, but every cost-based rule asks for the cost of overlapping subtrees, so an
        un-memoized recursion re-walks the whole tree once per node and makes planning
        quadratic in plan size. The entry holds a strong reference to its keyed node so a
        freed node's reused `id()` cannot produce a stale hit.
        """
        cached = self._cost_cache.get(id(node))
        if cached is not None and cached[0] is node:
            return cached[1]
        result = self._cost_uncached(node)
        self._cost_cache[id(node)] = (node, result)
        return result

    def _cost_uncached(self, node: LogicalPlan) -> Cost:
        own = self.op_cost(node)
        child_costs = [self.cost(child) for child in children(node)]
        summed = own
        peak_mem = own.mem
        for cc in child_costs:
            summed = summed + cc
            peak_mem = max(peak_mem, cc.mem)
        return replace(summed, mem=peak_mem)


def _sort_comparisons(n: float, heap: float) -> float:
    """Comparisons a sort of `n` rows keeping `heap` of them performs.

    A **full** sort is the textbook `n·log2(n)`.

    A **top-N** (a fused `Sort` + `Limit`) is not `n·log2(k)`, which is what charging every
    row a heap sift-down assumes. Every row is compared once against the heap's root, but only
    a row that beats it is inserted — and over a randomly ordered input the expected number of
    such rows is `k·(1 + ln(n/k))`, because the `i`-th row displaces the root only if it lands
    in the running top `k`, with probability `min(1, k/i)`. So the real cost is `n` root
    comparisons plus `k·(1 + ln(n/k))` sift-downs of `log2(k)` each.

    The difference is the whole reason to fuse a limit into a sort: for `n = 10^8, k = 10`,
    `n·log2(k)` charges 3.3x the input while the true cost is a little over one pass. Costing
    top-N as a discounted full sort made the optimizer nearly indifferent to the fusion, and
    over-charged a `LIMIT 10` over a large scan by more than three times.
    """
    if heap >= n:
        return n * math.log2(max(2.0, n))
    insertions = heap * (1.0 + math.log(n / heap))
    return n + insertions * math.log2(max(2.0, heap))
