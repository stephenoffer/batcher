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
from batcher.config import CostCoefficients, CostWeights, active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.expr_cost import expr_cost, expr_cost_factor
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

    def _rows(self, node: LogicalPlan) -> float:
        return self._est.estimate(node).rows

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
            # Hash-aggregate: build over the input; state size ~ number of groups.
            return Cost(
                cpu=c.hash_build_row * in_rows + c.output_row * out_rows,
                mem=self.row_bytes(node) * out_rows,
            )

        if isinstance(node, Sort):
            n = max(1.0, self._rows(node.input))
            # Top-N (fused limit) avoids a full sort: one pass over `n` maintaining a heap
            # of size `min(limit, n)`. The heap can never hold more rows than exist, so a
            # `LIMIT` larger than the input degenerates to a full sort — it must not be
            # costed *above* one, which `log2(limit)` did whenever `limit > n`.
            heap = min(node.limit, n) if node.limit else n
            sort_factor = math.log2(max(2.0, heap))
            return Cost(
                cpu=c.sort_row * n * sort_factor,
                mem=self.row_bytes(node) * heap,
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
            return Cost(
                cpu=c.hash_build_row * build + c.hash_probe_row * probe + c.output_row * out_rows,
                # Hash table is built over the right side, so its byte width drives mem.
                mem=self.row_bytes(node.right) * build,
            )

        if isinstance(node, Distinct):
            in_rows = self._rows(node.input)
            return Cost(
                cpu=c.distinct_row * in_rows,
                mem=self.row_bytes(node) * out_rows,
            )

        if isinstance(node, Window):
            in_rows = self._rows(node.input)
            # Partition + order ≈ a sort over the input.
            return Cost(
                cpu=c.sort_row * in_rows * max(1.0, math.log2(max(1.0, in_rows))),
                mem=self.row_bytes(node) * in_rows,
            )

        if isinstance(node, Unnest):
            # Explode is stateless and row-wise, but it *emits* the fanned-out rows: the
            # cost is per output row, not per input row. Costing it at zero would let
            # Kyber move an explode below a filter or ahead of an inference stage, where
            # it multiplies the rows that stage must pay for.
            return Cost(cpu=c.project_row * out_rows)

        if isinstance(node, Union):
            return Cost(cpu=c.union_row * out_rows)

        if isinstance(node, Limit):
            return Cost(cpu=c.project_row * out_rows)

        return Cost()

    def cost(self, node: LogicalPlan) -> Cost:
        """Total cost of the subtree rooted at `node` (this op + all inputs).

        `mem` accumulates as the max single-operator peak rather than the sum:
        breakers run at different times, so peak memory is the tallest, not the
        total. cpu/io/net sum (they're throughput work)."""
        own = self.op_cost(node)
        child_costs = [self.cost(child) for child in children(node)]
        summed = own
        peak_mem = own.mem
        for cc in child_costs:
            summed = summed + cc
            peak_mem = max(peak_mem, cc.mem)
        return replace(summed, mem=peak_mem)
