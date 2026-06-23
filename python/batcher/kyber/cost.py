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

    def row_bytes(self, node: LogicalPlan) -> float:
        """Estimated average bytes per output row of `node` — the byte-true width
        the memory/IO axes need. Uses learned per-column widths, falling back to
        the flat `bytes_per_row` coefficient when nothing is measured yet (so
        cold-start cost is unchanged). Public so the SELECTION rule can size
        broadcast eligibility in bytes."""
        return self._est.row_width(node, self._c.bytes_per_row)

    def op_cost(self, node: LogicalPlan) -> Cost:
        """Cost of `node` itself, excluding its inputs."""
        c = self._c
        out_rows = self._rows(node)

        if isinstance(node, Scan):
            return Cost(cpu=c.scan_row * out_rows, io=self.row_bytes(node) * out_rows)

        if isinstance(node, Filter):
            in_rows = self._rows(node.input)
            return Cost(cpu=c.filter_row * in_rows)

        if isinstance(node, Project):
            return Cost(cpu=c.project_row * out_rows)

        if isinstance(node, MapBatches):
            # A GPU model forward pass is orders of magnitude costlier per row than a
            # trivial column map, and scales with model size. Costing it as the
            # bottleneck it is makes Kyber prefer to filter/sample *before* inference
            # (predicate pushdown below a map stage) — the key win for AI pipelines.
            factor = 1.0
            if node.num_gpus > 0:
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
            limit = node.limit
            # Top-N (fused limit) avoids a full sort: heap of size `limit`.
            sort_factor = math.log2(limit) if limit else math.log2(n)
            return Cost(
                cpu=c.sort_row * n * max(1.0, sort_factor),
                mem=self.row_bytes(node) * (limit if limit else n),
            )

        if isinstance(node, Join):
            build = self._rows(node.right)  # right is the build side by convention
            probe = self._rows(node.left)
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
