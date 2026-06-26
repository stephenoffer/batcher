"""`PhysicalPlan` — what Kyber emits and Core executes.

A physical plan is the relational IR (ready to ship to the engine) plus the
metadata Kyber attaches: per-operator resource bounds and cardinality/cost
estimates tagged with their *provenance* (how much to trust them). Carbonite and
the adaptive controller read provenance to decide how defensively to budget and
how eagerly to re-optimize when reality diverges.

The bootstrap `PhysicalPlan` carries the lowered IR document directly; a richer
per-operator `PhysicalOp` DAG is filled in as the optimizer and runtime grow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from batcher.plan.ids import OpId
from batcher.plan.resource import ResourceBounds
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["PhysicalOp", "PhysicalPlan", "PlanProperties", "Provenance"]


@dataclass(frozen=True, slots=True)
class PlanProperties:
    """Estimated properties of an operator's output.

    `provenance` is the unified trust scale from `plan.stats`; `column_stats`
    carries the per-column statistics the estimator propagated, so Carbonite and
    the adaptive controller see not just *how many* rows but *what* the columns
    look like and how much to trust it.
    """

    est_rows: float = float("nan")
    row_size: float = float("nan")
    confidence: float = 0.0
    provenance: Provenance = Provenance.DEFAULT
    column_stats: Mapping[str, ColumnStat] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhysicalOp:
    """A physical operator: a relational op bound to a backend/algorithm + bounds."""

    op_id: OpId
    kind: str
    backend: str
    algorithm: str
    bounds: ResourceBounds
    inputs: tuple[OpId, ...]
    properties: PlanProperties = PlanProperties()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    """An executable plan: the lowered IR plus Kyber's annotations."""

    ir: dict[str, Any]
    output_schema: SchemaRef | None
    ops: tuple[PhysicalOp, ...] = ()
    # Per scan `source_id`, the column projection to read (projection pushdown).
    # Empty/absent means "read all columns".
    source_projections: dict[int, list[str]] = field(default_factory=dict)
    # Per scan `source_id`, the predicate IR of a `Filter` directly above it
    # (predicate pushdown). A pushdown-capable source translates the pushable
    # subset to its backend filter to skip I/O; the engine keeps the `Filter`
    # operator as a safe re-check, so an absent/partial translation is correct.
    source_predicates: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the relational IR for the engine."""
        return json.dumps(self.ir)

    def op_budgets(self) -> dict[int, int]:
        """Per-operator spill budgets (bytes) keyed by pre-order `op_id`.

        Kyber sizes each stateful operator's peak memory envelope
        (`ResourceBounds.m_max_bytes`); this surfaces those bounds as the side map
        Core ships to the engine so the data plane budgets each operator
        individually instead of applying one global `memory_budget_bytes` to every
        operator. Only positively-sized operators are included — an absent entry
        means "fall back to the global budget", which is exactly the behaviour for
        unsized (streaming/unknown) operators that Kyber leaves at `m_max_bytes=0`.
        """
        return {
            int(op.op_id): op.bounds.m_max_bytes for op in self.ops if op.bounds.m_max_bytes > 0
        }
