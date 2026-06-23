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
