"""Kyber — the query optimizer. **Optimization and planning only.**

Responsibility boundary (enforced by the layer-import contract):
  * Kyber turns a `LogicalPlan` into a `PhysicalPlan`: it runs the pass pipeline,
    estimates cardinality/cost from sketches + learned corrections, orders joins,
    selects algorithms/backends, and annotates each operator with `ResourceBounds`.
  * Kyber does NOT manage memory or move data (that is Carbonite), and it does
    NOT execute anything (that is Core). It may read the MetadataHub for learned
    state and consume Carbonite's `FeasibilityVerdict`, but it never imports
    `carbonite` or `core`.

The bootstrap optimizer is an identity lowering (logical → IR); the pass pipeline,
estimators, and join ordering land on top of this seam.
"""

from __future__ import annotations

from batcher.kyber.learning import (
    load_learned_stats,
    record_column_stats,
    record_execution,
    record_selectivity,
)
from batcher.kyber.metadata_answer import (
    answer_aggregate,
    answer_count,
    answer_is_empty,
    approx_count_distinct,
)
from batcher.kyber.optimizer import Optimizer, optimize, optimize_traced

__all__ = [
    "Optimizer",
    "answer_aggregate",
    "answer_count",
    "answer_is_empty",
    "approx_count_distinct",
    "load_learned_stats",
    "optimize",
    "optimize_traced",
    "record_column_stats",
    "record_execution",
    "record_selectivity",
]
