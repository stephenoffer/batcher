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

from batcher.kyber.column_tables import (
    AVG_BYTES_KEY,
    MCV_KEY,
    NDV_KEY,
    QUANTILES_KEY,
    columns_for,
)
from batcher.kyber.correction import estimate_is_reliable
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
from batcher.kyber.metadata_filter_count import (
    answer_filter_any,
    answer_filter_count,
    answer_filter_is_empty,
)
from batcher.kyber.metadata_summary import answer_column_summary
from batcher.kyber.optimizer import (
    Optimizer,
    optimize,
    optimize_full,
    optimize_logical,
    optimize_traced,
)
from batcher.kyber.rules.projections import required_columns_per_source
from batcher.kyber.stats import hot_join_values

__all__ = [
    "AVG_BYTES_KEY",
    "MCV_KEY",
    "NDV_KEY",
    "QUANTILES_KEY",
    "Optimizer",
    "answer_aggregate",
    "answer_column_summary",
    "answer_count",
    "answer_filter_any",
    "answer_filter_count",
    "answer_filter_is_empty",
    "answer_is_empty",
    "approx_count_distinct",
    "columns_for",
    "estimate_is_reliable",
    "hot_join_values",
    "load_learned_stats",
    "optimize",
    "optimize_full",
    "optimize_logical",
    "optimize_traced",
    "record_column_stats",
    "record_execution",
    "record_selectivity",
    "required_columns_per_source",
]
