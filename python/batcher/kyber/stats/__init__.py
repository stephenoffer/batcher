"""`kyber.stats` — cardinality + column-statistics estimation.

`StatsEstimator.estimate(node)` propagates a `RelStats` (row count + per-column
`ColumnStat`, each with `Provenance`) through a logical plan. Row logic lives in
`estimator`; column-stat propagation in `columns`; predicate selectivity in
`selectivity`. The neutral stat types themselves live in `batcher.plan.stats`.
"""

from __future__ import annotations

from batcher.kyber.stats.estimator import StatsEstimator
from batcher.kyber.stats.skew import hot_join_values

__all__ = ["StatsEstimator", "hot_join_values"]
