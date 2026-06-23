"""Back-compat shim — cardinality estimation moved to `kyber.stats`.

`StatsEstimator` (in `batcher.kyber.stats`) supersedes the old
`CardinalityEstimator`, returning the richer `RelStats` (row count + per-column
`ColumnStat` with `Provenance`) instead of the old rows-only `Estimate`. This
module preserves the historical import path; new code should import from
`batcher.kyber.stats` and `batcher.plan.stats` directly.
"""

from __future__ import annotations

from batcher.kyber.stats import StatsEstimator
from batcher.plan.stats import RelStats

# Historical names. `Estimate` was a rows-only record; `RelStats` is its superset
# (same `.rows`/`.provenance` surface, plus column stats), so it serves both.
CardinalityEstimator = StatsEstimator
Estimate = RelStats

__all__ = ["CardinalityEstimator", "Estimate"]
