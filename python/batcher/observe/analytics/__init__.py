"""Cross-run analysis — percentiles, trends, rollups, comparisons, and a health verdict.

The store answers "what happened in this run". This package answers the questions that only
exist once there are *many* runs: is this pipeline getting slower, which operator kind costs
the session the most, what changed between these two runs, and is anything wrong right now.

Pure functions over the store's records, computed on request. None of it is cached: the
inputs are a few hundred small dicts, the arithmetic is microseconds, and a cache would add
an invalidation bug in exchange for nothing measurable.
"""

from __future__ import annotations

from .comparison import compare_runs
from .health import health_report
from .pipeline import pipeline_report
from .rollups import failure_groups, operator_rollup
from .series import percentiles, throughput_series

__all__ = [
    "compare_runs",
    "failure_groups",
    "health_report",
    "operator_rollup",
    "percentiles",
    "pipeline_report",
    "throughput_series",
]
