"""Query profiles — the planned plan joined to the measured run, for `EXPLAIN`.

Batcher measures every operator (`bc-interp` emits `ExecMetrics`, keyed by a pre-order
`op_id`) and estimates every operator (Kyber's `PhysicalOp` carries the same `op_id`).
This package joins the two into one `QueryProfile` that renders as a Spark-style
`EXPLAIN` / `EXPLAIN ANALYZE` tree and serializes to JSON for the per-query event log.

Neutral by construction: it imports no subsystem (`kyber`/`carbonite`/`core`), only the
`plan` contract types — so `api`, `Dataset.stats()`, and the event-log writer all share
one renderer without crossing a layer boundary.
"""

from __future__ import annotations

from batcher.plan.profile.collect import (
    ProfileCollector,
    build_op_profiles,
    merge_metric_ops,
    worker_op_profiles,
)
from batcher.plan.profile.types import Decision, OpProfile, QueryProfile

__all__ = [
    "Decision",
    "OpProfile",
    "ProfileCollector",
    "QueryProfile",
    "build_op_profiles",
    "merge_metric_ops",
    "worker_op_profiles",
]
