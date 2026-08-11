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
    walk_ir,
    worker_op_profiles,
)
from batcher.plan.profile.stages import (
    StageRecorder,
    logical_op_ids,
    logical_preorder,
    metered,
    stage_kind,
)
from batcher.plan.profile.types import Decision, OpProfile, QueryProfile, QueryUsage
from batcher.plan.profile.usage import UsageStopwatch

__all__ = [
    "Decision",
    "OpProfile",
    "ProfileCollector",
    "QueryProfile",
    "QueryUsage",
    "StageRecorder",
    "UsageStopwatch",
    "build_op_profiles",
    "logical_op_ids",
    "logical_preorder",
    "merge_metric_ops",
    "metered",
    "stage_kind",
    "walk_ir",
    "worker_op_profiles",
]
