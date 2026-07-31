"""Carbonite fault tolerance: Spark-style recompute-from-lineage on worker loss.

Groups the resilience primitives the resource manager coordinates — `ShuffleLineage`
(how to regenerate a lost output, and at what epoch), `ShuffleRecovery` (the
policy-bounded recompute→retry loop), and the preemption/deadline watchers that let a
node drain before it is taken away. Re-exports only; the logic lives in the sibling
modules.
"""

from __future__ import annotations

from batcher.carbonite.resilience.lineage import ShuffleLineage, SourcePlacement
from batcher.carbonite.resilience.preemption import (
    PreemptionMonitor,
    preemption_monitor,
    termination_probe,
)
from batcher.carbonite.resilience.recovery import RecoveryPolicy, ShuffleRecovery
from batcher.carbonite.resilience.replication import assign_replica_hosts
from batcher.carbonite.resilience.speculative import (
    SpeculationPolicy,
    gather_with_backups,
    stragglers_to_backup,
)

__all__ = [
    "PreemptionMonitor",
    "RecoveryPolicy",
    "ShuffleLineage",
    "ShuffleRecovery",
    "SourcePlacement",
    "SpeculationPolicy",
    "assign_replica_hosts",
    "gather_with_backups",
    "preemption_monitor",
    "stragglers_to_backup",
    "termination_probe",
]
