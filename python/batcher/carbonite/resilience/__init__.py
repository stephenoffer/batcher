"""Carbonite fault tolerance: Spark-style recompute-from-lineage on worker loss.

Groups the resilience primitives the resource manager coordinates — `ShuffleLineage`
(how to regenerate a lost output, and at what epoch) and `ShuffleRecovery` (the
policy-bounded recompute→retry loop). Re-exports only; the logic lives in the
sibling modules.
"""

from __future__ import annotations

from batcher.carbonite.resilience.lineage import ShuffleLineage
from batcher.carbonite.resilience.preemption import (
    PreemptionMonitor,
    preemption_monitor,
)
from batcher.carbonite.resilience.recovery import RecoveryPolicy, ShuffleRecovery
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
    "SpeculationPolicy",
    "gather_with_backups",
    "preemption_monitor",
    "stragglers_to_backup",
]
