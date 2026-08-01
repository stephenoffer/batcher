"""Carbonite fault tolerance: surviving a fleet where nodes and devices fail.

Groups the resilience primitives the resource manager coordinates. They divide into two
halves that answer different questions.

Recovering from a loss that already happened: `ShuffleLineage` (how to regenerate a lost
output, and at what epoch), `ShuffleRecovery` (the policy-bounded recompute→retry loop), and
the preemption/deadline watchers that let a node drain before it is taken away.

Not repeating it: `classify` (what kind of failure this was, and therefore whether a retry
should happen at all and whether it must move), `FaultLedger` (which nodes and devices have
been failing, learned from outcomes rather than telemetry), and `RetryBudget` (a ceiling on
how much of a job may be spent retrying, so a systematically broken fleet fails fast with the
real error instead of slowly with the last one).

And two that make a failure *reportable* in the first place, which everything above depends
on: `preflight` (the node conditions that fail every task while every device reads healthy —
a scratch directory that cannot be written, a kernel that has already killed a worker here)
and `collectives` (the settings that turn a multi-GPU collective's indefinite hang into an
ordinary task failure, without which none of the machinery above ever runs).

Re-exports only; the logic lives in the sibling modules.
"""

from __future__ import annotations

from batcher.carbonite.resilience.blocklist import (
    FaultLedger,
    QuarantinePolicy,
    TargetHealth,
    default_ledger,
    reset_default_ledger,
)
from batcher.carbonite.resilience.budget import BudgetState, RetryBudget
from batcher.carbonite.resilience.classify import (
    CATEGORIES,
    FailureClass,
    classify_failure,
    failure_class,
    is_retryable,
    must_move,
    results_untrusted,
)
from batcher.carbonite.resilience.collectives import (
    STABILITY_VARS,
    collective_findings,
    stability_env,
)
from batcher.carbonite.resilience.lineage import ShuffleLineage, SourcePlacement
from batcher.carbonite.resilience.preemption import (
    PreemptionMonitor,
    preemption_monitor,
    termination_probe,
)
from batcher.carbonite.resilience.preflight import (
    CheckResult,
    PreflightReport,
    preflight_check,
)
from batcher.carbonite.resilience.recovery import RecoveryPolicy, ShuffleRecovery
from batcher.carbonite.resilience.replication import assign_replica_hosts
from batcher.carbonite.resilience.speculative import (
    STALL_WARN_AFTER_S,
    SpeculationPolicy,
    gather_with_backups,
    stragglers_to_backup,
    warn_barrier_stalled,
)

__all__ = [
    "CATEGORIES",
    "STABILITY_VARS",
    "STALL_WARN_AFTER_S",
    "BudgetState",
    "CheckResult",
    "FailureClass",
    "FaultLedger",
    "PreemptionMonitor",
    "PreflightReport",
    "QuarantinePolicy",
    "RecoveryPolicy",
    "RetryBudget",
    "ShuffleLineage",
    "ShuffleRecovery",
    "SourcePlacement",
    "SpeculationPolicy",
    "TargetHealth",
    "assign_replica_hosts",
    "classify_failure",
    "collective_findings",
    "default_ledger",
    "failure_class",
    "gather_with_backups",
    "is_retryable",
    "must_move",
    "preemption_monitor",
    "preflight_check",
    "reset_default_ledger",
    "results_untrusted",
    "stability_env",
    "stragglers_to_backup",
    "termination_probe",
    "warn_barrier_stalled",
]
