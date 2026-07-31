"""Tunables for surviving an unstable fleet — quarantine and retry budgets.

The rest of the distributed section tunes a cluster that mostly works. This one tunes what
happens when it does not: which nodes to stop using, and how much of a job may be spent
retrying before the failure it keeps hitting is raised instead.

Every default is chosen so that turning nothing on changes nothing. The quarantine thresholds
are permissive enough that a healthy fleet never trips them, and the retry budget is generous
enough that only a systematically broken run exhausts it. A deployment that configures none of
this behaves as it did before any of it existed, which is what makes the section safe to ship
enabled.

Kept beside `config.py` rather than inside it, following `accelerator.py`: that module is
already at its size limit, and these tunables carry their own range checks which would push
`validation/sections.py` over its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.errors import ConfigError

__all__ = [
    "FaultToleranceConfig",
    "QuarantineConfig",
    "validate_fault_tolerance",
]


@dataclass(frozen=True, slots=True)
class QuarantineConfig:
    """When repeated task failures take a node or device out of rotation.

    Examples:
        .. doctest::

            >>> from batcher.config import QuarantineConfig
            >>> QuarantineConfig().enabled
            True
    """

    #: Learn which nodes and devices are bad from task outcomes, not only from telemetry. On
    #: by default: the failures that telemetry cannot see — a mismatched driver, a
    #: half-deployed image, a disk returning `EIO` — are the ones that turn a single bad node
    #: into a job that never finishes, because a scheduler with a free slot there keeps
    #: offering it and the retries walk the whole queue onto it.
    enabled: bool = True
    #: Decayed failure weight at which a target stops being scheduled. Weights are per
    #: failure and vary by cause, so the default is roughly three placement-blaming failures
    #: in quick succession. One failure is almost never the node's fault.
    failure_threshold: float = 3.0
    #: How long a recorded failure keeps half its weight. This is what lets a node recover on
    #: its own after a bad patch, and what stops the ledger from becoming a permanent record
    #: of everything that ever went wrong on a long run.
    half_life_s: float = 300.0
    #: How long a first quarantine lasts before the target is put back on probation.
    cooldown_s: float = 60.0
    #: Ceiling on the per-offense doubling of the cooldown, so a repeatedly bad node is
    #: retried occasionally rather than never — a target that is out forever is a capacity
    #: loss nobody is tracking.
    max_cooldown_s: float = 900.0
    #: Share of the fleet that may be quarantined at once. The safety valve on the whole
    #: mechanism: when the cause is systemic — an expired credential, a bad image, a model
    #: file that 404s — every node fails every task, and without a cap the ledger condemns
    #: the entire cluster in the first minute and turns a degraded job into a dead one.
    max_blocked_fraction: float = 0.34


@dataclass(frozen=True, slots=True)
class FaultToleranceConfig:
    """Fleet-instability tunables: quarantine, retry budget, and OOM backoff.

    Named `fault_tolerance` rather than `resilience` because `distributed.resilience` already
    exists and means something different — it selects a *profile* (`default`/`spot`) that
    presets several distributed knobs at once. Two settings a level apart, both called
    resilience, with one a string and the other a section, is a mistake waiting to be made in
    a config file nobody can diff by eye.

    Examples:
        .. doctest::

            >>> from batcher.config import FaultToleranceConfig
            >>> FaultToleranceConfig().retry_budget_fraction
            0.1
    """

    quarantine: QuarantineConfig = QuarantineConfig()
    #: Share of attempted work a job may spend on retries before the next failure is raised
    #: instead of retried. Per-task retry limits do not bound a job: `max_retries=2` over a
    #: hundred thousand tasks authorizes two hundred thousand retries, and a fleet broken in
    #: some way no probe catches will use every one of them — so what an operator sees is
    #: hours at a fraction of the rate, then a failure with whatever error happened to be
    #: last. `0.0` disables the budget and restores that behavior.
    retry_budget_fraction: float = 0.1
    #: Retries authorized regardless of job size, so a short job is not left with a budget of
    #: zero and failed by one flaky node.
    retry_budget_floor: int = 16
    #: Fail a run when a device reports a fault that means data already computed on it is
    #: wrong, rather than retrying past it. On by default and deliberately not a performance
    #: trade: retrying past an uncontained ECC fault produces a job that completes
    #: successfully and writes out corruption, which is worse than the crash it avoided.
    fail_on_untrusted_results: bool = True


def validate_fault_tolerance(cfg: FaultToleranceConfig) -> None:
    """Raise `ConfigError` when a resilience tunable is out of range.

    Args:
        cfg: The resilience section to check.

    Raises:
        ConfigError: On the first out-of-range value, naming the field and its bound.
    """
    q = cfg.quarantine
    checks: tuple[tuple[bool, str], ...] = (
        (q.failure_threshold > 0, "fault_tolerance.quarantine.failure_threshold must be > 0"),
        (q.half_life_s > 0, "fault_tolerance.quarantine.half_life_s must be > 0"),
        (q.cooldown_s > 0, "fault_tolerance.quarantine.cooldown_s must be > 0"),
        (
            q.max_cooldown_s >= q.cooldown_s,
            "fault_tolerance.quarantine.max_cooldown_s must be >= cooldown_s",
        ),
        (
            0.0 < q.max_blocked_fraction <= 1.0,
            "fault_tolerance.quarantine.max_blocked_fraction must be in (0, 1]",
        ),
        (cfg.retry_budget_fraction >= 0.0, "fault_tolerance.retry_budget_fraction must be >= 0"),
        (cfg.retry_budget_floor >= 0, "fault_tolerance.retry_budget_floor must be >= 0"),
    )
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)
