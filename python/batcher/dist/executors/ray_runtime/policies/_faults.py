"""Config-driven fault-tolerance, recovery, and skew policies for the distributed
executor.

These are pure ``active_config()`` → policy/option builders plus the map-stage
resilience helpers (``gather_map_results``, ``map_barrier``, ``draining_workers``). They
hold no Ray lifecycle state, so they import nothing from the rest of the package.

The *classification* they build on lives in `carbonite.resilience.classify`, not here: the
single-node executor and this scheduler have to agree about what a failure means, and a
retry rule that two subsystems each keep their own copy of is a retry rule that behaves
differently depending on which path a failure took to reach it. What stays here is the part
that is genuinely about Ray — which of its own exception types are deaths and which are not.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

# Fallback in-flight cap for the map submission window when the live cluster size is
# unreadable (Ray down / test stubs) — bounds a high-fan-out job without a topology read.
_DEFAULT_PENDING_WINDOW = 1024

# `RayError` subclasses that are NOT worker loss and that a retry cannot fix. A recovery
# loop that reads these as a dead worker blames a perfectly healthy host, marks the whole
# fleet dead one retry at a time, and finally reports "no surviving worker" instead of the
# real cause. Deliberately narrow: an `OutOfMemoryError` (Ray's memory monitor killing a
# task under node pressure) stays retryable, because rescheduling it onto a less-loaded
# node genuinely can succeed. Matched by name so a Ray version lacking one — or a test's
# stub `ray.exceptions` — degrades to the old catch-all instead of failing at import.
_FATAL_RAY_ERROR_NAMES = (
    "RuntimeEnvSetupError",  # the worker environment is broken — every retry re-breaks it
    "TaskCancelledError",  # we cancelled it (e.g. a speculation loser); not a death
    "GetTimeoutError",  # a caller-imposed deadline, not a death
)


def is_recoverable_task_failure(exc: BaseException) -> bool:
    """Whether a `RayTaskError` reports lost data rather than a deterministic bug.

    `gather_map_results` can re-raise every `RayTaskError` outright, because a map task
    that fails reports worker loss as a *Ray* error. The combiner tree cannot: a combine
    fetches from its upstreams inside the task, so a genuinely-lost peer surfaces as a
    `RetryableShuffleError` **wrapped in** a `RayTaskError` — indistinguishable, by type
    alone, from a user's UDF raising `ZeroDivisionError`.

    Treating them alike in either direction is a real failure. Retrying everything makes a
    deterministic bug burn the whole recovery budget and then surface as
    `ResourceError("shuffle did not recover...")` with the original traceback gone — a
    resource error for a Python bug. Re-raising everything would abort a query whose only
    problem was a preempted peer, which is exactly what recovery exists to survive.

    So the transport's own classification decides. `RetryableShuffleError` (the Rust
    `FetchFault::Retryable`, i.e. an unreachable peer) and `ResourceError` (a spill file
    that vanished with an ephemeral disk) are recoverable; `FatalShuffleError` and every
    application exception are not. Ray fuses the original type into the raised class, so
    the instance check usually matches directly; `cause` covers the versions where it does
    not.
    """
    from batcher._internal.errors import ResourceError, RetryableShuffleError

    recoverable = (RetryableShuffleError, ResourceError)
    return isinstance(exc, recoverable) or isinstance(getattr(exc, "cause", None), recoverable)


def _is_fatal_ray_error(exc: BaseException) -> bool:
    """Whether `exc` is a Ray error that a worker-loss retry must NOT absorb."""
    try:
        import ray.exceptions as ray_exc
    except Exception:  # pragma: no cover - ray optional
        return False
    fatal = tuple(
        t for t in (getattr(ray_exc, n, None) for n in _FATAL_RAY_ERROR_NAMES) if t is not None
    )
    return bool(fatal) and isinstance(exc, fatal)


def _is_transient_udf_error(exc: BaseException) -> bool:
    """Whether a `RayTaskError` looks like a retryable resource/remote condition.

    A deterministic UDF bug fails identically on every worker, so retrying it only burns
    the budget and delays the error. A transient one — CUDA OOM under a concurrency spike,
    a throttled model endpoint — usually succeeds on the next attempt, and failing the whole
    job on it throws away hours of completed inference.

    The classification itself lives in `carbonite.resilience.classify`, which is the one
    taxonomy the single-node executor and this scheduler share. It used to be a marker list
    here and a second, different marker list nowhere — and a retry rule that two subsystems
    disagree about is a retry rule that behaves differently depending on which path a failure
    took to reach it. The classifier also answers the question this predicate structurally
    cannot: *where* the retry should land. See `must_move`.
    """
    from batcher.carbonite.resilience import is_retryable

    return is_retryable(exc)


def check_results_trusted(exc: BaseException) -> None:
    """Raise instead of retrying when a failure means data already computed is wrong.

    Almost every failure loses work, and losing work is what a retry is for. A handful do
    something else: an uncontained ECC fault, a double-bit ECC error, a device that kept
    running and answered incorrectly. Those do not lose a result, they *replace* it with a
    wrong one, and the tasks that already completed on that device are as suspect as the one
    that failed.

    Retrying past one of those produces a job that finishes successfully and writes out
    corruption, which is strictly worse than the crash it avoided — and, because nothing in
    the output says otherwise, it is discovered downstream by whoever trusted the numbers.
    So this is the one failure class the recovery loop refuses to absorb.

    Governed by `fault_tolerance.fail_on_untrusted_results`, which a deployment can turn off
    where the workload is itself tolerant (a resumable job that re-derives everything, a
    checksum downstream). It is not a performance knob and is on by default.

    Args:
        exc: The failure about to be retried.

    Raises:
        ExecutionError: When the failure indicates the device returned corrupted data.
    """
    from batcher.carbonite.resilience import results_untrusted

    if not active_config().fault_tolerance.fail_on_untrusted_results:
        return
    if not results_untrusted(exc):
        return
    from batcher._internal.errors import ExecutionError

    raise ExecutionError(
        "an accelerator reported a fault that corrupts data already computed on it, so this "
        "run is not being retried past it: results produced on that device cannot be trusted. "
        "Reset or replace the device and re-run. Set "
        "fault_tolerance.fail_on_untrusted_results=False to retry anyway."
    ) from exc


def retry_budget():
    """The job-wide retry budget, built from the active config.

    Per-task retry limits do not bound a job: `task_max_retries=2` over a hundred thousand
    partitions authorizes two hundred thousand retries, and a fleet broken in a way no probe
    catches will use every one of them. What an operator then sees is hours at a fraction of
    the rate followed by whatever error happened to be last, long after the first one said
    exactly what was wrong.

    Returns:
        A `RetryBudget` sized from `fault_tolerance.retry_budget_*`.
    """
    from batcher.carbonite.resilience import RetryBudget

    ft = active_config().fault_tolerance
    return RetryBudget(
        fraction=ft.retry_budget_fraction,
        floor=ft.retry_budget_floor,
        label="map",
    )


def node_ledger():
    """The process-wide ledger of which workers have been failing, or `None` when disabled.

    Returns:
        The shared `FaultLedger`, or `None` when `fault_tolerance.quarantine.enabled` is off —
        in which case every caller keeps the placement behavior it had before the ledger
        existed, rather than consulting a ledger that records nothing.
    """
    if not active_config().fault_tolerance.quarantine.enabled:
        return None
    from batcher.carbonite.resilience import default_ledger

    return default_ledger("worker")


def speculation_policy():
    """Build the straggler-speculation policy from the active config.

    `max_backups` defaults to 1 (`DistributedConfig.speculation_max_backups`), so one
    straggler per barrier gets a duplicate. Setting it to 0 turns speculation off and makes
    the barrier a plain `ray.get`. Either way distributed *results* are unchanged: a backup
    is a duplicate of a deterministic task and the barrier keeps whichever copy finishes
    first.

    The straggler *factor* (how many multiples of the median task time before a backup fires) is
    tuned from the measured task-time spread of prior shuffle stages (`_learned_straggler_factor`):
    a stage whose tasks finish uniformly gets a higher factor (rarely back up), a heavy-tailed one
    a lower factor (back up sooner). A cold store keeps the config default. Speculation only
    duplicates a slow task and keeps whichever copy finishes first, so the result is unchanged
    regardless of the factor.
    """
    from batcher.carbonite.resilience import SpeculationPolicy

    d = active_config().distributed
    return SpeculationPolicy(
        straggler_factor=_learned_straggler_factor(d.speculation_straggler_factor),
        min_finished_frac=d.speculation_min_finished_frac,
        max_backups=d.speculation_max_backups,
    )


def _learned_straggler_factor(default: float) -> float:
    """The learned straggler factor from measured shuffle-family task-time variance, else
    `default`. Best-effort read of the process-wide MetadataHub; any failure returns `default`."""
    try:
        from batcher.core import default_hub
        from batcher.dist.adaptive_sizing import learned_straggler_factor

        for family in ("aggregate", "hash_join", "sort", "window"):
            learned = learned_straggler_factor(default_hub(), family)
            if learned is not None:
                return learned
    except Exception as exc:  # pragma: no cover - learning is best-effort
        note_suppressed("dist", "read learned straggler factor", exc)
    return default


def skew_join_salt() -> tuple[int, float]:
    """Return ``(salt_buckets, hot_fraction)`` for skew-aware join salting.

    ``salt_buckets == 0`` (default) means salting is off → the shuffle join is the
    plain co-partition, so single-node==distributed is bit-identical.
    """
    d = active_config().distributed
    return int(d.skew_join_salt), float(d.skew_join_fraction)


def runtime_bloom_join() -> bool | str:
    """The runtime bloom-filter join policy: ``True``/``False``/``"auto"``.

    When engaged, a shuffle join prunes the probe side by a bloom built over the build
    side's keys before shuffling. Always correct; a network-volume optimization for
    selective joins. ``"auto"`` (the default) defers the per-join decision to a
    cardinality estimate (see `dist.executors.join._bloom_beneficial`); ``True``/
    ``False`` force it on/off (see `DistributedConfig.runtime_bloom_join`)."""
    return active_config().distributed.runtime_bloom_join


def recovery_policy():
    """Build the shuffle recompute-on-worker-loss recovery policy from the config.

    Bounds the recompute→retry rounds and the exponential backoff between them, so a
    larger cluster's failure rate is tunable rather than a hardcoded 3 attempts.
    """
    from batcher.carbonite.resilience import RecoveryPolicy

    d = active_config().distributed
    return RecoveryPolicy(
        max_attempts=d.recovery_max_attempts,
        backoff_base_s=d.recovery_backoff_base_s,
    )


def fault_options() -> dict:
    """Ray task fault-tolerance kwargs from config — the first line of defense.

    `max_retries` reruns a failed shuffle task (deterministic, recomputed from a
    durable source, so a rerun is safe) so a transient node/connection failure
    self-heals before the heavier app-level recompute loop engages. With
    `retry_on_transient`, retries also cover application exceptions, not just worker
    death; a deterministic failure still re-fails and surfaces once retries exhaust.
    """
    d = active_config().distributed
    opts: dict = {"max_retries": int(d.task_max_retries)}
    if d.retry_on_transient:
        opts["retry_exceptions"] = True
    return opts


def actor_fault_options() -> dict:
    """Ray actor fault-tolerance kwargs from config for compute actors (the map /
    inference pool): respawn a crashed actor (`max_restarts`) and rerun its in-flight
    call on the respawned actor (`max_task_retries`).

    Not applied to the Flight shuffle-server actors — their loss is recovered by the
    lineage recompute loop, and letting Ray restart them out from under it would race
    that recovery.
    """
    d = active_config().distributed
    return {
        "max_restarts": int(d.actor_max_restarts),
        "max_task_retries": int(d.actor_max_task_retries),
    }
