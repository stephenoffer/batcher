"""Config-driven fault-tolerance, recovery, and skew policies for the distributed
executor.

These are pure ``active_config()`` → policy/option builders plus the map-stage
resilience helpers (``gather_map_results``, ``map_barrier``, ``draining_workers``). They
hold no Ray lifecycle state, so they import nothing from the rest of the package.
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


# A UDF failure whose cause is a *resource or remote-service* condition rather than a bug in
# the UDF. These are the failure modes of large-scale GPU inference: a CUDA OOM when several
# actors peak together, a model server returning 429/503, a socket timeout to a weight store.
# Retrying one on a fresh worker routinely succeeds, whereas a `TypeError` in the UDF never will.
_TRANSIENT_UDF_ERROR_MARKERS = (
    "cuda out of memory",
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "nccl timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "timed out",
    "timeout",
    "too many requests",
    "service unavailable",
    "temporarily unavailable",
    "slow_down",
    "internal server error",
    "bad gateway",
    "502",
    "503",
    "429",
)


def _is_transient_udf_error(exc: BaseException) -> bool:
    """Whether a `RayTaskError` looks like a retryable resource/remote condition.

    A deterministic UDF bug fails identically on every worker, so retrying it only burns
    the budget and delays the error. A transient one — CUDA OOM under a concurrency spike,
    a throttled model endpoint — usually succeeds on the next attempt, and failing the whole
    job on it throws away hours of completed inference. Matching is on the message text
    because the real cause is raised by torch / an HTTP client / a vendor SDK and arrives
    here already wrapped in Ray's `RayTaskError`.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__} {cur}".lower()
        if any(marker in text for marker in _TRANSIENT_UDF_ERROR_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


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
