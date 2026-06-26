"""Straggler mitigation — speculative backup tasks for shuffle barriers.

A distributed stage waits on every task at a barrier (`ray.get`). One slow
*survivor* (a hot partition, a degraded node) then stalls the whole stage even
though every other task finished long ago. `ShuffleRecovery` handles a task that
*dies*; this handles one that is merely *slow*: once most tasks have finished, any
task running far longer than the median gets a **backup copy** launched, and the
barrier takes whichever copy finishes first.

This is safe because shuffle map/reduce tasks are pure, deterministic functions of
their on-disk input partition (the same property `ShuffleRecovery` relies on to
recompute), so a backup produces byte-identical output — speculation changes *when*
a partial arrives, never *what* it contains.

Carbonite owns the policy and the decision; the distributed layer supplies the
`relaunch` closure that re-issues task *i*. The decision (`stragglers_to_backup`)
is a pure function so it is tested without Ray. Speculation is **opt-in**:
`max_backups == 0` makes `gather_with_backups` behave exactly like `ray.get`
(gather all results in order), so wiring it into a barrier is a no-op until a
policy enables it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["SpeculationPolicy", "gather_with_backups", "stragglers_to_backup"]


@dataclass(frozen=True, slots=True)
class SpeculationPolicy:
    """When to launch a backup for a straggler.

    `straggler_factor`: back up a still-running task whose elapsed time exceeds this
    multiple of the median *finished* task's time. `min_finished_frac`: only start
    speculating once this fraction of tasks have finished (so the median is
    meaningful and we don't backup an entire slow-but-uniform stage). `max_backups`:
    hard cap on concurrent backups in flight — bounded by Carbonite's scheduling
    grant so speculation never oversubscribes the cluster. `0` disables speculation.
    """

    straggler_factor: float = 1.5
    min_finished_frac: float = 0.75
    max_backups: int = 0


def stragglers_to_backup(
    n: int,
    finished: dict[int, float],
    elapsed: dict[int, float],
    policy: SpeculationPolicy,
) -> list[int]:
    """Indices of still-running tasks that warrant a backup, slowest first.

    Pure (no Ray, no clock): `finished` maps a finished task index to its completion
    time, `elapsed` maps a still-running task index to its current elapsed time.
    Returns `[]` until `min_finished_frac` of the `n` tasks have finished, then the
    running tasks slower than `straggler_factor x median(finished)`, capped at
    `max_backups`. Empty when speculation is disabled (`max_backups <= 0`).
    """
    if policy.max_backups <= 0 or not finished:
        return []
    if len(finished) < max(1, math.ceil(policy.min_finished_frac * n)):
        return []
    threshold = policy.straggler_factor * statistics.median(finished.values())
    laggards = [i for i, e in elapsed.items() if e > threshold]
    laggards.sort(key=lambda i: elapsed[i], reverse=True)  # slowest first
    return laggards[: policy.max_backups]


def gather_with_backups(
    refs: list[Any],
    relaunch: Callable[[int], Any],
    policy: SpeculationPolicy | None = None,
    poll_seconds: float = 0.5,
    on_failure: Callable[[int, Any, Exception], Any] | None = None,
) -> list[Any]:
    """Gather `len(refs)` Ray results, launching backups for stragglers.

    Returns each slot's first-to-finish result, in the original order — identical
    to `ray.get(refs)` when `policy.max_backups == 0` (the default). `relaunch(i)`
    must re-issue task *i* and return a new `ObjectRef` whose result is equivalent
    (the task is deterministic). Backup losers are cancelled best-effort.

    `on_failure(i, ref, exc)`, when given, turns a task *error* (e.g. a dead actor)
    into slot `i`'s result instead of re-raising — so a barrier that must recover from
    a *lost* task (the shuffle reduce path) can speculate on stragglers AND classify a
    death in one pass. It is called only once **every** live copy of slot `i` has
    failed, with the last-failed `ref` so the caller can attribute the loss to the
    right host (a dying *backup* never finalizes a slot whose original is still
    running). `None` (the default) re-raises on the first error — the pure-straggler
    behavior for tasks that do not fail.
    """
    import time

    import ray

    policy = policy or SpeculationPolicy()
    n = len(refs)
    if n == 0:
        return []

    now = time.monotonic()
    started: dict[int, float] = dict.fromkeys(range(n), now)
    ref_to_idx: dict[Any, int] = {r: i for i, r in enumerate(refs)}
    result_of: dict[int, Any] = {}
    finished_times: dict[int, float] = {}
    backed_up: set[int] = set()
    pending = list(refs)
    # Live copies (original + backups) per slot — consulted only on the failure-tolerant
    # path so a slot is finalized as failed *only* when every copy has died.
    alive: dict[int, set] = {i: {refs[i]} for i in range(n)} if on_failure is not None else {}

    while len(result_of) < n:
        # Drain *all* currently-ready refs per wake (not one): a burst of completions
        # is collected in a single iteration instead of one Python wakeup each. The
        # `poll_seconds` timeout still bounds the wake, so the straggler-backup cadence
        # below is unchanged (it re-evaluates at most once per poll window).
        done, pending = ray.wait(pending, num_returns=len(pending), timeout=poll_seconds)
        now = time.monotonic()
        for r in done:
            i = ref_to_idx[r]
            if i in result_of:  # slot already won by another copy
                continue
            try:
                result_of[i] = ray.get(r)  # first copy to finish wins
                finished_times[i] = now - started[i]
            except Exception as exc:
                if on_failure is None:
                    raise
                alive[i].discard(r)
                if not alive[i]:  # every copy of this slot has now failed
                    result_of[i] = on_failure(i, r, exc)
                    finished_times[i] = now - started[i]
        if policy.max_backups > 0 and len(result_of) < n:
            elapsed = {i: now - started[i] for i in range(n) if i not in result_of}
            in_flight = len(backed_up) - sum(1 for i in backed_up if i in result_of)
            for i in stragglers_to_backup(n, finished_times, elapsed, policy):
                if i not in backed_up and in_flight < policy.max_backups:
                    backed_up.add(i)
                    in_flight += 1
                    backup = relaunch(i)
                    ref_to_idx[backup] = i
                    if on_failure is not None:
                        alive[i].add(backup)
                    pending.append(backup)

    import contextlib

    for r in pending:  # cancel any backups still running after their slot finished
        with contextlib.suppress(Exception):  # cancellation is best-effort
            # force=True: the winning copy's result is already in hand and tasks are
            # deterministic, so killing the loser outright reclaims the resource a
            # soft cancel would leave wedged on a stuck straggler.
            ray.cancel(r, force=True)
    return [result_of[i] for i in range(n)]
