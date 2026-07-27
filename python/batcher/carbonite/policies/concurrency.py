"""Bounding how many queries run at once, and how wide each one gets.

# The measurement this exists for

`BENCHMARK_RESULTS.md` records the worst result in the file: going from 1 concurrent
client to 16, throughput *fell* from 124 QPS to 88 while p50 latency rose from 7.6 ms to
178 ms. Not "failed to scale" — actively worse.

A ~3-4 ms fixed control-plane cost per query explains a *ceiling* on throughput. It does
not explain a **decline**, and the 23x p50 blow-up is the tell. That shape is
oversubscription: each concurrent query asks the Rust executor for a rayon pool sized to
every core, so sixteen of them on a fifteen-core box put ~240 runnable threads on 15 cores.
They then spend their time context-switching and evicting each other's cache lines.

Two knobs fix it, and both are Carbonite's job — this is the "protects" verb:

- **Admit fewer at once.** A bounded slot count with a FIFO queue, so query seventeen waits
  rather than joining the scrum.
- **Give each one less.** `width_for` divides the cores among the queries actually running,
  so N concurrent queries request N-way-narrower pools instead of N full ones.

# Why both

Either alone is half a fix. Slots without width still oversubscribe within the admitted
set. Width without slots leaves an unbounded queue of 1-core queries, which is fair but
starves everyone equally.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from batcher._internal.errors import ResourceError

__all__ = [
    "AdmissionTimeout",
    "ConcurrencyLimiter",
    "ExecutionGrant",
    "process_limiter",
    "reset_process_limiter",
]


class AdmissionTimeout(ResourceError):
    """A query waited for an execution slot longer than the configured deadline."""


@dataclass(frozen=True, slots=True)
class ExecutionGrant:
    """Permission to execute, and how much of the machine to use while doing it."""

    #: Rayon pool width this query should request. 0 means "unbounded", i.e. all cores —
    #: which is the historical behavior and what a single concurrent query still gets.
    workers: int
    #: How many queries (including this one) held a slot when it was granted. Carried for
    #: observability: a rising number is what an operator needs to see before latency does.
    concurrent: int


class ConcurrencyLimiter:
    """A bounded, FIFO-fair slot pool with a queue cap and an acquire deadline.

    One instance per process. `acquire` blocks until a slot frees, then hands back an
    `ExecutionGrant` sized to the number of queries actually running.

    **Re-entrancy is the trap here, and it is handled explicitly.** A `map_batches` UDF may
    call `collect()` on another `Dataset`; that inner query would ask for a slot while the
    outer one holds it, and with every slot taken the process deadlocks against itself
    forever. The limiter tracks depth per thread and lets a nested acquire through without
    consuming a slot — the outer query already paid for the machine, and the inner one is
    part of its work, not competition for it.
    """

    def __init__(self, slots: int, *, queue_depth: int = 1000, cores: int = 0) -> None:
        """Build a limiter.

        Args:
            slots: Maximum queries executing at once. 0 or less means unbounded.
            queue_depth: Maximum queries waiting. A further arrival raises rather than
                joining an unbounded queue, because a queue nobody drains is an outage
                that looks like slowness.
            cores: Cores to divide among running queries; 0 reads the machine.
        """
        self._slots = max(0, slots)
        self._queue_depth = max(1, queue_depth)
        self._cores = cores
        self._lock = threading.Condition()
        self._active = 0
        self._waiting = 0
        self._depth = threading.local()

    @property
    def active(self) -> int:
        """Queries currently holding a slot."""
        return self._active

    def width_for(self, active: int) -> int:
        """Rayon pool width for one query when `active` are running.

        Cores divided by concurrency, floored at 1: with one query running this is every
        core (unchanged from today), and with sixteen it is a sixteenth each instead of
        sixteen full pools fighting over the same silicon.

        Args:
            active: How many queries are running, including this one.

        Returns:
            The worker count to request, or 0 for "unbounded" when nothing is contended.
        """
        if self._slots <= 0 or active <= 1:
            return 0  # unbounded: the historical single-query behavior
        cores = self._cores
        if cores <= 0:
            from batcher._internal.hardware import available_cpu_count

            cores = available_cpu_count()
        return max(1, cores // active)

    def acquire(self, timeout: float = 0.0) -> ExecutionGrant:
        """Wait for a slot and return the grant. Releases via `release`.

        Args:
            timeout: Seconds to wait before raising. 0 waits indefinitely.

        Returns:
            The `ExecutionGrant` for this query.

        Raises:
            AdmissionTimeout: If the queue is full, or the deadline passes.
        """
        if self._slots <= 0:
            return ExecutionGrant(workers=0, concurrent=1)

        # A nested acquire (a `collect()` inside a UDF) must not take a second slot. The
        # outer query already holds the machine; making the inner one queue behind it
        # deadlocks the process against itself.
        depth = getattr(self._depth, "value", 0)
        if depth > 0:
            self._depth.value = depth + 1
            return ExecutionGrant(workers=self.width_for(self._active), concurrent=self._active)

        deadline = (time.monotonic() + timeout) if timeout > 0 else None
        with self._lock:
            if self._waiting >= self._queue_depth:
                raise AdmissionTimeout(
                    f"{self._waiting} queries are already queued for an execution slot "
                    f"(cap {self._queue_depth}).",
                    hint=(
                        "Raise `execution.admission_queue_depth`, raise "
                        "`execution.max_concurrent_queries`, or shed load upstream."
                    ),
                )
            self._waiting += 1
            try:
                while self._active >= self._slots:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise AdmissionTimeout(
                            f"waited {timeout:.1f}s for an execution slot "
                            f"({self._active} of {self._slots} in use).",
                            hint="Raise `execution.admission_timeout_s`, or add capacity.",
                        )
                    self._lock.wait(remaining)
            finally:
                self._waiting -= 1
            self._active += 1
            active = self._active
        self._depth.value = 1
        return ExecutionGrant(workers=self.width_for(active), concurrent=active)

    def release(self) -> None:
        """Give the slot back. Safe to call for a nested acquire, which held none."""
        if self._slots <= 0:
            return
        depth = getattr(self._depth, "value", 0)
        if depth > 1:
            self._depth.value = depth - 1
            return
        self._depth.value = 0
        with self._lock:
            if self._active > 0:
                self._active -= 1
            # `notify` rather than `notify_all`: exactly one slot freed, so waking every
            # waiter to have all but one go back to sleep is a thundering herd on precisely
            # the path that is already contended.
            self._lock.notify()


# One limiter per process, because the resource it protects — the machine's cores — is
# per-process. Rebuilt when the configured slot count changes, mirroring how
# `memory.pool.process_pool` reconciles its budget, so a `config_context` that raises the
# limit takes effect rather than being silently ignored.
_LIMITER: ConcurrencyLimiter | None = None
_LIMITER_SLOTS: int = -1
_LIMITER_LOCK = threading.Lock()


def process_limiter(config) -> ConcurrencyLimiter:
    """The process-wide concurrency limiter, sized from `config.execution`.

    Args:
        config: The active `Config`.

    Returns:
        The shared `ConcurrencyLimiter`.
    """
    global _LIMITER, _LIMITER_SLOTS

    execution = config.execution
    slots = int(getattr(execution, "max_concurrent_queries", 0) or 0)
    if _LIMITER is not None and slots == _LIMITER_SLOTS:
        return _LIMITER
    with _LIMITER_LOCK:
        if _LIMITER is None or slots != _LIMITER_SLOTS:
            # Rebuilding drops the old limiter's in-flight accounting. That is acceptable
            # only because the slot count is deployment configuration, changed at startup
            # — not something a running workload flips. A rebuild under load would
            # briefly over-admit, never deadlock.
            _LIMITER = ConcurrencyLimiter(
                slots,
                queue_depth=int(getattr(execution, "admission_queue_depth", 1000) or 1000),
            )
            _LIMITER_SLOTS = slots
        return _LIMITER


def reset_process_limiter() -> None:
    """Drop the process limiter, so the next call rebuilds it. For tests."""
    global _LIMITER, _LIMITER_SLOTS
    with _LIMITER_LOCK:
        _LIMITER = None
        _LIMITER_SLOTS = -1
