"""The buffer pool — Carbonite's reserve-before-allocate accounting.

A single process-wide pool bounds how much memory the engine holds at once. A
caller reserves *before* it materializes; a reservation that would push past the
limit is denied, and the caller spills (or back-pressures) instead of OOMing.

The accounting lives in Rust (`bc-resource::MemoryPool`, surfaced as
`batcher._native.MemoryPool`) so the operators and the transfer layer can later
enforce against the *same* envelope. Carbonite — the control plane — sets the
limit and reserves at operator/query granularity (coarse, never per row), so the
import of the engine here is the governor driving its data plane, not a hot-path
tuple touch. When the engine isn't built (pure-Python tooling, unit tests), the
pool degrades to an equivalent in-process accounting fallback.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from batcher._internal.native import engine_or_none

__all__ = [
    "BufferPool",
    "current_process_pool",
    "engine_pool_stats",
    "engine_pool_utilization",
    "process_pool",
    "reset_process_pool",
]


def engine_pool_stats() -> dict[str, int | float] | None:
    """What the **data plane's** process-wide pool is holding, or `None` if none exists.

    There are two pools, and this is the other one. `BufferPool` below wraps a
    `MemoryPool` the control plane constructs, and charges its own coarse per-query
    reservations to it. The engine constructs a *separate* process-wide pool inside
    `execute_plan` and charges operator state and the Flight transit buffers to that — and
    those are most of the footprint on exactly the queries anyone asks about. Neither
    counter could see the other, so the control plane's pressure reading could only infer
    the engine's memory from process RSS.

    Reading rather than merging is deliberate. Carbonite reserves a plan's estimated peak
    for the duration of execution and the engine then reserves the same operator's actual
    bytes; one shared counter would charge both and spill a query at half its budget.

    Returns:
        The engine pool's accounting, or `None` when no query has run under a memory budget
        in this process (or the extension is not built). `None` is distinct from a dict of
        zeros, which would assert something about a pool that has never existed.
    """
    reader = _engine_pool_reader()
    return None if reader is None else reader()


#: "Not looked up yet", distinct from a lookup that found nothing.
_UNRESOLVED = object()
#: The resolved `_native.engine_pool_stats`, or `None` when the extension is absent or
#: predates it.
_ENGINE_READER: object = _UNRESOLVED


def _engine_pool_reader():
    """The engine-pool accessor, resolved once.

    Which extension is loaded cannot change within a process, but the pressure monitor asks
    for this reading on every classification — once per query, and once per AIMD round on
    every shuffle channel. Re-importing the module and re-`getattr`ing through it each time
    is most of the call's cost on a build that does not have the function at all, which is
    precisely the build that gains nothing from asking.
    """
    global _ENGINE_READER
    if _ENGINE_READER is _UNRESOLVED:
        mod = engine_or_none()
        _ENGINE_READER = getattr(mod, "engine_pool_stats", None) if mod is not None else None
    return _ENGINE_READER


def engine_pool_utilization() -> float | None:
    """`used / limit` in the data plane's pool right now, or `None` when it has no envelope.

    The one figure the pressure monitor needs from the other pool. A zero limit reads as
    `None` rather than `1.0`: an unbounded engine pool governs nothing, and reporting it as
    full would pin every pressure reading at CRITICAL for a process that opted out of the
    budget entirely.
    """
    stats = engine_pool_stats()
    if not stats or not stats.get("limit_bytes"):
        return None
    return float(stats["utilization"])


class _FallbackPool:
    """Pure-Python mirror of `bc_resource::MemoryPool` for when `_native` is absent.

    Same greedy semantics (admit growth only when it fits; clamp release) so
    behavior is identical with or without the compiled engine.
    """

    def __init__(self, limit_bytes: int) -> None:
        self._limit = limit_bytes
        self._used = 0
        self._peak_used = 0
        self._denied = 0
        self._lock = threading.Lock()

    def try_reserve(self, n_bytes: int) -> bool:
        with self._lock:
            if self._used + n_bytes > self._limit:
                self._denied += 1
                return False
            self._used += n_bytes
            self._peak_used = max(self._peak_used, self._used)
            return True

    def release(self, n_bytes: int) -> None:
        with self._lock:
            self._used -= min(self._used, n_bytes)

    def set_limit(self, limit_bytes: int) -> None:
        with self._lock:
            self._limit = limit_bytes

    @property
    def used(self) -> int:
        return self._used

    @property
    def available(self) -> int:
        return max(0, self._limit - self._used)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def peak_used(self) -> int:
        return self._peak_used

    @property
    def denied(self) -> int:
        return self._denied


def _make_native_pool(limit_bytes: int):
    """The Rust `MemoryPool` if the engine is built, else the Python fallback."""
    # Through the one sanctioned accessor: a direct `import batcher._native` is attributed
    # by the import graph to the root `batcher` package, forging a phantom cycle that breaks
    # the layer-independence contracts (see `batcher._internal.native`).
    mod = engine_or_none()
    if mod is None:
        return _FallbackPool(limit_bytes)
    return mod.MemoryPool(limit_bytes)


class BufferPool:
    """Reserve-before-allocate accounting against a fixed memory limit.

    Backed by the Rust `MemoryPool` (one shared envelope) with a transparent
    Python fallback. Use `reserve` as a context manager: it accounts `n_bytes` for
    the duration of the block and releases them on exit, even if the block raises.
    The yielded bool says whether the reservation fit — a `False` means the pool
    is over budget and the caller should already be on the spill path.
    """

    def __init__(self, limit_bytes: int) -> None:
        self._pool = _make_native_pool(limit_bytes)
        # High-water mark of concurrently-reserved bytes — how close this envelope actually
        # came to its limit over the process's life. `peak_used / limit` is the measured
        # memory pressure a workload hit: near 1 means it ran at the edge (spill/OOM risk),
        # low means the budget was oversized. Measurement only; never gates a reservation.
        self._peak_used = 0
        # How many reservations were refused for lack of headroom. A non-zero count is the
        # direct evidence that the envelope is the binding constraint on this workload —
        # the thing `peak_used` near `limit` only *suggests*.
        self._denied = 0
        # Guards the two measurement counters. The accounting itself is atomic inside the
        # Rust pool; these are Python-side read-modify-writes, and several queries reserve
        # against the one process pool concurrently.
        self._stats_lock = threading.Lock()

    @contextmanager
    def reserve(self, n_bytes: int) -> Iterator[bool]:
        """Account `n_bytes` for the block; release on exit. Yields whether it fit.

        A non-positive request is granted without touching the accounting. Passing a
        *negative* count would otherwise **reduce** `used` on reserve and grow it again on
        release, which silently hands the caller memory the envelope never had — the pool
        admits growth, so nothing else in it checks the sign.

        Args:
            n_bytes: Bytes to hold for the duration of the block.

        Yields:
            Whether the reservation fit inside the envelope.
        """
        if n_bytes <= 0:
            yield True
            return
        granted = self._pool.try_reserve(n_bytes)
        with self._stats_lock:
            if granted:
                self._peak_used = max(self._peak_used, self._pool.used)
            else:
                self._denied += 1
        try:
            yield granted
        finally:
            if granted:
                self._pool.release(n_bytes)

    @property
    def peak_used(self) -> int:
        """The high-water mark of concurrently-reserved bytes over this pool's life — the
        measured memory pressure (`peak_used / limit`) the workload actually hit.

        **This pool's** peak, and only this pool's. The wrapped `MemoryPool` is one the
        control plane constructed, so its counters cover the coarse per-query reservations
        Carbonite makes and nothing else. Operator state and the Flight transit buffers are
        charged to a *different* process-wide pool inside the engine; read that one through
        `engine_pool_stats`, which `ResourceManager.stats` reports beside this.

        (The two are deliberately not summed into one figure. Carbonite reserves a plan's
        estimated peak and the engine then reserves the same operator's actual bytes, so a
        sum double-counts every query — and a `max` across pools with different limits is
        not a quantity at all.)
        """
        return max(self._peak_used, int(getattr(self._pool, "peak_used", 0) or 0))

    @property
    def denied(self) -> int:
        """Reservations **this pool** refused for lack of headroom.

        The max of the control plane's own count and the wrapped pool's, not their sum: a
        refusal Carbonite made is recorded on both sides, so adding them reports twice the
        refusals that happened.

        Denials in the engine's own pool — an operator that could not reserve its hash
        table — are counted there, not here. `engine_pool_stats()["denied"]` is that
        figure, and it is the one that says a query spilled because the *envelope* was
        binding rather than because Carbonite's estimate was.
        """
        return max(self._denied, int(getattr(self._pool, "denied", 0) or 0))

    @property
    def spill_requests(self) -> int:
        """Times the cooperative path made a consumer spill to grant a reservation.

        Effectively always `0` here. The cooperative path's only registered consumer is the
        Flight shuffle store, which registers with the **engine's** process-wide pool, and
        the control plane never reserves cooperatively anyway. The number worth reading is
        `engine_pool_stats()["spill_requests"]`, which says how often other operators had
        to pay for a query's memory; this one is kept so the two pools report the same
        shape.
        """
        return int(getattr(self._pool, "spill_requests", 0) or 0)

    @property
    def utilization(self) -> float:
        """`used / limit` right now, in ``[0, 1]``; `1.0` for a zero-limit pool.

        The one place this ratio is computed, so the pressure monitor and any telemetry
        reader agree on what "how full is the engine's envelope" means.
        """
        limit = self._pool.limit
        if limit <= 0:
            return 1.0
        return min(1.0, self._pool.used / limit)

    def stats(self) -> dict[str, int | float]:
        """A snapshot of this envelope: limit, live usage, lifetime peak, denials.

        Returns:
            The accounting figures, with `utilization` and `peak_utilization` as the two
            ratios a reader actually acts on.
        """
        limit = self._pool.limit
        peak = self.peak_used
        return {
            "limit_bytes": limit,
            "used_bytes": self._pool.used,
            "available_bytes": self._pool.available,
            "peak_used_bytes": peak,
            "denied": self.denied,
            "spill_requests": self.spill_requests,
            "utilization": self.utilization,
            "peak_utilization": min(1.0, peak / limit) if limit > 0 else 1.0,
        }

    def set_limit(self, limit_bytes: int) -> None:
        """Resize the envelope. Existing reservations are untouched; only the cap
        future reservations admit against changes (an autoscaler grew/shrank RAM)."""
        self._pool.set_limit(limit_bytes)

    @property
    def used(self) -> int:
        """Bytes currently reserved across the process."""
        return self._pool.used

    @property
    def available(self) -> int:
        """Bytes currently free in the envelope."""
        return self._pool.available

    @property
    def limit(self) -> int:
        """The pool's hard limit in bytes."""
        return self._pool.limit


_process_pool: BufferPool | None = None
_process_pool_lock = threading.Lock()
# A shrink that could not be applied because the pool was busy. Remembered, not dropped:
# see `process_pool`.
_pending_shrink: int | None = None


def process_pool(limit_bytes: int) -> BufferPool:
    """The process-wide buffer pool, created once and reconciled to `limit_bytes`.

    One envelope per process so concurrent queries and the transfer layer account
    against the same budget. The pool is created on first call; later calls reset
    the limit to `limit_bytes` (an autoscaler or a differently-configured query can
    grow/shrink the envelope) without dropping the live `used` accounting.

    A *shrink* is deferred while the pool still holds reservations. Several pipelines share
    this one envelope, so applying a cheap query's smaller budget mid-flight would shrink
    the envelope an expensive concurrent query is already running inside — pushing it into
    spurious spilling, or failing a reservation for work Carbonite had correctly admitted.
    Growth always applies at once (capacity the autoscaler just added must not wait).

    A deferred shrink is **remembered**. It used to be discarded outright, so the smaller
    budget only ever landed if some later call happened to ask for that same figure while
    the pool was idle — which for a shrink caused by an autoscaler taking RAM away is not
    something that happens at all. The envelope then stayed at the larger limit for the
    life of the process and admitted against memory the box no longer had. The pending
    figure is applied on the first subsequent call that finds the pool idle, and is
    superseded by any later reconcile so it can never resurrect a stale budget.

    Args:
        limit_bytes: The envelope this caller wants in force.

    Returns:
        The shared `BufferPool`.
    """
    global _process_pool, _pending_shrink
    pool = _process_pool
    if pool is None:
        with _process_pool_lock:
            if _process_pool is None:
                _process_pool = BufferPool(limit_bytes)
                _pending_shrink = None
                return _process_pool
            pool = _process_pool
    with _process_pool_lock:
        if limit_bytes > pool.limit:
            pool.set_limit(limit_bytes)
            _pending_shrink = None
        elif limit_bytes < pool.limit:
            if pool.used == 0:
                pool.set_limit(limit_bytes)
                _pending_shrink = None
            else:
                # Busy: remember *this* request so it lands the moment the envelope is
                # free. The latest reconcile wins, rather than the smallest ever seen.
                #
                # Taking the `min` of the pending figure and this one looked conservative
                # and was the opposite. A single small caller — a unit test, a deliberately
                # tiny `config_context`, one cheap query — pinned the pending figure at its
                # budget forever, because nothing but a *growth* past the live limit ever
                # cleared it. Every later reconcile then made it smaller or left it alone,
                # so the first moment the pool went idle the process-wide envelope
                # collapsed to that stale figure and stayed there: every subsequent query
                # spilled against a budget the machine had not had for hours.
                _pending_shrink = limit_bytes
        elif _pending_shrink is not None and pool.used == 0:
            pool.set_limit(_pending_shrink)
            _pending_shrink = None
    return pool


def current_process_pool() -> BufferPool | None:
    """The process-wide buffer pool if one has been created, else `None`.

    Lets a reader (the pressure monitor) observe how much the engine currently
    holds against its envelope without forcing a pool into existence.
    """
    return _process_pool


def reset_process_pool() -> None:
    """Drop the process-wide pool so the next `process_pool` call builds a fresh one.

    For tests, which otherwise inherit whatever envelope and lifetime high-water an
    earlier test left in this process — the same reason `reset_process_limiter` exists
    for the concurrency limiter.
    """
    global _process_pool, _pending_shrink
    with _process_pool_lock:
        _process_pool = None
        _pending_shrink = None
