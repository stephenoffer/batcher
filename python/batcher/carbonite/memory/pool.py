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
    "process_pool",
    "reset_process_pool",
]


class _FallbackPool:
    """Pure-Python mirror of `bc_resource::MemoryPool` for when `_native` is absent.

    Same greedy semantics (admit growth only when it fits; clamp release) so
    behavior is identical with or without the compiled engine.
    """

    def __init__(self, limit_bytes: int) -> None:
        self._limit = limit_bytes
        self._used = 0
        self._lock = threading.Lock()

    def try_reserve(self, n_bytes: int) -> bool:
        with self._lock:
            if self._used + n_bytes > self._limit:
                return False
            self._used += n_bytes
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
        measured memory pressure (`peak_used / limit`) the workload actually hit."""
        return self._peak_used

    @property
    def denied(self) -> int:
        """Reservations this pool refused for lack of headroom over its life."""
        return self._denied

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
        return {
            "limit_bytes": limit,
            "used_bytes": self._pool.used,
            "available_bytes": self._pool.available,
            "peak_used_bytes": self._peak_used,
            "denied": self._denied,
            "utilization": self.utilization,
            "peak_utilization": min(1.0, self._peak_used / limit) if limit > 0 else 1.0,
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
                # Busy: hold the smallest requested shrink so it lands the moment the
                # envelope is free, rather than being forgotten.
                _pending_shrink = min(_pending_shrink or limit_bytes, limit_bytes)
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
