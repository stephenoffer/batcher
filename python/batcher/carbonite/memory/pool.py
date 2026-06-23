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

__all__ = ["BufferPool", "current_process_pool", "process_pool"]


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
    try:
        from batcher._native import MemoryPool as _NativePool
    except ImportError:
        return _FallbackPool(limit_bytes)
    return _NativePool(limit_bytes)


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

    @contextmanager
    def reserve(self, n_bytes: int) -> Iterator[bool]:
        """Account `n_bytes` for the block; release on exit. Yields whether it fit."""
        granted = self._pool.try_reserve(n_bytes)
        try:
            yield granted
        finally:
            if granted:
                self._pool.release(n_bytes)

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


def process_pool(limit_bytes: int) -> BufferPool:
    """The process-wide buffer pool, created once and reconciled to `limit_bytes`.

    One envelope per process so concurrent queries and the transfer layer account
    against the same budget. The pool is created on first call; later calls reset
    the limit to `limit_bytes` (an autoscaler or a differently-configured query can
    grow/shrink the envelope) without dropping the live `used` accounting.
    """
    global _process_pool
    if _process_pool is None:
        with _process_pool_lock:
            if _process_pool is None:
                _process_pool = BufferPool(limit_bytes)
                return _process_pool
    if _process_pool.limit != limit_bytes:
        _process_pool.set_limit(limit_bytes)
    return _process_pool


def current_process_pool() -> BufferPool | None:
    """The process-wide buffer pool if one has been created, else `None`.

    Lets a reader (the pressure monitor) observe how much the engine currently
    holds against its envelope without forcing a pool into existence.
    """
    return _process_pool
