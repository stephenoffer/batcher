"""What the device tier actually did, per process — the counterpart to the fallback contract.

`backend="gpu"` is documented as always safe: an unsupported shape, a device out of memory or a
cluster with no GPU returns the same rows from the CPU engine. That safety has a cost, and it is
the reason this module exists: **the only signal a silent fallback leaves is the running time**.
A tier that has stopped translating anything is indistinguishable, from the outside, from one
that is translating everything and is merely slow.

`note_gpu_failure` already separates a defect from a decline in the *log*. This separates them
in a form a caller can read back: a small in-process ledger of how many plans reached a device,
how many declined, and why. It is what a benchmark needs to report "the GPU ran 9 of 22
queries" rather than 22 timings that silently include 13 CPU runs, and what a user needs to
answer "did that actually use my GPU".

Deliberately a counter and not a log sink: it holds bounded, aggregate counts keyed by a short
reason string, so a fan-out declining a thousand shards for one reason costs one entry.
"""

from __future__ import annotations

import threading

__all__ = [
    "GpuLedger",
    "gpu_ledger",
    "note_gpu_declined",
    "note_gpu_ran",
    "reset_gpu_ledger",
]


#: Distinct decline reasons a ledger keeps before folding the rest into `"other"`. The tier has
#: well under this many *kinds* of decline; the cap exists for reasons that carry a measurement.
_MAX_REASONS = 32


class GpuLedger:
    """Per-process counts of device runs and declines, keyed by reason.

    Thread-safe: a distributed collect drives its fan-out from several driver threads, and two
    of them recording a decline at once must not lose one.
    """

    __slots__ = ("_declined", "_lock", "_ran")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ran = 0
        self._declined: dict[str, int] = {}

    @property
    def ran(self) -> int:
        """Plans this process ran on a device."""
        return self._ran

    @property
    def declined(self) -> dict[str, int]:
        """Declines by reason, as a copy — the caller cannot mutate the ledger through it."""
        with self._lock:
            return dict(self._declined)

    def record_ran(self) -> None:
        """Count one plan that reached a device."""
        with self._lock:
            self._ran += 1

    def record_declined(self, reason: str) -> None:
        """Count one plan the device tier declined, under a short stable `reason`.

        Bounded at `_MAX_REASONS` distinct keys, everything past that folded into `"other"`.
        Not defensive padding: one of the reasons this ledger is handed is
        `kyber.gpu.policy`'s working-set veto, which formats a *measured* size into its text
        (`"~3.4GB per device: ..."`). Every distinct size is then a distinct key, so a fan-out
        over a thousand differently-sized shards would put a thousand entries in a structure
        whose whole purpose is to be a small summary.
        """
        with self._lock:
            if reason not in self._declined and len(self._declined) >= _MAX_REASONS:
                reason = "other"
            self._declined[reason] = self._declined.get(reason, 0) + 1

    def reset(self) -> None:
        """Forget everything counted so far."""
        with self._lock:
            self._ran = 0
            self._declined.clear()

    def snapshot(self) -> dict[str, object]:
        """`{"ran": n, "declined": {reason: n}}` — the whole ledger as plain data."""
        with self._lock:
            return {"ran": self._ran, "declined": dict(self._declined)}


_LEDGER = GpuLedger()


def gpu_ledger() -> GpuLedger:
    """This process's device-tier ledger."""
    return _LEDGER


def note_gpu_ran() -> None:
    """Record that a plan ran on a device."""
    _LEDGER.record_ran()


def note_gpu_declined(reason: str) -> None:
    """Record that the device tier declined a plan.

    Args:
        reason: A short, stable phrase — `"no visible device"`, `"untranslatable shape"`,
            `"kyber routed to cpu"`. Bounded vocabulary: it is a dictionary key, and an
            unbounded one (a formatted exception message, a plan hash) makes the ledger grow
            with the workload instead of with the number of distinct failure modes.
    """
    _LEDGER.record_declined(reason)


def reset_gpu_ledger() -> None:
    """Clear the ledger — for a benchmark measuring one query at a time, and for tests."""
    _LEDGER.reset()
