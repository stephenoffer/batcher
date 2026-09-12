"""The device tier must be able to say what it did — a silent fallback leaves only a timing.

`backend="gpu"` is documented as always safe: an unsupported shape, a device out of memory or
a cluster with no GPU returns the same rows from the CPU engine. The cost of that safety is
that a tier which has stopped translating anything looks, from outside, exactly like one that
is translating everything and is merely slow. Two whole-path outages have shipped here behind
that indistinguishability.

`note_gpu_failure` separates a defect from a decline in the *log*. The ledger separates them in
a form a caller can read back, which is what a benchmark needs to report "the device ran 9 of
22 queries" rather than 22 timings that silently include 13 CPU runs.
"""

from __future__ import annotations

import threading

import pytest

from batcher.api.terminal.gpu_backend.audit import (
    _MAX_REASONS,
    GpuLedger,
    gpu_ledger,
    note_gpu_declined,
    note_gpu_ran,
    reset_gpu_ledger,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    reset_gpu_ledger()
    yield
    reset_gpu_ledger()


def test_a_fresh_ledger_counts_nothing():
    assert gpu_ledger().snapshot() == {"ran": 0, "declined": {}}


def test_a_device_run_and_a_decline_are_counted_separately():
    note_gpu_ran()
    note_gpu_ran()
    note_gpu_declined("no visible device")
    assert gpu_ledger().snapshot() == {"ran": 2, "declined": {"no visible device": 1}}


def test_the_same_reason_accumulates():
    for _ in range(3):
        note_gpu_declined("untranslatable shape")
    assert gpu_ledger().declined == {"untranslatable shape": 3}


def test_the_declined_view_is_a_copy():
    """A caller mutating what it read must not be able to rewrite the ledger."""
    note_gpu_declined("untranslatable shape")
    view = gpu_ledger().declined
    view["untranslatable shape"] = 99
    view["invented"] = 1
    assert gpu_ledger().declined == {"untranslatable shape": 1}


def test_reset_clears_both_halves():
    note_gpu_ran()
    note_gpu_declined("untranslatable shape")
    reset_gpu_ledger()
    assert gpu_ledger().snapshot() == {"ran": 0, "declined": {}}


def test_reasons_are_bounded_so_a_measured_reason_cannot_grow_the_ledger():
    """`kyber.gpu.policy` formats a measured size into its veto text (`"~3.4GB per device"`),
    so every distinct shard size is a distinct key. Unbounded, a wide fan-out would put a
    thousand entries in a structure whose whole purpose is to be a small summary."""
    ledger = GpuLedger()
    for i in range(_MAX_REASONS * 3):
        ledger.record_declined(f"~{i}.0GB per device: will not fit")
    declined = ledger.declined
    assert len(declined) == _MAX_REASONS + 1
    assert declined["other"] == _MAX_REASONS * 2
    assert sum(declined.values()) == _MAX_REASONS * 3


def test_a_known_reason_still_accumulates_after_the_cap_is_reached():
    """Folding into `"other"` must not stop an already-known reason from being counted."""
    ledger = GpuLedger()
    ledger.record_declined("first")
    for i in range(_MAX_REASONS * 2):
        ledger.record_declined(f"filler {i}")
    ledger.record_declined("first")
    assert ledger.declined["first"] == 2


def test_concurrent_records_lose_nothing():
    """A distributed collect drives its fan-out from several driver threads."""
    ledger = GpuLedger()

    def _work():
        for _ in range(200):
            ledger.record_ran()
            ledger.record_declined("untranslatable shape")

    threads = [threading.Thread(target=_work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledger.ran == 1600
    assert ledger.declined == {"untranslatable shape": 1600}
