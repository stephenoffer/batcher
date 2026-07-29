"""Spilling is costed against the device it actually spills to.

Whether to accept a plan that spills is the optimizer's most consequential memory decision,
and the right answer depends on the storage underneath. On local NVMe a spilled plan is
frequently better than a contorted one that avoids spilling; on a network-attached volume the
same plan can run an order of magnitude longer. One constant across both makes the optimizer
confidently right on one class of machine and confidently wrong on the rest.

The constraint on the fix is that it must be a strict refinement: a device nobody can identify
has to cost exactly what it always did, so an unreadable `/sys` cannot re-rank a single plan.
"""

from __future__ import annotations

import pytest

from batcher.kyber import cost as cost_mod
from batcher.kyber import storage_cost

pytestmark = pytest.mark.unit


@pytest.fixture
def model():
    """A `CostModel` with no estimator — only the device-factor arithmetic is under test."""
    return cost_mod.CostModel.__new__(cost_mod.CostModel)


@pytest.fixture
def spill_device(monkeypatch):
    """Pin the class of the device backing the spill directory."""

    def configure(device: str):
        monkeypatch.setattr(storage_cost, "device_class", lambda path: device)

    return configure


def test_local_flash_is_the_unchanged_baseline(model, spill_device):
    # `_SPILL_WRITE_READ_PASSES` was calibrated on local flash, so flash must stay at 1.0.
    # Anything else would be a silent re-tuning of the whole spill cost term.
    for device in ("nvme", "ssd", "raid", "mapped"):
        spill_device(device)
        assert storage_cost.spill_device_factor() == 1.0, device


def test_an_unidentifiable_device_changes_nothing(model, spill_device):
    # The property that makes this a refinement rather than a re-tuning: absence of a
    # measurement must never move a plan. A container with no /sys access must plan exactly
    # as it did before this model existed.
    spill_device("unknown")
    assert storage_cost.spill_device_factor() == 1.0
    spill_device("a-class-that-does-not-exist-yet")
    assert storage_cost.spill_device_factor() == 1.0


def test_slow_devices_cost_more(model, spill_device):
    # The whole point. A network volume and a spinning disk are genuinely far slower, and the
    # ordering between them must hold too.
    spill_device("network")
    network = storage_cost.spill_device_factor()
    spill_device("rotational")
    rotational = storage_cost.spill_device_factor()
    assert 1.0 < network < rotational


def test_tmpfs_is_costed_honestly_as_fast(model, spill_device):
    # Spilling to RAM relieves no memory pressure, which is a real trap — but it is a
    # feasibility question, not a ranking one. Encoding the warning as a fake throughput
    # number would put it in the one place nobody reads, and would make the cost model lie
    # about a device that is genuinely fast.
    spill_device("memory")
    assert storage_cost.spill_device_factor() == 1.0


def test_the_factor_scales_the_spill_io_term(model, spill_device, monkeypatch):
    # A factor nothing consults changes nothing. Both spill sites — the flat write-and-read
    # term and the external-merge term — must scale, or a sort and an aggregate would
    # disagree about the same disk.
    monkeypatch.setattr(model, "_memory_budget", lambda: 1_000.0)
    spill_device("nvme")
    flash = model._spill_io(state_bytes=5_000.0)
    spill_device("rotational")
    disk = model._spill_io(state_bytes=5_000.0)
    assert disk == flash * storage_cost.SPILL_DEVICE_FACTOR["rotational"]
    # Below the budget nothing spills, so the device is irrelevant on both.
    assert model._spill_io(state_bytes=500.0) == 0.0


def test_an_unbounded_budget_still_charges_no_spill(model, spill_device, monkeypatch):
    # `spill_budget_bytes() == 0` means the user opted out of bounded memory: nothing spills,
    # so the device factor must not conjure an IO cost out of a disabled term.
    monkeypatch.setattr(model, "_memory_budget", lambda: 0.0)
    spill_device("rotational")
    assert model._spill_io(state_bytes=10**12) == 0.0
