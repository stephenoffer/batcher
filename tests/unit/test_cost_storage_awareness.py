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

from batcher.kyber import storage_cost
from batcher.kyber.cost import terms

pytestmark = pytest.mark.unit


@pytest.fixture
def spill_device(monkeypatch):
    """Pin the class of the device backing the spill directory."""

    def configure(device: str):
        # Patched where the probe *lives* rather than where it is used: the class table moved
        # down to layer 0 so Carbonite could read the same figures Kyber does without the two
        # subsystems importing each other, and `device_cost_factor` calls it from there.
        monkeypatch.setattr("batcher._internal.hardware.storage.device_class", lambda path: device)

    return configure


def test_local_flash_is_the_unchanged_baseline(spill_device):
    # `_SPILL_WRITE_READ_PASSES` was calibrated on local flash, so flash must stay at 1.0.
    # Anything else would be a silent re-tuning of the whole spill cost term.
    for device in ("nvme", "ssd", "raid", "mapped"):
        spill_device(device)
        assert storage_cost.spill_device_factor() == 1.0, device


def test_an_unidentifiable_device_changes_nothing(spill_device):
    # The property that makes this a refinement rather than a re-tuning: absence of a
    # measurement must never move a plan. A container with no /sys access must plan exactly
    # as it did before this model existed.
    spill_device("unknown")
    assert storage_cost.spill_device_factor() == 1.0
    spill_device("a-class-that-does-not-exist-yet")
    assert storage_cost.spill_device_factor() == 1.0


def test_slow_devices_cost_more(spill_device):
    # The whole point. A network volume and a spinning disk are genuinely far slower, and the
    # ordering between them must hold too.
    spill_device("network")
    network = storage_cost.spill_device_factor()
    spill_device("rotational")
    rotational = storage_cost.spill_device_factor()
    assert 1.0 < network < rotational


def test_tmpfs_is_costed_honestly_as_fast(spill_device):
    # Spilling to RAM relieves no memory pressure, which is a real trap — but it is a
    # feasibility question, not a ranking one. Encoding the warning as a fake throughput
    # number would put it in the one place nobody reads, and would make the cost model lie
    # about a device that is genuinely fast.
    spill_device("memory")
    assert storage_cost.spill_device_factor() == 1.0


def test_the_factor_scales_the_spill_io_term(spill_device, monkeypatch):
    # A factor nothing consults changes nothing. Both spill sites — the flat write-and-read
    # term and the external-merge term — must scale, or a sort and an aggregate would
    # disagree about the same disk.
    #
    # Patched at the `terms` module rather than on a `CostModel` instance: the budget is one
    # rule shared by every spill site, so binding the fake to the module is what proves all
    # of them move together. (A per-instance patch silently became a no-op when the term
    # moved out of the class — the failure mode `.claude/rules/concurrent-agents.md` names.)
    monkeypatch.setattr(terms, "memory_budget", lambda: 1_000.0)
    spill_device("nvme")
    flash = terms.spill_io(state_bytes=5_000.0)
    spill_device("rotational")
    disk = terms.spill_io(state_bytes=5_000.0)
    assert disk == flash * storage_cost.SPILL_DEVICE_FACTOR["rotational"]
    # Below the budget nothing spills, so the device is irrelevant on both.
    assert terms.spill_io(state_bytes=500.0) == 0.0


def test_an_unbounded_budget_still_charges_no_spill(spill_device, monkeypatch):
    # `spill_budget_bytes() == 0` means the user opted out of bounded memory: nothing spills,
    # so the device factor must not conjure an IO cost out of a disabled term.
    monkeypatch.setattr(terms, "memory_budget", lambda: 0.0)
    spill_device("rotational")
    assert terms.spill_io(state_bytes=10**12) == 0.0
    assert terms.merge_io(state_bytes=10**12) == 0.0


# --- The same table, read by Carbonite ------------------------------------------------------


def test_compression_on_local_flash_is_the_size_rule_it_always_was():
    from batcher.carbonite.policies.spill_shape import SPILL_COMPRESS_ABOVE, should_compress

    assert should_compress(0) is None
    assert should_compress(SPILL_COMPRESS_ABOVE) is True
    assert should_compress(SPILL_COMPRESS_ABOVE // 4) is False
    # An unidentified device is local flash for this purpose, so nothing moves.
    assert should_compress(SPILL_COMPRESS_ABOVE // 4, 1.0) is False


def test_a_slow_device_compresses_a_state_the_size_rule_would_write_raw():
    # On a network volume every byte not written is time not spent, so the trade pays well
    # below the size at which it pays on flash.
    from batcher._internal.hardware.storage import SPILL_DEVICE_FACTOR
    from batcher.carbonite.policies.spill_shape import SPILL_COMPRESS_ABOVE, should_compress

    modest = min(SPILL_COMPRESS_ABOVE // 4, 1 << 30)
    assert should_compress(modest, SPILL_DEVICE_FACTOR["network"]) is True
    assert should_compress(modest, SPILL_DEVICE_FACTOR["rotational"]) is True


def test_a_tiny_state_is_not_compressed_however_slow_the_device():
    # The whole spill is a handful of buffers; the codec's own setup dominates.
    from batcher._internal.hardware.storage import SPILL_DEVICE_FACTOR
    from batcher.carbonite.policies.spill_shape import should_compress

    assert should_compress(1 << 20, SPILL_DEVICE_FACTOR["network"]) is False


def test_both_subsystems_read_one_table_rather_than_two():
    # The failure this pins is the one the layer rules call out by name: Kyber and Carbonite
    # cannot import each other, so a shared figure pasted into both is the only wrong way to
    # share it — and the two copies then drift.
    from batcher._internal.hardware import storage as layer0
    from batcher.kyber import storage_cost as kyber_side

    assert kyber_side.SPILL_DEVICE_FACTOR is layer0.SPILL_DEVICE_FACTOR
    assert kyber_side.SPILL_DEVICE_FACTOR_DEFAULT == layer0.SPILL_DEVICE_FACTOR_DEFAULT


# --- The worker's device, not the driver's -------------------------------------------------
#
# Everything above resolves the class from *this process's* spill directory, which is exactly
# right single-node and describes the wrong machine on a cluster: the driver spills nothing,
# the workers do, to their own volumes. A driver on local NVMe planning for workers on a
# network volume under-stated a spilled byte tenfold, in the one term that decides whether an
# out-of-core plan is acceptable at all. `HardwareProfile.storage_class` carries the binding
# worker's measured class so the prediction is about the node that will pay for it.


def test_a_named_class_overrides_the_local_probe(spill_device):
    """The explicit class wins, whatever this process happens to be sitting on."""
    spill_device("nvme")
    assert storage_cost.spill_device_factor("network") == 10.0
    assert storage_cost.spill_device_factor("rotational") == 30.0


def test_no_named_class_still_reads_this_machine(spill_device):
    """The default is the pre-existing behavior, which every single-node caller takes."""
    spill_device("rotational")
    assert storage_cost.spill_device_factor("") == storage_cost.spill_device_factor()


def test_an_unrecognized_class_reports_no_opinion(spill_device):
    """A class this build has no figure for must not re-rank anything."""
    spill_device("nvme")
    assert storage_cost.spill_device_factor("some-future-device") == 1.0


def test_the_spill_terms_take_the_workers_device(spill_device):
    """Both out-of-core terms scale, and the merge is where it compounds per pass."""
    spill_device("nvme")
    budget = terms.memory_budget(8 << 30)
    state = 32e9
    assert terms.spill_io(state, budget, "network") == pytest.approx(
        10.0 * terms.spill_io(state, budget)
    )
    assert terms.merge_io(state, budget, "rotational") == pytest.approx(
        30.0 * terms.merge_io(state, budget)
    )


def test_the_cost_model_reads_the_profile(spill_device):
    """End to end: two models over one plan, differing only in the volume they spill to."""
    import batcher as bt
    from batcher import col
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel
    from batcher.plan.resource import HardwareProfile

    spill_device("nvme")
    frame = bt.from_pydict({"k": list(range(5_000)), "v": list(range(5_000))})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    # A worker small enough that this aggregate spills, so the device factor has something to
    # scale. Only the device differs between the two.
    tiny = 1
    flash = CostModel(estimator, hardware=HardwareProfile(memory_bytes=tiny, storage_class="nvme"))
    remote = CostModel(
        estimator, hardware=HardwareProfile(memory_bytes=tiny, storage_class="network")
    )
    # The `io` axis carries scanned bytes as well as spilled ones, and only the spill half is
    # device-scaled — so the ratio to assert is of the *spill component*, isolated by a third
    # model whose worker is large enough that nothing spills at all.
    resident = CostModel(
        estimator, hardware=HardwareProfile(memory_bytes=1 << 50, storage_class="nvme")
    )
    scanned = resident.cost(plan).io
    flash_spill = flash.cost(plan).io - scanned
    remote_spill = remote.cost(plan).io - scanned
    assert flash_spill > 0.0
    assert remote_spill == pytest.approx(10.0 * flash_spill)


def test_a_profile_that_could_not_probe_the_device_is_unchanged(spill_device):
    """`""` on the profile is "unprobeable", and must read as the local answer."""
    from batcher.kyber.cost.model import CostModel
    from batcher.kyber.stats.estimator import StatsEstimator
    from batcher.plan.resource import HardwareProfile

    spill_device("nvme")
    model = CostModel(StatsEstimator([]), hardware=HardwareProfile(cpu_cores=8))
    assert model._storage_class == ""
