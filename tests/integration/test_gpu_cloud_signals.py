"""The GPU-cloud signals end to end: measure the node, then watch a decision change.

Each probe has unit tests that pin what it reads off `/sys` or NVML. What those cannot show is
that the reading reaches the thing that was supposed to act on it — and every gap of that kind
here is silent, because the query still returns the right rows. A fabric nobody prices, a
condemned device nobody avoids, a scratch volume nobody spills to: all of them look exactly
like a healthy fast fleet from inside a job.

So each test below fakes one node-level fact and asserts on a *decision*, never on the probe
that produced it. No engine, no GPU, no cluster: every layer in the chain is control plane,
which is why the whole thing runs on a CPU-only host.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric.pcie import PcieLink
from batcher._internal.hardware.faults.counters import DeviceFaults
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel import assess_faults, assess_fleet, health
from batcher.kyber.cost import Cost
from batcher.kyber.cost import fabric as cost_fabric

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _fresh_fabric_weight():
    cost_fabric.reset_fabric_weight()
    yield
    cost_fabric.reset_fabric_weight()


def _healthy(index: int = 0, **kw) -> DeviceTelemetry:
    base = {
        "index": index,
        "uuid": f"GPU-{index}",
        "temperature_c": 45.0,
        "memory_used_bytes": 1,
        "memory_total_bytes": 100,
    }
    return DeviceTelemetry(**{**base, **kw})


def test_the_fabric_a_node_measures_reaches_the_plan_ranking(monkeypatch):
    # The whole chain: /sys reports the ports, the weight is derived from their rate, and the
    # scalar a plan is ranked by moves. A shuffle-heavy plan that loses on a 10 Gb/s VM wins
    # on an InfiniBand node, and nothing between the two is aware of the other.
    shuffle_heavy = Cost(cpu=10.0, io=5.0, net=100.0)
    local_heavy = Cost(cpu=200.0, io=5.0, net=0.0)

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 10.0)
    cost_fabric.reset_fabric_weight()
    assert shuffle_heavy.total() > local_heavy.total()

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 3200.0)
    cost_fabric.reset_fabric_weight()
    assert shuffle_heavy.total() < local_heavy.total()


def test_a_container_that_cannot_see_the_fabric_can_declare_it(monkeypatch):
    # The deployment that most needs the measurement is the one that cannot take it: a pod
    # without the host's `/sys` has a real fabric it simply cannot read.
    from batcher.config import AcceleratorConfig, Config, config_context

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 0.0)
    cost_fabric.reset_fabric_weight()
    assert cost_fabric.fabric_net_weight() is None

    with config_context(Config(accelerator=AcceleratorConfig(fabric_gbps=3200.0))):
        cost_fabric.reset_fabric_weight()
        assert cost_fabric.fabric_net_weight() == pytest.approx(1.0)


def test_a_condemned_device_is_avoided_by_every_layer_that_places_work(monkeypatch):
    # One fault, three consumers: the verdict, the schedulable count, and the pool's choice of
    # device. A device that has run out of spare memory rows reports *all* of its memory free,
    # which makes it the most attractive placement on the node and the only one certain to fail.
    from batcher.carbonite.accel import VramPool

    readings = (_healthy(0), _healthy(1))
    faults = (
        DeviceFaults(index=0, uuid="GPU-0", readable=True),
        DeviceFaults(index=1, uuid="GPU-1", remap_failure=True, readable=True),
    )
    verdicts = assess_fleet(readings, faults=faults)
    assert [v.state for v in verdicts] == ["healthy", "quarantine"]

    schedulable = tuple(v.device_index for v in verdicts if v.schedulable)
    assert schedulable == (0,)

    gib = 1 << 30
    pool = VramPool(capacity_bytes=80 * gib, device_count=2, headroom=0.0)
    pool.reserve(70 * gib, device=0)  # the healthy device is nearly full
    assert pool.best_device() == 1  # free memory alone would pick the condemned one
    assert pool.best_device(exclude=[v.device_index for v in verdicts if not v.schedulable]) == 0


def test_a_degraded_host_link_moves_the_device_decision(monkeypatch):
    # The link is measured on the node; the veto is computed in Kyber. A board that came up at
    # a quarter width passes every health check and feeds at a quarter of the rate.
    from batcher.kyber.gpu.energy import device_energy_advice

    kwargs = {"bytes_per_row": 8192.0, "flops_per_row": 4.0}
    full = device_energy_advice("NVIDIA_H100", **kwargs, link_efficiency=1.0)
    degraded = PcieLink("0000:0c:00.0", gen=3, width=8, max_gen=5, max_width=16)
    quarter = device_energy_advice(
        "NVIDIA_H100", **kwargs, link_efficiency=degraded.degradation_ratio
    )
    assert degraded.degraded is True
    assert quarter.speedup < full.speedup
    assert quarter.transfer_share > full.transfer_share


def test_the_scratch_volume_a_node_has_reaches_both_the_spill_and_its_price(monkeypatch, tmp_path):
    # Two consumers that must agree: where the bytes go, and what the optimizer thinks they
    # cost to put there. They resolve the directory the same three ways for that reason.
    from batcher.dist.spill.scratch import _work_dir
    from batcher.kyber import storage_cost

    volume = tmp_path / "ephemeral"
    volume.mkdir()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: str(volume))
    seen: list[str] = []
    monkeypatch.setattr(
        "batcher._internal.hardware.storage.device_class",
        lambda path: seen.append(path) or "network",
    )
    work, owned = _work_dir(None, "probe_")
    assert work.startswith(str(volume))
    assert owned is True
    assert storage_cost.spill_device_factor() == storage_cost.SPILL_DEVICE_FACTOR["network"]
    assert seen[-1] == str(volume)


def test_a_slow_scratch_device_also_changes_how_the_spill_is_written(monkeypatch, tmp_path):
    # The same measurement, read by Carbonite rather than Kyber: on a slow volume a state
    # well under the size threshold is still worth compressing.
    from batcher._internal.hardware.storage import SPILL_DEVICE_FACTOR
    from batcher.carbonite.policies.spill_shape import SPILL_COMPRESS_ABOVE, should_compress

    modest = min(SPILL_COMPRESS_ABOVE // 4, 1 << 30)
    assert should_compress(modest) is False
    assert should_compress(modest, SPILL_DEVICE_FACTOR["network"]) is True


def test_a_fatal_xid_reaches_the_verdict_that_stops_the_retry_storm():
    # The failure this closes end to end: a device that has fallen off the bus enumerates,
    # reports a temperature, accepts work, and fails every task placed on it — so the retries
    # walk the whole queue onto the one bad board.
    readings = (_healthy(0), _healthy(1))
    faults = (
        DeviceFaults(index=0, uuid="GPU-0", pci_address="0000:0c:00.0", readable=True),
        DeviceFaults(index=1, uuid="GPU-1", pci_address="0000:1a:00.0", readable=True),
    )
    clean = assess_fleet(readings, faults=faults)
    assert all(v.schedulable for v in clean)
    after = health.xid_verdicts(clean, faults, {"0000:1a:00.0": (79,)})
    assert [v.schedulable for v in after] == [True, False]
    assert "xid_79" in after[1].reasons


def test_nothing_readable_leaves_every_decision_exactly_where_it_was(monkeypatch):
    # The property that matters more than any single signal: a container that can read none of
    # this must behave as the engine did before any of it existed. Wrong in this direction
    # would quarantine a fleet the day a base image stopped shipping `pynvml`.
    from batcher.carbonite.accel import VramPool

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 0.0)
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    cost_fabric.reset_fabric_weight()

    from batcher.config import CostWeights

    weights = CostWeights()
    assert cost_fabric.fabric_adjusted_weights(weights) is weights

    unreadable = DeviceFaults(index=0, uuid="GPU-0", remap_failure=True, readable=False)
    verdict = assess_faults(health.assess_device(_healthy()), unreadable)
    assert verdict.state == "healthy"

    gib = 1 << 30
    assert VramPool(capacity_bytes=80 * gib, headroom=0.0).usable_bytes() == 80 * gib
