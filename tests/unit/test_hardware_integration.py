"""The seams where a hardware fact changes a real decision.

A device table nothing reads is documentation. These cover the points where one is consulted:
the packing fraction comes from the device's own MIG profiles, the GPU grant is clamped by the
devices that are healthy as well as the ones that exist, a stage the host copy would lose is
kept on the CPU, and an inference stage is sized against the memory a co-tenant left free.

Each one also pins the *inert* direction, because every switch here defaults off: an unlabelled
fleet, an unrecognized device, and absent telemetry must all reproduce the pre-existing
behavior exactly.
"""

from __future__ import annotations

import pytest

from batcher._internal.device_specs import (
    device_host_link,
    device_host_link_gbps,
    host_transfer_seconds,
)
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.policies.scheduling import DefaultSchedulingPolicy
from batcher.config import (
    AcceleratorConfig,
    Config,
    DeviceHealthConfig,
    EnergyConfig,
    config_context,
)
from batcher.dist.executors.ray_runtime.fabric import GpuNodeTopology, plan_collective
from batcher.governance import DataResidency, ResidencyCatalog, set_residency
from batcher.kyber.gpu.policy import _mig_fraction, _transfer_veto
from batcher.plan.energy import configured_grid
from batcher.plan.resource import HardwareProfile

pytestmark = pytest.mark.unit


# --- the host link, the fact a data engine's GPU decision turns on -----------------------


def test_the_host_link_is_recorded_per_generation() -> None:
    assert device_host_link("NVIDIA_H100") == "pcie5"
    assert device_host_link("NVIDIA_TESLA_T4") == "pcie3"
    assert device_host_link("NVIDIA_GB200") == "nvlink-c2c"
    assert device_host_link("MADE_UP") == ""


def test_a_coherent_package_is_an_order_of_magnitude_off_pcie() -> None:
    assert device_host_link_gbps("NVIDIA_GB200") > 8 * device_host_link_gbps("NVIDIA_H100")


def test_transfer_time_is_bytes_over_the_link() -> None:
    ten_gib = 10 * (1 << 30)
    assert host_transfer_seconds(ten_gib, "NVIDIA_H100") == pytest.approx(ten_gib / 50e9, rel=1e-6)
    assert host_transfer_seconds(ten_gib, "NVIDIA_H100", round_trip=True) == pytest.approx(
        2 * ten_gib / 50e9, rel=1e-6
    )


def test_an_unknown_link_models_no_transfer_rather_than_a_penalty() -> None:
    assert host_transfer_seconds(1 << 30, "MADE_UP") == 0.0
    assert host_transfer_seconds(0, "NVIDIA_H100") == 0.0


# --- the packing fraction comes from the device's own profiles ---------------------------


def test_a_partitionable_device_packs_on_its_instance_boundary() -> None:
    assert _mig_fraction(6.0, "NVIDIA_H100") == pytest.approx(1 / 7)


def test_an_unlabelled_or_unpartitionable_device_keeps_the_quanta() -> None:
    assert _mig_fraction(6.0, "") is None, "an unlabelled fleet packs exactly as before"
    assert _mig_fraction(6.0, "NVIDIA_L40S") is None, "no MIG on this part"
    assert _mig_fraction(70.0, "NVIDIA_H100") is None, "needs the whole device"


def test_the_switch_turns_partition_packing_off() -> None:
    with config_context(Config().replace(accelerator=AcceleratorConfig(prefer_mig=False))):
        assert _mig_fraction(6.0, "NVIDIA_H100") is None


# --- the grant is clamped by health as well as by inventory ------------------------------


def _health_on(**kw) -> Config:
    return Config().replace(
        accelerator=AcceleratorConfig(health=DeviceHealthConfig(enabled=True, **kw))
    )


def test_a_quarantined_device_is_not_granted(monkeypatch) -> None:
    readings = (
        DeviceTelemetry(index=0, memory_total_bytes=80 << 30),
        DeviceTelemetry(index=1, memory_total_bytes=80 << 30, ecc_uncorrected=3),
    )
    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: readings)
    with config_context(_health_on()):
        env = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=1.0, n_tasks=8, gpu_count=2, accelerator_type="NVIDIA_H100"
        )
    assert env.n_tasks == 1, "the device returning wrong tensors is not schedulable"


def test_absent_telemetry_grants_the_full_inventory() -> None:
    # No probe is not evidence of an unhealthy fleet; the alternative takes a cluster offline
    # the day pynvml stops being installed.
    with config_context(_health_on()):
        env = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=1.0, n_tasks=8, gpu_count=4, accelerator_type="NVIDIA_H100"
        )
    assert env.n_tasks == 4


def test_health_checking_is_off_by_default(monkeypatch) -> None:
    def _fail():  # pragma: no cover - the point is that it is never called
        raise AssertionError("telemetry must not be read when health checking is off")

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", _fail)
    env = DefaultSchedulingPolicy.gpu_envelope(
        num_gpus=1.0, n_tasks=8, gpu_count=4, accelerator_type="NVIDIA_H100"
    )
    assert env.n_tasks == 4


# --- a stage the host copy would lose stays on the CPU -----------------------------------


def test_a_slow_link_vetoes_the_device() -> None:
    reason = _transfer_veto("NVIDIA_TESLA_T4", working_set_gb=10.0, rows=100_000_000)
    assert reason is not None
    assert "CPU wins" in reason
    assert "transfer" in reason


def test_a_fast_link_does_not_veto() -> None:
    assert _transfer_veto("NVIDIA_H100", working_set_gb=10.0, rows=100_000_000) is None
    assert _transfer_veto("NVIDIA_GB200", working_set_gb=10.0, rows=100_000_000) is None


def test_an_unknown_device_never_vetoes() -> None:
    assert _transfer_veto("MADE_UP", working_set_gb=10.0, rows=100_000_000) is None


# --- placement composes residency, power, and efficiency ---------------------------------


_FLEET = (
    GpuNodeTopology("us", 8, "NVIDIA_H100", region="us-east-1", power_zone="bw1"),
    GpuNodeTopology("eu", 8, "NVIDIA_H100", region="eu-north-1", power_zone="bw2"),
)


@pytest.fixture
def eu_only():
    catalog = ResidencyCatalog(mode="strict").register(
        DataResidency("s3://eu/", frozenset({"eu-north-1"}), "GDPR Art. 44")
    )
    previous = set_residency(catalog)
    yield
    set_residency(previous)


def test_placement_removes_a_forbidden_region_before_choosing(eu_only) -> None:
    plan = plan_collective(8, _FLEET, datasets=["s3://eu/orders"])
    assert plan.node_ids == ("eu",), "the US node is not a candidate, not merely a worse one"


def test_placement_ignores_residency_when_no_dataset_is_named(eu_only) -> None:
    # A rule constrains the datasets it governs, not the fleet: a stage that names no input
    # keeps every node, even with a strict catalog installed.
    assert set(plan_collective(16, _FLEET).node_ids) == {"eu", "us"}


def test_a_zone_with_no_power_left_is_skipped() -> None:
    plan = plan_collective(8, _FLEET, zone_budget_watts=1_000.0)
    assert plan.bundles == ()
    assert "no eligible node of 2" in plan.reason
    assert "unreadable" not in plan.reason, "a filtered fleet is not an unreadable one"


def test_an_unreadable_fleet_still_says_so() -> None:
    assert "unreadable" in plan_collective(8, ()).reason


def test_efficiency_first_orders_the_candidates(monkeypatch) -> None:
    mixed = (
        GpuNodeTopology("old", 8, "NVIDIA_TESLA_V100"),
        GpuNodeTopology("new", 8, "NVIDIA_H100"),
    )
    assert plan_collective(8, mixed).node_ids == ("new",), "widest domain first, then node id"
    cfg = Config().replace(accelerator=AcceleratorConfig(efficiency_first_placement=True))
    with config_context(cfg):
        assert plan_collective(8, mixed).node_ids == ("new",), "and the efficient part first"


# --- the contract carries the device model, and the grid comes from config ---------------


def test_the_hardware_profile_carries_the_device_model() -> None:
    assert HardwareProfile().accelerator_type == "", "unknown by default"
    profile = HardwareProfile.for_cluster(
        cpu_cores=8, memory_bytes=1 << 30, worker_count=2, accelerator_type="NVIDIA_H100"
    )
    assert profile.accelerator_type == "NVIDIA_H100"


def test_the_grid_profile_is_read_from_configuration() -> None:
    assert not configured_grid().configured, "unconfigured by default, so no invented figures"
    cfg = Config().replace(
        accelerator=AcceleratorConfig(
            energy=EnergyConfig(carbon_intensity=20.0, price_per_kwh=0.05, pue=1.15, region="no")
        )
    )
    with config_context(cfg):
        grid = configured_grid()
    assert grid.configured
    assert grid.region == "no"
    assert grid.carbon_grams(3.6e6) == pytest.approx(23.0)
