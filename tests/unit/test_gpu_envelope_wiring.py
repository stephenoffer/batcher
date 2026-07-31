"""The GPU envelope the conductor actually builds, not the one a test constructs directly.

The defect this covers is the one that hides best: a clamp written on a function nothing calls.
`DefaultSchedulingPolicy.gpu_envelope` held the inventory, power, and health ceilings for
several commits while the live path — `api.executors._map_scheduling_envelope` — sized a GPU
stage's fan-out from the *CPU* count and never consulted any of them.

So these drive the conductor's function and assert on what it produces. A GPU stage's fan-out
must be bounded by devices; a CPU stage must be untouched; and every ceiling beyond inventory
must stay off until it is configured.
"""

from __future__ import annotations

import importlib

import pytest

import batcher as bt
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.config import (
    AcceleratorConfig,
    Config,
    DeviceHealthConfig,
    EnergyConfig,
    config_context,
)

pytestmark = pytest.mark.unit

executors = importlib.import_module("batcher.api.executors")


def _envelope(monkeypatch, *, devices: int, model_gb: float = 8.0, num_gpus: float = 1.0):
    """The envelope the conductor builds for a one-stage GPU map pipeline."""
    monkeypatch.setattr(executors, "_gpu_device_count", lambda: devices)
    plan = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(lambda b: b, num_gpus=num_gpus, model_memory_gb=model_gb)
        ._plan
    )
    return executors._map_scheduling_envelope(plan, num_workers=64, hub=None)


def test_a_gpu_stage_is_bounded_by_devices_not_by_cores(monkeypatch) -> None:
    env = _envelope(monkeypatch, devices=4)
    assert env.num_gpus > 0
    assert env.n_tasks <= 4, "64 actors on a 4-GPU cluster is 60 requests that never schedule"


def test_a_fractional_request_packs_more_actors_than_devices(monkeypatch) -> None:
    # Packing several light models onto one device is the point of a fractional request, so
    # the clamp must be devices/fraction rather than devices.
    env = _envelope(monkeypatch, devices=4, model_gb=1.0, num_gpus=0.25)
    assert env.num_gpus <= 0.25
    assert env.n_tasks > 4


def test_a_cluster_with_no_visible_gpu_keeps_the_requested_fan_out(monkeypatch) -> None:
    # An unreadable inventory must not shrink a stage the user sized deliberately.
    env = _envelope(monkeypatch, devices=0)
    assert env.n_tasks == 64


def test_a_cpu_stage_is_untouched(monkeypatch) -> None:
    monkeypatch.setattr(executors, "_gpu_device_count", lambda: 4)
    plan = bt.from_pydict({"x": [1, 2, 3]}).map_batches(lambda b: b)._plan
    env = executors._map_scheduling_envelope(plan, num_workers=64, hub=None)
    assert env.num_gpus == 0.0
    assert env.n_tasks == 64, "no GPU request, no GPU ceiling"


def test_the_power_budget_binds_through_the_conductor(monkeypatch) -> None:
    monkeypatch.setattr(executors, "_gpu_device_count", lambda: 64)
    plan = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(
            lambda b: b, num_gpus=1.0, model_memory_gb=40.0, accelerator_type="NVIDIA_H100"
        )
        ._plan
    )
    budget = Config().replace(
        accelerator=AcceleratorConfig(energy=EnergyConfig(power_budget_watts=10_000.0))
    )
    with config_context(budget):
        env = executors._map_scheduling_envelope(plan, num_workers=64, hub=None)
    assert env.n_tasks == 10, "a 10 kW budget powers ten 700 W devices, not sixty-four"


def test_an_unhealthy_device_is_not_granted_through_the_conductor(monkeypatch) -> None:
    monkeypatch.setattr(executors, "_gpu_device_count", lambda: 2)
    monkeypatch.setattr(
        "batcher._internal.hardware.nvml.device_telemetry",
        lambda: (
            DeviceTelemetry(index=0, memory_total_bytes=80 << 30),
            DeviceTelemetry(index=1, memory_total_bytes=80 << 30, ecc_uncorrected=1),
        ),
    )
    plan = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(
            lambda b: b, num_gpus=1.0, model_memory_gb=40.0, accelerator_type="NVIDIA_H100"
        )
        ._plan
    )
    with config_context(
        Config().replace(accelerator=AcceleratorConfig(health=DeviceHealthConfig(enabled=True)))
    ):
        env = executors._map_scheduling_envelope(plan, num_workers=8, hub=None)
    assert env.n_tasks == 1, "the device reporting uncorrectable ECC is not schedulable"


def test_neither_ceiling_applies_until_it_is_configured(monkeypatch) -> None:
    def _fail():  # pragma: no cover - the point is that it is never called
        raise AssertionError("telemetry must not be read with health checking off")

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", _fail)
    env = _envelope(monkeypatch, devices=8)
    assert env.n_tasks == 8, "inventory only: no budget, no health probe"


def test_the_device_count_and_the_binding_vram_stay_separate_reads(monkeypatch) -> None:
    # Fusing them into one topology call routed around `cluster_gpu_memory_gb`, whose whole
    # job is to report the *smallest* device in the fleet — and packing a fraction derived
    # from the driver's larger device onto a smaller worker is an OOM on every one of them.
    seen: list[str] = []
    monkeypatch.setattr(executors, "_gpu_device_count", lambda: seen.append("count") or 8)
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.accelerators.cluster_gpu_memory_gb",
        lambda: seen.append("vram") or 16.0,
    )
    plan = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(lambda b: b, num_gpus=1.0, model_memory_gb=4.0)
        ._plan
    )
    executors._map_scheduling_envelope(plan, num_workers=16, hub=None)
    assert "vram" in seen, "the binding-device accessor must still be the source of VRAM"
    assert "count" in seen
