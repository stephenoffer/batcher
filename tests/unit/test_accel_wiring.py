"""The accelerator primitives where they actually take effect.

A power model nothing consults and a fabric map nothing places against are documentation, not
behavior. These cover the three points where the new facts change a decision: Carbonite's GPU
grant is clamped by the power budget as well as by inventory, a gang-scheduled collective is
reported when it is wider than any fabric domain, and an inference stage's concurrency comes
from its KV cache rather than from its weights.
"""

from __future__ import annotations

import logging

import pytest

from batcher.carbonite.policies.scheduling import DefaultSchedulingPolicy
from batcher.config import AcceleratorConfig, Config, EnergyConfig, config_context
from batcher.dist.executors.ray_runtime.scheduling import _report_collective_fabric
from batcher.ml.llm.sizing import kv_cache_concurrency
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit


def _budget(watts: float) -> Config:
    return Config().replace(
        accelerator=AcceleratorConfig(energy=EnergyConfig(power_budget_watts=watts))
    )


# --- Carbonite's GPU grant -------------------------------------------------------------


def test_inventory_still_binds_when_no_budget_is_configured() -> None:
    env = DefaultSchedulingPolicy.gpu_envelope(
        num_gpus=1.0, n_tasks=64, gpu_count=32, accelerator_type="NVIDIA_H100"
    )
    assert env.n_tasks == 32
    assert env.accelerator_type == "NVIDIA_H100"


def test_a_power_budget_binds_before_inventory() -> None:
    with config_context(_budget(10_000.0)):
        env = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=1.0, n_tasks=64, gpu_count=32, accelerator_type="NVIDIA_H100"
        )
    assert env.n_tasks == 10, "a 10 kW budget powers ten 700 W devices with their host share"


def test_fractional_requests_still_pack_under_a_budget() -> None:
    with config_context(_budget(10_000.0)):
        env = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=0.25, n_tasks=64, gpu_count=32, accelerator_type="NVIDIA_H100"
        )
    assert env.n_tasks == 40, "ten powered devices, four 0.25-GPU actors each"


def test_an_unknown_device_model_is_never_power_clamped() -> None:
    with config_context(_budget(1_000.0)):
        known = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=1.0, n_tasks=64, gpu_count=32, accelerator_type=None
        )
        made_up = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=1.0, n_tasks=64, gpu_count=32, accelerator_type="MADE_UP"
        )
    assert known.n_tasks == 32
    assert made_up.n_tasks == 32


def test_a_cluster_with_no_gpus_still_gets_no_gpu_grant() -> None:
    with config_context(_budget(10_000.0)):
        env = DefaultSchedulingPolicy.gpu_envelope(num_gpus=1.0, n_tasks=8, gpu_count=0)
    assert env.num_gpus == 0.0
    assert env.n_tasks == 8, "the stage runs on CPU rather than pending forever"


# --- the collective fabric report -------------------------------------------------------


def test_a_collective_wider_than_the_fabric_is_reported(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.largest_local_domain", lambda *a, **k: 8
    )
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        _report_collective_fabric(16, SchedulingEnvelope(gpu_collective=True))
    assert "wider than the fleet's fabric domain" in caplog.text


def test_a_collective_that_fits_is_not_warned_about(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.largest_local_domain", lambda *a, **k: 8
    )
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        _report_collective_fabric(8, SchedulingEnvelope(gpu_collective=True))
    assert caplog.text == ""


def test_an_unreadable_topology_reports_nothing(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.largest_local_domain", lambda *a, **k: 0
    )
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        _report_collective_fabric(16, SchedulingEnvelope(gpu_collective=True))
    assert caplog.text == ""


def test_a_non_collective_stage_is_not_examined(monkeypatch, caplog) -> None:
    def _fail(*a, **k):  # pragma: no cover - the point is that it is never called
        raise AssertionError("topology must not be read for a non-collective stage")

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.largest_local_domain", _fail
    )
    with caplog.at_level(logging.DEBUG, logger="batcher.dist"):
        _report_collective_fabric(16, SchedulingEnvelope())
        _report_collective_fabric(1, SchedulingEnvelope(gpu_collective=True))
    assert caplog.text == ""


def test_the_report_never_fails_a_placement(monkeypatch, caplog) -> None:
    def _boom(*a, **k):
        raise RuntimeError("topology exploded")

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.largest_local_domain", _boom
    )
    _report_collective_fabric(16, SchedulingEnvelope(gpu_collective=True))  # must not raise


def test_the_report_is_off_when_fabric_placement_is_disabled(monkeypatch) -> None:
    def _fail(*a, **k):  # pragma: no cover - the point is that it is never called
        raise AssertionError("topology must not be read when the switch is off")

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.largest_local_domain", _fail
    )
    cfg = Config().replace(accelerator=AcceleratorConfig(fabric_aware_placement=False))
    with config_context(cfg):
        _report_collective_fabric(16, SchedulingEnvelope(gpu_collective=True))


# --- inference concurrency --------------------------------------------------------------


def test_concurrency_comes_from_the_cache_not_the_weights() -> None:
    common = {
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16 << 30,
        "device_bytes": 80 << 30,
        "dtype": "fp16",
    }
    assert kv_cache_concurrency(context_tokens=8192, **common) == 56
    assert kv_cache_concurrency(context_tokens=4096, **common) == 112, "half the context, twice"


def test_an_fp8_cache_doubles_concurrency() -> None:
    common = {
        "context_tokens": 8192,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16 << 30,
        "device_bytes": 80 << 30,
    }
    assert kv_cache_concurrency(dtype="fp8", **common) == 2 * kv_cache_concurrency(
        dtype="fp16", **common
    )


def test_the_configured_cache_dtype_is_the_default() -> None:
    cfg = Config().replace(accelerator=AcceleratorConfig(kv_cache_dtype="fp8"))
    args = {
        "context_tokens": 8192,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16 << 30,
        "device_bytes": 80 << 30,
    }
    with config_context(cfg):
        assert kv_cache_concurrency(**args) == kv_cache_concurrency(dtype="fp8", **args)


def test_a_configured_context_cap_overrides_the_request() -> None:
    cfg = Config().replace(accelerator=AcceleratorConfig(max_context_tokens=4096))
    args = {
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16 << 30,
        "device_bytes": 80 << 30,
        "dtype": "fp16",
    }
    with config_context(cfg):
        assert kv_cache_concurrency(context_tokens=8192, **args) == 112


def test_weights_that_do_not_fit_report_no_concurrency() -> None:
    assert (
        kv_cache_concurrency(
            context_tokens=8192,
            layers=80,
            kv_heads=8,
            head_dim=128,
            weight_bytes=40 << 30,
            device_bytes=24 << 30,
            dtype="fp16",
        )
        == 0
    )


def test_no_visible_device_reports_zero_rather_than_a_default() -> None:
    assert (
        kv_cache_concurrency(
            context_tokens=8192,
            layers=32,
            kv_heads=8,
            head_dim=128,
            weight_bytes=16 << 30,
            device_bytes=0,
            dtype="fp16",
        )
        == 0
    )
