"""Residency where it reaches the scheduler, plus the health count and the feed advice.

The filter's two failure directions are asymmetric in the same way the policy's are. Dropping
a node it should have kept costs capacity for a reason nobody can see; keeping one it should
have dropped breaks the obligation the rule exists for. These pin both, and pin the reporting
that makes a shrunken fleet distinguishable from a busy one.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel import configured_thresholds, schedulable_device_count
from batcher.config import AcceleratorConfig, Config, DeviceHealthConfig, config_context
from batcher.dist.executors.ray_runtime.fabric import (
    GpuNodeTopology,
    fleet_regions,
    permitted_nodes,
    residency_report,
)
from batcher.governance import DataResidency, ResidencyCatalog
from batcher.ml.devices import device_feed_advice

pytestmark = pytest.mark.unit

_FLEET = (
    GpuNodeTopology(node_id="eu-a", gpus=8, accelerator_type="NVIDIA_H100", region="eu-north-1"),
    GpuNodeTopology(node_id="us-a", gpus=8, accelerator_type="NVIDIA_H100", region="us-east-1"),
    GpuNodeTopology(node_id="unlabelled", gpus=4, accelerator_type="NVIDIA_H100"),
)


def _catalog(mode: str = "strict") -> ResidencyCatalog:
    return ResidencyCatalog(mode=mode).register(
        DataResidency("s3://eu/", frozenset({"eu-north-1"}), "GDPR Art. 44")
    )


def test_fleet_regions_ignores_unlabelled_nodes() -> None:
    assert fleet_regions(_FLEET) == ("eu-north-1", "us-east-1")
    assert fleet_regions(()) == ()


def test_a_regulated_input_narrows_the_fleet() -> None:
    nodes = permitted_nodes(_catalog(), ["s3://eu/orders"], _FLEET)
    assert {n.node_id for n in nodes} == {"eu-a", "unlabelled"}


def test_an_unlabelled_node_is_never_filtered_out() -> None:
    # An unreadable label is not evidence of a violation, and dropping it would take a
    # cluster offline the day a label was missed.
    nodes = permitted_nodes(_catalog(), ["s3://eu/orders"], _FLEET)
    assert any(n.node_id == "unlabelled" for n in nodes)


def test_an_unregistered_input_restricts_nothing() -> None:
    nodes = permitted_nodes(_catalog(), ["s3://public/reference"], _FLEET)
    assert len(nodes) == len(_FLEET)


def test_mode_off_restricts_nothing() -> None:
    nodes = permitted_nodes(_catalog("off"), ["s3://eu/orders"], _FLEET)
    assert len(nodes) == len(_FLEET)


def test_the_report_makes_a_shrunken_fleet_visible() -> None:
    report = residency_report(_catalog(), ["s3://eu/orders"], _FLEET)
    assert report["mode"] == "strict"
    assert report["permitted_regions"] == ["eu-north-1"]
    assert report["fleet_regions"] == ["eu-north-1", "us-east-1"]
    assert report["gpus_total"] == 20
    assert report["gpus_permitted"] == 12
    assert report["excluded_nodes"] == 1


def test_the_report_says_unrestricted_rather_than_listing_every_region() -> None:
    report = residency_report(_catalog(), ["s3://public/x"], _FLEET)
    assert report["permitted_regions"] is None
    assert report["excluded_nodes"] == 0


# --- health count and feed advice --------------------------------------------------------


def _reading(**kw) -> DeviceTelemetry:
    base = {"index": 0, "memory_used_bytes": 8 << 30, "memory_total_bytes": 80 << 30}
    return DeviceTelemetry(**{**base, **kw})


def test_absent_telemetry_reports_unknown_not_zero() -> None:
    assert schedulable_device_count() is None, "no probe is not evidence of an unhealthy fleet"


def test_the_health_count_follows_the_configured_thresholds(monkeypatch) -> None:
    readings = (_reading(index=0), _reading(index=1, ecc_uncorrected=2))
    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: readings)
    assert schedulable_device_count() == 1
    tolerant = Config().replace(
        accelerator=AcceleratorConfig(health=DeviceHealthConfig(quarantine_on_ecc=False))
    )
    with config_context(tolerant):
        assert schedulable_device_count() == 2


def test_configured_thresholds_mirror_the_config() -> None:
    cfg = Config().replace(
        accelerator=AcceleratorConfig(health=DeviceHealthConfig(max_temperature_c=70.0))
    )
    with config_context(cfg):
        assert configured_thresholds().max_temperature_c == 70.0
    assert configured_thresholds().max_temperature_c == 87.0


def test_feed_advice_says_when_there_is_no_telemetry() -> None:
    assert "no device telemetry" in device_feed_advice()


def test_feed_advice_distinguishes_starved_from_saturated(monkeypatch) -> None:
    def _readings(util: float, **kw):
        return (_reading(sm_utilization=util, **kw),)

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: _readings(0.2))
    assert "starving" in device_feed_advice()
    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: _readings(0.95))
    assert "saturated" in device_feed_advice()
    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: _readings(0.6))
    assert "headroom" in device_feed_advice()


def test_a_clamped_device_changes_the_diagnosis(monkeypatch) -> None:
    monkeypatch.setattr(
        "batcher._internal.hardware.nvml.device_telemetry",
        lambda: (_reading(sm_utilization=0.3, throttle_reasons=("thermal",)),),
    )
    advice = device_feed_advice()
    assert "clamped" in advice
    assert "thermal" in advice
    assert "starving" not in advice, "a clamped device is not a feeding problem"
