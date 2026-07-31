"""`bt.accelerators()`: report what is there, omit what cannot be answered.

The report is the first thing someone reads when a GPU pipeline misbehaves, so its contract is
that every key present is a fact. A CPU-only host produces a small honest report rather than a
large one full of zeros, and a key whose source could not answer is absent rather than
defaulted.
"""

from __future__ import annotations

import sys

import pytest

import batcher as bt
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.config import AcceleratorConfig, Config, EnergyConfig, config_context

pytestmark = pytest.mark.unit

# The modules, not the functions the package re-exports over them (the same shadowing
# `versions` has). An `import a.b.c as m` would bind the function here.
#
# `accelerators` became a package when the report outgrew one file: `rows` builds each
# device's row and `report` decides what a reader is shown, so a patch belongs on whichever
# of the two owns the name.
accel_mod = sys.modules["batcher.api.session.accelerators.report"]
rows_mod = sys.modules["batcher.api.session.accelerators.rows"]


def test_a_cpu_only_host_reports_a_backend_and_no_devices() -> None:
    report = bt.accelerators()
    assert report["backend"] in {"cuda", "rocm", "xpu", "mps", "tpu", "neuron", "hpu", "cpu"}
    assert isinstance(report["devices"], list)
    assert isinstance(report["power"], dict)


def test_a_fleet_key_appears_only_when_the_cluster_has_accelerator_nodes() -> None:
    # Ray is not running here, so the topology cannot answer and the key must be absent
    # rather than present with zeros.
    assert "fleet" not in bt.accelerators()


def test_a_configured_budget_is_reported() -> None:
    cfg = Config().replace(
        accelerator=AcceleratorConfig(energy=EnergyConfig(power_budget_watts=10_000.0))
    )
    with config_context(cfg):
        assert bt.accelerators()["power"]["budget_watts"] == 10_000.0
    assert "budget_watts" not in bt.accelerators()["power"], "unbounded is not zero"


def test_device_rows_carry_nameplate_and_live_figures(monkeypatch) -> None:
    monkeypatch.setattr(
        accel_mod,
        "device_rows",
        lambda: [
            {
                "index": 0,
                "name": "NVIDIA H100 80GB HBM3",
                "memory_gib": 80.0,
                "tdp_watts": 700.0,
                "nvlink_domain": 8,
                "power_watts": 512.0,
                "sm_utilization": 0.91,
            }
        ],
    )
    row = bt.accelerators()["devices"][0]
    assert row["nvlink_domain"] == 8
    assert row["sm_utilization"] == 0.91


def test_rows_are_built_from_inventory_and_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(
        "batcher._internal.hardware.gpu_inventory",
        lambda: [
            {"name": "NVIDIA H100", "memory_bytes": 80 << 30, "accelerator_type": "NVIDIA_H100"}
        ],
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.device_telemetry",
        lambda: (
            DeviceTelemetry(
                index=0,
                name="NVIDIA H100",
                power_watts=512.0,
                sm_utilization=0.91,
                temperature_c=64.0,
                throttle_reasons=("power",),
                ecc_uncorrected=2,
            ),
        ),
    )
    row = rows_mod.device_rows()[0]
    assert row["memory_gib"] == 80.0
    assert row["tdp_watts"] == 700.0, "nameplate figures come from the device table"
    assert row["host_link"] == "pcie5", "why a stage is transfer-bound, from the report alone"
    assert row["host_link_gbps"] == 50.0
    assert row["nvlink_gbps"] == 900.0
    assert row["power_watts"] == 512.0
    assert row["throttled"] == ["power"]
    assert row["ecc_uncorrected"] == 2


def test_an_unrecognized_device_reports_only_what_is_known(monkeypatch) -> None:
    monkeypatch.setattr(
        "batcher._internal.hardware.gpu_inventory",
        lambda: [{"name": "Some Future GPU", "memory_bytes": 0}],
    )
    monkeypatch.setattr("batcher._internal.hardware.device_telemetry", tuple)
    row = rows_mod.device_rows()[0]
    assert row == {"index": 0, "name": "Some Future GPU"}, "no invented memory, power, or domain"


def test_show_accelerators_prints_without_a_device(capsys) -> None:
    bt.show_accelerators()
    out = capsys.readouterr().out
    assert "backend:" in out
    assert "devices: none visible" in out
