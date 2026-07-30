"""What `bt.accelerators()` reports about the site, the fabric, and a silently sick device.

The report is what someone pastes into a bug report from a GPU node, so the additions here are
the conditions that are otherwise invisible: a host link that renegotiated to half width, a
device whose memory has repaired itself as far as it can, and which fabric the node is on. The
countervailing property, tested just as hard, is that none of it appears on a machine where
none of it is readable — a report that prints "provider: unknown" on a laptop has cost the
reader a line and told them nothing.
"""

from __future__ import annotations

import importlib

import pytest

from batcher._internal.hardware.fabric.pcie import PcieLink
from batcher._internal.hardware.faults.counters import DeviceFaults

# The module, not the function of the same name the package re-exports. `import a.b.c as m`
# binds the *attribute*, which the package's own re-export has already replaced with the
# function — so patching on it silently does nothing.
report_mod = importlib.import_module("batcher.api.session.accelerators")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quiet_environment(monkeypatch):
    """A machine with no site, no fabric, and no devices — the CI shape."""
    monkeypatch.setattr(report_mod, "_device_rows", lambda: [])
    for name in ("BATCHER_PROVIDER", "SLURM_JOB_ID", "KUBERNETES_SERVICE_HOST", "RAY_ADDRESS"):
        monkeypatch.delenv(name, raising=False)
    from batcher._internal.site import provider

    provider.reset_provider_probe()
    yield
    provider.reset_provider_probe()


def test_a_machine_with_nothing_to_say_reports_nothing_extra():
    report = report_mod.accelerators()
    assert "site" not in report
    assert "fabric" not in report
    assert sorted(report) == ["backend", "devices", "power"]


def test_a_gpu_cloud_node_reports_its_site(monkeypatch):
    monkeypatch.setenv("BATCHER_PROVIDER", "coreweave")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-02]")
    from batcher._internal.site import provider

    provider.reset_provider_probe()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: "/ephemeral")
    site = report_mod.accelerators()["site"]
    assert site["provider"] == "coreweave"
    assert site["scheduler"] == "slurm"
    assert site["scratch_dir"] == "/ephemeral"


def test_a_scheduled_job_reports_its_site_even_on_an_unnamed_platform(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    site = report_mod.accelerators()["site"]
    assert site["provider"] == "unknown"
    assert site["scheduler"] == "kubernetes"
    assert "scratch_dir" not in site


def test_the_fabric_section_appears_only_when_there_is_a_fabric(monkeypatch):
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma_summary",
        lambda: {
            "ports": 8,
            "active_ports": 8,
            "bandwidth_gbps": 3200.0,
            "link_layers": {"InfiniBand": 8},
            "rdma_available": True,
            "partition": "0xffff",
            "devices": [],
            "numa_nodes": [],
        },
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.nvlink_summary",
        lambda: {
            "devices": 8,
            "links": 144,
            "active_links": 144,
            "degraded_devices": 0,
            "errors": {},
        },
    )
    fabric = report_mod.accelerators()["fabric"]
    assert fabric["rdma"]["bandwidth_gbps"] == 3200.0
    assert fabric["nvlink"]["active_links"] == 144


def test_a_degraded_host_link_is_added_to_the_device_row(monkeypatch):
    link = PcieLink("0000:0c:00.0", gen=3, width=8, max_gen=5, max_width=16, numa_node=1)
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.device_links.device_pcie_links", lambda: (link,)
    )
    row: dict = {"index": 0}
    report_mod._add_measured_link(row, 0)
    assert row["numa_node"] == 1
    assert row["link_degraded"] == "gen3 x8 of gen5 x16"
    assert row["link_efficiency"] == pytest.approx(0.125, rel=1e-2)


def test_a_healthy_link_adds_no_noise(monkeypatch):
    link = PcieLink("0000:0c:00.0", gen=5, width=16, max_gen=5, max_width=16, numa_node=0)
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.device_links.device_pcie_links", lambda: (link,)
    )
    row: dict = {"index": 0}
    report_mod._add_measured_link(row, 0)
    assert row == {"index": 0, "numa_node": 0}


def test_memory_faults_reach_the_device_row_only_when_readable():
    row: dict = {"index": 0}
    report_mod._add_faults(row, DeviceFaults(index=0, remap_failure=True, readable=False))
    assert row == {"index": 0}
    report_mod._add_faults(
        row, DeviceFaults(index=0, remap_pending=True, pcie_replay=9, readable=True)
    )
    assert row["reset_pending"] is True
    assert row["pcie_replay"] == 9
    assert "remap_failure" not in row


def test_the_silent_faults_are_called_out_by_device(capsys):
    report_mod._show_silent_faults(
        [
            {"index": 0, "link_degraded": "gen3 x8 of gen5 x16", "link_efficiency": 0.125},
            {"index": 1, "remap_failure": True},
            {"index": 2, "reset_pending": True},
            {"index": 3},
        ]
    )
    out = capsys.readouterr().out
    assert "gpu 0  host link at gen3 x8 of gen5 x16 (12% of nameplate bandwidth)" in out
    assert "gpu 1  memory row remapping has FAILED" in out
    assert "gpu 2  memory repair pending" in out
    assert "gpu 3" not in out


def test_show_accelerators_prints_the_site_and_fabric(monkeypatch, capsys):
    monkeypatch.setenv("BATCHER_PROVIDER", "lambda")
    from batcher._internal.site import provider

    provider.reset_provider_probe()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: "/scratch")
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma_summary",
        lambda: {
            "ports": 8,
            "active_ports": 6,
            "bandwidth_gbps": 2400.0,
            "link_layers": {"InfiniBand": 6},
            "rdma_available": True,
            "partition": "",
            "devices": [],
            "numa_nodes": [],
        },
    )
    report_mod.show_accelerators()
    out = capsys.readouterr().out
    assert "site: lambda" in out
    assert "local scratch /scratch" in out
    assert "6/8 RDMA port(s) up (2400 Gb/s, 6 x InfiniBand)" in out


# --- The live device table ------------------------------------------------------------


def _reading(**kw):
    from batcher._internal.hardware.nvml import DeviceTelemetry

    base = {
        "index": 0,
        "name": "NVIDIA H100",
        "temperature_c": 44.0,
        "memory_total_bytes": 80 << 30,
        "memory_used_bytes": 3 << 30,
    }
    return DeviceTelemetry(**{**base, **kw})


@pytest.mark.parametrize(
    ("faults", "link", "expected"),
    [
        (None, None, "ok"),
        (DeviceFaults(index=0, remap_failure=True, readable=True), None, "rma:row-remap-failed"),
        (DeviceFaults(index=0, remap_pending=True, readable=True), None, "reset-pending"),
        (None, PcieLink("a", gen=3, width=8, max_gen=5, max_width=16), "link:12%"),
        # An unreadable counter set says nothing, which is not the same as saying "ok" — but
        # it looks the same here, and that is why `bt.accelerators()` carries `readable`.
        (DeviceFaults(index=0, remap_failure=True, readable=False), None, "ok"),
    ],
)
def test_the_device_state_cell_names_the_worst_condition(faults, link, expected):
    from batcher.observe.energy import _device_state

    fault_map = {0: faults} if faults is not None and faults.readable else {}
    link_map = {0: link} if link is not None else {}
    assert _device_state(_reading(), fault_map, link_map) == expected


def test_a_clamp_outranks_a_degraded_link_and_ecc_outranks_both():
    from batcher.observe.energy import _device_state

    link = {0: PcieLink("a", gen=3, width=8, max_gen=5, max_width=16)}
    assert _device_state(_reading(throttle_reasons=("thermal",)), {}, link) == "thermal"
    hot_and_wrong = _reading(ecc_uncorrected=2, throttle_reasons=("thermal",))
    assert _device_state(hot_and_wrong, {}, link) == "ecc:2"


def test_the_table_never_fails_when_the_probes_do(monkeypatch):
    from batcher.observe import energy

    def _boom():
        raise RuntimeError("driver gone")

    monkeypatch.setattr("batcher._internal.hardware.faults.device_faults", _boom)
    assert energy._device_conditions() == ({}, {})
    assert "NVIDIA H100" in energy.format_device_table([_reading()])
