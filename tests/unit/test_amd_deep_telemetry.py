"""The AMD readings that give a mixed fleet one set of answers instead of a vendor branch.

The properties under test are the two the NVML side holds to, plus one that is specific to
reading a vendor through sysfs rather than through a library.

**Unreadable is never healthy.** An older kernel, a consumer part, and a container that mounted
`/sys` without the `amdgpu` tree all produce zeros, and `readable` is what separates them from a
device that answered.

**A counter is not a rate.** `pcie_bw` publishes lifetime packet counts. Deriving a bandwidth
from one reading would report the driver's whole uptime of traffic as though it happened in the
last second, so the rate takes two readings and refuses when it cannot have them.

**The active clock is marked, not last.** `pp_dpm_*` lists every level the part supports and
marks the running one. Taking the last line reports every idle board at its boost clock.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.hardware.amd import telemetry as amd_telemetry_mod
from batcher._internal.hardware.amd.telemetry import AmdTelemetry, pcie_bytes_per_second

pytestmark = pytest.mark.unit


def _reading(**fields) -> AmdTelemetry:
    return AmdTelemetry(index=0, readable=True, **fields)


# --- host-link rate -----------------------------------------------------------------------


def test_bytes_are_packets_times_the_payload_size():
    reading = _reading(
        pcie_packets_received=1_000,
        pcie_packets_sent=1_000,
        pcie_max_payload_bytes=256,
    )
    assert reading.pcie_bytes_total == 512_000


def test_a_rate_needs_two_readings_and_an_interval():
    before = _reading(pcie_packets_received=10, pcie_packets_sent=10, pcie_max_payload_bytes=256)
    after = _reading(
        pcie_packets_received=1_010, pcie_packets_sent=1_010, pcie_max_payload_bytes=256
    )
    assert pcie_bytes_per_second(before, after, 2.0) == pytest.approx(256_000.0)


def test_a_zero_interval_reports_nothing_rather_than_dividing():
    before = _reading(pcie_packets_received=10, pcie_max_payload_bytes=256)
    after = _reading(pcie_packets_received=1_010, pcie_max_payload_bytes=256)
    assert pcie_bytes_per_second(before, after, 0.0) == 0.0


def test_a_driver_reload_between_readings_reports_no_rate():
    # The counters restart, so the delta goes negative. Reporting that as a rate would be a
    # nonsense figure; reporting zero sends the caller back to whatever it had.
    high = _reading(pcie_packets_received=10_000, pcie_max_payload_bytes=256)
    low = _reading(pcie_packets_received=5, pcie_max_payload_bytes=256)
    assert pcie_bytes_per_second(high, low, 1.0) == 0.0


def test_an_unreadable_device_yields_no_rate():
    before = AmdTelemetry(index=0, pcie_packets_received=10, pcie_max_payload_bytes=256)
    after = AmdTelemetry(index=0, pcie_packets_received=1_010, pcie_max_payload_bytes=256)
    assert before.readable is False
    assert pcie_bytes_per_second(before, after, 1.0) == 0.0


# --- host-mappable aperture ---------------------------------------------------------------


def test_visible_vram_pressure_needs_an_aperture_reading():
    pressured = _reading(visible_vram_total_bytes=1_000, visible_vram_used_bytes=900)
    unknown = _reading(visible_vram_total_bytes=0, visible_vram_used_bytes=0)
    assert pressured.visible_vram_pressured is True
    # Zero total is unknown, not an exhausted aperture, and certainly not headroom.
    assert unknown.visible_vram_pressured is False
    assert unknown.visible_vram_utilization == 0.0


def test_pressured_devices_are_filtered_from_a_supplied_list():
    pressured = _reading(visible_vram_total_bytes=100, visible_vram_used_bytes=95)
    roomy = _reading(visible_vram_total_bytes=100, visible_vram_used_bytes=10)
    assert amd_telemetry_mod.visible_vram_pressured((pressured, roomy)) == (pressured,)


# --- temperatures -------------------------------------------------------------------------


def test_the_hottest_sensor_wins_rather_than_the_edge_one():
    # An Instinct clamps on its junction sensor, which runs well above the edge sensor a
    # single-temperature check reads. Checking the edge alone finds the clamp after it has
    # already cost the stage.
    reading = _reading(junction_temperature_c=96.0, memory_temperature_c=88.0)
    assert reading.hottest_c == 96.0


def test_a_board_with_no_extra_sensors_reports_no_temperature():
    assert _reading().hottest_c == 0.0


# --- clock table parsing ------------------------------------------------------------------


def _dpm(tmp_path, body: str) -> str:
    path = tmp_path / "pp_dpm_sclk"
    path.write_text(body)
    return str(path)


def test_the_marked_level_is_the_current_clock(tmp_path):
    path = _dpm(tmp_path, "0: 500Mhz\n1: 1200Mhz *\n2: 1700Mhz\n")
    # Not 1700: the last line is the highest level the part supports, not the one it is on.
    assert amd_telemetry_mod._current_clock_mhz(path) == 1200


def test_a_table_with_no_marked_level_reports_nothing(tmp_path):
    path = _dpm(tmp_path, "0: 500Mhz\n1: 1200Mhz\n")
    assert amd_telemetry_mod._current_clock_mhz(path) == 0


def test_a_missing_table_reports_nothing(tmp_path):
    assert amd_telemetry_mod._current_clock_mhz(str(tmp_path / "absent")) == 0


def test_a_malformed_level_reports_nothing_rather_than_raising(tmp_path):
    path = _dpm(tmp_path, "0: not-a-numberMhz *\n")
    assert amd_telemetry_mod._current_clock_mhz(path) == 0


# --- the pcie_bw attribute ----------------------------------------------------------------


def test_pcie_bw_is_parsed_as_three_integers(tmp_path):
    device_dir = tmp_path / "device"
    device_dir.mkdir()
    (device_dir / "pcie_bw").write_text("1234 5678 256\n")
    assert amd_telemetry_mod._pcie_bw(str(device_dir)) == (1234, 5678, 256)


def test_a_short_or_missing_pcie_bw_reports_nothing(tmp_path):
    device_dir = tmp_path / "device"
    device_dir.mkdir()
    assert amd_telemetry_mod._pcie_bw(str(device_dir)) == (0, 0, 0)
    (device_dir / "pcie_bw").write_text("1234\n")
    assert amd_telemetry_mod._pcie_bw(str(device_dir)) == (0, 0, 0)
    (device_dir / "pcie_bw").write_text("a b c\n")
    assert amd_telemetry_mod._pcie_bw(str(device_dir)) == (0, 0, 0)


# --- the probe ----------------------------------------------------------------------------


def test_a_host_with_no_amd_devices_reports_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(amd_telemetry_mod, "AMDGPU_SYSFS_ROOT", str(tmp_path))
    assert amd_telemetry_mod.amd_telemetry() == ()


def test_the_probe_reads_a_faked_card(monkeypatch, tmp_path):
    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "mem_busy_percent").write_text("42\n")
    (device_dir / "mem_info_vis_vram_total").write_text("268435456\n")
    (device_dir / "mem_info_vis_vram_used").write_text("134217728\n")
    (device_dir / "pcie_bw").write_text("100 200 512\n")
    (device_dir / "pp_dpm_mclk").write_text("0: 1600Mhz *\n")
    monkeypatch.setattr(amd_telemetry_mod, "AMDGPU_SYSFS_ROOT", str(tmp_path))
    monkeypatch.setattr(amd_telemetry_mod, "_pci_vendor", lambda d: 0x1002)
    monkeypatch.setattr(amd_telemetry_mod, "_hwmon_dir", lambda d: "")
    readings = amd_telemetry_mod.amd_telemetry()
    assert len(readings) == 1
    reading = readings[0]
    assert reading.card == "card0"
    assert reading.memory_busy_percent == 42
    assert reading.visible_vram_utilization == pytest.approx(0.5)
    assert reading.mclk_mhz == 1600
    assert reading.pcie_bytes_total == (100 + 200) * 512
    assert reading.readable is True


def test_a_device_that_publishes_nothing_is_flagged_unreadable(monkeypatch, tmp_path):
    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)
    monkeypatch.setattr(amd_telemetry_mod, "AMDGPU_SYSFS_ROOT", str(tmp_path))
    monkeypatch.setattr(amd_telemetry_mod, "_pci_vendor", lambda d: 0x1002)
    monkeypatch.setattr(amd_telemetry_mod, "_hwmon_dir", lambda d: "")
    readings = amd_telemetry_mod.amd_telemetry()
    assert len(readings) == 1
    # A record is still reported, so a caller can tell an idle device from an unreadable one.
    assert readings[0].readable is False


def test_connectors_and_non_amd_cards_are_skipped(monkeypatch, tmp_path):
    for name in ("card0", "card0-DP-1", "card1"):
        (tmp_path / name / "device").mkdir(parents=True)
    monkeypatch.setattr(amd_telemetry_mod, "AMDGPU_SYSFS_ROOT", str(tmp_path))
    # Only card1 is an AMD device; card0 is another vendor and card0-DP-1 is a connector.
    monkeypatch.setattr(
        amd_telemetry_mod,
        "_pci_vendor",
        lambda d: 0x1002 if os.path.basename(os.path.dirname(d)) == "card1" else 0x10DE,
    )
    monkeypatch.setattr(amd_telemetry_mod, "_hwmon_dir", lambda d: "")
    readings = amd_telemetry_mod.amd_telemetry()
    assert [r.card for r in readings] == ["card1"]
    # Indices are dense over AMD cards, not over DRM nodes, matching how ROCm numbers them.
    assert readings[0].index == 0
