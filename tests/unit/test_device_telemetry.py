"""NVML telemetry: readable when the driver is there, silent when it is not.

The contract that matters here is failure behavior. Telemetry is read on paths that must not
fail a query — sizing, health, energy accounting — and NVML fails in three distinct ways: not
installed, installed but refusing to initialize, and initialized but refusing individual
queries. All three have to read as "not reported", and the third must not erase the fields that
*did* answer. A fake `pynvml` stands in for the driver so this runs on any machine.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher._internal.hardware import nvml

pytestmark = pytest.mark.unit


class _Util:
    def __init__(self, gpu: int, memory: int) -> None:
        self.gpu = gpu
        self.memory = memory


class _Mem:
    def __init__(self, used: int, total: int) -> None:
        self.used = used
        self.total = total


def _fake_pynvml(*, devices: int = 2, refuse: frozenset[str] = frozenset()) -> types.ModuleType:
    """A `pynvml` stand-in; names in `refuse` raise the way a real driver refuses a query."""
    mod = types.ModuleType("pynvml")

    def guard(name):
        if name in refuse:
            raise RuntimeError(f"NVML refuses {name}")

    mod.nvmlInit = lambda: guard("nvmlInit")
    mod.nvmlDeviceGetCount = lambda: devices
    mod.nvmlDeviceGetHandleByIndex = lambda i: i
    mod.nvmlDeviceGetUUID = lambda h: f"GPU-{h}".encode()
    mod.nvmlDeviceGetName = lambda h: b"NVIDIA H100 80GB HBM3"

    def power(h):
        guard("nvmlDeviceGetPowerUsage")
        return 420_000 + h * 1000  # milliwatts

    mod.nvmlDeviceGetPowerUsage = power
    mod.nvmlDeviceGetEnforcedPowerLimit = lambda h: 600_000
    mod.nvmlDeviceGetTemperature = lambda h, sensor: 61 + h
    mod.nvmlDeviceGetUtilizationRates = lambda h: _Util(90 - h * 10, 40)
    mod.nvmlDeviceGetMemoryInfo = lambda h: _Mem(20 << 30, 80 << 30)
    mod.nvmlDeviceGetTotalEccErrors = lambda h, kind, scope: h  # device 1 has one
    mod.nvmlDeviceGetClockInfo = lambda h, kind: 1755
    mod.nvmlClocksThrottleReasonSwPowerCap = 0x4
    mod.nvmlClocksThrottleReasonHwThermalSlowdown = 0x40
    mod.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 0x4 if h == 0 else 0
    return mod


@pytest.fixture
def driver(monkeypatch):
    """Install a fake driver and clear the memoized handshake around the test."""

    def install(**kwargs):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(**kwargs))
        nvml.reset_nvml_probe()

    yield install
    nvml.reset_nvml_probe()


def test_absent_driver_reads_as_no_telemetry(monkeypatch) -> None:
    # `None` in `sys.modules` is how CPython spells "this import fails", so this exercises the
    # real not-installed path rather than stubbing the accessor that path feeds.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    nvml.reset_nvml_probe()
    assert not nvml.nvml_available()
    assert nvml.device_telemetry() == ()
    assert nvml.total_power_watts() == 0.0
    assert nvml.throttled_devices() == ()


def test_driver_that_refuses_to_initialize_is_unavailable(driver) -> None:
    driver(refuse=frozenset({"nvmlInit"}))
    assert not nvml.nvml_available()
    assert nvml.device_telemetry() == ()


def test_readings_are_reported_per_device(driver) -> None:
    driver()
    assert nvml.nvml_available()
    devices = nvml.device_telemetry()
    assert len(devices) == 2
    first = devices[0]
    assert first.uuid == "GPU-0"
    assert first.name == "NVIDIA H100 80GB HBM3"
    assert first.power_watts == pytest.approx(420.0)
    assert first.power_limit_watts == pytest.approx(600.0)
    assert first.power_headroom_watts == pytest.approx(180.0)
    assert first.sm_utilization == pytest.approx(0.9)
    assert first.memory_utilization == pytest.approx(0.4)
    assert first.memory_free_bytes == 60 << 30
    assert first.graphics_clock_mhz == 1755


def test_a_refused_field_does_not_erase_the_record(driver) -> None:
    driver(refuse=frozenset({"nvmlDeviceGetPowerUsage"}))
    devices = nvml.device_telemetry()
    assert len(devices) == 2, "one unsupported query must not drop the device"
    assert devices[0].power_watts == 0.0, "the refused field reads as not reported"
    assert devices[0].sm_utilization == pytest.approx(0.9), "the others still answer"
    assert devices[0].power_headroom_watts == 0.0, "no draw means no headroom figure"


def test_throttling_is_surfaced_by_reason(driver) -> None:
    driver()
    throttled = nvml.throttled_devices()
    assert [d.index for d in throttled] == [0]
    assert throttled[0].throttle_reasons == ("power",)
    assert not nvml.device_telemetry()[1].throttled


def test_ecc_counts_are_carried_per_device(driver) -> None:
    driver()
    assert [d.ecc_uncorrected for d in nvml.device_telemetry()] == [0, 1]


def test_total_power_sums_the_fleet(driver) -> None:
    driver()
    assert nvml.total_power_watts() == pytest.approx(841.0)


def test_handshake_is_memoized_but_readings_are_not(driver) -> None:
    driver()
    calls: list[int] = []
    real = sys.modules["pynvml"].nvmlDeviceGetUtilizationRates

    def counting(h):
        calls.append(h)
        return real(h)

    sys.modules["pynvml"].nvmlDeviceGetUtilizationRates = counting
    nvml.device_telemetry()
    nvml.device_telemetry()
    assert len(calls) == 4, "a cached utilization reading would be worse than none"
