"""Device settings that cost throughput or correctness without raising anything.

A rented GPU arrives configured by whoever had it last. ECC off makes every guard built on
the uncorrectable-error counter read a counter that cannot move; an exclusive compute mode
makes a second worker fail to open the device rather than share it; persistence off charges
every task a driver initialization before its first line runs. None of them is an error, and
the tests here pin both halves of the contract: the findings a configured device produces,
and the silence an unreadable one produces — a driver that will not answer must not be
reported as well-configured any more than as broken.
"""

from __future__ import annotations

import importlib

import pytest

from batcher._internal.hardware.faults import modes as modes_mod
from batcher._internal.hardware.faults.modes import DeviceModes
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel import health

pytestmark = pytest.mark.unit


class _FakeNvml:
    """NVML answering the configuration queries, with each individually refusable."""

    def __init__(
        self,
        *,
        ecc=(1, 1),
        persistence=1,
        compute=0,
        limit=300_000,
        constraints=(100_000, 400_000),
        refuse=(),
    ):
        self._values = {
            "ecc": ecc,
            "persistence": persistence,
            "compute": compute,
            "limit": limit,
            "constraints": constraints,
        }
        self._refuse = set(refuse)

    def _get(self, key):
        if key in self._refuse:
            raise RuntimeError("not supported on this device")
        return self._values[key]

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-abc"

    def nvmlDeviceGetEccMode(self, handle):
        return self._get("ecc")

    def nvmlDeviceGetPersistenceMode(self, handle):
        return self._get("persistence")

    def nvmlDeviceGetComputeMode(self, handle):
        return self._get("compute")

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return self._get("limit")

    def nvmlDeviceGetPowerManagementLimitConstraints(self, handle):
        return self._get("constraints")


def _use(monkeypatch, fake):
    monkeypatch.setattr(modes_mod, "_nvml", lambda: fake)
    monkeypatch.setattr(modes_mod, "_device_count", lambda nv: nv.nvmlDeviceGetCount())


def test_a_well_configured_device_has_no_findings(monkeypatch):
    _use(monkeypatch, _FakeNvml())
    (record,) = modes_mod.device_modes()
    assert record.readable is True
    assert record.ecc_enabled is True
    assert record.persistence is True
    assert record.compute_mode == "default"
    assert record.findings == ()
    assert modes_mod.misconfigured_devices() == ()


def test_ecc_off_is_reported_because_it_silences_the_guard(monkeypatch):
    # Every check built on `ecc_uncorrected` is then reading a counter that cannot move.
    _use(monkeypatch, _FakeNvml(ecc=(0, 0)))
    (record,) = modes_mod.device_modes()
    assert record.ecc_enabled is False
    assert record.findings == ("ecc_disabled",)


def test_an_exclusive_compute_mode_means_a_second_worker_cannot_open_the_device(monkeypatch):
    _use(monkeypatch, _FakeNvml(compute=3))
    (record,) = modes_mod.device_modes()
    assert record.compute_mode == "exclusive_process"
    assert record.single_tenant is True
    assert record.findings == ("compute_mode_exclusive_process",)
    _use(monkeypatch, _FakeNvml(compute=0))
    assert modes_mod.device_modes()[0].single_tenant is False


def test_a_power_limit_at_the_floor_is_a_setting_not_slow_hardware(monkeypatch):
    _use(monkeypatch, _FakeNvml(limit=100_000, constraints=(100_000, 400_000)))
    (record,) = modes_mod.device_modes()
    assert record.power_limit_watts == pytest.approx(100.0)
    assert record.power_limit_floor_watts == pytest.approx(100.0)
    assert record.power_capped_to_floor is True
    assert "power_at_floor" in record.findings


def test_a_normal_datacenter_cap_is_not_a_finding(monkeypatch):
    _use(monkeypatch, _FakeNvml(limit=250_000, constraints=(100_000, 400_000)))
    assert modes_mod.device_modes()[0].power_capped_to_floor is False


def test_an_unreported_constraint_cannot_produce_a_power_finding(monkeypatch):
    _use(monkeypatch, _FakeNvml(refuse=("constraints",)))
    record = modes_mod.device_modes()[0]
    assert record.power_limit_floor_watts == 0.0
    assert record.power_capped_to_floor is False


def test_a_driver_refusing_everything_is_unreadable_not_well_configured(monkeypatch):
    _use(monkeypatch, _FakeNvml(refuse=("ecc", "persistence", "compute", "limit", "constraints")))
    (record,) = modes_mod.device_modes()
    assert record.readable is False
    assert record.findings == ()
    assert modes_mod.misconfigured_devices() == ()


def test_no_driver_reports_nothing(monkeypatch):
    monkeypatch.setattr(modes_mod, "_nvml", lambda: None)
    assert modes_mod.device_modes() == ()
    assert modes_mod.misconfigured_devices() == ()


def test_findings_are_ordered_by_consequence(monkeypatch):
    _use(monkeypatch, _FakeNvml(ecc=(0, 0), compute=3, persistence=0, limit=100_000))
    (record,) = modes_mod.device_modes()
    assert record.findings == (
        "ecc_disabled",
        "compute_mode_exclusive_process",
        "power_at_floor",
        "persistence_off",
    )


# --- The device's own thermal threshold ----------------------------------------------------


def _reading(**kw) -> DeviceTelemetry:
    base = {"index": 0, "uuid": "GPU-0", "memory_used_bytes": 1, "memory_total_bytes": 100}
    return DeviceTelemetry(**{**base, **kw})


def test_heat_is_judged_against_the_part_not_a_fleet_wide_constant():
    # A part that clamps at 83 must not be judged against a fleet-wide 87 it never reaches:
    # the warning would arrive after the clamp it exists to precede.
    hot = _reading(temperature_c=80.0, slowdown_temperature_c=83.0)
    assert "hot" in health.assess_device(hot).reasons
    # The same temperature on a part that clamps at 95 is unremarkable.
    cool = _reading(temperature_c=80.0, slowdown_temperature_c=95.0)
    assert cool.temperature_c == 80.0
    assert health.assess_device(cool).state == "healthy"


def test_the_configured_ceiling_still_applies_when_the_part_says_nothing():
    assert health.assess_device(_reading(temperature_c=90.0)).reasons == ("hot",)
    assert health.assess_device(_reading(temperature_c=50.0)).state == "healthy"


def test_the_part_threshold_is_a_floor_on_strictness_not_a_licence_to_run_hotter():
    # A device whose published slowdown is 120 does not get to ignore the configured ceiling.
    reading = _reading(temperature_c=90.0, slowdown_temperature_c=120.0)
    assert "hot" in health.assess_device(reading).reasons


# --- What an operator sees -----------------------------------------------------------------


def test_the_report_carries_the_findings_and_explains_them(capsys):
    report_mod = importlib.import_module("batcher.api.session.accelerators")
    row: dict = {"index": 2}
    report_mod._add_modes(row, DeviceModes(index=2, ecc_enabled=False, readable=True))
    assert row["config"] == ["ecc_disabled"]
    report_mod._show_silent_faults([row])
    out = capsys.readouterr().out
    assert "gpu 2  ECC is OFF" in out
    assert "will not be reported" in out


def test_a_clean_device_adds_nothing_to_its_row():
    report_mod = importlib.import_module("batcher.api.session.accelerators")
    row: dict = {"index": 0}
    report_mod._add_modes(row, DeviceModes(index=0, ecc_enabled=True, readable=True))
    report_mod._add_modes(row, None)
    assert row == {"index": 0}


def test_every_finding_has_advice_written_for_it():
    # A reason code with no explanation makes the reader look it up, which on a report they
    # are reading because something is slow is exactly the wrong moment.
    report_mod = importlib.import_module("batcher.api.session.accelerators")
    produced = set()
    for ecc, compute, limit, persistence in [
        (False, 0, 300_000, True),
        (True, 3, 300_000, True),
        (True, 1, 300_000, True),
        (True, 2, 300_000, True),
        (True, 0, 300_000, False),
    ]:
        record = DeviceModes(
            index=0,
            ecc_enabled=ecc,
            persistence=persistence,
            compute_mode=modes_mod._COMPUTE_MODES[compute],
            power_limit_watts=limit / 1000.0,
            power_limit_floor_watts=100.0,
            readable=True,
        )
        produced.update(record.findings)
    produced.add("power_at_floor")
    assert produced <= set(report_mod._CONFIG_ADVICE)


# --- Partitioning, which changes what every other number means -----------------------------


class _MigNvml(_FakeNvml):
    """A device reporting MIG mode and a fixed number of instances."""

    def __init__(self, *, mig=(1, 1), instances=4, **kw):
        super().__init__(**kw)
        self._mig, self._instances = mig, instances

    def nvmlDeviceGetMigMode(self, handle):
        if self._mig is None:
            raise RuntimeError("not a MIG-capable part")
        return self._mig

    def nvmlDeviceGetMigDeviceHandleByIndex(self, handle, index):
        if index >= self._instances:
            raise RuntimeError("no such instance")
        return (handle, index)


def test_a_partitioned_device_reports_its_instances(monkeypatch):
    _use(monkeypatch, _MigNvml(mig=(1, 1), instances=4))
    (record,) = modes_mod.device_modes()
    assert record.mig_enabled is True
    assert record.mig_instances == 4


def test_an_unpartitioned_device_counts_no_instances(monkeypatch):
    # The walk must not run at all when MIG is off: on a card that cannot partition, every
    # index would be a refused query.
    _use(monkeypatch, _MigNvml(mig=(0, 0), instances=7))
    (record,) = modes_mod.device_modes()
    assert record.mig_enabled is False
    assert record.mig_instances == 0


def test_a_part_that_cannot_partition_says_nothing(monkeypatch):
    _use(monkeypatch, _MigNvml(mig=None))
    (record,) = modes_mod.device_modes()
    assert record.mig_enabled is None
    assert record.mig_instances == 0


def test_partitioning_reaches_the_device_row_but_is_not_a_finding():
    report_mod = importlib.import_module("batcher.api.session.accelerators")
    row: dict = {"index": 0}
    report_mod._add_modes(
        row, DeviceModes(index=0, mig_enabled=True, mig_instances=7, readable=True)
    )
    assert row["mig_instances"] == 7
    assert "config" not in row  # deliberate partitioning is not a misconfiguration
