"""Core's stage energy meter: measure where possible, model where not, and say which.

The lane matters as much as the arithmetic. This is Core measuring, so the meter must never
change a decision, never fail a stage, and never present a modelled figure as a measured one.
It must also stay inert when nothing asked for accounting, because it sits on a path that
usually runs without it.
"""

from __future__ import annotations

import pytest

from batcher.config import AcceleratorConfig, Config, EnergyConfig, config_context
from batcher.core import energy as core_energy
from batcher.core.energy import active_ledger, energy_scope, measure_stage

pytestmark = pytest.mark.unit


def test_outside_a_scope_the_meter_records_nothing() -> None:
    assert active_ledger() is None
    with measure_stage("Infer#1", accelerator_type="NVIDIA_H100", device_count=8) as meter:
        meter.add_rows(1000)
    assert active_ledger() is None, "no ledger, no recording, no error"


def test_a_scope_collects_every_stage_inside_it() -> None:
    with energy_scope() as ledger:
        with measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=8) as m:
            m.add_rows(500)
        with measure_stage("B#2", accelerator_type="NVIDIA_H100", device_count=8) as m:
            m.add_tokens(2000)
    assert [s.stage for s in ledger.stages] == ["A#1", "B#2"]
    assert ledger.total_rows == 500
    assert ledger.total_tokens == 2000
    assert active_ledger() is None, "the scope is restored on the way out"


def test_counts_accumulate_rather_than_overwrite() -> None:
    with energy_scope() as ledger, measure_stage("A#1") as meter:
        meter.add_rows(10)
        meter.add_rows(15)
        meter.add_tokens(3)
        meter.add_rows(-5)
        meter.add_tokens(0)  # both ignored: a count only ever grows
    assert ledger.stages[0].rows == 25
    assert ledger.stages[0].tokens == 3


def test_a_stage_that_raises_is_still_recorded() -> None:
    with (
        energy_scope() as ledger,
        pytest.raises(RuntimeError),
        measure_stage("A#1", accelerator_type="NVIDIA_H100"),
    ):
        raise RuntimeError("stage failed")
    assert len(ledger.stages) == 1, "a failed stage still drew power, and still cost money"


def test_without_telemetry_the_figure_is_modelled_and_marked() -> None:
    with (
        energy_scope() as ledger,
        measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=8, utilization=0.9),
    ):
        pass
    record = ledger.stages[0]
    assert not record.measured, "a datasheet estimate must never read as a measurement"
    assert record.joules > 0
    assert record.utilization == pytest.approx(0.9)
    assert ledger.measured_joules == 0.0


def test_with_telemetry_the_figure_is_measured(monkeypatch) -> None:
    monkeypatch.setattr(core_energy, "_draw", lambda: (1400.0, 0.8, True))
    with (
        energy_scope() as ledger,
        measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=2),
    ):
        pass
    record = ledger.stages[0]
    assert record.measured
    assert record.utilization == pytest.approx(0.8), "measured, not the assumed default"
    assert record.joules == pytest.approx(1400.0 * record.seconds)
    assert ledger.measured_joules == record.joules


def test_a_telemetry_failure_falls_back_instead_of_raising(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: _boom())
    with (
        energy_scope() as ledger,
        measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=1),
    ):
        pass
    assert not ledger.stages[0].measured


def test_accounting_can_be_turned_off() -> None:
    cfg = Config().replace(accelerator=AcceleratorConfig(energy=EnergyConfig(accounting=False)))
    with (
        config_context(cfg),
        energy_scope() as ledger,
        measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=8) as meter,
    ):
        meter.add_rows(10)
    assert ledger.stages == []


def test_a_cpu_stage_records_duration_without_inventing_power() -> None:
    with energy_scope() as ledger, measure_stage("Sort#3") as meter:
        meter.add_rows(100)
    record = ledger.stages[0]
    assert record.accelerator_type == ""
    assert record.joules == 0.0, "no device, no fabricated watts"
    assert record.seconds >= 0.0


def test_the_summary_reports_how_much_was_measured(monkeypatch) -> None:
    with energy_scope() as ledger:
        with measure_stage("modelled#1", accelerator_type="NVIDIA_H100", device_count=1):
            pass
        monkeypatch.setattr(core_energy, "_draw", lambda: (700.0, 0.9, True))
        with measure_stage("measured#2", accelerator_type="NVIDIA_H100", device_count=1):
            pass
    fraction = ledger.summary()["measured_fraction"]
    assert 0.0 < fraction < 1.0


def test_the_gpu_kernel_is_bracketed_so_a_stage_is_actually_recorded(monkeypatch) -> None:
    # The meter is only worth having if something on the live path calls it. `gpu_groupby_agg`
    # is the one point every GPU relational stage passes through — local dispatch and Ray
    # worker alike — so this pins that the bracket is there and that it counts the rows.
    import pyarrow as pa

    from batcher.core import gpu_transform

    table = pa.table({"g": [1, 1, 2], "v": [10, 20, 30]})
    result = pa.table({"g": [1, 2], "total": [30, 30]})
    monkeypatch.setattr(gpu_transform, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_transform, "accelerator_backend", lambda: "cuda")
    monkeypatch.setattr(gpu_transform, "_dispatch_groupby", lambda *a, **k: result)
    monkeypatch.setattr(gpu_transform, "_local_device_model", lambda: "NVIDIA_H100")

    with energy_scope() as ledger:
        out = gpu_transform.gpu_groupby_agg(table, "g", {"total": ("v", "sum")})

    assert out is result
    assert len(ledger.stages) == 1
    record = ledger.stages[0]
    assert record.stage.startswith("GpuGroupBy#")
    assert record.accelerator_type == "NVIDIA_H100"
    assert record.rows == 2
    assert record.joules > 0


def test_the_untraced_gpu_path_is_unchanged(monkeypatch) -> None:
    import pyarrow as pa

    from batcher.core import gpu_transform

    result = pa.table({"g": [1], "total": [30]})
    monkeypatch.setattr(gpu_transform, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_transform, "accelerator_backend", lambda: "cuda")
    monkeypatch.setattr(gpu_transform, "_dispatch_groupby", lambda *a, **k: result)
    assert active_ledger() is None
    out = gpu_transform.gpu_groupby_agg(
        pa.table({"g": [1], "v": [30]}), "g", {"total": ("v", "sum")}
    )
    assert out is result, "no scope open: the kernel returns exactly what it returned before"


def test_the_public_scope_is_the_same_ledger() -> None:
    import batcher as bt

    with (
        bt.measure_energy() as ledger,
        measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=1) as meter,
    ):
        meter.add_rows(7)
    assert ledger.total_rows == 7
    assert active_ledger() is None
