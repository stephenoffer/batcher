"""The GPU-datacenter loop end to end: decide, admit, place, measure, report.

Each piece has its own unit tests. What those cannot show is that the pieces agree — that the
device class Kyber picks is the one Carbonite prices, that the fan-out admission allows is the
one the grant hands out, that the stage the meter records is the one the report reads, and that
a distributed run's merged energy equals the single-node figure. A disagreement anywhere in
that chain is silent: the query still returns the right rows.

No engine and no GPU: every layer here is control plane, which is exactly why the whole chain
can be exercised on a CPU-only host.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel import (
    KvCacheBudget,
    assess_device,
    devices_within_budget,
    kv_bytes_per_token,
    mig_plan,
    validate_fleet_power,
)
from batcher.carbonite.policies.scheduling import DefaultSchedulingPolicy
from batcher.config import AcceleratorConfig, Config, EnergyConfig, config_context
from batcher.core.energy import energy_scope, measure_stage
from batcher.dist.executors.ray_runtime.fabric import (
    GpuNodeTopology,
    permitted_nodes,
    plan_collective,
    power_zone_load,
)
from batcher.governance import DataResidency, ResidencyCatalog
from batcher.kyber.gpu import power_bounded_devices, select_device_class, stage_joules
from batcher.observe import energy_metrics, format_energy_report
from batcher.plan.energy import GridProfile, merge_ledgers

pytestmark = [pytest.mark.integration, pytest.mark.unit]

_GIB = 1 << 30

#: A two-rack fleet on one busway per rack, one rack of H100s and one of older V100s.
_FLEET = (
    GpuNodeTopology(
        "h-1", 8, "NVIDIA_H100", rack="r1", fabric="ib0", power_zone="bw1", region="eu-north-1"
    ),
    GpuNodeTopology(
        "h-2", 8, "NVIDIA_H100", rack="r1", fabric="ib0", power_zone="bw1", region="eu-north-1"
    ),
    GpuNodeTopology(
        "v-1", 8, "NVIDIA_TESLA_V100", rack="r2", fabric="ib1", power_zone="bw2", region="us-east-1"
    ),
)


def _config(watts: float = 0.0, *, efficiency_first: bool = False) -> Config:
    return Config().replace(
        accelerator=AcceleratorConfig(
            energy=EnergyConfig(
                power_budget_watts=watts, carbon_intensity=20.0, price_per_kwh=0.05
            ),
            efficiency_first_placement=efficiency_first,
        )
    )


def test_the_device_class_kyber_picks_is_one_carbonite_can_price() -> None:
    models = sorted({n.accelerator_type for n in _FLEET})
    chosen = select_device_class(models, model_gib=30.0, prefer_efficiency=True)
    assert chosen == "NVIDIA_H100", "the V100 cannot hold a 30 GiB model"
    with config_context(_config(10_000.0)):
        # Whatever Kyber picks, Carbonite must be able to price it, or the budget silently
        # stops binding for the exact device the plan chose.
        assert devices_within_budget(chosen, 32) == 10
        assert not validate_fleet_power(chosen, 32).feasible


def test_the_fan_out_admission_allows_is_the_one_the_grant_hands_out() -> None:
    with config_context(_config(10_000.0)):
        allowed = power_bounded_devices(64, "NVIDIA_H100")
        envelope = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=1.0, n_tasks=64, gpu_count=64, accelerator_type="NVIDIA_H100"
        )
        verdict = validate_fleet_power("NVIDIA_H100", 64)
    assert allowed == envelope.n_tasks == 10
    assert verdict.suggested_bounds is not None
    assert verdict.suggested_bounds.n_max_parallelism == allowed, (
        "the counter-offer and the grant must be the same number, or a plan is refused for "
        "one fan-out and then scheduled at another"
    )


def test_a_collective_is_placed_inside_a_fabric_and_within_its_zone_budget() -> None:
    placement = plan_collective(8, _FLEET)
    assert placement.strategy == "STRICT_PACK"
    assert not placement.spans_fabric
    load = power_zone_load(_FLEET)
    # Two 8-GPU H100 nodes on one busway draw more than a 10 kW circuit delivers, which is the
    # case the budget exists to catch.
    assert load["bw1"] > 10_000.0
    assert load["bw2"] < load["bw1"], "the older, lower-TDP rack draws less"


def test_residency_narrows_the_fleet_before_placement_rather_than_after() -> None:
    catalog = ResidencyCatalog(mode="strict").register(
        DataResidency("s3://eu/", frozenset({"eu-north-1"}), "GDPR Art. 44")
    )
    allowed = permitted_nodes(catalog, ["s3://eu/orders"], _FLEET)
    assert {n.node_id for n in allowed} == {"h-1", "h-2"}
    placement = plan_collective(16, allowed)
    assert placement.node_ids == ("h-1", "h-2"), "the collective is planned on permitted nodes"


def test_an_inference_stage_is_sized_by_cache_and_partitioned_when_it_fits() -> None:
    per_token = kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, dtype="fp16")
    budget = KvCacheBudget(
        device_bytes=80 * _GIB,
        weight_bytes=16 * _GIB,
        bytes_per_token=per_token,
        context_tokens=4096,
    )
    assert budget.fits
    assert budget.max_sequences > 0
    # A model this small also fits a partition, so the same stage should be planned onto
    # instances rather than whole devices.
    plan = mig_plan(16.0, "NVIDIA_H100", concurrency=6)
    assert plan.profile is not None
    assert plan.devices_needed < 6


def test_a_degraded_device_is_derated_and_a_corrupt_one_is_removed() -> None:
    clamped = DeviceTelemetry(index=0, throttle_reasons=("thermal",), memory_total_bytes=80 * _GIB)
    corrupt = DeviceTelemetry(index=1, ecc_uncorrected=1, memory_total_bytes=80 * _GIB)
    assert assess_device(clamped).schedulable
    assert assess_device(clamped).derate < 1.0
    assert not assess_device(corrupt).schedulable


def test_the_run_is_measured_reported_and_priced_from_one_ledger() -> None:
    with config_context(_config()), energy_scope() as ledger:
        with measure_stage("Decode#1", accelerator_type="NVIDIA_H100", device_count=8) as meter:
            meter.add_rows(1_000_000)
        with measure_stage("Generate#2", accelerator_type="NVIDIA_H100", device_count=8) as meter:
            meter.add_tokens(4_000_000)

    assert len(ledger.stages) == 2
    grid = GridProfile(region="eu-north-1", gco2e_per_kwh=20.0, price_per_kwh=0.05, pue=1.15)
    report = format_energy_report(ledger, grid)
    assert "Decode#1" in report and "Generate#2" in report
    assert "modelled" in report, "no NVML here, and the report must say so"
    metrics = energy_metrics(ledger, grid)
    assert metrics["energy.stages"] == 2.0
    assert metrics["energy.joules"] == pytest.approx(ledger.total_joules)


def test_a_distributed_run_reports_the_single_node_energy() -> None:
    def _worker(rows: int):
        with (
            energy_scope() as ledger,
            measure_stage(
                "Agg#1", accelerator_type="NVIDIA_H100", device_count=8, utilization=0.9
            ) as meter,
        ):
            meter.add_rows(rows)
        return ledger

    workers = [_worker(1000), _worker(2000), _worker(3000)]
    merged = merge_ledgers(workers)
    assert merged.total_rows == 6000
    assert merged.total_joules == pytest.approx(sum(w.total_joules for w in workers))
    assert merge_ledgers(list(reversed(workers))).total_joules == pytest.approx(
        merged.total_joules
    ), "energy folds in any order, like every other mergeable quantity here"


def test_the_planned_and_recorded_energy_agree_in_magnitude() -> None:
    # Kyber estimates a stage's draw before it runs; Core records what it drew. They use the
    # same power model, so an estimate for the recorded duration must match the record.
    with (
        energy_scope() as ledger,
        measure_stage("Agg#1", accelerator_type="NVIDIA_H100", device_count=4, utilization=1.0),
    ):
        pass
    record = ledger.stages[0]
    planned = stage_joules(record.seconds, "NVIDIA_H100", 4, 1.0)
    assert record.joules == pytest.approx(planned)
