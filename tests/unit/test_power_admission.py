"""Power as an admission decision, and energy as a mergeable quantity.

Two contracts here. Admission must never *fail* a query on a modelled figure — it steers, so
its refusals are advisory and carry a device count the budget can actually run. And the ledger
must merge the way the engine's stateful operators do: a distributed run's energy is the fold
of its workers' and equals what one node would have reported, in any order.
"""

from __future__ import annotations

import pytest

from batcher._internal.device_specs import (
    device_fp8_tflops,
    device_generation,
    device_spec,
)
from batcher.carbonite.accel import devices_within_budget, validate_fleet_power
from batcher.config import AcceleratorConfig, Config, EnergyConfig, config_context
from batcher.plan.energy import (
    EnergyLedger,
    StageEnergy,
    configured_power_envelope,
    merge_ledgers,
)

pytestmark = pytest.mark.unit


def _budget(watts: float) -> Config:
    return Config().replace(
        accelerator=AcceleratorConfig(energy=EnergyConfig(power_budget_watts=watts))
    )


# --- the envelope and its verdict ---------------------------------------------------------


def test_no_budget_admits_everything() -> None:
    assert configured_power_envelope().unbounded
    assert validate_fleet_power("NVIDIA_H100", 1024).feasible
    assert devices_within_budget("NVIDIA_H100", 1024) == 1024


def test_a_fleet_over_budget_is_refused_with_a_counter_offer() -> None:
    with config_context(_budget(10_000.0)):
        verdict = validate_fleet_power("NVIDIA_H100", 32)
    assert not verdict.feasible
    assert verdict.binding_constraint == "power"
    assert verdict.suggested_bounds is not None
    assert verdict.suggested_bounds.n_max_parallelism == 10
    assert verdict.binding_op == "NVIDIA_H100"


def test_a_power_refusal_is_advisory_not_fatal() -> None:
    # The draw is modelled at planning time, and a plan Kyber could not size may well fit.
    with config_context(_budget(10_000.0)):
        assert validate_fleet_power("NVIDIA_H100", 32).advisory


def test_a_fleet_inside_the_budget_is_admitted() -> None:
    with config_context(_budget(10_000.0)):
        assert validate_fleet_power("NVIDIA_H100", 8).feasible


def test_headroom_is_charged_against_the_budget() -> None:
    tight = Config().replace(
        accelerator=AcceleratorConfig(
            energy=EnergyConfig(power_budget_watts=9_000.0, power_headroom=0.5)
        )
    )
    with config_context(tight):
        assert configured_power_envelope().usable_watts == pytest.approx(4_500.0)
        assert not validate_fleet_power("NVIDIA_H100", 8).feasible


def test_an_unknown_device_is_admitted_rather_than_judged() -> None:
    with config_context(_budget(100.0)):
        assert validate_fleet_power("MADE_UP", 64).feasible
        assert validate_fleet_power(None, 64).feasible
        assert devices_within_budget("MADE_UP", 64) == 64


def test_lower_utilization_admits_more_devices() -> None:
    with config_context(_budget(10_000.0)):
        assert devices_within_budget("NVIDIA_H100", 64, utilization=0.5) > devices_within_budget(
            "NVIDIA_H100", 64, utilization=1.0
        )


def test_the_expected_draw_can_be_carried_on_the_envelope() -> None:
    with config_context(_budget(10_000.0)):
        assert not configured_power_envelope(expected_watts=9_999.0).fits()
        assert configured_power_envelope(expected_watts=1_000.0).fits()


# --- mergeable energy ----------------------------------------------------------------------


def _ledger(joules: float, rows: int) -> EnergyLedger:
    out = EnergyLedger()
    out.record(StageEnergy("Agg#1", "NVIDIA_H100", 8, 10.0, 0.9, joules=joules, rows=rows))
    return out


def test_merging_is_commutative_and_associative() -> None:
    a, b, c = _ledger(100.0, 5), _ledger(200.0, 7), _ledger(50.0, 3)
    left = merge_ledgers([merge_ledgers([a, b]), c]).total_joules
    right = merge_ledgers([a, merge_ledgers([b, c])]).total_joules
    swapped = merge_ledgers([c, b, a]).total_joules
    assert left == right == swapped == 350.0


def test_a_merged_run_equals_the_single_node_figures() -> None:
    single = EnergyLedger()
    single.record(StageEnergy("Agg#1", "NVIDIA_H100", 8, 10.0, 0.9, joules=100.0, rows=5))
    single.record(StageEnergy("Agg#1", "NVIDIA_H100", 8, 10.0, 0.9, joules=200.0, rows=7))
    merged = merge_ledgers([_ledger(100.0, 5), _ledger(200.0, 7)])
    assert merged.total_joules == single.total_joules
    assert merged.total_rows == single.total_rows
    assert merged.rows_per_joule() == single.rows_per_joule()
    assert merged.by_device() == single.by_device()


def test_merge_leaves_its_inputs_alone() -> None:
    a, b = _ledger(100.0, 5), _ledger(200.0, 7)
    merge_ledgers([a, b])
    assert len(a.stages) == 1
    assert len(b.stages) == 1


def test_merging_nothing_yields_an_empty_ledger() -> None:
    assert merge_ledgers([]).stages == []
    assert merge_ledgers([]).total_joules == 0.0


def test_in_place_merge_chains() -> None:
    a = _ledger(100.0, 5)
    assert a.merge(_ledger(200.0, 7)).merge(_ledger(50.0, 3)).total_joules == 350.0


# --- the new device accessors ---------------------------------------------------------------


def test_fp8_is_reported_only_where_the_unit_exists() -> None:
    assert device_fp8_tflops("NVIDIA_H100") > 0
    assert device_fp8_tflops("NVIDIA_A100_80G") == 0.0, "Ampere has no FP8 unit"
    assert device_fp8_tflops("MADE_UP") == 0.0


def test_the_generation_is_reported_or_empty() -> None:
    # The key learned statistics fall back to: parts of one generation share a capability set.
    assert device_generation("NVIDIA_H200") == "hopper"
    assert device_generation("NVIDIA_H100") == "hopper"
    assert device_generation("NVIDIA_A100_80G") == "ampere"
    assert device_generation("MADE_UP") == ""
    assert device_generation(None) == ""


def test_the_added_ampere_and_cdna_parts_are_consistent() -> None:
    for name in ("NVIDIA_A30", "NVIDIA_A40", "AMD_INSTINCT_MI325X"):
        spec = device_spec(name)
        assert spec is not None
        assert spec.memory_gib > 0
        assert spec.idle_watts <= spec.tdp_watts
    assert device_spec("NVIDIA_A30").mig_slices == 4, "the A30 partitions four ways, not seven"
    assert (
        device_spec("AMD_INSTINCT_MI325X").memory_gib
        > device_spec("AMD_INSTINCT_MI300X").memory_gib
    )
