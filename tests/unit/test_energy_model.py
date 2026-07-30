"""The energy contract: power draw, grid conversion, and the per-stage ledger.

Power is a plannable resource here, which means a wrong figure does not fail loudly — it
quietly admits work a rack cannot power, or refuses work it could. These pin the three
behaviors that keep that safe: an unknown device yields "no opinion" rather than a bound, an
unconfigured grid reports zero rather than an invented emissions figure, and an efficiency
ratio with no work in the numerator is `None` rather than zero.
"""

from __future__ import annotations

import pytest

from batcher.config import AcceleratorConfig, Config, config_context
from batcher.plan.energy import (
    EnergyLedger,
    GridProfile,
    PowerEnvelope,
    StageEnergy,
    carbon_grams,
    configured_power_envelope,
    device_power_watts,
    energy_cost,
    energy_joules,
    fleet_power_watts,
    host_overhead_watts,
    joules_to_kwh,
    kwh_to_joules,
    max_concurrent_devices,
)

pytestmark = pytest.mark.unit


def test_power_interpolates_between_idle_and_tdp() -> None:
    idle = device_power_watts("NVIDIA_H100", 0.0)
    full = device_power_watts("NVIDIA_H100", 1.0)
    assert idle == 150.0
    assert full == 700.0
    assert device_power_watts("NVIDIA_H100", 0.5) == pytest.approx((idle + full) / 2)


def test_a_reserved_idle_device_is_not_free() -> None:
    assert device_power_watts("NVIDIA_H100", 0.0) > 0


def test_utilization_is_clamped_not_extrapolated() -> None:
    assert device_power_watts("NVIDIA_H100", 4.0) == device_power_watts("NVIDIA_H100", 1.0)
    assert device_power_watts("NVIDIA_H100", -1.0) == device_power_watts("NVIDIA_H100", 0.0)


def test_host_share_scales_with_the_device() -> None:
    assert host_overhead_watts("NVIDIA_H100") > host_overhead_watts("NVIDIA_L4")
    with_host = device_power_watts("NVIDIA_H100", 1.0, include_host=True)
    assert with_host == pytest.approx(700.0 + host_overhead_watts("NVIDIA_H100"))


def test_unknown_device_draws_no_watts_rather_than_a_default() -> None:
    assert device_power_watts("MADE_UP", 1.0) == 0.0
    assert host_overhead_watts("MADE_UP") == 0.0


def test_fleet_power_is_a_lower_bound_under_unknown_hardware() -> None:
    known = fleet_power_watts({"NVIDIA_H100": 8}, 1.0)
    mixed = fleet_power_watts({"NVIDIA_H100": 8, "MADE_UP": 8}, 1.0)
    assert mixed == known, "unknown hardware contributes nothing, never a guess"
    assert known == pytest.approx(5600.0)


def test_energy_is_watt_seconds() -> None:
    assert energy_joules(700.0, 10.0) == 7000.0
    assert energy_joules(0.0, 10.0) == 0.0
    assert energy_joules(700.0, -1.0) == 0.0


def test_power_capped_concurrency_binds_before_the_slot_count() -> None:
    # A 208V/60A rack circuit is ~10 kW; eight 700 W devices plus their host share exceed it.
    fits = max_concurrent_devices(10_000.0, "NVIDIA_H100", 1.0)
    assert fits == 11
    assert max_concurrent_devices(100.0, "NVIDIA_H100") == 0


def test_unknown_device_yields_no_opinion_not_a_bound_of_zero() -> None:
    assert max_concurrent_devices(10_000.0, "MADE_UP") == -1


def test_envelope_defaults_to_unbounded() -> None:
    env = PowerEnvelope()
    assert env.unbounded
    assert env.fits()
    assert env.scale_to_fit(700.0) == -1


def test_envelope_reserves_headroom() -> None:
    env = PowerEnvelope(budget_watts=10_000.0, expected_watts=9_500.0, headroom_fraction=0.1)
    assert env.usable_watts == pytest.approx(9_000.0)
    assert not env.fits()
    assert PowerEnvelope(budget_watts=10_000.0, expected_watts=8_000.0).fits()


def test_envelope_passes_an_unknown_expected_draw() -> None:
    # An unsizable plan must not be refused on power grounds — that would fail queries whose
    # hardware this build simply does not recognize.
    assert PowerEnvelope(budget_watts=10.0, expected_watts=0.0).fits()


def test_one_clamp_serves_both_subsystems() -> None:
    # Kyber sizes against this and Carbonite admits against it. A second copy of the
    # arithmetic is how a plan gets sized for one fan-out and granted another.
    envelope = PowerEnvelope(budget_watts=10_000.0, headroom_fraction=0.1)
    assert envelope.devices_that_fit("NVIDIA_H100") == 10
    assert envelope.clamp_devices(64, "NVIDIA_H100") == 10
    assert envelope.clamp_devices(4, "NVIDIA_H100") == 4, "a smaller request is left alone"


def test_an_unbounded_envelope_has_no_opinion_on_device_count() -> None:
    assert PowerEnvelope().devices_that_fit("NVIDIA_H100") == -1
    assert PowerEnvelope().clamp_devices(64, "NVIDIA_H100") == 64


def test_an_unknown_device_is_never_clamped() -> None:
    envelope = PowerEnvelope(budget_watts=100.0)
    assert envelope.devices_that_fit("MADE_UP") == -1
    assert envelope.clamp_devices(64, "MADE_UP") == 64


def test_a_budget_too_small_for_one_device_still_plans_one() -> None:
    assert PowerEnvelope(budget_watts=50.0).clamp_devices(8, "NVIDIA_H100") == 1


def test_the_configured_envelope_reads_the_active_config() -> None:
    from batcher.config import EnergyConfig

    assert configured_power_envelope().unbounded
    cfg = Config().replace(
        accelerator=AcceleratorConfig(
            energy=EnergyConfig(power_budget_watts=10_000.0, power_headroom=0.2)
        )
    )
    with config_context(cfg):
        envelope = configured_power_envelope(expected_watts=500.0)
    assert envelope.budget_watts == 10_000.0
    assert envelope.usable_watts == pytest.approx(8_000.0)
    assert envelope.expected_watts == 500.0


def test_kwh_round_trips() -> None:
    assert joules_to_kwh(3.6e6) == pytest.approx(1.0)
    assert kwh_to_joules(joules_to_kwh(1234.0)) == pytest.approx(1234.0)
    assert joules_to_kwh(-5.0) == 0.0


def test_unconfigured_grid_reports_zero_not_an_invented_figure() -> None:
    assert carbon_grams(3.6e6, 0.0) == 0.0
    assert energy_cost(3.6e6, 0.0) == 0.0
    assert not GridProfile().configured


def test_pue_grosses_up_carbon_and_cost() -> None:
    grid = GridProfile(region="nordic", gco2e_per_kwh=20.0, price_per_kwh=0.05, pue=1.2)
    assert grid.configured
    assert grid.carbon_grams(3.6e6) == pytest.approx(24.0)
    assert grid.cost(3.6e6) == pytest.approx(0.06)
    assert grid.facility_joules(3.6e6) == pytest.approx(4.32e6)


def test_pue_below_one_is_impossible_and_clamped() -> None:
    assert GridProfile(gco2e_per_kwh=100.0, pue=0.5).carbon_grams(3.6e6) == pytest.approx(100.0)


def test_ledger_rolls_up_energy_and_efficiency() -> None:
    ledger = EnergyLedger()
    ledger.record(
        StageEnergy("Decode#1", "NVIDIA_H100", 8, 100.0, 0.9, joules=500_000.0, rows=1_000_000)
    )
    ledger.record(
        StageEnergy("Generate#2", "NVIDIA_H100", 8, 50.0, 0.8, joules=250_000.0, tokens=2_000_000)
    )
    assert ledger.total_joules == 750_000.0
    assert ledger.total_rows == 1_000_000
    assert ledger.total_tokens == 2_000_000
    assert ledger.hottest_stage() is not None
    assert ledger.hottest_stage().stage == "Decode#1"
    assert ledger.by_device() == {"NVIDIA_H100": 750_000.0}
    assert ledger.tokens_per_joule() == pytest.approx(2_000_000 / 750_000.0)


def test_efficiency_is_none_rather_than_zero_when_undefined() -> None:
    stage = StageEnergy("Filter#1", "NVIDIA_H100", 1, 10.0, 0.5, joules=1000.0, rows=0)
    assert stage.rows_per_joule is None, "a stage that emitted nothing has no efficiency figure"
    assert stage.tokens_per_joule is None
    assert EnergyLedger().tokens_per_joule() is None
    assert "tokens_per_joule" not in EnergyLedger().summary()


def test_idle_energy_is_what_a_scheduler_could_reclaim() -> None:
    busy = StageEnergy("A#1", "NVIDIA_H100", 1, 100.0, 1.0, joules=70_000.0)
    starved = StageEnergy("B#1", "NVIDIA_H100", 1, 100.0, 0.1, joules=205_000.0)
    assert busy.idle_joules == 0.0, "a fully fed device wastes nothing"
    assert starved.idle_joules > 0
    ledger = EnergyLedger()
    ledger.record(busy)
    ledger.record(starved)
    assert 0 < ledger.idle_fraction() < 1


def test_summary_is_json_safe() -> None:
    ledger = EnergyLedger()
    ledger.record(StageEnergy("A#1", "NVIDIA_H100", 1, 1.0, 1.0, joules=700.0, rows=10, tokens=5))
    assert all(isinstance(v, float) for v in ledger.summary().values())
