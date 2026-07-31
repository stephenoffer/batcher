"""The accelerator config section and the Kyber decisions that read it.

Two properties matter more than any individual number here. First, every default is inert: a
deployment that configures nothing must get exactly the placement and fan-out it got before
energy was plannable, because a switch that changes behavior by default is one an operator
cannot adopt incrementally. Second, an unknown device produces no opinion rather than a
default, so a fleet this build does not recognize is never scheduled against fabricated watts.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ConfigError
from batcher.config import (
    AcceleratorConfig,
    Config,
    DeviceHealthConfig,
    EnergyConfig,
    config_context,
    config_to_dict,
)
from batcher.kyber.gpu import (
    device_energy_advice,
    power_bounded_devices,
    select_device_class,
    stage_joules,
)

pytestmark = pytest.mark.unit


# --- configuration --------------------------------------------------------------------


def test_defaults_are_inert() -> None:
    accel = Config().accelerator
    assert accel.energy.power_budget_watts == 0.0, "unbounded"
    assert accel.energy.carbon_intensity == 0.0, "unconfigured, not a shipped national average"
    assert accel.energy.pue == 1.0, "IT load only"
    assert not accel.health.enabled, "telemetry-driven health is opt-in"
    assert not accel.efficiency_first_placement, "placement is unconstrained by default"


def test_section_reads_from_the_environment_by_path() -> None:
    cfg = Config.from_env(
        {
            "BATCHER_ACCELERATOR_ENERGY_POWER_BUDGET_WATTS": "12000",
            "BATCHER_ACCELERATOR_HEALTH_ENABLED": "1",
            "BATCHER_ACCELERATOR_KV_CACHE_DTYPE": "fp8",
        }
    )
    assert cfg.accelerator.energy.power_budget_watts == 12_000.0
    assert cfg.accelerator.health.enabled
    assert cfg.accelerator.kv_cache_dtype == "fp8"


def test_section_round_trips_through_the_dict_form() -> None:
    cfg = Config().replace(accelerator=AcceleratorConfig(energy=EnergyConfig(pue=1.2)))
    assert config_to_dict(cfg, only_non_default=True) == {"accelerator": {"energy": {"pue": 1.2}}}


def test_out_of_range_values_are_refused_with_the_field_named() -> None:
    with pytest.raises(ConfigError, match="pue"):
        Config().replace(accelerator=AcceleratorConfig(energy=EnergyConfig(pue=0.5))).validate()
    with pytest.raises(ConfigError, match="kv_cache_dtype"):
        Config().replace(accelerator=AcceleratorConfig(kv_cache_dtype="int4")).validate()
    with pytest.raises(ConfigError, match="max_memory_fraction"):
        Config().replace(
            accelerator=AcceleratorConfig(health=DeviceHealthConfig(max_memory_fraction=1.5))
        ).validate()


# --- device selection -----------------------------------------------------------------


def test_smallest_that_fits_is_the_default_ordering() -> None:
    chosen = select_device_class(["NVIDIA_L4", "NVIDIA_A100_80G", "NVIDIA_H200"], 30.0)
    assert chosen == "NVIDIA_A100_80G", "an H200 would fit too, and would be wasted here"


def test_efficiency_first_picks_a_different_device() -> None:
    fleet = ["NVIDIA_TESLA_V100", "NVIDIA_A100_80G", "NVIDIA_H100"]
    assert select_device_class(fleet, 30.0, prefer_efficiency=False) == "NVIDIA_A100_80G"
    assert select_device_class(fleet, 30.0, prefer_efficiency=True) == "NVIDIA_H100"


def test_efficiency_first_follows_the_active_config() -> None:
    fleet = ["NVIDIA_TESLA_V100", "NVIDIA_A100_80G", "NVIDIA_H100"]
    cfg = Config().replace(accelerator=AcceleratorConfig(efficiency_first_placement=True))
    with config_context(cfg):
        assert select_device_class(fleet, 30.0) == "NVIDIA_H100"


def test_no_pin_when_a_pin_would_only_constrain() -> None:
    assert select_device_class(["NVIDIA_H100", "NVIDIA_H200"], 4.0) is None, "everything fits"
    assert select_device_class(["NVIDIA_L4", "NVIDIA_H100"], 300.0) is None, "nothing fits"
    assert select_device_class(["NVIDIA_H100"], 30.0) is None, "homogeneous"
    assert select_device_class(["MADE_UP", "ALSO_MADE_UP"], 30.0) is None, "unknowable"


# --- power-bounded fan-out ------------------------------------------------------------


def test_no_budget_leaves_the_requested_fan_out_alone() -> None:
    assert power_bounded_devices(64, "NVIDIA_H100") == 64


def test_a_budget_clamps_fan_out_below_the_slot_count() -> None:
    budget = AcceleratorConfig(energy=EnergyConfig(power_budget_watts=10_000.0))
    with config_context(Config().replace(accelerator=budget)):
        assert power_bounded_devices(64, "NVIDIA_H100") == 10


def test_an_unknown_device_is_never_clamped_on_fabricated_watts() -> None:
    budget = AcceleratorConfig(energy=EnergyConfig(power_budget_watts=1_000.0))
    with config_context(Config().replace(accelerator=budget)):
        assert power_bounded_devices(64, "MADE_UP") == 64


def test_a_budget_too_small_for_one_device_still_plans_one() -> None:
    # Surfacing that as a zero-device plan here would produce a confusing empty stage; the
    # right place to refuse it is admission, with the budget named.
    budget = AcceleratorConfig(energy=EnergyConfig(power_budget_watts=100.0))
    with config_context(Config().replace(accelerator=budget)):
        assert power_bounded_devices(8, "NVIDIA_H100") == 1


# --- the roofline / energy verdict ----------------------------------------------------


def test_a_compute_bound_stage_is_worth_the_watts() -> None:
    advice = device_energy_advice("NVIDIA_H100", bytes_per_row=1.0, flops_per_row=100_000.0)
    assert advice.worth_it
    assert advice.energy_ratio < 1.0
    assert "compute-bound" in advice.reason


def test_a_bandwidth_bound_stage_gains_bandwidth_not_flops() -> None:
    advice = device_energy_advice(
        "NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0, resident=True
    )
    assert "bandwidth-bound" in advice.reason
    assert advice.speedup == pytest.approx(3350.0 / 20.0), "resident data pays no copy"
    assert advice.transfer_share == 0.0


def test_the_host_copy_is_charged_and_usually_dominates_a_scan() -> None:
    # The term a data engine forgets and then cannot explain. A scan's bytes cross PCIe
    # before a kernel sees them, and that copy is slower than the device's own memory by two
    # orders of magnitude — so the 167x an on-device roofline promises collapses to ~2x.
    advice = device_energy_advice("NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0)
    assert advice.transfer_share > 0.5
    assert "host copy dominates" in advice.reason
    assert advice.speedup < 5.0


def test_a_coherent_package_changes_the_answer_rather_than_shading_it() -> None:
    # An NVLink-C2C host link is an order of magnitude faster than PCIe, which is what makes
    # scan-shaped work worth offloading on those parts and not on their PCIe siblings.
    pcie = device_energy_advice("NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0)
    c2c = device_energy_advice("NVIDIA_GB200", bytes_per_row=64.0, flops_per_row=4.0)
    assert c2c.speedup > 5 * pcie.speedup


def test_a_slow_link_can_make_a_device_lose_outright() -> None:
    advice = device_energy_advice("NVIDIA_TESLA_T4", bytes_per_row=64.0, flops_per_row=4.0)
    assert advice.speedup < 1.0, "a PCIe 3.0 copy is slower than the CPU scanning the data"
    assert not advice.worth_it


def test_the_cpu_path_is_scored_on_the_same_roofline() -> None:
    # Charging the CPU only its memory bandwidth made a compute-heavy row look free there and
    # cost the device its entire advantage — the verdict said inference was not worth a GPU.
    advice = device_energy_advice("NVIDIA_H100", bytes_per_row=64.0, flops_per_row=2_000_000.0)
    assert advice.worth_it
    assert advice.speedup > 100
    assert "compute-bound" in advice.reason


def test_a_kernel_far_off_peak_stops_being_worth_the_watts() -> None:
    advice = device_energy_advice(
        "NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0, achieved_fraction=0.01
    )
    assert not advice.worth_it
    assert advice.energy_ratio > 1.0


def test_an_unknown_device_leaves_the_existing_decision_untouched() -> None:
    advice = device_energy_advice("MADE_UP", bytes_per_row=64.0, flops_per_row=4.0)
    assert advice.worth_it, "no energy opinion must not become a veto"
    assert advice.speedup == 0.0
    assert "no energy opinion" in advice.reason


def test_stage_energy_includes_the_host_share() -> None:
    joules = stage_joules(60.0, "NVIDIA_H100", 8)
    assert joules == pytest.approx(60.0 * 8 * 875.0)
    assert stage_joules(60.0, "MADE_UP", 8) == 0.0


def test_a_non_reducing_shape_pays_the_full_return_trip() -> None:
    # The default result fraction describes the *reducing* shapes this decision routes — a
    # group-by returns a group count, not a row count. A projection returns roughly what it
    # reads, and inheriting the reducing default would understate its copy tenfold.
    reducing = device_energy_advice("NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0)
    projection = device_energy_advice(
        "NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0, result_fraction=1.0
    )
    assert projection.speedup < reducing.speedup
    assert projection.transfer_share >= reducing.transfer_share


def test_a_negative_result_fraction_cannot_make_the_copy_free() -> None:
    advice = device_energy_advice(
        "NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0, result_fraction=-5.0
    )
    assert advice.transfer_share > 0
