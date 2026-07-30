"""Data residency verdicts and the energy report.

Residency is a compliance control, so its failure modes are asymmetric: refusing a placement it
should allow takes a cluster offline, and allowing one it should refuse breaks an obligation
silently. These pin both directions — unregistered and unlabelled inputs never refuse, and a
registered dataset in the wrong region never passes in strict mode.

The report is held to the reporting rule the module states: an unknown figure is omitted rather
than printed as zero, because a zero in a cost report is a claim.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import AccessDeniedError
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.governance import DataResidency, ResidencyCatalog
from batcher.observe import energy_metrics, format_device_table, format_energy_report
from batcher.plan.energy import EnergyLedger, GridProfile, StageEnergy

pytestmark = pytest.mark.unit

_EU = DataResidency("s3://eu-cust/", frozenset({"eu-north-1", "eu-west-1"}), "GDPR Art. 44")


def _catalog(mode: str = "strict") -> ResidencyCatalog:
    return ResidencyCatalog(mode=mode).register(_EU)


def test_off_is_the_default_and_checks_nothing() -> None:
    catalog = ResidencyCatalog().register(_EU)
    assert catalog.check("s3://eu-cust/orders", "us-east-1").allowed
    assert catalog.permitted_regions(["s3://eu-cust/orders"]) is None


def test_a_registered_dataset_is_refused_outside_its_regions() -> None:
    verdict = _catalog().check("s3://eu-cust/orders", "us-east-1")
    assert not verdict.allowed
    assert verdict.enforced
    assert "GDPR Art. 44" in verdict.message()
    assert "eu-north-1, eu-west-1" in verdict.message()


def test_an_unregistered_dataset_is_unrestricted() -> None:
    assert _catalog().check("s3://public/reference", "us-east-1").allowed
    assert _catalog().check("s3://public/reference", "us-east-1").message() == ""


def test_an_unlabelled_region_never_refuses() -> None:
    # A dropped node label must not take a cluster offline.
    assert _catalog().check("s3://eu-cust/orders", "").allowed


def test_the_longest_matching_prefix_wins() -> None:
    catalog = _catalog().register(
        DataResidency("s3://eu-cust/public/", frozenset({"eu-north-1", "us-east-1"}), "published")
    )
    assert catalog.check("s3://eu-cust/public/prices", "us-east-1").allowed
    assert not catalog.check("s3://eu-cust/orders", "us-east-1").allowed


def test_an_empty_allowed_set_quarantines_rather_than_unrestricting() -> None:
    catalog = ResidencyCatalog(mode="strict").register(
        DataResidency("s3://quarantine/", frozenset(), "under investigation")
    )
    assert not catalog.check("s3://quarantine/x", "eu-north-1").allowed


def test_advisory_reports_the_refusal_without_enforcing_it() -> None:
    verdict = _catalog("advisory").enforce("s3://eu-cust/orders", "us-east-1")
    assert not verdict.allowed
    assert not verdict.enforced, "advisory exists so a fleet can measure before it blocks"


def test_strict_enforcement_raises_with_the_obligation_named() -> None:
    with pytest.raises(AccessDeniedError, match="GDPR"):
        _catalog().enforce("s3://eu-cust/orders", "us-east-1")


def test_a_multi_input_job_runs_only_where_every_input_may() -> None:
    catalog = _catalog().register(
        DataResidency("s3://uk-cust/", frozenset({"eu-west-2"}), "UK GDPR")
    )
    both = catalog.permitted_regions(["s3://eu-cust/orders", "s3://uk-cust/orders"])
    assert both == frozenset(), "an empty intersection is a real answer: split the job"
    assert catalog.permitted_regions(["s3://eu-cust/orders"]) == _EU.allowed_regions


def test_scheduler_filtering_preserves_preference_order() -> None:
    regions = _catalog().filter_regions(
        ["us-east-1", "eu-west-1", "eu-north-1"], ["s3://eu-cust/orders"]
    )
    assert regions == ("eu-west-1", "eu-north-1")
    assert _catalog().filter_regions(["us-east-1"], ["s3://public/x"]) == ("us-east-1",)


# --- the energy report ----------------------------------------------------------------


def _ledger() -> EnergyLedger:
    ledger = EnergyLedger()
    ledger.record(
        StageEnergy("Decode#1", "NVIDIA_H100", 8, 120.0, 0.35, joules=1_900_000.0, rows=4_000_000)
    )
    ledger.record(
        StageEnergy(
            "Generate#2", "NVIDIA_H100", 8, 400.0, 0.92, joules=2_400_000.0, tokens=88_000_000
        )
    )
    return ledger


def test_an_empty_ledger_says_so_rather_than_printing_zero() -> None:
    report = format_energy_report(EnergyLedger())
    assert "nothing recorded" in report
    assert "0 J" not in report


def test_the_report_names_the_hottest_stage() -> None:
    report = format_energy_report(_ledger())
    assert "hottest Generate#2" in report
    assert "4.3 MJ" in report
    assert "tokens/J" in report


def test_cost_and_carbon_lines_appear_only_when_the_grid_is_configured() -> None:
    plain = format_energy_report(_ledger())
    assert "carbon" not in plain, "an unconfigured grid must not report zero emissions"
    priced = format_energy_report(_ledger(), GridProfile("nordic", 20.0, 0.05, 1.15))
    assert "carbon" in priced
    assert "cost" in priced


def test_metrics_omit_undefined_figures() -> None:
    rows = energy_metrics(EnergyLedger())
    assert "energy.tokens_per_joule" not in rows
    assert rows["energy.joules"] == 0.0
    full = energy_metrics(_ledger(), GridProfile("nordic", 20.0, 0.05))
    assert full["energy.device.NVIDIA_H100"] == 4_300_000.0
    assert full["energy.carbon_grams"] > 0
    assert all(isinstance(v, float) for v in full.values())


def test_device_table_says_when_telemetry_is_absent() -> None:
    assert "no telemetry" in format_device_table([])


def test_device_table_surfaces_a_clamp_and_an_ecc_error() -> None:
    readings = [
        DeviceTelemetry(
            index=0,
            name="NVIDIA H100 80GB HBM3",
            power_watts=690.0,
            power_limit_watts=700.0,
            sm_utilization=0.97,
            memory_used_bytes=60 << 30,
            memory_total_bytes=80 << 30,
            temperature_c=78.0,
            throttle_reasons=("power",),
        ),
        DeviceTelemetry(index=1, name="NVIDIA H100 80GB HBM3", ecc_uncorrected=3),
    ]
    table = format_device_table(readings)
    assert "power" in table
    assert "ecc:3" in table
    assert "60/80 GiB" in table
