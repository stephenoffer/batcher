"""Carbonite's accelerator layer: VRAM pooling, MIG planning, KV budgets, and health verdicts.

Each of these decides whether work is admitted to a device, so the failure they exist to
prevent is silent: an over-granted pool OOMs a worker, a wrong MIG plan holds seven devices for
one model's worth of work, an over-optimistic KV budget makes a serving engine thrash, and a
health verdict that fires on absent telemetry takes a whole fleet offline. These pin the
conservative direction of each.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ResourceError
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel import (
    HealthThresholds,
    KvCacheBudget,
    VramPool,
    assess_device,
    assess_fleet,
    kv_bytes_per_token,
    max_concurrent_sequences,
    mig_plan,
    mig_profiles,
    mig_supported,
    schedulable_devices,
    smallest_profile_for,
)

pytestmark = pytest.mark.unit

_GIB = 1 << 30


# --- VRAM pool ------------------------------------------------------------------------


def test_pool_holds_headroom_back() -> None:
    pool = VramPool(capacity_bytes=80 * _GIB, headroom=0.15)
    assert pool.usable_bytes() == int(80 * _GIB * 0.85)
    assert not pool.fits(80 * _GIB)
    assert pool.fits(60 * _GIB)


def test_reservations_accumulate_across_claimants() -> None:
    # The failure a bare free-memory check cannot see: three stages each find the same free
    # bytes and each proceed.
    pool = VramPool(capacity_bytes=80 * _GIB, headroom=0.0)
    a = pool.reserve(30 * _GIB, device=0, owner="weights")
    pool.reserve(30 * _GIB, device=0, owner="kv-cache")
    assert pool.held_bytes(0) == 60 * _GIB
    with pytest.raises(ResourceError, match="cannot hold"):
        pool.reserve(30 * _GIB, device=0)
    pool.release(a)
    assert pool.fits(30 * _GIB)


def test_release_beyond_what_is_held_clamps_at_zero() -> None:
    pool = VramPool(capacity_bytes=80 * _GIB)
    res = pool.reserve(10 * _GIB, device=0)
    pool.release(res)
    pool.release(res)  # a double release is a bug worth surviving
    assert pool.held_bytes(0) == 0


def test_external_usage_binds_the_budget() -> None:
    pool = VramPool(capacity_bytes=80 * _GIB, headroom=0.0)
    pool.observe_external(0, 70 * _GIB)
    assert pool.available_bytes(0) == 10 * _GIB
    with pytest.raises(ResourceError, match="external"):
        pool.reserve(20 * _GIB, device=0)


def test_external_observation_excludes_this_pools_own_holdings() -> None:
    pool = VramPool(capacity_bytes=80 * _GIB, headroom=0.0)
    pool.reserve(20 * _GIB, device=0)
    pool.observe_external(0, 30 * _GIB)  # 20 of that 30 is ours
    assert pool.external_bytes[0] == 10 * _GIB
    assert pool.available_bytes(0) == 50 * _GIB


def test_placement_picks_the_emptiest_device_deterministically() -> None:
    pool = VramPool(capacity_bytes=80 * _GIB, device_count=4, headroom=0.0)
    pool.reserve(60 * _GIB, device=0)
    pool.reserve(40 * _GIB, device=1)
    assert pool.best_device() == 2, "ties break low so a repeated run places the same way"
    assert pool.reserve(10 * _GIB).device == 2


def test_pool_tracks_the_high_water_mark() -> None:
    pool = VramPool(capacity_bytes=80 * _GIB, headroom=0.0)
    res = pool.reserve(50 * _GIB, device=0)
    pool.release(res)
    assert pool.peak_bytes(0) == 50 * _GIB
    assert pool.pressure(0) == 0.0
    assert pool.summary()["peak_bytes"] == float(50 * _GIB)


def test_a_non_positive_reservation_is_rejected() -> None:
    with pytest.raises(ResourceError, match="must be positive"):
        VramPool(capacity_bytes=_GIB).reserve(0)


# --- MIG ------------------------------------------------------------------------------


def test_derived_profiles_match_the_published_families() -> None:
    a100 = [p.name for p in mig_profiles("NVIDIA_A100_40G")]
    assert a100 == ["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"]
    h100 = [p.name for p in mig_profiles("NVIDIA_H100")]
    assert h100 == ["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]


def test_instances_per_device_follow_the_compute_slices() -> None:
    by_name = {p.name: p for p in mig_profiles("NVIDIA_H100")}
    assert by_name["1g.10gb"].instances == 7
    assert by_name["2g.20gb"].instances == 3
    assert by_name["7g.80gb"].instances == 1
    assert by_name["1g.10gb"].gpu_fraction == pytest.approx(1 / 7)


def test_non_partitionable_devices_report_no_profiles() -> None:
    assert not mig_supported("NVIDIA_L40S")
    assert mig_profiles("NVIDIA_L40S") == ()
    assert not mig_supported("MADE_UP")
    assert mig_profiles(None) == ()


def test_small_model_gets_the_smallest_instance_that_fits() -> None:
    profile = smallest_profile_for(6.0, "NVIDIA_H100")
    assert profile is not None
    assert profile.name == "1g.10gb"


def test_a_model_needing_the_whole_device_is_not_partitioned() -> None:
    assert smallest_profile_for(70.0, "NVIDIA_H100") is None
    plan = mig_plan(70.0, "NVIDIA_H100", concurrency=4)
    assert plan.profile is None
    assert plan.gpu_fraction == 1.0
    assert plan.devices_needed == 4


def test_partitioning_collapses_devices_needed() -> None:
    plan = mig_plan(6.0, "NVIDIA_H100", concurrency=14)
    assert plan.profile is not None
    assert plan.instances_per_device == 7
    assert plan.devices_needed == 2, "14 small workers fit on two devices, not fourteen"
    assert plan.gpu_fraction == pytest.approx(1 / 7)
    assert "1g.10gb" in plan.reason


def test_headroom_is_charged_against_the_instance() -> None:
    # 9 GiB fits a 10 GiB instance only if headroom is ignored, which is how a stage OOMs.
    assert smallest_profile_for(9.0, "NVIDIA_H100", headroom=0.15).name == "2g.20gb"
    assert smallest_profile_for(9.0, "NVIDIA_H100", headroom=0.0).name == "1g.10gb"


# --- KV cache -------------------------------------------------------------------------


def test_grouped_query_attention_shrinks_the_cache() -> None:
    mha = kv_bytes_per_token(layers=32, kv_heads=32, head_dim=128, dtype="fp16")
    gqa = kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, dtype="fp16")
    assert mha == 2 * 32 * 32 * 128 * 2
    assert gqa * 4 == mha, "an eighth of the heads is an eighth of the cache"


def test_cache_dtype_is_the_largest_concurrency_lever() -> None:
    fp16 = kv_bytes_per_token(32, 8, 128, "fp16")
    fp8 = kv_bytes_per_token(32, 8, 128, "fp8")
    assert fp8 * 2 == fp16


def test_unknown_dtype_reports_zero_rather_than_guessing() -> None:
    assert kv_bytes_per_token(32, 8, 128, "int4") == 0
    assert max_concurrent_sequences(80 * _GIB, 4096, 0) == 0


def test_budget_sets_concurrency_from_what_is_left_after_weights() -> None:
    per_token = kv_bytes_per_token(80, 8, 128, "fp16")
    budget = KvCacheBudget(
        device_bytes=80 * _GIB,
        weight_bytes=40 * _GIB,
        bytes_per_token=per_token,
        context_tokens=8192,
        headroom=0.1,
    )
    assert budget.fits
    assert budget.max_sequences == budget.cache_bytes // (8192 * per_token)
    assert budget.sequences_at(2048) > budget.max_sequences, "shorter context, more sequences"
    assert 0 < budget.cache_fraction < 1


def test_a_model_that_does_not_fit_reports_zero_sequences_not_a_negative_cache() -> None:
    budget = KvCacheBudget(
        device_bytes=24 * _GIB,
        weight_bytes=40 * _GIB,
        bytes_per_token=kv_bytes_per_token(80, 8, 128, "fp16"),
        context_tokens=8192,
    )
    assert budget.cache_bytes == 0
    assert budget.max_sequences == 0
    assert not budget.fits
    assert budget.devices_for(64) == 0, "replication cannot fix a model that does not fit"


def test_devices_for_rounds_up() -> None:
    budget = KvCacheBudget(
        device_bytes=80 * _GIB,
        weight_bytes=16 * _GIB,
        bytes_per_token=kv_bytes_per_token(32, 8, 128, "fp16"),
        context_tokens=4096,
    )
    per_device = budget.max_sequences
    assert per_device > 0
    assert budget.devices_for(per_device) == 1
    assert budget.devices_for(per_device + 1) == 2


# --- health ---------------------------------------------------------------------------


def _reading(**kw) -> DeviceTelemetry:
    base = {
        "index": 0,
        "uuid": "GPU-0",
        "temperature_c": 60.0,
        "memory_used_bytes": 10 * _GIB,
        "memory_total_bytes": 80 * _GIB,
    }
    return DeviceTelemetry(**{**base, **kw})


def test_a_normal_device_is_healthy() -> None:
    verdict = assess_device(_reading())
    assert verdict.state == "healthy"
    assert verdict.derate == 1.0
    assert verdict.reasons == ()


def test_uncorrectable_ecc_quarantines_regardless_of_throughput() -> None:
    verdict = assess_device(_reading(ecc_uncorrected=1))
    assert verdict.state == "quarantine"
    assert verdict.reasons == ("ecc",)
    assert not verdict.schedulable


def test_a_power_capped_device_is_derated_not_removed() -> None:
    # The cap is usually the datacenter's own limit working as intended.
    verdict = assess_device(_reading(throttle_reasons=("power",)))
    assert verdict.state == "degraded"
    assert verdict.schedulable
    assert verdict.derate == 0.75
    assert verdict.reasons == ("power_clamp",)


def test_hardware_thermal_slowdown_derates_harder() -> None:
    verdict = assess_device(_reading(throttle_reasons=("thermal",)))
    assert verdict.reasons == ("thermal_throttle",)
    assert verdict.derate == 0.5
    assert verdict.schedulable


def test_a_device_another_tenant_filled_is_not_scheduled() -> None:
    verdict = assess_device(_reading(memory_used_bytes=79 * _GIB))
    assert verdict.state == "quarantine"
    assert "memory_full" in verdict.reasons


def test_empty_telemetry_never_quarantines_a_fleet() -> None:
    assert assess_fleet([]) == ()
    assert schedulable_devices([]) == ()
    assert assess_device(DeviceTelemetry(index=3)).state == "healthy"


def test_thresholds_are_configurable_not_hardcoded() -> None:
    strict = HealthThresholds(max_temperature_c=55.0)
    assert assess_device(_reading(), strict).state == "degraded"
    assert assess_device(_reading()).state == "healthy"


def test_schedulable_excludes_only_the_quarantined() -> None:
    readings = [
        _reading(index=0),
        _reading(index=1, throttle_reasons=("thermal",)),
        _reading(index=2, ecc_uncorrected=4),
    ]
    assert schedulable_devices(readings) == (0, 1)
