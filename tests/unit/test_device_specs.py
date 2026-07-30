"""The datacenter device table: internally consistent, conservative, and honest about gaps.

`device_specs` is read by placement, power budgeting, and the roofline check, so a wrong or
inconsistent row propagates into decisions nothing downstream can second-guess. These pin the
properties that hold across every row rather than restating individual datasheet figures: the
table stays the single source of device memory, unknown devices report unknown instead of a
default, and the derived ratios refuse to divide by an absent figure.
"""

from __future__ import annotations

import pytest

from batcher._internal.accelerators import accelerator_memory_bytes
from batcher._internal.device_specs import (
    device_half_tflops,
    device_idle_watts,
    device_memory_bandwidth_gbps,
    device_mig_slices,
    device_nvlink_domain,
    device_nvlink_gbps,
    device_spec,
    device_tdp_watts,
    device_tflops_per_watt,
    known_device_names,
    rank_devices_by_efficiency,
    resolve_device_name,
)

pytestmark = pytest.mark.unit


def test_every_row_is_internally_consistent() -> None:
    for name in known_device_names():
        spec = device_spec(name)
        assert spec is not None
        assert spec.memory_gib > 0, f"{name}: memory is the one figure that must be known"
        assert spec.memory_bandwidth_gbps > 0, name
        assert spec.idle_watts <= spec.tdp_watts, f"{name}: idle above TDP is impossible"
        assert spec.nvlink_domain >= 1, f"{name}: a device is always in a domain of at least itself"
        assert spec.mig_slices >= 0
        assert spec.vendor in {"nvidia", "amd", "intel", "google"}, name


def test_fp8_only_on_generations_that_have_an_fp8_unit() -> None:
    # An FP8 figure on a pre-Hopper part would silently double its planned throughput.
    for name in known_device_names():
        spec = device_spec(name)
        assert spec is not None
        if spec.fp8_tflops > 0:
            assert spec.generation in {"hopper", "blackwell", "ada", "cdna3"}, name
            assert spec.fp8_tflops >= spec.half_tflops, f"{name}: FP8 is never slower than BF16"


def test_unknown_device_reports_unknown_not_a_default() -> None:
    for probe in (None, "", "NVIDIA_MADE_UP_9000"):
        assert device_spec(probe) is None
        assert device_tdp_watts(probe) == 0.0
        assert device_idle_watts(probe) == 0.0
        assert device_memory_bandwidth_gbps(probe) == 0.0
        assert device_half_tflops(probe) == 0.0
        assert device_nvlink_domain(probe) == 0
        assert device_nvlink_gbps(probe) == 0.0
        assert device_mig_slices(probe) == 0
        assert device_tflops_per_watt(probe) == 0.0


def test_name_lookup_is_case_insensitive() -> None:
    assert device_spec("nvidia_h100") == device_spec("NVIDIA_H100")


def test_accelerator_memory_reads_this_table() -> None:
    # One source of truth: the memory accessor in `accelerators` must agree row for row,
    # because two tables of the same fact drift apart in the direction nobody is looking.
    for name in known_device_names():
        spec = device_spec(name)
        assert spec is not None
        assert accelerator_memory_bytes(name) == spec.memory_gib * (1 << 30)


def test_efficiency_ranking_drops_the_unrankable() -> None:
    ranked = rank_devices_by_efficiency(["NVIDIA_H100", "NVIDIA_TESLA_K80", "MADE_UP", "TPU-V4"])
    assert "MADE_UP" not in ranked, "an unknown device has no position, so it gets none"
    assert "TPU-V4" not in ranked, "no published power figure means no efficiency figure"
    assert "NVIDIA_TESLA_K80" not in ranked, "a part with no tensor path has no half-precision rate"
    assert ranked == ["NVIDIA_H100"]


def test_efficiency_ranking_orders_newer_parts_ahead() -> None:
    ranked = rank_devices_by_efficiency(["NVIDIA_TESLA_V100", "NVIDIA_H100", "NVIDIA_A100_80G"])
    assert ranked == ["NVIDIA_H100", "NVIDIA_A100_80G", "NVIDIA_TESLA_V100"]


def test_nvlink_domain_distinguishes_fabric_from_pcie() -> None:
    assert device_nvlink_domain("NVIDIA_L40S") == 1, "PCIe-only: no coherent fabric"
    assert device_nvlink_gbps("NVIDIA_L40S") == 0.0
    assert device_nvlink_domain("NVIDIA_H100") == 8
    assert device_nvlink_domain("NVIDIA_GB200") == 72, "rack-scale NVLink domain"


def test_a_driver_reported_name_resolves_to_a_table_key() -> None:
    # Nothing in the fleet spells a device the way the table does: NVML reports a board
    # variant and a memory size, Ray reports a label, and neither is the key.
    assert resolve_device_name("NVIDIA H100 80GB HBM3") == "NVIDIA_H100"
    assert resolve_device_name("Tesla T4") == "NVIDIA_TESLA_T4"
    assert resolve_device_name("AMD Instinct MI300X") == "AMD_INSTINCT_MI300X"
    assert resolve_device_name("TPU v4") == "TPU-V4"


def test_the_part_token_wins_over_a_shared_memory_token() -> None:
    # `80G` prefixes the `80GB` in an H100's name, so a substring match resolves an H100 to
    # an A100 — a 2x error in bandwidth, power, and tensor rate, silently.
    assert resolve_device_name("NVIDIA H100 80GB HBM3") != "NVIDIA_A100_80G"
    assert resolve_device_name("NVIDIA A100-SXM4-80GB") == "NVIDIA_A100_80G"
    assert resolve_device_name("NVIDIA A100-SXM4-40GB") == "NVIDIA_A100_40G"


def test_a_name_without_a_memory_size_takes_the_conservative_entry() -> None:
    resolved = resolve_device_name("NVIDIA A100")
    assert resolved == "NVIDIA_A100"
    assert device_spec(resolved).memory_gib == 40, "the smallest shipping variant"


def test_an_unrecognized_part_resolves_to_unknown_not_a_neighbour() -> None:
    assert resolve_device_name("NVIDIA GeForce RTX 4090") is None
    assert resolve_device_name("Some Future GPU") is None
    assert resolve_device_name("NVIDIA") is None, "a vendor alone identifies no part"
    assert resolve_device_name("") is None
    assert resolve_device_name(None) is None


def test_every_key_resolves_to_itself() -> None:
    # The round trip the resolver's precomputed token table must preserve. A key that no
    # longer resolves to itself means the precomputation and the table have drifted, and the
    # symptom downstream is a device silently reading as unknown.
    for name in known_device_names():
        assert resolve_device_name(name) == name, name


def test_every_key_resolves_from_a_driver_style_spelling() -> None:
    # Drivers report names with spaces and hyphens where the table uses underscores.
    for name in known_device_names():
        spelled = name.replace("_", " ").title()
        assert resolve_device_name(spelled) is not None, spelled


def test_every_row_is_physically_plausible() -> None:
    # The table is hand-entered from datasheets and nothing downstream can second-guess a
    # figure, so these are the bounds a transposed or fat-fingered column would violate.
    # Deliberately wide: they catch a typo, not a device that is merely unusual.
    for name in known_device_names():
        spec = device_spec(name)
        assert 100 <= spec.memory_bandwidth_gbps <= 10_000, f"{name}: HBM/GDDR rate"
        assert 4 <= spec.memory_gib <= 1024, f"{name}: device memory"
        assert spec.tdp_watts == 0 or 50 <= spec.tdp_watts <= 1500, f"{name}: board power"
        assert 0 <= spec.half_tflops <= 5000, f"{name}: dense half rate"
        assert 0 <= spec.nvlink_gbps <= 5000, f"{name}: fabric bandwidth"


def test_the_host_link_is_always_slower_than_device_memory() -> None:
    # Physically true of every accelerator ever shipped, and the one invariant that catches a
    # host-link figure entered in the wrong column: if a device could reach host memory as
    # fast as its own, it would not have its own.
    for name in known_device_names():
        spec = device_spec(name)
        if spec.host_link_gbps:
            assert spec.host_link_gbps < spec.memory_bandwidth_gbps, name


def test_the_fabric_is_always_faster_than_the_host_link() -> None:
    # The reason a collective is worth keeping inside an NVLink domain at all. A part where
    # this did not hold would make every placement decision in this package pointless.
    for name in known_device_names():
        spec = device_spec(name)
        if spec.nvlink_gbps and spec.host_link_gbps:
            assert spec.nvlink_gbps > spec.host_link_gbps, name
