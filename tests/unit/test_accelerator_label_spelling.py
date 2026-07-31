"""A decision must not change because the device was named the way Ray names it.

The device table's row keys are `ray.util.accelerators` constant *identifiers*
(`NVIDIA_TESLA_T4`); a node is labelled with the constant's *value* (`"T4"`), and the label is
what reaches a decision at runtime. When only the identifier resolved, every accelerator-aware
decision on a real cluster silently took its unknown-device branch: no transfer veto, no energy
opinion, no VRAM, a fabricated fabric width. Each of those is a legitimate answer for a device
nobody has heard of, which is why nothing failed and nothing was logged.

So the property worth pinning is not "the table has a T4 row" -- it is that *both spellings
reach the same decision*, at every layer that takes an `accelerator_type`. A new call site that
looks the name up strictly will fail here rather than in production.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

#: `(what Ray labels the node, what the table calls the same part)`. Only parts whose two
#: spellings genuinely differ are useful here.
_SPELLINGS = [
    ("T4", "NVIDIA_TESLA_T4"),
    ("V100", "NVIDIA_TESLA_V100"),
    ("A100", "NVIDIA_A100"),
    ("A10G", "NVIDIA_A10G"),
    ("L4", "NVIDIA_L4"),
    ("H100", "NVIDIA_H100"),
]


@pytest.mark.parametrize(("label", "key"), _SPELLINGS)
def test_device_memory_agrees_across_spellings(label: str, key: str) -> None:
    from batcher._internal.accelerators import accelerator_memory_bytes

    assert accelerator_memory_bytes(label) == accelerator_memory_bytes(key) > 0


@pytest.mark.parametrize(("label", "key"), _SPELLINGS)
def test_every_device_fact_agrees_across_spellings(label: str, key: str) -> None:
    from batcher._internal import device_specs as ds

    for fact in (
        ds.device_generation,
        ds.device_half_tflops,
        ds.device_host_link,
        ds.device_host_link_gbps,
        ds.device_idle_watts,
        ds.device_memory_bandwidth_gbps,
        ds.device_mig_slices,
        ds.device_nvlink_domain,
        ds.device_nvlink_gbps,
        ds.device_tdp_watts,
    ):
        assert fact(label) == fact(key), f"{fact.__name__} disagrees for {label!r} vs {key!r}"


@pytest.mark.parametrize(("label", "key"), _SPELLINGS)
def test_the_fabric_width_agrees_across_spellings(label: str, key: str) -> None:
    """The one that fabricated a fabric: an unresolved part fell back to the node's device
    count, so four PCIe T4s were reported as one coherent domain of four."""
    from batcher.dist.executors.ray_runtime.fabric import nvlink_domain_size

    assert nvlink_domain_size(label, 4) == nvlink_domain_size(key, 4)


def test_a_pcie_part_does_not_claim_a_fabric_it_has_not_got() -> None:
    from batcher.dist.executors.ray_runtime.fabric import nvlink_domain_size

    assert nvlink_domain_size("T4", 4) == 1, "a T4 has no NVLink; four of them are still four"
    assert nvlink_domain_size("UNHEARD_OF", 4) == 4, "unknown still degrades to node-local"


@pytest.mark.parametrize(("label", "key"), _SPELLINGS)
def test_the_energy_opinion_agrees_across_spellings(label: str, key: str) -> None:
    from batcher.kyber.gpu.energy import device_energy_advice

    lhs = device_energy_advice(label, bytes_per_row=200.0, flops_per_row=10.0)
    rhs = device_energy_advice(key, bytes_per_row=200.0, flops_per_row=10.0)
    assert lhs.worth_it == rhs.worth_it
    assert lhs.speedup == rhs.speedup


@pytest.mark.parametrize(("label", "key"), _SPELLINGS)
def test_the_transfer_veto_agrees_across_spellings(label: str, key: str) -> None:
    """A veto that fires for `NVIDIA_TESLA_T4` and not for `T4` is a veto that never fires."""
    from batcher.kyber.gpu import policy

    lhs = policy._transfer_veto(label, working_set_gb=10.0, rows=100_000_000)
    rhs = policy._transfer_veto(key, working_set_gb=10.0, rows=100_000_000)
    assert (lhs is None) == (rhs is None)


def test_a_slow_link_actually_vetoes_scan_shaped_work() -> None:
    """The veto is only worth agreeing about if it fires at all on a PCIe-3 part."""
    from batcher.kyber.gpu import policy

    veto = policy._transfer_veto("T4", working_set_gb=10.0, rows=100_000_000)
    assert veto is not None, "a T4's host copy dominates a scan; the CPU should win"
    assert "transfer" in veto


def test_a_fast_link_does_not_veto() -> None:
    from batcher.kyber.gpu import policy

    assert policy._transfer_veto("H100", working_set_gb=10.0, rows=100_000_000) is None
