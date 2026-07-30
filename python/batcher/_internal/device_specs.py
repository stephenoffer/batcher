"""Datacenter accelerator specifications — the hardware facts a cluster cannot report.

`accelerators.py` answers "how much VRAM does this model have?" because Ray reports a device
*count* and a *model name* and nothing else. Running Batcher inside a GPU datacenter needs
more of the same shape of fact, and for the same reason: nothing in the runtime reports a
device's power draw, its memory bandwidth, the size of its NVLink domain, or whether it can be
partitioned. Every one of those decides something real —

* **power** decides energy-aware placement and the tokens-per-joule figures a datacenter bills
  and schedules on;
* **memory bandwidth** decides whether a scan-shaped stage is worth moving to a device at all,
  since a bandwidth-bound kernel scales with HBM, not with FLOPS;
* **the NVLink domain** decides where a tensor-parallel shard may land — spanning two domains
  turns an all-reduce from an on-package copy into a PCIe/network round trip;
* **MIG partitionability** decides whether a small model can share a device instead of holding
  a whole one idle.

This module is that table, keyed by the same `ray.util.accelerators` model names
`accelerators.py` uses, and it is the single source of truth for device memory as well —
`accelerator_memory_bytes` reads it rather than keeping a second copy.

**The numbers are vendor nameplate figures for the dense tensor path**, not measured
throughput: peak half-precision (BF16 on every generation that has it, FP16 on Turing and
older, which have no BF16 unit) and FP8 where the generation has an FP8 unit, both *without*
the 2x structured-sparsity multiplier vendors headline. They are used as *ratios* — is this
device 3x that one, is this stage bandwidth-bound or compute-bound — so consistency of basis
matters more than absolute accuracy, and a sparsity figure mixed into a dense table would
silently double one device against its neighbours.

**Unknown stays unknown.** Where one Ray name covers several configurations the smallest
shipping variant is recorded (the same conservative rule `accelerators.py` states), and an
unrecognized name yields `None` from `device_spec` and `0.0` from every scalar accessor, so a
caller falls back to whatever default it had instead of acting on a fabricated figure.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DeviceSpec",
    "device_arithmetic_intensity",
    "device_fp8_tflops",
    "device_generation",
    "device_half_tflops",
    "device_idle_watts",
    "device_memory_bandwidth_gbps",
    "device_mig_slices",
    "device_nvlink_domain",
    "device_nvlink_gbps",
    "device_spec",
    "device_tdp_watts",
    "device_tflops_per_watt",
    "device_vendor",
    "devices_by_generation",
    "known_device_names",
    "rank_devices_by_efficiency",
]


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """One accelerator model's datacenter-relevant nameplate figures.

    Attributes:
        name: The canonical (uppercased) `ray.util.accelerators` model name.
        vendor: `nvidia`, `amd`, `intel`, or `google`.
        generation: Vendor architecture family, used to group devices that share a
            capability set (`hopper`, `blackwell`, `ampere`, ...).
        memory_gib: Device memory of the smallest shipping variant, in GiB.
        memory_bandwidth_gbps: Peak device-memory bandwidth in GB/s.
        tdp_watts: Board power limit at full load.
        idle_watts: Board power at idle with the driver loaded — what a reserved but
            unused device still costs.
        half_tflops: Peak dense BF16 (FP16 pre-Ampere) tensor throughput in TFLOP/s.
        fp8_tflops: Peak dense FP8 tensor throughput, or `0.0` on a generation with no
            FP8 unit.
        nvlink_domain: Accelerators reachable over the coherent vendor fabric from one
            device (NVLink/NVSwitch, Infinity Fabric). `1` means PCIe-attached only.
        nvlink_gbps: Per-device bidirectional fabric bandwidth in GB/s, `0.0` for none.
        mig_slices: Maximum hardware partitions (NVIDIA MIG), `0` when not partitionable.
    """

    name: str
    vendor: str
    generation: str
    memory_gib: int
    memory_bandwidth_gbps: float
    tdp_watts: float
    idle_watts: float
    half_tflops: float
    fp8_tflops: float
    nvlink_domain: int
    nvlink_gbps: float
    mig_slices: int


# name, vendor, generation, GiB, GB/s, TDP W, idle W, dense half TFLOPS, dense FP8 TFLOPS,
# NVLink domain, NVLink GB/s, MIG slices. Written with bare integers where a figure is whole;
# `_SPECS` widens them to the dataclass's float fields, which keeps every row on one line.
_ROWS: tuple[tuple, ...] = (
    # NVIDIA datacenter, newest first.
    ("NVIDIA_GB200", "nvidia", "blackwell", 186, 8000, 1200, 300, 2250, 4500, 72, 1800, 7),
    ("NVIDIA_B200", "nvidia", "blackwell", 180, 8000, 1000, 250, 2250, 4500, 8, 1800, 7),
    ("NVIDIA_H200", "nvidia", "hopper", 141, 4800, 700, 150, 989, 1979, 8, 900, 7),
    ("NVIDIA_H100", "nvidia", "hopper", 80, 3350, 700, 150, 989, 1979, 8, 900, 7),
    ("NVIDIA_L40S", "nvidia", "ada", 48, 864, 350, 40, 181, 362, 1, 0, 0),
    ("NVIDIA_L4", "nvidia", "ada", 24, 300, 72, 15, 121, 242, 1, 0, 0),
    ("NVIDIA_A100_80G", "nvidia", "ampere", 80, 2039, 400, 80, 312, 0, 8, 600, 7),
    ("NVIDIA_A100_40G", "nvidia", "ampere", 40, 1555, 400, 80, 312, 0, 8, 600, 7),
    ("NVIDIA_A100", "nvidia", "ampere", 40, 1555, 400, 80, 312, 0, 8, 600, 7),
    ("NVIDIA_A40", "nvidia", "ampere", 48, 696, 300, 35, 75, 0, 1, 0, 0),
    ("NVIDIA_A30", "nvidia", "ampere", 24, 933, 165, 30, 165, 0, 2, 200, 4),
    ("NVIDIA_A10G", "nvidia", "ampere", 24, 600, 300, 30, 70, 0, 1, 0, 0),
    ("NVIDIA_A10", "nvidia", "ampere", 24, 600, 150, 25, 62.5, 0, 1, 0, 0),
    ("NVIDIA_TESLA_T4", "nvidia", "turing", 16, 320, 70, 12, 65, 0, 1, 0, 0),
    ("NVIDIA_TESLA_V100", "nvidia", "volta", 16, 900, 300, 50, 125, 0, 8, 300, 0),
    ("NVIDIA_TESLA_P100", "nvidia", "pascal", 16, 732, 300, 40, 19, 0, 4, 160, 0),
    ("NVIDIA_TESLA_P4", "nvidia", "pascal", 8, 192, 75, 10, 0, 0, 1, 0, 0),
    ("NVIDIA_TESLA_K80", "nvidia", "kepler", 12, 240, 300, 45, 0, 0, 1, 0, 0),
    # AMD Instinct.
    ("AMD_INSTINCT_MI325X", "amd", "cdna3", 256, 6000, 1000, 180, 1307, 2615, 8, 896, 0),
    ("AMD_INSTINCT_MI300X", "amd", "cdna3", 192, 5300, 750, 150, 1307, 2615, 8, 896, 0),
    ("AMD_INSTINCT_MI250X", "amd", "cdna2", 128, 3200, 560, 100, 383, 0, 8, 800, 0),
    ("AMD_INSTINCT_MI210", "amd", "cdna2", 64, 1600, 300, 60, 181, 0, 2, 300, 0),
    # Intel Data Center GPU Max.
    ("INTEL_MAX_1550", "intel", "ponte-vecchio", 128, 3200, 600, 120, 832, 0, 8, 0, 0),
    ("INTEL_MAX_1100", "intel", "ponte-vecchio", 48, 1229, 300, 60, 362, 0, 1, 0, 0),
    # Google Cloud TPU — HBM per chip, the unit Ray's `TPU` resource counts. Power and fabric
    # figures are not published per chip in a comparable form, so they read unknown (0) rather
    # than guessed; the memory column is what callers use these entries for.
    ("TPU-V6E", "google", "trillium", 32, 1640, 0, 0, 0, 0, 256, 0, 0),
    ("TPU-V5P", "google", "tpu-v5", 95, 2765, 0, 0, 0, 0, 8960, 0, 0),
    ("TPU-V5E", "google", "tpu-v5", 16, 819, 0, 0, 0, 0, 256, 0, 0),
    ("TPU-V5LITEPOD", "google", "tpu-v5", 16, 819, 0, 0, 0, 0, 256, 0, 0),
    ("TPU-V4", "google", "tpu-v4", 32, 1200, 0, 0, 0, 0, 4096, 0, 0),
    ("TPU-V3", "google", "tpu-v3", 16, 900, 0, 0, 0, 0, 1024, 0, 0),
    ("TPU-V2", "google", "tpu-v2", 8, 700, 0, 0, 0, 0, 256, 0, 0),
)

_SPECS: dict[str, DeviceSpec] = {
    row[0]: DeviceSpec(
        name=row[0],
        vendor=row[1],
        generation=row[2],
        memory_gib=int(row[3]),
        memory_bandwidth_gbps=float(row[4]),
        tdp_watts=float(row[5]),
        idle_watts=float(row[6]),
        half_tflops=float(row[7]),
        fp8_tflops=float(row[8]),
        nvlink_domain=int(row[9]),
        nvlink_gbps=float(row[10]),
        mig_slices=int(row[11]),
    )
    for row in _ROWS
}


def device_spec(accelerator_type: str | None) -> DeviceSpec | None:
    """The full specification for a Ray accelerator-type name, or `None` when unrecognized.

    Args:
        accelerator_type: A `ray.util.accelerators` model name such as `"NVIDIA_H100"`,
            matched case-insensitively. `None` and the empty string report unknown.

    Returns:
        The `DeviceSpec`, or `None` when the model is not in the table.
    """
    if not accelerator_type:
        return None
    return _SPECS.get(accelerator_type.upper())


def known_device_names() -> tuple[str, ...]:
    """Every accelerator model name the table recognizes, newest generation first.

    Returns:
        The canonical uppercased names, in table order.
    """
    return tuple(_SPECS)


def _field(accelerator_type: str | None, attr: str) -> float:
    spec = device_spec(accelerator_type)
    return float(getattr(spec, attr)) if spec is not None else 0.0


def device_tdp_watts(accelerator_type: str | None) -> float:
    """Board power limit at full load in watts, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Watts at the device's power limit, `0.0` if unrecognized or unpublished.
    """
    return _field(accelerator_type, "tdp_watts")


def device_idle_watts(accelerator_type: str | None) -> float:
    """Board power at idle in watts — what a reserved but unused device still burns.

    This is the figure that makes an idle GPU expensive rather than free, and it is why
    holding a device across a long CPU-bound stage is a real cost rather than a bookkeeping
    one.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Idle watts, `0.0` if unrecognized or unpublished.
    """
    return _field(accelerator_type, "idle_watts")


def device_memory_bandwidth_gbps(accelerator_type: str | None) -> float:
    """Peak device-memory bandwidth in GB/s, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Peak HBM/GDDR bandwidth in GB/s.
    """
    return _field(accelerator_type, "memory_bandwidth_gbps")


def device_half_tflops(accelerator_type: str | None) -> float:
    """Peak dense BF16/FP16 tensor throughput in TFLOP/s, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Dense half-precision TFLOP/s, without the structured-sparsity multiplier.
    """
    return _field(accelerator_type, "half_tflops")


def device_nvlink_domain(accelerator_type: str | None) -> int:
    """Accelerators reachable over one coherent vendor fabric, or `0` when unknown.

    `1` means the device is PCIe-attached only, so any multi-device collective crosses the
    host bus. A figure above one bounds how wide a tensor-parallel shard may go before its
    all-reduce leaves the fabric.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Devices in one NVLink/NVSwitch (or Infinity Fabric) domain, `0` if unrecognized.
    """
    spec = device_spec(accelerator_type)
    return spec.nvlink_domain if spec is not None else 0


def device_nvlink_gbps(accelerator_type: str | None) -> float:
    """Per-device bidirectional fabric bandwidth in GB/s, or `0.0` for PCIe-only devices.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        NVLink/Infinity Fabric bandwidth in GB/s.
    """
    return _field(accelerator_type, "nvlink_gbps")


def device_mig_slices(accelerator_type: str | None) -> int:
    """Maximum hardware partitions of one device, or `0` when it cannot be partitioned.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Maximum MIG instances per device, `0` for a device with no partitioning support.
    """
    spec = device_spec(accelerator_type)
    return spec.mig_slices if spec is not None else 0


def device_tflops_per_watt(accelerator_type: str | None) -> float:
    """Dense half-precision TFLOP/s per watt of board power, or `0.0` when unknown.

    The efficiency figure a power-constrained datacenter schedules on: two devices that
    deliver the same throughput are not equivalent if one draws twice the power, because the
    binding constraint on a full rack is the breaker, not the slot.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        TFLOP/s per watt, `0.0` when either figure is unknown.
    """
    spec = device_spec(accelerator_type)
    if spec is None or spec.tdp_watts <= 0 or spec.half_tflops <= 0:
        return 0.0
    return spec.half_tflops / spec.tdp_watts


def device_arithmetic_intensity(accelerator_type: str | None) -> float:
    """FLOPs per byte of memory traffic at which the device stops being bandwidth-bound.

    The ridge point of the device's roofline: a kernel below it is limited by HBM and gains
    nothing from a faster tensor core, which is exactly the case for the scan, filter, and
    shuffle shapes a data engine runs. Comparing a stage's intensity against this number is
    how the optimizer decides a stage is not worth a device.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        FLOPs per byte at the roofline ridge, `0.0` when either figure is unknown.
    """
    spec = device_spec(accelerator_type)
    if spec is None or spec.memory_bandwidth_gbps <= 0 or spec.half_tflops <= 0:
        return 0.0
    return spec.half_tflops * 1e12 / (spec.memory_bandwidth_gbps * 1e9)


def rank_devices_by_efficiency(names: list[str] | tuple[str, ...]) -> list[str]:
    """Recognized device names ordered most to least TFLOP/s per watt.

    Unrecognized names and devices with no published power figure are dropped rather than
    sorted to one end, because their position would be an invention: a datacenter placing
    work by efficiency needs the devices it can actually rank, not a list padded with
    guesses.

    Args:
        names: Candidate Ray accelerator-type names, in any order.

    Returns:
        The rankable subset, most efficient first; ties break on the name for determinism.
    """
    rankable = [(n, device_tflops_per_watt(n)) for n in names]
    return [n for n, eff in sorted(rankable, key=lambda p: (-p[1], p[0])) if eff > 0]


def device_fp8_tflops(accelerator_type: str | None) -> float:
    """Peak dense FP8 tensor throughput in TFLOP/s, `0.0` on a generation with no FP8 unit.

    The figure that decides whether quantizing a model buys throughput or only memory: on a
    part with an FP8 unit it roughly doubles the compute rate as well as halving the weights,
    and on one without it buys the memory alone.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Dense FP8 TFLOP/s, `0.0` when unsupported or unknown.
    """
    return _field(accelerator_type, "fp8_tflops")


def device_vendor(accelerator_type: str | None) -> str:
    """The device's vendor (`nvidia`, `amd`, `intel`, `google`), or `""` when unrecognized.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        The vendor name, lowercase.
    """
    spec = device_spec(accelerator_type)
    return spec.vendor if spec is not None else ""


def device_generation(accelerator_type: str | None) -> str:
    """The device's architecture family, or `""` when unrecognized.

    The right key for anything learned *per capability set* rather than per model: an H100 and
    an H200 differ in memory and bandwidth but share a instruction set and an FP8 unit, so a
    measurement from one transfers to the other in a way an Ampere measurement does not.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        The generation name (`"hopper"`, `"blackwell"`, `"ampere"`, ...), lowercase.
    """
    spec = device_spec(accelerator_type)
    return spec.generation if spec is not None else ""


def devices_by_generation(generation: str) -> tuple[str, ...]:
    """Every recognized device model in one architecture family, in table order.

    Args:
        generation: A generation name, matched case-insensitively.

    Returns:
        The model names, empty when the generation is not recognized.
    """
    want = generation.lower()
    return tuple(name for name, spec in _SPECS.items() if spec.generation == want)
