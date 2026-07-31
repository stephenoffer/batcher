"""The device table itself: one row per accelerator model, and the host link per part.

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

__all__ = ["SPECS", "DeviceSpec"]


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
        host_link: How the device reaches host memory — `pcie3`/`pcie4`/`pcie5` or
            `nvlink-c2c` for a coherent CPU-GPU package. `""` when not comparable.
        host_link_gbps: *Effective* one-way host-to-device bandwidth in GB/s, not the
            interface's theoretical rate: a PCIe 5.0 x16 link rates 64 GB/s and delivers
            around 50 on a pinned-memory copy, and unpinned host memory halves that again.
            This is the figure that decides whether a device is worth using at all for a
            scan-shaped stage, because the copy, not the kernel, is the cost.
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
    host_link: str = ""
    host_link_gbps: float = 0.0


# name, vendor, generation, GiB, GB/s, TDP W, idle W, dense half TFLOPS, dense FP8 TFLOPS,
# NVLink domain, NVLink GB/s, MIG slices. Written with bare integers where a figure is whole;
# `_SPECS` widens them to the dataclass's float fields, which keeps every row on one line.
_ROWS: tuple[tuple, ...] = (
    # NVIDIA datacenter, newest first.
    ("NVIDIA_GB200", "nvidia", "blackwell", 186, 8000, 1200, 300, 2250, 4500, 72, 1800, 7),
    ("NVIDIA_B200", "nvidia", "blackwell", 180, 8000, 1000, 250, 2250, 4500, 8, 1800, 7),
    ("NVIDIA_H200", "nvidia", "hopper", 141, 4800, 700, 150, 989, 1979, 8, 900, 7),
    ("NVIDIA_H100", "nvidia", "hopper", 80, 3350, 700, 150, 989, 1979, 8, 900, 7),
    ("NVIDIA_H20", "nvidia", "hopper", 96, 4000, 400, 0, 0, 0, 8, 900, 7),
    ("NVIDIA_L40S", "nvidia", "ada", 48, 864, 350, 40, 181, 362, 1, 0, 0),
    ("NVIDIA_L40", "nvidia", "ada", 48, 864, 300, 0, 0, 0, 1, 0, 0),
    ("NVIDIA_RTX_6000_ADA", "nvidia", "ada", 48, 960, 300, 0, 0, 0, 1, 0, 0),
    ("NVIDIA_L4", "nvidia", "ada", 24, 300, 72, 15, 121, 242, 1, 0, 0),
    ("NVIDIA_A100_80G", "nvidia", "ampere", 80, 2039, 400, 80, 312, 0, 8, 600, 7),
    ("NVIDIA_A100_40G", "nvidia", "ampere", 40, 1555, 400, 80, 312, 0, 8, 600, 7),
    ("NVIDIA_A100", "nvidia", "ampere", 40, 1555, 400, 80, 312, 0, 8, 600, 7),
    ("NVIDIA_A40", "nvidia", "ampere", 48, 696, 300, 35, 75, 0, 1, 0, 0),
    ("NVIDIA_A30", "nvidia", "ampere", 24, 933, 165, 30, 165, 0, 2, 200, 4),
    ("NVIDIA_A10G", "nvidia", "ampere", 24, 600, 300, 30, 70, 0, 1, 0, 0),
    ("NVIDIA_A10", "nvidia", "ampere", 24, 600, 150, 25, 62.5, 0, 1, 0, 0),
    ("NVIDIA_RTX_A6000", "nvidia", "ampere", 48, 768, 300, 0, 0, 0, 2, 112, 0),
    ("NVIDIA_TESLA_T4", "nvidia", "turing", 16, 320, 70, 12, 65, 0, 1, 0, 0),
    ("NVIDIA_TESLA_V100", "nvidia", "volta", 16, 900, 300, 50, 125, 0, 8, 300, 0),
    ("NVIDIA_TESLA_P100", "nvidia", "pascal", 16, 732, 300, 40, 19, 0, 4, 160, 0),
    ("NVIDIA_TESLA_P4", "nvidia", "pascal", 8, 192, 75, 10, 0, 0, 1, 0, 0),
    ("NVIDIA_TESLA_K80", "nvidia", "kepler", 12, 240, 300, 45, 0, 0, 1, 0, 0),
    # NVIDIA workstation and consumer parts, which the GPU-rental market runs on a great deal
    # of. Memory, bandwidth and board power are the columns these are consulted for — VRAM
    # sizing, the MIG answer (none of them partition), and the fabric width (none has more
    # than a two-way bridge). Their tensor throughput and idle draw are left unknown rather
    # than filled from a marketing figure whose basis (sparsity, boost, TF32 vs FP16) differs
    # by source, which is the one thing a table like this must not do.
    ("NVIDIA_RTX_5090", "nvidia", "blackwell", 32, 1792, 575, 0, 0, 0, 1, 0, 0),
    ("NVIDIA_RTX_4090", "nvidia", "ada", 24, 1008, 450, 0, 0, 0, 1, 0, 0),
    ("NVIDIA_RTX_3090", "nvidia", "ampere", 24, 936, 350, 0, 0, 0, 2, 112, 0),
    # AMD Instinct.
    ("AMD_INSTINCT_MI325X", "amd", "cdna3", 256, 6000, 1000, 180, 1307, 2615, 8, 896, 0),
    ("AMD_INSTINCT_MI300X", "amd", "cdna3", 192, 5300, 750, 150, 1307, 2615, 8, 896, 0),
    ("AMD_INSTINCT_MI300A", "amd", "cdna3", 128, 5300, 550, 0, 0, 0, 4, 896, 0),
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

#: How each part reaches host memory, and the *effective* one-way bandwidth of that link in
#: GB/s — measured-copy figures, not the interface's theoretical rate: PCIe 5.0 x16 rates
#: 64 GB/s and delivers around 50 on a pinned copy, PCIe 4.0 rates 32 and delivers about 25.
#: Kept out of the row table because it is a property of the *generation*, not of the part:
#: every Hopper board is PCIe 5.0, and stating that once per row invites the day one row says
#: otherwise. A part absent here has no comparable host link (the TPUs) and reads as unknown.
_HOST_LINK: dict[str, tuple[str, float]] = {
    "NVIDIA_GB200": ("nvlink-c2c", 450.0),
    "NVIDIA_B200": ("pcie5", 50.0),
    "NVIDIA_H200": ("pcie5", 50.0),
    "NVIDIA_H100": ("pcie5", 50.0),
    "NVIDIA_H20": ("pcie5", 50.0),
    "NVIDIA_L40S": ("pcie4", 25.0),
    "NVIDIA_L40": ("pcie4", 25.0),
    "NVIDIA_RTX_6000_ADA": ("pcie4", 25.0),
    "NVIDIA_RTX_5090": ("pcie5", 50.0),
    "NVIDIA_RTX_4090": ("pcie4", 25.0),
    "NVIDIA_RTX_3090": ("pcie4", 25.0),
    "NVIDIA_RTX_A6000": ("pcie4", 25.0),
    "NVIDIA_L4": ("pcie4", 25.0),
    "NVIDIA_A100_80G": ("pcie4", 25.0),
    "NVIDIA_A100_40G": ("pcie4", 25.0),
    "NVIDIA_A100": ("pcie4", 25.0),
    "NVIDIA_A40": ("pcie4", 25.0),
    "NVIDIA_A30": ("pcie4", 25.0),
    "NVIDIA_A10G": ("pcie4", 25.0),
    "NVIDIA_A10": ("pcie4", 25.0),
    "NVIDIA_TESLA_T4": ("pcie3", 12.0),
    "NVIDIA_TESLA_V100": ("pcie3", 12.0),
    "NVIDIA_TESLA_P100": ("pcie3", 12.0),
    "NVIDIA_TESLA_P4": ("pcie3", 12.0),
    "NVIDIA_TESLA_K80": ("pcie3", 12.0),
    "AMD_INSTINCT_MI325X": ("pcie5", 50.0),
    "AMD_INSTINCT_MI300X": ("pcie5", 50.0),
    # An APU: the GPU and the CPU share one package and one memory pool, so there is no host
    # link to cross at all. Recorded as the coherent-fabric case rather than as a fast PCIe
    # one, because a copy that does not happen is a different thing from a quick copy.
    "AMD_INSTINCT_MI300A": ("coherent", 0.0),
    "AMD_INSTINCT_MI250X": ("pcie4", 25.0),
    "AMD_INSTINCT_MI210": ("pcie4", 25.0),
    "INTEL_MAX_1550": ("pcie5", 50.0),
    "INTEL_MAX_1100": ("pcie5", 50.0),
}

SPECS: dict[str, DeviceSpec] = {
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
        host_link=_HOST_LINK.get(row[0], ("", 0.0))[0],
        host_link_gbps=_HOST_LINK.get(row[0], ("", 0.0))[1],
    )
    for row in _ROWS
}
