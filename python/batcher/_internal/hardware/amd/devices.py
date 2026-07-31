"""What an AMD Instinct node can be asked about itself, without ROCm installed.

Every device-level fact Batcher reads today comes from NVML, and NVML is NVIDIA-only. On an
MI300X node — which is a large and growing share of rented GPU capacity, and the part several
GPU-specialist clouds are built around — the telemetry probe reports no devices, the fault
probe reports no faults, and the mode probe reports no misconfiguration. None of those is
"this node is fine". They are all "this node was never looked at", and the two are
indistinguishable from the values alone. That is the exact failure this package exists to
prevent elsewhere, reproduced for a whole vendor.

The fix is not a second optional dependency. `amdsmi` ships with ROCm, is versioned against
the driver, and is absent from every container that runs a framework wheel rather than a full
ROCm install. Everything below instead reads `/sys/class/drm/card*/device`, which the in-tree
`amdgpu` driver publishes on any machine with the kernel module loaded, needs no runtime, no
root, and no device context:

* `mem_info_vram_total` / `mem_info_vram_used` — device memory, in bytes.
* `gpu_busy_percent` — occupancy, the same figure `rocm-smi` prints as GPU%.
* `product_name`, `unique_id`, `serial_number` — identity, for matching against the device
  table and for naming a board in an RMA.
* `hwmon/hwmon*/` — `temp1_input` and `temp1_crit` in millidegrees, `power1_average` and
  `power1_cap` in microwatts. The same hwmon contract every other Linux sensor uses.
* `ras/*_err_count` — the RAS blocks' correctable and uncorrectable counts. An uncorrectable
  error in the memory controller block is AMD's equivalent of a double-bit ECC Xid, and it is
  the one signal here that should take a device out of service.
* `current_compute_partition` / `current_memory_partition` — how the board is carved up. This
  is AMD's MIG, and it matters for the same reason: an MI300X in `CPX` presents eight logical
  devices with an eighth of the compute each, so every other figure on the row is a slice
  rather than a board. Reported unconditionally when published, because partitioning is
  usually deliberate and is never irrelevant.

Two deliberate omissions. There is no attempt to read the XGMI fabric: the sysfs names for it
have moved between kernel releases and a fabric figure that is wrong is worse than one that is
absent, so `fabric.nvlink` stays NVIDIA-only and an AMD node reports an unknown fabric. And
there is no PCIe reading here either, because there does not need to be — the DRM device is a
symlink to its PCI device, so [`address`][AmdDevice.address] feeds the existing
`fabric.pcie` probes and an AMD card's renegotiated link is already visible through those.

Off Linux, in a container without `/sys/class/drm` mounted, and on any NVIDIA-only host, every
entry point returns empty and nothing downstream changes. `readable()` is how a caller asks
whether an empty result means healthy or means blind.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

import functools
import glob
import os
import re
from dataclasses import dataclass, field, replace

__all__ = [
    "AMDGPU_SYSFS_ROOT",
    "AMD_PCI_VENDOR",
    "AmdDevice",
    "RasCounts",
    "amd_devices",
    "amd_power_watts",
    "amd_present",
    "ecc_faulted_amd_devices",
    "readable",
    "reset_amd_probe",
    "throttled_amd_devices",
]

#: Where the kernel publishes DRM devices. A constant so a test can point it at a fake tree.
AMDGPU_SYSFS_ROOT = "/sys/class/drm"

#: AMD's PCI vendor ID. `/sys/class/drm` also carries Intel and NVIDIA cards, and on an MI300A
#: it carries the integrated display device too, so the vendor check is what keeps this from
#: reporting a laptop's iGPU as a datacenter accelerator.
AMD_PCI_VENDOR = 0x1002

#: A `ras/<block>_err_count` file reports two labeled counts. The block names differ by part
#: and by kernel version, so the blocks are discovered by glob rather than listed.
_RAS_LINE = re.compile(r"^\s*(ue|ce)\s*:\s*(\d+)\s*$", re.MULTILINE)

#: RAS blocks whose uncorrectable errors mean the *memory* is failing rather than a transient
#: engine fault. `umc` is the unified memory controller — HBM. An uncorrectable count here is
#: the AMD analogue of Xid 48/64/94: the data is gone, and the board needs replacing, not
#: resetting. Other blocks (gfx, sdma, mmhub) can log an uncorrectable error from a single bad
#: command and recover on a reset, so they are reported but do not condemn a board.
MEMORY_RAS_BLOCKS = frozenset({"umc"})


@dataclass(frozen=True)
class RasCounts:
    """Correctable and uncorrectable error counts for one RAS block.

    Attributes:
        block: The block's name as the driver spells it, such as `"umc"` or `"gfx"`.
        correctable: Errors the hardware repaired. A rising count is a wear signal, not a
            fault; a device with thousands is worth watching and is still correct.
        uncorrectable: Errors the hardware could not repair. In a memory block this means
            data was lost.
    """

    block: str
    correctable: int = 0
    uncorrectable: int = 0


@dataclass(frozen=True)
class AmdDevice:
    """One `amdgpu` device as sysfs describes it; every field is `0`/empty when unreadable.

    The zero-means-unknown convention matches `nvml.DeviceTelemetry`, and for the same reason:
    a caller that wants to distinguish an idle device from an unreadable one asks
    [`readable`][readable] rather than inspecting a value.

    Attributes:
        index: Position in card-number order, which is the order ROCm enumerates in.
        card: The DRM node's name, such as `"card0"`.
        address: PCI address in `0000:c1:00.0` form, or `""`. Feeds `fabric.pcie`.
        name: The board's marketing name from `product_name`, or `""` on a kernel that does
            not publish it.
        unique_id: The device's own 64-bit ID, stable across reboots, or `""`.
        serial_number: Board serial, or `""`. The figure an RMA is filed against.
        memory_total_bytes: HBM size.
        memory_used_bytes: HBM currently allocated, across every process.
        busy_percent: Occupancy, 0-100.
        temperature_c: Edge temperature.
        temperature_limit_c: The part's own critical threshold, published by the board rather
            than assumed. Comparing against this instead of a constant is what makes the
            check correct on a part whose limit is not the one a table guessed.
        power_watts: Instantaneous board draw.
        power_cap_watts: The cap the board is running under, which on a rented node is
            whatever the operator set and is frequently below the part's rating.
        ras: Per-block error counts, empty when the driver publishes no RAS tree.
        compute_partition: How the compute is carved up (`"SPX"` whole-board, `"CPX"`
            per-die), or `""` on a part or kernel that does not publish it.
        memory_partition: How the memory is interleaved (`"NPS1"` through `"NPS8"`), or `""`.
    """

    index: int
    card: str
    address: str = ""
    name: str = ""
    unique_id: str = ""
    serial_number: str = ""
    memory_total_bytes: int = 0
    memory_used_bytes: int = 0
    busy_percent: int = 0
    temperature_c: float = 0.0
    temperature_limit_c: float = 0.0
    power_watts: float = 0.0
    power_cap_watts: float = 0.0
    ras: tuple[RasCounts, ...] = field(default_factory=tuple)
    compute_partition: str = ""
    memory_partition: str = ""

    @property
    def partitioned(self) -> bool:
        """Whether this board is presenting slices rather than itself.

        `SPX` is the whole board and is not partitioning. Anything else published here is,
        and a caller reading memory or occupancy off this row is reading a slice.
        """
        return bool(self.compute_partition) and self.compute_partition.upper() != "SPX"

    @property
    def memory_free_bytes(self) -> int:
        """Unallocated HBM, or `0` when the total was unreadable."""
        return max(0, self.memory_total_bytes - self.memory_used_bytes)

    @property
    def uncorrectable_errors(self) -> int:
        """Uncorrectable errors summed over every RAS block."""
        return sum(block.uncorrectable for block in self.ras)

    @property
    def memory_uncorrectable_errors(self) -> int:
        """Uncorrectable errors in a memory block, which is the count that condemns a board."""
        return sum(b.uncorrectable for b in self.ras if b.block in MEMORY_RAS_BLOCKS)

    @property
    def power_headroom(self) -> float:
        """Fraction of the board's cap still unused, or `1.0` when either figure is unknown.

        Near zero means the cap, not the workload, is setting the clock — so a slow job on
        this device will not get faster by giving it more work.
        """
        if self.power_cap_watts <= 0.0 or self.power_watts <= 0.0:
            return 1.0
        return max(0.0, 1.0 - self.power_watts / self.power_cap_watts)

    @property
    def thermal_headroom_c(self) -> float:
        """Degrees below the board's own critical limit, or `0.0` when either is unknown."""
        if self.temperature_limit_c <= 0.0 or self.temperature_c <= 0.0:
            return 0.0
        return self.temperature_limit_c - self.temperature_c


def _read_text(path: str) -> str:
    """One sysfs attribute's contents, stripped, or `""` when it cannot be read.

    Sysfs reads fail in three ordinary ways that are all "unknown" rather than an error: the
    attribute does not exist on this driver version, the container did not mount the tree, and
    the driver returns `EINVAL` for a figure the part does not support.
    """
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _read_int(path: str, scale: float = 1.0) -> float:
    """One numeric sysfs attribute, divided by `scale`, or `0.0`."""
    raw = _read_text(path)
    if not raw:
        return 0.0
    try:
        return int(raw) / scale
    except ValueError:
        return 0.0


def _pci_vendor(device_dir: str) -> int:
    """The PCI vendor ID behind a DRM node, or `0` when unreadable."""
    raw = _read_text(os.path.join(device_dir, "vendor"))
    try:
        return int(raw, 16)
    except ValueError:
        return 0


def _pci_address(device_dir: str) -> str:
    """The PCI address a DRM device symlinks to, or `""`.

    `/sys/class/drm/card0/device` is a symlink into `/sys/bus/pci/devices`, so the address is
    the resolved link's basename. Under a fake tree in a test the link may be a plain
    directory, in which case the basename is still the right answer.
    """
    resolved = os.path.realpath(device_dir)
    base = os.path.basename(resolved)
    return base if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d", base) else ""


def _hwmon_dir(device_dir: str) -> str:
    """The device's hwmon directory, or `""`. There is exactly one on an `amdgpu` device."""
    found = sorted(glob.glob(os.path.join(device_dir, "hwmon", "hwmon*")))
    return found[0] if found else ""


def _ras_counts(device_dir: str) -> tuple[RasCounts, ...]:
    """Every RAS block's counts, in name order, or `()` when the driver publishes no tree.

    Blocks are discovered rather than listed because which ones exist depends on the part and
    the kernel: an MI300X publishes blocks an MI210 does not, and a block that a future
    driver adds should be counted the day it appears rather than the day this list is updated.
    """
    blocks: list[RasCounts] = []
    for path in sorted(glob.glob(os.path.join(device_dir, "ras", "*_err_count"))):
        name = os.path.basename(path)[: -len("_err_count")]
        counts = {kind: int(value) for kind, value in _RAS_LINE.findall(_read_text(path))}
        if counts:
            blocks.append(
                RasCounts(
                    block=name,
                    correctable=counts.get("ce", 0),
                    uncorrectable=counts.get("ue", 0),
                )
            )
    return tuple(blocks)


def _card_number(card: str) -> int:
    """Sort key for a DRM node name, so `card10` follows `card9` rather than `card1`."""
    digits = "".join(ch for ch in card if ch.isdigit())
    return int(digits) if digits else 0


def _probe() -> tuple[AmdDevice, ...]:
    """Walk `/sys/class/drm` once and build a device for every AMD card found."""
    devices: list[AmdDevice] = []
    nodes = glob.glob(os.path.join(AMDGPU_SYSFS_ROOT, "card*"))
    # `card0-DP-1` and friends are connectors, not devices, and they carry no `device/vendor`.
    cards = sorted((c for c in nodes if "-" not in os.path.basename(c)), key=_card_number)
    for index, card_dir in enumerate(cards):
        device_dir = os.path.join(card_dir, "device")
        if _pci_vendor(device_dir) != AMD_PCI_VENDOR:
            continue
        hwmon = _hwmon_dir(device_dir)
        devices.append(
            AmdDevice(
                index=index,
                card=os.path.basename(card_dir),
                address=_pci_address(device_dir),
                name=_read_text(os.path.join(device_dir, "product_name")),
                unique_id=_read_text(os.path.join(device_dir, "unique_id")),
                serial_number=_read_text(os.path.join(device_dir, "serial_number")),
                memory_total_bytes=int(_read_int(os.path.join(device_dir, "mem_info_vram_total"))),
                memory_used_bytes=int(_read_int(os.path.join(device_dir, "mem_info_vram_used"))),
                busy_percent=int(_read_int(os.path.join(device_dir, "gpu_busy_percent"))),
                temperature_c=(
                    _read_int(os.path.join(hwmon, "temp1_input"), 1000.0) if hwmon else 0.0
                ),
                temperature_limit_c=(
                    _read_int(os.path.join(hwmon, "temp1_crit"), 1000.0) if hwmon else 0.0
                ),
                power_watts=(
                    _read_int(os.path.join(hwmon, "power1_average"), 1_000_000.0) if hwmon else 0.0
                ),
                power_cap_watts=(
                    _read_int(os.path.join(hwmon, "power1_cap"), 1_000_000.0) if hwmon else 0.0
                ),
                ras=_ras_counts(device_dir),
                compute_partition=_read_text(
                    os.path.join(device_dir, "current_compute_partition")
                ).upper(),
                memory_partition=_read_text(
                    os.path.join(device_dir, "current_memory_partition")
                ).upper(),
            )
        )
    # Re-index so the numbering is dense over AMD cards, not over every DRM node. A host with
    # a display adapter at card0 and eight Instincts after it must report those as 0-7, which
    # is how ROCm and every operator tool number them.
    return tuple(replace(d, index=i) for i, d in enumerate(devices))


@functools.lru_cache(maxsize=1)
def _cached_identity() -> tuple[str, ...]:
    """The card names present at first probe.

    Identity is fixed for the life of a process — a device does not appear or vanish — so it
    is memoized. The *readings* are not, because temperature and occupancy are the whole point
    of asking.
    """
    return tuple(device.card for device in _probe())


def reset_amd_probe() -> None:
    """Forget the memoized device set so the next call re-walks sysfs.

    For tests that point [`AMDGPU_SYSFS_ROOT`][AMDGPU_SYSFS_ROOT] at a fake tree, and for a
    process that has just changed its device visibility.
    """
    _cached_identity.cache_clear()


def amd_present() -> bool:
    """Whether any AMD accelerator is visible through sysfs on this host.

    Returns:
        True when at least one `/sys/class/drm/card*` device reports AMD's PCI vendor.
    """
    return bool(_cached_identity())


def readable() -> bool:
    """Whether AMD device state can be read here at all.

    The question to ask before treating an empty fault list as good news. False on an
    NVIDIA-only host, off Linux, and in a container without `/sys/class/drm` — three cases
    where "no faults" means nobody looked.

    Returns:
        True when the DRM tree exists and at least one AMD device is in it.
    """
    return os.path.isdir(AMDGPU_SYSFS_ROOT) and amd_present()


def amd_devices() -> tuple[AmdDevice, ...]:
    """Live readings for every AMD accelerator on this host, in ROCm's enumeration order.

    Returns:
        One entry per device, empty when there are none or sysfs is unreadable.
    """
    return _probe() if amd_present() else ()


def amd_power_watts(devices: tuple[AmdDevice, ...] | None = None) -> float:
    """Instantaneous draw across this host's AMD accelerators, in watts.

    The counterpart of `nvml.total_power_watts`. A power envelope that summed only the NVIDIA
    devices reported zero on an Instinct node, and an admission check reading zero draw against
    a real breaker is the one error here with a physical consequence.

    Args:
        devices: Readings to sum. Probed when omitted.

    Returns:
        Summed board power, `0.0` when no board publishes one.
    """
    probed = amd_devices() if devices is None else devices
    return sum(device.power_watts for device in probed)


def ecc_faulted_amd_devices(
    devices: tuple[AmdDevice, ...] | None = None,
) -> tuple[AmdDevice, ...]:
    """Devices that have lost data to an unrepairable memory error.

    This is the AMD counterpart of a fatal Xid, and it carries the same consequence: the
    board's HBM has failed in a way a reset does not clear, so the device should stop taking
    work and the node should be reported. Errors in non-memory blocks are deliberately not
    included, because those recover.

    Args:
        devices: Readings to judge. Probed when omitted.

    Returns:
        The subset with a non-zero uncorrectable count in a memory block.
    """
    probed = amd_devices() if devices is None else devices
    return tuple(device for device in probed if device.memory_uncorrectable_errors > 0)


def throttled_amd_devices(
    devices: tuple[AmdDevice, ...] | None = None,
    *,
    power_headroom: float = 0.02,
    thermal_headroom_c: float = 3.0,
) -> tuple[AmdDevice, ...]:
    """Devices whose clock is being held down by their power cap or their temperature.

    A throttled device is not broken and must not be quarantined for it. It is the answer to
    "why is this node slower than the identical one next to it", which on rented hardware is
    usually a cap the operator set rather than anything about the job.

    Both thresholds are compared against figures the *board* publishes rather than a table's
    idea of the part, so this stays correct on a device whose limits are not what a datasheet
    says. A device that publishes neither figure is never reported, since unknown is not
    throttled.

    Args:
        devices: Readings to judge. Probed when omitted.
        power_headroom: Fraction of the cap still unused, below which the board counts as
            power-limited.
        thermal_headroom_c: Degrees below the critical limit, under which the board counts as
            thermally limited.

    Returns:
        The subset currently clock-limited, in device order.
    """
    probed = amd_devices() if devices is None else devices
    limited = []
    for device in probed:
        by_power = device.power_cap_watts > 0.0 and device.power_headroom <= power_headroom
        by_heat = 0.0 < device.thermal_headroom_c <= thermal_headroom_c
        if by_power or by_heat:
            limited.append(device)
    return tuple(limited)
