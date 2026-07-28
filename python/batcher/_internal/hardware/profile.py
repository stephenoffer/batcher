"""The machine's identity — one record of what this hardware is, and a key that names it.

Every adaptive mechanism in Batcher learns a number from a measurement: how many nanoseconds a
join costs per probe row, how many bytes an aggregate holds per group, how large a UDF batch
should be, how much VRAM a model needs. Those numbers are properties of *a workload on a
machine*, not of the workload alone. Learned on a 4-core container with 8 GiB and a spinning
disk, none of them is true on a 64-core host with 512 GiB and NVMe — some are wrong by an
order of magnitude.

Every one of those learned values was previously keyed by workload signature alone. In a
single-machine deployment that is harmless, because there is only ever one machine. Point two
different node shapes at one shared metadata store — a heterogeneous Ray cluster, a laptop and
CI sharing a checkout, an autoscaling group that mixes instance generations — and the store
blends them into a model that is wrong for both, with no error and no way to notice from the
outside.

`fingerprint()` is the fix: a short, stable name for "this class of machine", added to the
key of anything learned from a measurement. Machines that are genuinely alike share it, so
learning still transfers across the nodes of a homogeneous fleet and across restarts. Machines
that are genuinely different do not, so their models stay separate.

Choosing what goes into it is a real trade-off. Too coarse and unlike machines are merged
again; too fine and the fingerprint changes on a kernel upgrade or a cgroup tweak, silently
discarding everything the engine had learned. The fields below are the ones that both change
performance materially and stay put across reboots, kernel updates, and instances of the same
shape.
"""

from __future__ import annotations

import hashlib
import math
import platform
import tempfile
from dataclasses import dataclass, field
from typing import Any

from batcher._internal.accelerators import gpu_devices_absent, gpu_inventory
from batcher._internal.hardware.cache import cache_hierarchy
from batcher._internal.hardware.cpu import available_cpu_count
from batcher._internal.hardware.isa import cpu_model_name, cpu_vendor, simd_width_bits
from batcher._internal.hardware.memory import machine_memory_bytes, page_size_bytes
from batcher._internal.hardware.storage import device_class
from batcher._internal.hardware.topology import numa_node_count, physical_core_count

__all__ = ["HardwareProfile", "fingerprint", "hardware_profile"]

_GIB = 1 << 30

# Length of the hex digest kept as the fingerprint. Twelve hex characters is 48 bits: enough
# that a collision between two machine shapes in one deployment is not a practical concern,
# and short enough to read in a log line or a metadata key without wrapping.
_DIGEST_CHARS = 12


def _nearest_power_of_two(value: int) -> int:
    """Round `value` to the nearest power of two in log space (`0` stays `0`).

    Used for memory capacity, which must bucket rather than compare exactly. Two nodes of the
    same instance type rarely report byte-identical memory: the cgroup limit differs by
    whatever the kubelet reserved, the host reserves a different amount for firmware, a
    hypervisor rounds differently. Comparing raw bytes would give every node its own
    fingerprint and destroy the sharing that makes fleet-wide learning work.

    Nearest in log space rather than floor, so a node reporting slightly under a round
    capacity (60 GiB of a 64 GiB box, the usual case) buckets with its peers instead of
    dropping a whole level away from them.
    """
    if value <= 0:
        return 0
    exponent = round(math.log2(value))
    return 1 << max(0, exponent)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """A structured description of the machine this process runs on.

    Assembled once per process from the individual probes. Every field is a *measured* fact
    with a `0`/`""` fallback meaning "this platform could not report it" — never a plausible
    default, because a fabricated figure here would propagate into the fingerprint and split
    or merge machine classes on a value nobody measured.
    """

    #: Logical CPUs this process may use (cgroup- and affinity-aware).
    logical_cpus: int = 0
    #: Physical cores backing those CPUs; equals `logical_cpus` when SMT is off or unreadable.
    physical_cores: int = 0
    #: NUMA nodes holding usable CPUs; `1` on a uniform machine.
    numa_nodes: int = 1
    #: The binding memory ceiling in bytes: `min(host RAM, cgroup limit)`.
    memory_bytes: int = 0
    #: Cache sizes in bytes keyed by level (`l1d`, `l2`, `l3`) plus `line`.
    caches: dict[str, int] = field(default_factory=dict)
    #: Widest SIMD register in bits.
    simd_bits: int = 128
    #: CPU vendor string, or the machine architecture when no vendor is published.
    vendor: str = ""
    #: CPU model string, `""` when undetectable.
    model: str = ""
    #: Virtual-memory page size in bytes.
    page_bytes: int = 4096
    #: Device class backing the default scratch directory (`nvme`, `ssd`, `rotational`, ...).
    storage_class: str = "unknown"
    #: Accelerator model names in sorted order, one entry per device.
    accelerators: tuple[str, ...] = ()
    #: Operating system name, since the same silicon behaves differently across kernels.
    platform_system: str = ""

    @property
    def total_cache_bytes(self) -> int:
        """Last-level cache size in bytes, or `0` when no level was detected."""
        return self.caches.get("l3") or self.caches.get("l2") or 0

    @property
    def memory_per_core_bytes(self) -> int:
        """Memory ceiling divided by the usable core count — the per-worker envelope shape.

        The ratio that decides whether a machine is memory-rich or core-rich, which is what
        actually determines whether a plan should trade memory for parallelism or the
        reverse. Two 64-core machines with 64 GiB and 1 TiB want different plans for the same
        query, and the core count alone does not say so.
        """
        if self.logical_cpus <= 0:
            return 0
        return self.memory_bytes // self.logical_cpus

    @property
    def has_accelerator(self) -> bool:
        """Whether any accelerator device is attached."""
        return bool(self.accelerators)

    def label(self) -> str:
        """A short human-readable name for this machine shape, for logs and `EXPLAIN`.

        Deliberately lossy and *not* the fingerprint: it is meant to be recognized at a glance
        in a log line ("oh, that ran on the small nodes"), which a hex digest never is. Two
        machines can share a label and differ in fingerprint.

        Returns:
            A compact machine-shape description such as ``x86_64/16c/64GiB/l3=32MiB/nvme``.
        """
        parts = [self.vendor or platform.machine() or "cpu", f"{self.logical_cpus}c"]
        if self.numa_nodes > 1:
            parts.append(f"{self.numa_nodes}numa")
        if self.memory_bytes:
            parts.append(f"{self.memory_bytes // _GIB}GiB")
        if self.total_cache_bytes:
            parts.append(f"l3={self.total_cache_bytes // (1 << 20)}MiB")
        if self.storage_class != "unknown":
            parts.append(self.storage_class)
        if self.accelerators:
            parts.append(f"{len(self.accelerators)}x{self.accelerators[0]}")
        return "/".join(parts)

    def fingerprint(self) -> str:
        """A stable short key naming this class of machine.

        The scoping key for every learned parameter. Built from the fields that change
        performance materially and stay put across reboots and across instances of the same
        shape, with capacity bucketed so near-identical nodes still share a key.

        Deliberately excluded, and why:

        * **The full CPU flag list.** A microcode update changes it without changing anything
          the engine can exploit, which would discard every coefficient learned on the host.
          The vector width it implies is included instead.
        * **Exact memory bytes.** Two nodes of one instance type differ by whatever the
          kubelet and firmware reserved, so raw bytes would give every node its own key.
        * **Load, temperature, clock speed.** Real and important, but they vary minute to
          minute; a fingerprint that changes under load would re-learn from scratch every
          time the box got busy, which is exactly when the learned values matter most.

        Returns:
            A 12-character hex digest identifying this machine class.
        """
        material = "|".join(
            (
                self.vendor,
                self.model,
                f"cpus={self.logical_cpus}",
                f"cores={self.physical_cores}",
                f"numa={self.numa_nodes}",
                f"simd={self.simd_bits}",
                f"page={self.page_bytes}",
                f"mem={_nearest_power_of_two(self.memory_bytes)}",
                f"l2={self.caches.get('l2', 0)}",
                f"l3={self.caches.get('l3', 0)}",
                f"disk={self.storage_class}",
                f"gpu={','.join(self.accelerators)}",
                f"os={self.platform_system}",
            )
        )
        return hashlib.sha256(material.encode()).hexdigest()[:_DIGEST_CHARS]

    def to_dict(self) -> dict[str, Any]:
        """The profile as a JSON-safe document, for the event log and diagnostics.

        Returns:
            Every measured field plus the derived `fingerprint` and `label`.
        """
        return {
            "fingerprint": self.fingerprint(),
            "label": self.label(),
            "logical_cpus": self.logical_cpus,
            "physical_cores": self.physical_cores,
            "numa_nodes": self.numa_nodes,
            "memory_bytes": self.memory_bytes,
            "memory_per_core_bytes": self.memory_per_core_bytes,
            "caches": dict(self.caches),
            "simd_bits": self.simd_bits,
            "vendor": self.vendor,
            "model": self.model,
            "page_bytes": self.page_bytes,
            "storage_class": self.storage_class,
            "accelerators": list(self.accelerators),
            "platform_system": self.platform_system,
        }


def _accelerator_names() -> tuple[str, ...]:
    """Attached accelerator model names, without paying for a framework import to find none.

    `gpu_inventory` falls back to `torch.cuda` when NVML is unavailable, and importing torch
    costs ~1.6 s. That is affordable on a machine that has a GPU and absurd on one that does
    not — and this runs on the first `fingerprint()` call in every process, which on a Ray
    cluster means every worker.

    `gpu_devices_absent` is the cheap device-node check that exists for exactly this: it
    returns `True` only when it can *prove* there is no accelerator, so a machine that has one
    (or a platform that cannot tell, such as macOS Metal) still gets the real inventory.

    Returns:
        Sorted accelerator model names, empty when the machine provably has none.
    """
    if gpu_devices_absent():
        return ()
    return tuple(sorted(str(d.get("name", "")) for d in gpu_inventory()))


# Assembled once per process. Not `functools.lru_cache`d because `HardwareProfile` holds a
# mutable `caches` dict, and a memo would hand the same dict to every caller; the module-level
# binding is cleared by `reset_hardware_probes` alongside the probes it is built from.
_PROFILE: HardwareProfile | None = None


def hardware_profile() -> HardwareProfile:
    """The assembled description of this machine, probed once per process.

    Every underlying probe is itself memoized, so the first call costs a handful of `/sys` and
    `/proc` reads and later calls cost a dictionary lookup. Safe to call on a query path.

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware import hardware_profile
            >>> profile = hardware_profile()
            >>> profile.logical_cpus >= 1 and len(profile.fingerprint()) == 12
            True

    Returns:
        The machine's `HardwareProfile`.
    """
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = HardwareProfile(
            logical_cpus=available_cpu_count(),
            physical_cores=physical_core_count(),
            numa_nodes=numa_node_count(),
            memory_bytes=machine_memory_bytes(),
            caches=dict(cache_hierarchy()),
            simd_bits=simd_width_bits(),
            vendor=cpu_vendor(),
            model=cpu_model_name(),
            page_bytes=page_size_bytes(),
            storage_class=device_class(tempfile.gettempdir()),
            accelerators=_accelerator_names(),
            platform_system=platform.system(),
        )
    return _PROFILE


def fingerprint() -> str:
    """The stable short key naming this machine class — the scoping key for learned state.

    Shorthand for ``hardware_profile().fingerprint()``. Prefer this at the call sites that
    only need the key, so the intent (scope this learned value to the hardware) reads
    directly.

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware import fingerprint
            >>> len(fingerprint())
            12

    Returns:
        A 12-character hex digest identifying this machine class.
    """
    return hardware_profile().fingerprint()


def _reset_profile() -> None:
    """Forget the assembled profile so the next call re-probes (test hook)."""
    global _PROFILE
    _PROFILE = None
