"""Device memory as a managed pool — the VRAM counterpart of the host buffer pool.

Carbonite already governs host memory: reservations are made against a pool, admission is
refused before an allocation would exceed it, and the operator spills instead of dying. Device
memory had none of that, and it is the scarcer resource by an order of magnitude — 80 GiB
against a terabyte of host RAM, with no swap behind it and a failure mode (a CUDA OOM inside a
worker) that kills the process rather than degrading it.

This is the same discipline for VRAM. Nothing here allocates device memory or touches a tensor;
it is bookkeeping that decides *whether* an allocation may be attempted, which is exactly the
control-plane half of the problem. The framework doing the allocating (torch, cuDF, an
inference engine) is what actually reserves the bytes.

Three things it does that a bare `torch.cuda.mem_get_info` check cannot:

* **It accounts across stages.** A pipeline that loads a model, builds a KV cache, and decodes
  images onto the same device has three claimants; each one checking free memory in isolation
  sees the same free bytes and all three proceed.
* **It reserves headroom.** A CUDA context, allocator fragmentation, and transient activation
  peaks are real and are not in any declared footprint. The pool holds a fraction back so
  "fits" means fits in practice.
* **It knows what is external.** On a shared device another process's resident memory is the
  binding constraint and is invisible to a per-process allocator; the pool takes a measured
  external figure and reserves against the remainder.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from batcher._internal.errors import ResourceError

__all__ = ["SPILL_TIERS", "VramPool", "VramReservation", "spill_tier"]

#: Where a working set that will not fit device memory should go, in order of cost. Device
#: memory is roughly an order of magnitude faster than the host bus and two more than disk, so
#: the tiers are not interchangeable and the choice is worth making deliberately.
SPILL_TIERS = ("device", "host", "disk")

#: Fraction of a device held back from reservation by default: CUDA context (~0.5-1 GiB),
#: allocator fragmentation, and activation peaks a declared model footprint never includes.
#: The same 15% the packing math in `ml.gpu` leaves, kept in agreement deliberately — two
#: different headroom constants would make "fits" mean two different things in one pipeline.
DEFAULT_HEADROOM = 0.15


@dataclass(frozen=True, slots=True)
class VramReservation:
    """A granted claim on device memory, released by `VramPool.release`.

    Attributes:
        device: Device index the claim is against.
        bytes_: Bytes reserved.
        owner: Free-form claimant label (`"Inference#3"`, `"kv-cache"`), used only for
            reporting which stage is holding a device's memory.
    """

    device: int
    bytes_: int
    owner: str = ""


@dataclass
class VramPool:
    """Per-device VRAM accounting with headroom, external usage, and a high-water mark.

    Attributes:
        capacity_bytes: Total device memory of one device.
        device_count: Devices the pool governs; each is accounted separately.
        headroom: Fraction of each device held back from reservation, in [0, 0.9].
        external_bytes: Per-device bytes already resident to processes outside this pool,
            typically measured from `DeviceTelemetry.memory_used_bytes`. Reservations are made
            against capacity minus this.
    """

    capacity_bytes: int
    device_count: int = 1
    headroom: float = DEFAULT_HEADROOM
    external_bytes: dict[int, int] = field(default_factory=dict)
    _held: dict[int, int] = field(default_factory=dict, repr=False)
    _peak: dict[int, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def usable_bytes(self, device: int = 0) -> int:
        """Bytes a caller may reserve on one device, after headroom and external usage.

        Args:
            device: Device index.

        Returns:
            Reservable bytes, `0` when the device is already oversubscribed by other processes.
        """
        budget = int(self.capacity_bytes * (1.0 - min(0.9, max(0.0, self.headroom))))
        return max(0, budget - self.external_bytes.get(device, 0))

    def held_bytes(self, device: int = 0) -> int:
        """Bytes currently reserved on one device by this pool."""
        return self._held.get(device, 0)

    def available_bytes(self, device: int = 0) -> int:
        """Bytes still reservable on one device right now."""
        return max(0, self.usable_bytes(device) - self.held_bytes(device))

    def peak_bytes(self, device: int = 0) -> int:
        """High-water reservation on one device, for reporting how close a run came."""
        return self._peak.get(device, 0)

    def fits(self, nbytes: int, device: int = 0) -> bool:
        """Whether a reservation of `nbytes` would be granted on one device.

        Args:
            nbytes: Bytes the caller intends to allocate.
            device: Device index.

        Returns:
            True when the reservation would succeed.
        """
        return nbytes <= self.available_bytes(device)

    def best_device(self) -> int:
        """The governed device with the most reservable memory right now.

        Ties break on the lowest index so placement is deterministic, which keeps a repeated
        run reproducible rather than drifting with dictionary order.
        """
        return min(range(max(1, self.device_count)), key=lambda d: (-self.available_bytes(d), d))

    def reserve(
        self, nbytes: int, *, device: int | None = None, owner: str = ""
    ) -> VramReservation:
        """Reserve device memory, or raise when it does not fit.

        Args:
            nbytes: Bytes to reserve; must be positive.
            device: Device index, or `None` to pick the emptiest governed device.
            owner: Claimant label carried on the reservation for reporting.

        Returns:
            The granted reservation.

        Raises:
            ResourceError: When the reservation does not fit, naming the device, the request,
                and what was actually available — the three figures needed to act on it.
        """
        if nbytes <= 0:
            raise ResourceError(f"VRAM reservation must be positive, got {nbytes} bytes")
        with self._lock:
            target = self.best_device() if device is None else device
            available = max(0, self.usable_bytes(target) - self._held.get(target, 0))
            if nbytes > available:
                raise ResourceError(
                    f"device {target} cannot hold {nbytes / (1 << 30):.2f} GiB: "
                    f"{available / (1 << 30):.2f} GiB reservable of "
                    f"{self.capacity_bytes / (1 << 30):.2f} GiB "
                    f"({self.headroom:.0%} headroom, "
                    f"{self.external_bytes.get(target, 0) / (1 << 30):.2f} GiB external)"
                )
            held = self._held.get(target, 0) + nbytes
            self._held[target] = held
            self._peak[target] = max(self._peak.get(target, 0), held)
            return VramReservation(device=target, bytes_=nbytes, owner=owner)

    def release(self, reservation: VramReservation) -> None:
        """Return a reservation's bytes to the pool.

        Releasing more than is held is clamped to zero rather than going negative: a
        double-release is a bug worth surviving, and a negative balance would silently grant
        the next claimant memory that does not exist.

        Args:
            reservation: A reservation previously returned by `reserve`.
        """
        with self._lock:
            held = self._held.get(reservation.device, 0)
            self._held[reservation.device] = max(0, held - reservation.bytes_)

    def observe_external(self, device: int, used_bytes: int) -> None:
        """Record memory resident to processes outside this pool, from live telemetry.

        Args:
            device: Device index.
            used_bytes: Total resident bytes on the device as the driver reports them; this
                pool's own holdings are subtracted, so passing a raw NVML figure is correct.
        """
        with self._lock:
            self.external_bytes[device] = max(0, used_bytes - self._held.get(device, 0))

    def pressure(self, device: int = 0) -> float:
        """Fraction of the usable budget currently reserved on one device, in [0, 1].

        The signal an admission policy watches: above roughly 0.9 the next stage should be
        sized down or serialized rather than admitted and left to fail inside CUDA.
        """
        usable = self.usable_bytes(device)
        return min(1.0, self.held_bytes(device) / usable) if usable > 0 else 1.0

    def summary(self) -> dict[str, float]:
        """A flat roll-up across governed devices, for logs and the dashboard.

        Returns:
            Capacity, held, available, and peak bytes summed across devices, plus the
            maximum per-device pressure.
        """
        devices = range(max(1, self.device_count))
        return {
            "capacity_bytes": float(self.capacity_bytes * max(1, self.device_count)),
            "usable_bytes": float(sum(self.usable_bytes(d) for d in devices)),
            "held_bytes": float(sum(self.held_bytes(d) for d in devices)),
            "available_bytes": float(sum(self.available_bytes(d) for d in devices)),
            "peak_bytes": float(sum(self.peak_bytes(d) for d in devices)),
            "max_pressure": max((self.pressure(d) for d in devices), default=0.0),
        }


def spill_tier(nbytes: int, pool: VramPool, *, device: int = 0, host_free_bytes: int = 0) -> str:
    """Where a working set of `nbytes` should live, given what the device has left.

    The device-memory analogue of the host spill decision, and it matters more here because
    the tiers are further apart: device memory is roughly an order of magnitude faster than the
    host bus, and the host bus two more than disk. A KV cache or an embedding table that will
    not fit a device is not automatically a disk problem — pinned host memory is often the
    right tier, and it is the one a bare "does it fit VRAM" check never considers.

    Args:
        nbytes: Bytes the working set needs.
        pool: The pool governing the device.
        device: Device index.
        host_free_bytes: Host memory available to hold the spill; `0` means unknown, which
            routes to disk rather than assuming the host can absorb it.

    Returns:
        One of `SPILL_TIERS`.
    """
    if nbytes <= 0 or pool.fits(nbytes, device):
        return "device"
    return "host" if 0 < nbytes <= host_free_bytes else "disk"
