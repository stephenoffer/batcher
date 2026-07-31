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
from collections.abc import Sequence
from dataclasses import dataclass, field

from batcher._internal.errors import ResourceError

__all__ = ["VramPool", "VramReservation"]

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
        share: Fraction of each device this pool may plan for, in (0, 1]. `1.0` — the default
            — is a device this process has to itself. A declared share is what covers the case
            `external_bytes` cannot: co-tenants that start *together*, each measuring an empty
            device, each sizing to all of it, and each discovering the conflict as a
            simultaneous out-of-memory error. Measurement can only see a tenant that has
            already allocated.
    """

    capacity_bytes: int
    device_count: int = 1
    headroom: float = DEFAULT_HEADROOM
    share: float = 1.0
    external_bytes: dict[int, int] = field(default_factory=dict)
    capacities: dict[int, int] = field(default_factory=dict)
    _held: dict[int, int] = field(default_factory=dict, repr=False)
    _peak: dict[int, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_devices(
        cls, capacities: dict[int, int], *, headroom: float = DEFAULT_HEADROOM, share: float = 1.0
    ) -> VramPool:
        """Build a pool over devices that are not all the same size.

        The scalar `capacity_bytes` describes a node whose devices are interchangeable, which
        is the node everyone develops on and not the node a fleet actually accumulates: a box
        part-way through an upgrade, an L4 beside an A100, a partitioned device beside a whole
        one. Governing those with one capacity is wrong in both directions at once — it strands
        the large device and over-admits onto the small one, and the over-admission surfaces as
        a job that fails only on certain nodes.

        Args:
            capacities: Total bytes per device index. Devices absent from it are not governed.
            headroom: Fraction of each device held back from reservation.
            share: Fraction of each device this pool may plan for.

        Returns:
            A pool whose per-device budgets follow each device's own capacity.

        Examples:
            .. doctest::

                >>> from batcher.carbonite.accel import VramPool
                >>> pool = VramPool.from_devices({0: 80 << 30, 1: 24 << 30})
                >>> pool.usable_bytes(0) > pool.usable_bytes(1)
                True
        """
        return cls(
            capacity_bytes=min(capacities.values()) if capacities else 0,
            device_count=max(1, len(capacities)),
            headroom=headroom,
            share=share,
            capacities=dict(capacities),
        )

    def capacity_of(self, device: int = 0) -> int:
        """Total memory of one governed device.

        Falls back to the scalar `capacity_bytes` for a device the per-device map does not
        name, so a pool built the original way is unchanged and a partially-populated map
        degrades to the uniform assumption rather than to zero — reporting a real device as
        having no memory would refuse every reservation on it.
        """
        return self.capacities.get(device, self.capacity_bytes)

    def usable_bytes(self, device: int = 0) -> int:
        """Bytes a caller may reserve on one device, after headroom and external usage.

        Args:
            device: Device index.

        Returns:
            Reservable bytes, `0` when the device is already oversubscribed by other processes.
        """
        share = min(1.0, max(0.0, self.share))
        budget = int(self.capacity_of(device) * share * (1.0 - min(0.9, max(0.0, self.headroom))))
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

    def best_device(self, exclude: Sequence[int] | None = None) -> int:
        """The governed device with the most reservable memory right now.

        Ties break on the lowest index so placement is deterministic, which keeps a repeated
        run reproducible rather than drifting with dictionary order.

        Args:
            exclude: Device indices to skip — the ones Carbonite's health verdicts have
                quarantined. Free memory is the wrong sole criterion on a fleet with a sick
                device in it: a board that has fallen off the bus or exhausted its spare
                memory rows reports *all* of its memory free, which makes it the most
                attractive placement on the node and the only one guaranteed to fail.

        Returns:
            The chosen device index. When every device is excluded the exclusion is ignored
            and the emptiest device wins, because refusing to place work at all is a worse
            answer than placing it on the least-bad option and letting the reservation fail
            with a reason.
        """
        devices = range(max(1, self.device_count))
        skip = set(exclude or ())
        eligible = [d for d in devices if d not in skip] or list(devices)
        return min(eligible, key=lambda d: (-self.available_bytes(d), d))

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
                    f"{self.capacity_of(target) / (1 << 30):.2f} GiB "
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

    def observe_external(
        self, device: int, used_bytes: int, *, own_bytes: int | None = None
    ) -> None:
        """Record memory resident to processes outside this pool, from live telemetry.

        Args:
            device: Device index.
            used_bytes: Total resident bytes on the device as the driver reports them; this
                pool's own share is subtracted, so passing a raw NVML figure is correct.
            own_bytes: This process's *measured* share of that total, when the driver could
                attribute it per process. Preferred over the pool's own accounting, which
                tracks what was admitted rather than what was allocated: the framework
                allocates, and the two diverge by the allocator's pool, the CUDA context, and
                every buffer nobody reserved. `None` keeps the accounting-based subtraction.
        """
        with self._lock:
            mine = self._held.get(device, 0) if own_bytes is None else own_bytes
            self.external_bytes[device] = max(0, used_bytes - mine)

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
            # Summed per device rather than `capacity_bytes * device_count`, which silently
            # reports a mixed node as though every board were the smallest one.
            "capacity_bytes": float(sum(self.capacity_of(d) for d in devices)),
            "usable_bytes": float(sum(self.usable_bytes(d) for d in devices)),
            "held_bytes": float(sum(self.held_bytes(d) for d in devices)),
            "available_bytes": float(sum(self.available_bytes(d) for d in devices)),
            "peak_bytes": float(sum(self.peak_bytes(d) for d in devices)),
            "max_pressure": max((self.pressure(d) for d in devices), default=0.0),
        }
