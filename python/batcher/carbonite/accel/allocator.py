"""The device allocator a GPU worker computes on — the pool in front of `cudaMalloc`.

`vram.py` decides *whether* an allocation may be attempted. This decides *how* the ones that
are attempted are served, and on a device the two are not the same problem. Unconfigured,
RAPIDS asks the CUDA driver for every intermediate column a query produces, and a driver
allocation is a synchronizing call: it stops the device, and it is not cheap enough to do
thousands of times. A translated chain of a dozen operators over a hundred shards does exactly
that, and the cost lands as a constant factor on every single query rather than as a visible
failure. Pointing RAPIDS at a suballocated pool is the largest single-line lever this package
has on GPU throughput, and it is off until asked for because it reserves memory a co-tenant on
the same device can then no longer see.

Three concerns, deliberately separated so only the last one needs a GPU:

* `plan_allocator` is arithmetic over a config and a device's reservable bytes. It is pure,
  which is why the sizing is tested rather than assumed.
* `configure_device_memory` applies a plan to this process, once, and reports whether it
  applied. It degrades in one direction only: a build with no RMM, an allocator RMM does not
  offer, or a device that refuses the reservation each leave the worker on the allocator it
  already had, because a GPU worker that computes slowly is worth more than one that will not
  start.
* `device_allocator_state` reads back what is configured and, when statistics were asked for,
  the device high-water mark — the figure `core.energy`'s stage meter reports as *measured*
  rather than declared.

The size the pool is asked for is `VramPool.usable_bytes`, not the device's capacity. That is
the whole point of routing it through Carbonite: the headroom the pool already holds back for
the CUDA context, fragmentation, and a co-tenant is the headroom the allocator must respect
too, and two different answers to "how much of this device is mine" is how a fleet ends up
with an allocator that succeeds and a stage that then cannot fit its model.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from batcher._internal.logging import note_suppressed

__all__ = [
    "AllocatorPlan",
    "configure_device_memory",
    "device_allocator_state",
    "plan_allocator",
    "prepare_device_memory",
    "reset_device_allocator",
]

#: Smallest pool worth reserving. Below this the reservation costs more in lost headroom than
#: it saves in driver calls, so the plan degrades to the driver allocator rather than carving
#: a pool out of a device that has nothing left to give.
MIN_POOL_BYTES = 64 << 20

#: RMM rounds a pool to the device's allocation granularity, so an initial size is aligned
#: down to a multiple of this rather than handed over as an arbitrary byte count.
_ALIGN = 256 << 20

_lock = threading.Lock()
_applied: AllocatorPlan | None = None
_statistics_adaptor = None


@dataclass(frozen=True, slots=True)
class AllocatorPlan:
    """What a GPU worker should point RAPIDS at, sized for one device.

    Attributes:
        allocator: `default`, `pool`, `async`, or `managed`. `default` means make no change.
        initial_bytes: Bytes reserved when the pool is created.
        maximum_bytes: Ceiling the pool may grow to, `0` for unbounded.
        spill_to_host: Let cuDF move columns to host memory instead of failing.
        statistics: Track allocation counts and the device high-water mark.
    """

    allocator: str = "default"
    initial_bytes: int = 0
    maximum_bytes: int = 0
    spill_to_host: bool = False
    statistics: bool = False

    @property
    def is_inert(self) -> bool:
        """Whether applying this plan would change nothing about the worker."""
        return self.allocator == "default" and not self.spill_to_host and not self.statistics


def plan_allocator(cfg, usable_bytes: int) -> AllocatorPlan:
    """Size a device allocator from the accelerator config and a device's reservable bytes.

    `usable_bytes` is what `VramPool.usable_bytes` reports: capacity less the VRAM headroom
    and less whatever another process already holds. Sizing from it rather than from capacity
    is what keeps the allocator and the admission check agreeing about the same device.

    A device whose reservable bytes are unknown (`0`, the answer on a host with no NVML and
    the honest answer on a device already full) plans no pool at all. Reserving a pool from a
    figure nobody measured is how a worker takes memory a co-tenant was using.

    Args:
        cfg: The `DeviceMemoryConfig` section to plan from.
        usable_bytes: Bytes of one device this worker may reserve.

    Returns:
        The plan to apply, inert when the config asks for nothing or the device cannot say
        how much it has.

    Examples:
        .. doctest::

            >>> from batcher.config import DeviceMemoryConfig
            >>> from batcher.carbonite.accel import plan_allocator
            >>> plan_allocator(DeviceMemoryConfig(), 40 << 30).is_inert
            True
            >>> pooled = DeviceMemoryConfig(allocator="pool")
            >>> plan_allocator(pooled, 40 << 30).initial_bytes >> 30
            20
    """
    tail = AllocatorPlan(spill_to_host=bool(cfg.spill_to_host), statistics=bool(cfg.statistics))
    if cfg.allocator == "default":
        return tail
    maximum = _align(int(usable_bytes * min(1.0, max(0.0, cfg.pool_max_fraction))))
    initial = _align(int(usable_bytes * min(1.0, max(0.0, cfg.pool_initial_fraction))))
    if maximum < MIN_POOL_BYTES:
        # Nothing worth pooling: no measurement, or a device a co-tenant has already filled.
        return tail
    return AllocatorPlan(
        allocator=cfg.allocator,
        initial_bytes=max(MIN_POOL_BYTES, min(initial, maximum)),
        maximum_bytes=maximum,
        spill_to_host=tail.spill_to_host,
        statistics=tail.statistics,
    )


def _align(nbytes: int) -> int:
    """`nbytes` rounded down to the allocation granularity RMM reserves in."""
    return max(0, nbytes) // _ALIGN * _ALIGN


def configure_device_memory(plan: AllocatorPlan) -> bool:
    """Point this process's RAPIDS allocator at `plan`, once per process.

    Idempotent by design: every GPU task body calls it, and a Ray worker runs many. Re-applying
    a pool would free the first one out from under the columns living in it, so the second call
    onwards is a no-op that reports the first call's answer.

    Args:
        plan: The allocator plan to apply, from `plan_allocator`.

    Returns:
        True when this process is now running on the planned allocator; False when the plan
        was inert, RMM is absent, or the device refused the reservation — in every one of
        which the worker keeps the allocator it already had and still computes correct results.
    """
    global _applied
    if plan.is_inert:
        return False
    with _lock:
        if _applied is not None:
            return _applied.allocator == plan.allocator
        applied = False
        if plan.allocator != "default" or plan.statistics:
            applied = _apply_resource(plan)
        if _apply_spill(plan):
            applied = True
        if applied:
            _applied = plan
        return applied


def _apply_resource(plan: AllocatorPlan) -> bool:
    """Install the planned RMM device resource, reporting whether it took."""
    global _statistics_adaptor
    try:
        import rmm
    except Exception as exc:  # pragma: no cover - exercised only where RAPIDS is installed
        note_suppressed("carbonite", "install a device allocator", exc)
        return False
    try:
        resource = _device_resource(rmm, plan)
        if plan.statistics:
            adaptor = getattr(rmm.mr, "StatisticsResourceAdaptor", None)
            if adaptor is not None:
                resource = adaptor(resource)
                _statistics_adaptor = resource
        rmm.mr.set_current_device_resource(resource)
    except Exception as exc:  # pragma: no cover - a device-side refusal
        note_suppressed("carbonite", f"install a {plan.allocator} device allocator", exc)
        _statistics_adaptor = None
        return False
    return True


def _device_resource(rmm, plan: AllocatorPlan):
    """The RMM device resource `plan` asks for.

    `default` returns the process's current resource unchanged, so asking for statistics alone
    measures the allocator the worker already had rather than quietly replacing it. An RMM
    with no stream-ordered pool falls back to a suballocated one, which is the same shape of
    win through a different mechanism.
    """
    mr = rmm.mr
    if plan.allocator == "default":
        return mr.get_current_device_resource()
    if plan.allocator == "async":
        async_mr = getattr(mr, "CudaAsyncMemoryResource", None)
        if async_mr is not None:
            return async_mr(initial_pool_size=plan.initial_bytes)
        note_suppressed(
            "carbonite",
            "use a stream-ordered device pool",
            RuntimeError("this RMM has no CudaAsyncMemoryResource"),
        )
    base = mr.ManagedMemoryResource() if plan.allocator == "managed" else mr.CudaMemoryResource()
    return mr.PoolMemoryResource(
        base, initial_pool_size=plan.initial_bytes, maximum_pool_size=plan.maximum_bytes or None
    )


def _apply_spill(plan: AllocatorPlan) -> bool:
    """Turn on cuDF's host spilling when the plan asks for it."""
    if not plan.spill_to_host:
        return False
    try:
        import cudf

        cudf.set_option("spill", True)
    except Exception as exc:  # pragma: no cover - exercised only with cuDF installed
        note_suppressed("carbonite", "enable device-to-host spilling", exc)
        return False
    return True


def device_allocator_state() -> dict:
    """What this process's device allocator is, and what it has peaked at.

    Returns:
        `allocator` (the applied strategy, `default` when none was), `pool_bytes` (the
        ceiling it was sized to), and `peak_bytes` (the device high-water mark, `0` when
        statistics were not requested — never a guess).

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel import device_allocator_state
            >>> device_allocator_state()["allocator"]
            'default'
    """
    applied = _applied
    return {
        "allocator": applied.allocator if applied else "default",
        "pool_bytes": applied.maximum_bytes if applied else 0,
        "peak_bytes": _peak_bytes(),
    }


def _peak_bytes() -> int:
    """Device high-water bytes from the statistics adaptor, or `0` when it is not tracking."""
    adaptor = _statistics_adaptor
    if adaptor is None:
        return 0
    try:
        counts = adaptor.allocation_counts
        peak = getattr(counts, "peak_bytes", None)
        return int(peak if peak is not None else counts["peak_bytes"])
    except Exception as exc:  # pragma: no cover - adaptor shape differs across RMM versions
        note_suppressed("carbonite", "read the device high-water mark", exc)
        return 0


def prepare_device_memory() -> bool:
    """Configure this worker's device allocator from the active config and its own device.

    The one call a GPU task body makes. It measures the device this process can actually see
    rather than taking the driver's word for the fleet, sizes the pool through a `VramPool` so
    the headroom matches what admission already reserved, and applies the plan once.

    Returns:
        True when this process is now running on a configured allocator.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel import prepare_device_memory
            >>> prepare_device_memory()  # no device on this host, so nothing to configure
            False
    """
    from batcher.config import active_config

    cfg = active_config().accelerator
    plan = plan_allocator(cfg.memory, _visible_device_usable_bytes(cfg.vram_headroom))
    return configure_device_memory(plan)


def _visible_device_usable_bytes(headroom: float) -> int:
    """Reservable bytes on the device this process is bound to, or `0` when it cannot tell.

    Reads capacity from the local inventory and subtracts what is already resident, so a
    worker sharing a device with another tenant plans a pool out of the remainder rather than
    out of a device it does not have to itself.
    """
    from batcher._internal.accelerators import gpu_inventory
    from batcher._internal.hardware.nvml import device_telemetry, own_device_memory
    from batcher.carbonite.accel.vram import VramPool

    devices = gpu_inventory()
    if not devices:
        return 0
    # Under MPS this process shares one device with its co-tenants, and they all start at
    # once against an empty device: without a declared share each would plan for the whole of
    # it and they would fail together. `mps_client_share` is `1.0` off MPS and wherever the
    # tenancy is unpublished, which is the sizing this pool has always done.
    from batcher.carbonite.accel.affinity import mps_client_share

    pool = VramPool(
        capacity_bytes=int(devices[0].get("memory_bytes", 0) or 0),
        headroom=headroom,
        share=mps_client_share(),
    )
    if not pool.capacity_bytes:
        return 0
    for telemetry in device_telemetry()[:1]:
        # Measured rather than accounted where the driver will attribute it: what this
        # process holds is not what this pool admitted, and on a shared device the
        # difference is charged to the co-tenant.
        pool.observe_external(
            0,
            int(telemetry.memory_used_bytes),
            own_bytes=own_device_memory(telemetry.index),
        )
    return pool.usable_bytes(0)


def reset_device_allocator() -> None:
    """Forget the applied plan so the next `configure_device_memory` acts again.

    For tests and for a worker deliberately re-pointed between stages. It does not free the
    pool: RMM owns that, and freeing it under live columns is a use-after-free.
    """
    global _applied, _statistics_adaptor
    with _lock:
        _applied = None
        _statistics_adaptor = None
