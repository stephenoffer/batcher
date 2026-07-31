"""Device memory as the driver actually divides it — reserved, resident, and host-mappable.

`nvml.DeviceTelemetry` reports device memory as `used` and `total`, which is the v1 shape of
the query, and the arithmetic every pool does with it is wrong by a fixed amount. The driver
reserves a slice of the framebuffer for itself — page tables, the context, ECC parity storage —
and the v1 query folds that into neither figure consistently across releases. A pool sizing
itself as `total - used` therefore believes it has several hundred megabytes it can never
allocate, and finds out at the moment it tries, which is deep inside a stage.

The v2 query separates the three, and this module reads it: `total`, `reserved`, and `used`,
with free derived from all three rather than from two.

Two further readings that only fail when something is already going wrong:

* **BAR1** — the aperture through which the host addresses device memory. It is small (a few
  hundred megabytes on many parts, resizable to the full framebuffer on others) and it is
  consumed by exactly the things a fast pipeline does: pinned host-mapped buffers, GPUDirect
  RDMA registrations, and peer mappings. Exhausting it does not report as an out-of-memory
  condition on the device — the device has plenty — it reports as a mapping failure several
  layers away from the cause.
* **Memory temperature** — HBM stacks have their own sensor, and they throttle on it
  independently of the core. A device whose core is cool while its memory is at its limit is
  memory-throttled, and every core-temperature-based health check reports it as healthy.

Every field degrades to `0` when the driver is absent, the query is refused, or the part
predates it — v2 memory info and the HBM sensor are both Ampere-and-later.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _device_count, _nvml, _read

__all__ = [
    "DeviceMemory",
    "allocatable_bytes",
    "bar1_pressured_devices",
    "device_memory",
]

#: `nvmlTemperatureSensors` value for the HBM/GDDR stack. Absent on parts without the sensor,
#: where the query is refused rather than returning the core temperature.
_SENSOR_MEMORY = 1

#: Fraction of BAR1 in use above which host mapping is treated as at risk. Well below full,
#: because the failure is not gradual: a registration either fits in the aperture or fails, and
#: the allocations that consume it arrive in large chunks.
_BAR1_PRESSURE = 0.85


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """One device's memory division, as the driver reports it rather than as arithmetic.

    Attributes:
        index: NVML device index on this host.
        total_bytes: Framebuffer size.
        reserved_bytes: Memory the driver holds for itself and will never hand out. `0` on a
            part or binding without the v2 query, where it is folded invisibly into the others.
        used_bytes: Memory resident across every process on the device, excluding the reserve.
        free_bytes: Memory the driver reports as allocatable right now. Taken from the query
            rather than derived, because the three figures do not sum the way a caller expects
            on every release, and the driver's own answer is the one an allocation is checked
            against.
        bar1_total_bytes: Size of the host-mappable aperture.
        bar1_used_bytes: Aperture currently mapped.
        memory_temperature_c: HBM/GDDR stack temperature, `0.0` when the part has no sensor.
        v2: Whether the reserved figure came from the v2 query. False means `reserved_bytes` is
            a default and not a measurement, so a caller must not subtract it twice.
        readable: Whether NVML answered any query.
    """

    index: int
    total_bytes: int = 0
    reserved_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    bar1_total_bytes: int = 0
    bar1_used_bytes: int = 0
    memory_temperature_c: float = 0.0
    v2: bool = False
    readable: bool = False

    @property
    def bar1_utilization(self) -> float:
        """Fraction of the host-mappable aperture in use, in [0, 1], `0.0` when unknown."""
        if self.bar1_total_bytes <= 0:
            return 0.0
        return min(1.0, self.bar1_used_bytes / self.bar1_total_bytes)

    @property
    def bar1_pressured(self) -> bool:
        """Whether host mapping on this device is close to failing.

        The precondition to check before registering a large pinned or peer buffer. A caller
        that ignores it does not get a slow path, it gets a mapping error attributed to
        whichever library happened to ask last.
        """
        return self.bar1_total_bytes > 0 and self.bar1_utilization >= _BAR1_PRESSURE

    @property
    def utilization(self) -> float:
        """Fraction of the usable framebuffer resident, in [0, 1], `0.0` when unknown.

        Measured against total less reserved, because the reserve is not available to anybody
        and counting it as capacity understates how full the device is — by exactly the amount
        that decides whether the next allocation fits.
        """
        usable = self.total_bytes - self.reserved_bytes
        if usable <= 0:
            return 0.0
        return min(1.0, max(0, self.used_bytes) / usable)


def _memory_info(nv, handle) -> tuple[int, int, int, int, bool]:
    """`(total, reserved, used, free, v2)` for one device, preferring the v2 query.

    The v2 query takes a version argument on some bindings and is a separate symbol on others.
    Both spellings are tried before falling back to v1, because the difference between them is
    the reserved figure, and without it a pool over-commits by the size of the driver's reserve
    on every device it manages.
    """
    for getter in ("nvmlDeviceGetMemoryInfo_v2", "nvmlDeviceGetMemoryInfo"):
        fn = getattr(nv, getter, None)
        if fn is None:
            continue
        info = _read(lambda f=fn: f(handle), None)
        if info is None:
            continue
        reserved = getattr(info, "reserved", None)
        return (
            int(getattr(info, "total", 0) or 0),
            int(reserved or 0),
            int(getattr(info, "used", 0) or 0),
            int(getattr(info, "free", 0) or 0),
            reserved is not None,
        )
    return (0, 0, 0, 0, False)


def _bar1(nv, handle) -> tuple[int, int]:
    """`(total, used)` bytes of the host-mappable aperture, `(0, 0)` when unreported."""
    fn = getattr(nv, "nvmlDeviceGetBAR1MemoryInfo", None)
    if fn is None:
        return (0, 0)
    info = _read(lambda: fn(handle), None)
    if info is None:
        return (0, 0)
    return (int(getattr(info, "bar1Total", 0) or 0), int(getattr(info, "bar1Used", 0) or 0))


def device_memory() -> tuple[DeviceMemory, ...]:
    """Memory division for every local device, in NVML index order.

    Not memoized: every field but the totals moves continuously.

    Returns:
        One record per device, empty when NVML is unavailable.
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[DeviceMemory] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        total, reserved, used, free, v2 = _memory_info(nv, handle)
        bar1_total, bar1_used = _bar1(nv, handle)
        out.append(
            DeviceMemory(
                index=index,
                total_bytes=total,
                reserved_bytes=reserved,
                used_bytes=used,
                free_bytes=free,
                bar1_total_bytes=bar1_total,
                bar1_used_bytes=bar1_used,
                memory_temperature_c=float(
                    _read(
                        lambda h=handle: nv.nvmlDeviceGetTemperature(h, _SENSOR_MEMORY),
                        0,
                    )
                    or 0
                ),
                v2=v2,
                readable=total > 0,
            )
        )
    return tuple(out)


def allocatable_bytes(index: int, headroom: float = 0.0) -> int:
    """Bytes one device will actually accept an allocation for, after a safety margin.

    The figure a pool should size itself against, and the one `total - used` gets wrong. It
    takes the driver's own free count rather than deriving it, so the driver's reserve, another
    process's allocations, and the fragmentation the driver already knows about are all
    accounted for by the party that can account for them.

    Args:
        index: NVML device index.
        headroom: Fraction of the *total* framebuffer to hold back, in [0, 1). Expressed
            against total rather than free deliberately: a margin against free shrinks as the
            device fills, which is exactly backwards — the margin exists to absorb the
            allocations of other tenants, and those grow as the device fills.

    Returns:
        Bytes, `0` when the device did not report or when the margin exceeds what is free.
    """
    record = next((m for m in device_memory() if m.index == index), None)
    if record is None or not record.readable:
        return 0
    margin = int(record.total_bytes * max(0.0, min(1.0, headroom)))
    return max(0, record.free_bytes - margin)


def bar1_pressured_devices(
    readings: tuple[DeviceMemory, ...] | None = None,
) -> tuple[DeviceMemory, ...]:
    """Devices whose host-mappable aperture is close to exhausted, in index order.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        The pressured subset. Empty when none are *or* when BAR1 was unreadable, which is
        common inside a container and must not be read as headroom.
    """
    records = device_memory() if readings is None else readings
    return tuple(r for r in records if r.bar1_pressured)
