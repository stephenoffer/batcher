"""The driver's own integrated joule counter, and the power limits a device is held to.

Energy accounting from power samples is an approximation with a known failure mode. Sampling
the board draw at the ends of a stage and multiplying by the duration assumes the draw between
the samples was the mean of them, and a GPU stage is precisely the workload where that is
false: a device idles at 60 W while a batch is staged across PCIe and pulls 700 W while the
kernel runs. Two samples taken during the staging phases of a stage that spent most of its time
computing under-report the energy by an order of magnitude, and nothing about the resulting
figure looks wrong.

NVML publishes the exact answer on Volta and later: a monotonic millijoule counter the driver
integrates continuously, in hardware, since the driver last loaded. Two readings and a
subtraction give the energy a stage actually consumed, including every transient between them.
That turns an estimate worth caveating into a figure worth billing, which is the difference
between energy accounting being a diagnostic and being a chargeback input.

This module also reads the *constraints* around the draw, which the instantaneous power reading
cannot express:

* **The enforced limit against the part's default** — a datacenter routinely caps boards below
  their TDP for rack power reasons. A device capped at 400 W of a 700 W part is not
  underperforming, it is configured, and a scheduler that does not know the difference will
  keep looking for a fault that is not there.
* **The settable range** — the floor and ceiling the driver will accept. A power-aware placement
  policy that wants to raise a limit needs to know whether the headroom it is asking for exists
  before it asks.

**The counter resets when the driver reloads**, which shows up as a negative delta. That is
reported as no measurement rather than as negative energy, so a caller falls back to the power
model instead of recording a nonsense figure.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _device_count, _nvml, _read

__all__ = [
    "DeviceEnergy",
    "capped_below_default",
    "device_energy",
    "energy_counter_available",
    "fleet_energy_joules",
    "interval_energy_joules",
]


@dataclass(frozen=True, slots=True)
class DeviceEnergy:
    """One device's integrated energy counter and the power envelope it runs inside.

    Attributes:
        index: NVML device index on this host.
        uuid: Stable device UUID, the key an energy series should be recorded against — the
            index is reassigned by `CUDA_VISIBLE_DEVICES` and by a driver reload.
        total_energy_joules: Energy the board has consumed since the driver loaded. Monotonic
            within a driver load. `0.0` when the part predates the counter, which is every
            Pascal and earlier device and most consumer parts.
        enforced_limit_watts: The limit actually in force, which is the lower of the management
            limit and any external cap.
        default_limit_watts: The limit the part ships with, before an operator changes it.
        min_limit_watts: Lowest limit the driver will accept for this part.
        max_limit_watts: Highest limit the driver will accept for this part.
        readable: Whether NVML answered any query.
    """

    index: int
    uuid: str = ""
    total_energy_joules: float = 0.0
    enforced_limit_watts: float = 0.0
    default_limit_watts: float = 0.0
    min_limit_watts: float = 0.0
    max_limit_watts: float = 0.0
    readable: bool = False

    @property
    def counted(self) -> bool:
        """Whether this device has a usable hardware energy counter.

        The flag an accounting path branches on: when False it must integrate power samples and
        say so, and when True it must not, because the sampled figure is strictly worse.
        """
        return self.total_energy_joules > 0.0

    @property
    def limit_headroom_watts(self) -> float:
        """Watts between the enforced limit and the highest the driver would accept.

        `0.0` when either figure is unknown or the device is already at its ceiling. Non-zero
        headroom is the precondition for a power-aware policy raising a cap; without it, the
        only lever left is placing work elsewhere.
        """
        if self.enforced_limit_watts <= 0 or self.max_limit_watts <= 0:
            return 0.0
        return max(0.0, self.max_limit_watts - self.enforced_limit_watts)

    @property
    def derated_fraction(self) -> float:
        """How far below its default the enforced limit sits, in [0, 1].

        `0.0` on an undererated device and when either figure is unknown. A device at 0.4 here
        will not reach its datasheet throughput no matter what is scheduled onto it, and every
        performance comparison against the same part elsewhere has to account for it.
        """
        if self.enforced_limit_watts <= 0 or self.default_limit_watts <= 0:
            return 0.0
        return max(0.0, 1.0 - self.enforced_limit_watts / self.default_limit_watts)


def _limit_constraints(nv, handle) -> tuple[float, float]:
    """`(minimum, maximum)` settable power limit in watts, `(0.0, 0.0)` when unreported."""
    fn = getattr(nv, "nvmlDeviceGetPowerManagementLimitConstraints", None)
    if fn is None:
        return (0.0, 0.0)
    value = _read(lambda: fn(handle), None)
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return (0.0, 0.0)
    return (float(value[0] or 0) / 1000.0, float(value[1] or 0) / 1000.0)


def device_energy() -> tuple[DeviceEnergy, ...]:
    """Energy counters and power envelopes for every local device, in NVML index order.

    Not memoized: the counter advancing is the entire point. The power-limit fields do not move
    within a run in practice, but they are read together because the whole record costs one
    handle lookup and five NVML calls, and splitting them would double the handle work to save
    nothing.

    Returns:
        One record per device, empty when NVML is unavailable.
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[DeviceEnergy] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        sentinel = object()
        getter = getattr(nv, "nvmlDeviceGetTotalEnergyConsumption", None)
        millijoules = (
            sentinel if getter is None else _read(lambda f=getter, h=handle: f(h), sentinel)
        )
        enforced = _read(lambda h=handle: nv.nvmlDeviceGetEnforcedPowerLimit(h), 0)
        default = _read(lambda h=handle: nv.nvmlDeviceGetPowerManagementDefaultLimit(h), 0)
        low, high = _limit_constraints(nv, handle)
        uuid = _read(lambda h=handle: nv.nvmlDeviceGetUUID(h), "")
        out.append(
            DeviceEnergy(
                index=index,
                uuid=uuid.decode() if isinstance(uuid, bytes) else str(uuid or ""),
                total_energy_joules=(
                    0.0 if millijoules is sentinel else float(millijoules or 0) / 1000.0
                ),
                enforced_limit_watts=float(enforced or 0) / 1000.0,
                default_limit_watts=float(default or 0) / 1000.0,
                min_limit_watts=low,
                max_limit_watts=high,
                readable=millijoules is not sentinel or bool(enforced),
            )
        )
    return tuple(out)


def energy_counter_available(readings: tuple[DeviceEnergy, ...] | None = None) -> bool:
    """Whether every local device can be metered exactly rather than by sampling.

    Deliberately *every* rather than *any*: a fleet where one device counts and the rest do not
    would produce a total that mixes an exact figure with an estimate and reports it as though
    the whole thing were measured. Mixed fleets fall back to sampling for all of it, and the
    ledger says so.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        True when at least one device is present and all of them expose the counter.
    """
    records = device_energy() if readings is None else readings
    return bool(records) and all(r.counted for r in records)


def fleet_energy_joules(readings: tuple[DeviceEnergy, ...] | None = None) -> float:
    """Total energy every local device has consumed since the driver loaded.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        Joules across the fleet, `0.0` when no device exposes the counter.
    """
    records = device_energy() if readings is None else readings
    return sum(r.total_energy_joules for r in records)


def interval_energy_joules(
    before: tuple[DeviceEnergy, ...],
    after: tuple[DeviceEnergy, ...],
) -> float | None:
    """Exact energy the fleet consumed between two readings, or `None` when unmeasurable.

    The counter is matched by UUID rather than by index, because a reading taken across a
    `CUDA_VISIBLE_DEVICES` change or a driver reload would otherwise subtract one device's
    counter from another's and report a plausible wrong number.

    Args:
        before: Readings taken at the start of the interval.
        after: Readings taken at the end.

    Returns:
        Joules consumed, or `None` when no device appears in both readings with a usable
        counter, or when any matched counter went backwards — the signature of a driver reload,
        after which the interval simply cannot be measured this way.
    """
    baseline = {r.uuid: r.total_energy_joules for r in before if r.uuid and r.counted}
    if not baseline:
        return None
    total = 0.0
    matched = 0
    for reading in after:
        start = baseline.get(reading.uuid)
        if start is None or not reading.counted:
            continue
        delta = reading.total_energy_joules - start
        if delta < 0:
            return None
        total += delta
        matched += 1
    return total if matched else None


def capped_below_default(
    readings: tuple[DeviceEnergy, ...] | None = None,
    tolerance: float = 0.05,
) -> tuple[DeviceEnergy, ...]:
    """Devices an operator has capped meaningfully below their default power limit.

    Not a fault, and the reason it is worth surfacing anyway: it is the most common explanation
    for the same part being measurably slower here than in a published benchmark, and it is
    invisible in every other reading. A run that reports a regression on capped hardware is
    reporting the cap.

    Args:
        readings: Records to inspect, or `None` to read them live.
        tolerance: Fraction below default to ignore. The default absorbs the small differences
            between a part's enforced and default limits that are normal on healthy hardware.

    Returns:
        The capped subset, in index order.
    """
    records = device_energy() if readings is None else readings
    return tuple(r for r in records if r.readable and r.derated_fraction > tolerance)
