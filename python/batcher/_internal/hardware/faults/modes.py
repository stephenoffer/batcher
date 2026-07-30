"""Device settings that cost throughput or correctness without ever raising anything.

A GPU arrives configured. Most of the time the configuration is right and nobody looks at it
again; on a rented node it is whatever the last tenant, the provisioning script, or the image
left behind, and four of those settings are expensive in ways that never surface as an error.

* **ECC disabled.** The device runs a few percent faster and gains a little memory, and an
  uncorrectable memory error stops being reported at all. Every guard built on
  `ecc_uncorrected` is then reading a counter that cannot move, so a corrupted tensor becomes
  a wrong answer with no signal anywhere. This is the one setting here that is a correctness
  matter rather than a performance one.
* **Persistence mode off.** The driver unloads its device state whenever no process holds the
  device, so the *next* process pays the initialization again — seconds, on a path a
  short-task fleet crosses thousands of times a day. Invisible in a task's own timing, since
  it is spent before the task's first line runs.
* **Exclusive compute mode.** Only one context may use the device at a time. Correct for a
  dedicated trainer and wrong for everything this engine does with fractional scheduling or
  MPS: a second worker does not share the device, it fails to open it.
* **A power limit below the part's floor.** A datacenter capping power is normal and often
  right. A cap at or under the device's own minimum is a misconfiguration that clamps the
  device permanently, and it reads as "this hardware is slow" rather than as a setting.

Everything degrades to unknown rather than to a default. `readable` says whether the driver
answered, and a caller must not treat an unanswered query as a well-configured device — the
queries are refused inside a MIG instance and in a container without the full driver, which
are common places to be.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _decode, _device_count, _nvml, _read

__all__ = [
    "DeviceModes",
    "device_modes",
    "misconfigured_devices",
]

#: NVML's `nvmlComputeMode_t`: default (any number of contexts), exclusive-thread (deprecated),
#: prohibited, and exclusive-process. Named rather than numbered at the point of use, because
#: "compute mode 3" in a report is a value a reader has to go and look up.
_COMPUTE_MODES = {
    0: "default",
    1: "exclusive_thread",
    2: "prohibited",
    3: "exclusive_process",
}

#: Compute modes under which a second worker cannot open the device at all. `prohibited` is
#: the extreme case and is included for completeness: it is not a mode a working fleet runs
#: in, and finding it means a provisioning step did not finish.
_SINGLE_TENANT_MODES = frozenset({"exclusive_thread", "exclusive_process", "prohibited"})


@dataclass(frozen=True, slots=True)
class DeviceModes:
    """One device's configuration, as opposed to its readings.

    Attributes:
        index: NVML device index.
        uuid: Stable device UUID.
        ecc_enabled: Whether ECC is on, `None` when the driver did not say. `False` is a
            correctness finding, not a preference.
        persistence: Whether persistence mode is on, `None` when unreported.
        compute_mode: One of `default`, `exclusive_thread`, `prohibited`,
            `exclusive_process`, or `""` when unreported.
        power_limit_watts: The enforced limit, `0.0` when unreported.
        power_limit_floor_watts: The lowest limit this part accepts, `0.0` when unreported.
        readable: Whether NVML answered any of it. False means every field above is a default
            rather than a measurement.
    """

    index: int
    uuid: str = ""
    ecc_enabled: bool | None = None
    persistence: bool | None = None
    compute_mode: str = ""
    power_limit_watts: float = 0.0
    power_limit_floor_watts: float = 0.0
    readable: bool = False

    @property
    def single_tenant(self) -> bool:
        """Whether the device's compute mode forbids a second context.

        The fact fractional scheduling and MPS both depend on being false. A caller that
        packs two workers onto such a device does not get contention, it gets a failure to
        initialize on the second one.
        """
        return self.compute_mode in _SINGLE_TENANT_MODES

    @property
    def power_capped_to_floor(self) -> bool:
        """Whether the enforced power limit is at or below what the part will accept.

        A datacenter cap is normal. A cap at the floor is a misconfiguration that clamps the
        device for the life of the node, and it reads as slow hardware rather than a setting.
        """
        if self.power_limit_watts <= 0.0 or self.power_limit_floor_watts <= 0.0:
            return False
        return self.power_limit_watts <= self.power_limit_floor_watts

    @property
    def findings(self) -> tuple[str, ...]:
        """Short reason codes for everything misconfigured here, most serious first.

        Ordered by consequence: a device that cannot report a memory error outranks one that
        cannot be shared, which outranks one paying an avoidable initialization.
        """
        out: list[str] = []
        if self.readable:
            if self.ecc_enabled is False:
                out.append("ecc_disabled")
            if self.single_tenant:
                out.append(f"compute_mode_{self.compute_mode}")
            if self.power_capped_to_floor:
                out.append("power_at_floor")
            if self.persistence is False:
                out.append("persistence_off")
        return tuple(out)


def _tri_state(value) -> bool | None:
    """NVML's enabled/disabled enum as a bool, or `None` when the query was refused."""
    if value is None:
        return None
    return bool(int(value))


def device_modes() -> tuple[DeviceModes, ...]:
    """Configuration for every device on this host, in NVML index order.

    Not memoized. These settings do change under a running process — `nvidia-smi -pm 1` and a
    power-limit change both take effect immediately — and they are read on a health cadence
    rather than a hot path, so a cached "ECC is on" is a worse answer than a fresh one.

    Returns:
        One record per device, empty when NVML is unavailable. A device whose queries are all
        refused still reports a record with `readable=False`, so a caller can tell a
        well-configured device from an unreadable one.
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[DeviceModes] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        sentinel = object()
        # NVML reports ECC as `(current, pending)`; the current setting is what is in force,
        # and a pending change only takes effect on the next reset.
        ecc = _read(lambda h=handle: nv.nvmlDeviceGetEccMode(h), sentinel)
        persistence = _read(lambda h=handle: nv.nvmlDeviceGetPersistenceMode(h), sentinel)
        mode = _read(lambda h=handle: nv.nvmlDeviceGetComputeMode(h), sentinel)
        limit = _read(lambda h=handle: nv.nvmlDeviceGetEnforcedPowerLimit(h), sentinel)
        floor = _read(lambda h=handle: nv.nvmlDeviceGetPowerManagementLimitConstraints(h), sentinel)
        readable = any(v is not sentinel for v in (ecc, persistence, mode, limit))
        out.append(
            DeviceModes(
                index=index,
                uuid=_decode(_read(lambda h=handle: nv.nvmlDeviceGetUUID(h), "")),
                ecc_enabled=_tri_state(_first_of(ecc, sentinel)),
                persistence=_tri_state(None if persistence is sentinel else persistence),
                compute_mode=("" if mode is sentinel else _COMPUTE_MODES.get(int(mode or 0), "")),
                power_limit_watts=0.0 if limit is sentinel else int(limit or 0) / 1000.0,
                power_limit_floor_watts=_floor_watts(floor, sentinel),
                readable=readable,
            )
        )
    return tuple(out)


def _first_of(value, sentinel):
    """The current half of NVML's `(current, pending)` pair, or `None` when refused."""
    if value is sentinel or value is None:
        return None
    return value[0] if isinstance(value, (tuple, list)) and value else value


def _floor_watts(value, sentinel) -> float:
    """The lower bound of NVML's `(min, max)` power constraint pair, in watts."""
    if value is sentinel or not isinstance(value, (tuple, list)) or not value:
        return 0.0
    return int(value[0] or 0) / 1000.0


def misconfigured_devices(modes: tuple[DeviceModes, ...] | None = None) -> tuple[DeviceModes, ...]:
    """Devices carrying a setting that costs throughput or correctness.

    Args:
        modes: Records to inspect, or `None` to read them live.

    Returns:
        The devices with at least one finding, in index order. Empty on a well-configured
        fleet *and* on one whose driver would not answer — `DeviceModes.readable` is what
        distinguishes those, and a fleet where it is False must not be reported as
        misconfigured any more than it should be reported as healthy.
    """
    records = device_modes() if modes is None else modes
    return tuple(m for m in records if m.findings)
