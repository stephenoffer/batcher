"""Live device telemetry through NVML — what a GPU is *doing*, not what it is.

`device_specs` says what a device model can do; this says what one particular device is doing
right now: how much power it is drawing, how hot it is, how busy its SMs are, how much of its
memory is resident, whether the driver is clamping its clocks, and whether its memory has
reported an uncorrectable error. That is the same source `nvidia-smi` and DCGM read, and it is
the only way a control plane learns any of it — Ray reports a device *count* and nothing else.

Four things this feeds, none of which can be answered without it:

* **Energy accounting.** `plan.energy` models power from a datasheet; a real board runs below
  its limit most of the time. A measured draw turns an estimate into a figure worth billing.
* **Utilization-driven sizing.** A device at 20% SM utilization is starved by the pipeline
  feeding it, and the fix (deeper prefetch, bigger batches) is not the one a device at 95%
  needs. Autobatching that cannot see utilization is tuning blind.
* **Health.** Uncorrectable ECC errors and thermal clamping are how a device fails in a
  datacenter: not by disappearing, but by silently running at a third of its rate or by
  corrupting a tensor. Both are readable here, and neither is visible from a task's own timings.
* **Real free memory.** The memory another process on the same device already holds is invisible
  to a CUDA allocator's own accounting, and it is the difference between a model that loads and
  one that OOMs on a shared device.

**Four private helpers here are shared, not local.** `hardware/fabric/{nvlink,pcie,rdma}`
import `_nvml`, `_read`, `_decode`, and `_device_count` from this module rather than opening
their own NVML handle — one handshake per process, one failure policy, and one place that
recovers from the library being torn down underneath us. They are underscore-prefixed because
they are private to `_internal`, not because they have a single caller: renaming or narrowing
one silently breaks the fabric probes, which fail by reporting an unlinked fleet rather than
by raising.

**Every entry point degrades to empty rather than raising.** NVML is absent on a CPU-only node,
absent inside a container that did not mount the driver, and present-but-refusing for some
queries on consumer parts and inside MIG instances. A telemetry source that can fail a query is
worse than no telemetry, so unavailability, permission errors, and per-field failures all read
as "not reported" and callers keep whatever default they had.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

__all__ = [
    "DeviceTelemetry",
    "device_processes",
    "device_telemetry",
    "host_pid",
    "nvml_available",
    "own_device_memory",
    "own_process_ids",
    "reset_nvml_probe",
    "throttled_devices",
    "total_power_watts",
]


@dataclass(frozen=True, slots=True)
class DeviceTelemetry:
    """One device's live readings; every field is `0`/empty when NVML did not report it.

    Attributes:
        index: NVML device index on this host.
        uuid: Stable device UUID, the only identifier that survives a reindex and the key
            health history should be recorded against.
        name: Device name as the driver reports it (`"NVIDIA H100 80GB HBM3"`).
        power_watts: Instantaneous board draw.
        power_limit_watts: Enforced power limit, which a datacenter may set well below TDP.
        temperature_c: GPU core temperature.
        sm_utilization: Fraction of the sample period the SMs were busy, in [0, 1]. This is a
            *duty cycle*, not occupancy — a kernel using one SM reads as fully busy — so it is
            a starvation signal, not an efficiency one.
        memory_utilization: Fraction of the sample period memory was being read or written.
        memory_used_bytes: Device memory resident across every process on the device.
        memory_total_bytes: Total device memory.
        ecc_uncorrected: Uncorrectable ECC errors since the driver last loaded — NVML's
            *volatile* counter, not its aggregate lifetime one. Volatile is the right signal
            for scheduling: it clears when a device is reset or replaced, so a repaired device
            returns to service instead of being quarantined by its own history. Anything above
            zero means data has already been read back wrong.
        throttle_reasons: Active clock-clamping reasons (`"thermal"`, `"power"`, `"hw_slowdown"`,
            `"sw_thermal"`, `"sync_boost"`), empty when the device is running unclamped.
        graphics_clock_mhz: Current graphics clock.
        slowdown_temperature_c: The temperature at which *this part* starts clamping itself,
            as the driver reports it, `0.0` when unreported. A constant cannot stand in for
            it: the threshold differs by tens of degrees across the parts one fleet runs, so
            a fixed figure is simultaneously too strict on one and too lax on another — and
            "too lax" means the warning arrives after the clamp it was supposed to precede.
    """

    index: int
    uuid: str = ""
    name: str = ""
    power_watts: float = 0.0
    power_limit_watts: float = 0.0
    temperature_c: float = 0.0
    sm_utilization: float = 0.0
    memory_utilization: float = 0.0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    ecc_uncorrected: int = 0
    throttle_reasons: tuple[str, ...] = ()
    graphics_clock_mhz: int = 0
    slowdown_temperature_c: float = 0.0

    @property
    def memory_free_bytes(self) -> int:
        """Device memory not resident to any process, `0` when memory was not reported."""
        return max(0, self.memory_total_bytes - self.memory_used_bytes)

    @property
    def power_headroom_watts(self) -> float:
        """Watts between the current draw and the enforced limit; `0.0` when either is unknown.

        A device consistently at zero headroom is power-limited, which caps its clocks: adding
        work to it buys nothing, and the useful move is another device or a higher limit.
        """
        if self.power_limit_watts <= 0 or self.power_watts <= 0:
            return 0.0
        return max(0.0, self.power_limit_watts - self.power_watts)

    @property
    def throttled(self) -> bool:
        """Whether the driver is currently clamping this device's clocks for any reason."""
        return bool(self.throttle_reasons)


#: NVML throttle-reason bits, as `(suffix of the pynvml constant, reported label)`. Read by
#: name rather than by literal value so a release that renumbers them cannot silently mislabel
#: a reason, and an unknown name is skipped instead of raising. The prefix varies by release,
#: which is why only the suffix is written here.
_THROTTLE_BITS = (
    ("SwPowerCap", "power"),
    ("HwPowerBrakeSlowdown", "power"),
    ("SwThermalSlowdown", "sw_thermal"),
    ("HwThermalSlowdown", "thermal"),
    ("HwSlowdown", "hw_slowdown"),
    ("SyncBoost", "sync_boost"),
)


@functools.lru_cache(maxsize=1)
def _nvml():
    """The initialized `pynvml` module, or `None` when it is unusable on this host.

    Memoized because `nvmlInit` is a driver handshake and the answer cannot change within a
    process: a driver that was absent at first call is absent for the run.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None  # driver absent, unmounted in the container, or refusing to initialize
    return pynvml


def nvml_available() -> bool:
    """Whether live device telemetry can be read on this host.

    Returns:
        True when NVML initialized; False on a CPU-only host, without the driver mounted, or
        without `pynvml` installed.
    """
    return _nvml() is not None


def reset_nvml_probe() -> None:
    """Forget the memoized NVML handshake so the next call re-initializes.

    The hook a test faking `pynvml` needs; there is nothing else in a running process that can
    change the answer. `host_pid` is cleared alongside it because a test that fakes NVML's
    process list has to be able to fake which PID counts as this one.
    """
    _nvml.cache_clear()
    host_pid.cache_clear()


def _read(fn, default):
    """Call one NVML getter, mapping any failure to `default`.

    NVML refuses individual queries per device and per driver version — power draw on some
    consumer parts, ECC counts with ECC disabled, utilization inside a MIG instance — so a
    per-field guard is what keeps one unsupported reading from erasing the whole record.
    """
    try:
        return fn()
    except Exception:
        return default


#: The throttle-reason getter, under both names it has had. NVML renamed it to
#: `...ClocksEventReasons` in the 12.x line, and a build carrying only the new name would
#: otherwise report a permanently unthrottled fleet — a silent loss of the one signal that
#: explains a device running at a third of its rate.
_THROTTLE_GETTERS = (
    "nvmlDeviceGetCurrentClocksEventReasons",
    "nvmlDeviceGetCurrentClocksThrottleReasons",
)

#: Same story for the reason *bits*: `nvmlClocksThrottleReason*` became `nvmlClocksEventReason*`.
_THROTTLE_PREFIXES = ("nvmlClocksEventReason", "nvmlClocksThrottleReason")


def _throttle_reasons(nv, handle) -> tuple[str, ...]:
    """Active clock-clamping reasons for one device, deduplicated and ordered stably."""
    bits = 0
    for getter in _THROTTLE_GETTERS:
        fn = getattr(nv, getter, None)
        if fn is not None:
            bits = _read(lambda f=fn: f(handle), 0)
            break
    if not bits:
        return ()
    out: list[str] = []
    for attr, label in _THROTTLE_BITS:
        mask = next(
            (m for m in (getattr(nv, p + attr, None) for p in _THROTTLE_PREFIXES) if m is not None),
            None,
        )
        if mask is not None and bits & mask and label not in out:
            out.append(label)
    return tuple(out)


def _decode(value) -> str:
    """NVML returns `bytes` on some versions and `str` on others; normalize to `str`."""
    return value.decode() if isinstance(value, bytes) else str(value or "")


#: The enum values `device_telemetry` passes, resolved by name with the documented value as the
#: fallback. `_THROTTLE_BITS` already argues the case — read by name so a release that renumbers
#: cannot silently mislabel — and then four selectors were passed as bare integers anyway. A
#: renumbering there is worse than a mislabel: the calls still succeed and return a *different
#: device property*, so the fleet would report, say, a memory clock as its graphics clock, or a
#: correctable ECC count as the uncorrectable one that quarantines a board.
#:
#: The fallback is what the value has been across every NVML release, so a binding too old to
#: publish the constant behaves exactly as it did before.
_ENUMS: dict[str, int] = {
    "NVML_TEMPERATURE_GPU": 0,
    "NVML_CLOCK_GRAPHICS": 0,
    "NVML_MEMORY_ERROR_TYPE_UNCORRECTED": 1,
    "NVML_VOLATILE_ECC": 0,
    "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN": 1,
}


def _enum(nv, name: str) -> int:
    """One NVML enum value, by name, falling back to its long-standing documented value."""
    value = getattr(nv, name, None)
    return int(value) if isinstance(value, int) else _ENUMS[name]


def device_telemetry() -> tuple[DeviceTelemetry, ...]:
    """Live readings for every device on this host, in NVML index order.

    Not memoized — every field is a live reading, and a cached utilization figure is worse than
    none. Costs one NVML call per field per device (tens of microseconds each), so it is fine
    on a per-stage or per-second cadence and wrong on a per-batch one.

    Returns:
        One record per device, or an empty tuple when telemetry is unavailable.
    """
    nv = _nvml()
    if nv is None:
        return ()
    count = _device_count(nv)
    temp_gpu = _enum(nv, "NVML_TEMPERATURE_GPU")
    clock_graphics = _enum(nv, "NVML_CLOCK_GRAPHICS")
    ecc_uncorrected = _enum(nv, "NVML_MEMORY_ERROR_TYPE_UNCORRECTED")
    ecc_volatile = _enum(nv, "NVML_VOLATILE_ECC")
    threshold_slowdown = _enum(nv, "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN")
    out: list[DeviceTelemetry] = []
    for index in range(count):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        util = _read(lambda h=handle: nv.nvmlDeviceGetUtilizationRates(h), None)
        mem = _read(lambda h=handle: nv.nvmlDeviceGetMemoryInfo(h), None)
        out.append(
            DeviceTelemetry(
                index=index,
                uuid=_decode(_read(lambda h=handle: nv.nvmlDeviceGetUUID(h), "")),
                name=_decode(_read(lambda h=handle: nv.nvmlDeviceGetName(h), "")),
                power_watts=_read(lambda h=handle: nv.nvmlDeviceGetPowerUsage(h), 0) / 1000.0,
                power_limit_watts=(
                    _read(lambda h=handle: nv.nvmlDeviceGetEnforcedPowerLimit(h), 0) / 1000.0
                ),
                temperature_c=float(
                    _read(lambda h=handle: nv.nvmlDeviceGetTemperature(h, temp_gpu), 0)
                ),
                sm_utilization=(getattr(util, "gpu", 0) or 0) / 100.0,
                memory_utilization=(getattr(util, "memory", 0) or 0) / 100.0,
                memory_used_bytes=int(getattr(mem, "used", 0) or 0),
                memory_total_bytes=int(getattr(mem, "total", 0) or 0),
                ecc_uncorrected=int(
                    _read(
                        lambda h=handle: nv.nvmlDeviceGetTotalEccErrors(
                            h, ecc_uncorrected, ecc_volatile
                        ),
                        0,
                    )
                    or 0
                ),
                throttle_reasons=_throttle_reasons(nv, handle),
                graphics_clock_mhz=int(
                    _read(lambda h=handle: nv.nvmlDeviceGetClockInfo(h, clock_graphics), 0)
                ),
                # `NVML_TEMPERATURE_THRESHOLD_SLOWDOWN`: the point the driver itself starts
                # clamping at, which is the only threshold that means the same thing on every
                # part in a mixed fleet.
                slowdown_temperature_c=float(
                    _read(
                        lambda h=handle: nv.nvmlDeviceGetTemperatureThreshold(
                            h, threshold_slowdown
                        ),
                        0,
                    )
                    or 0
                ),
            )
        )
    return tuple(out)


def _device_count(nv) -> int:
    """Device count, re-initializing NVML once if the library has been torn down under us.

    This process has a second NVML user: `accelerators.gpu_inventory` initializes the library,
    reads the device list, and calls `nvmlShutdown` in a `finally`. NVML reference-counts
    init/shutdown, so that is normally harmless — but any ordering that drops the count to zero
    leaves this module holding a memoized handle to a torn-down library, and every subsequent
    reading would degrade to "not reported" *silently and permanently*. One re-init recovers
    it; a second failure is a genuine absence and reads as zero devices.
    """
    try:
        return int(nv.nvmlDeviceGetCount())
    except Exception:
        try:
            nv.nvmlInit()
            return int(nv.nvmlDeviceGetCount())
        except Exception:
            return 0


def total_power_watts(sample: tuple[DeviceTelemetry, ...] | None = None) -> float:
    """Sum of every local device's instantaneous draw, or `0.0` when unreadable.

    The measured counterpart of `plan.energy.fleet_power_watts`: what a node is actually
    pulling, against what its hardware was budgeted to pull.

    Args:
        sample: A reading already taken, to derive this view from instead of probing again.
            `None` — the default, and what every caller had — takes its own.

    Returns:
        Watts across all local devices.
    """
    return sum(d.power_watts for d in (device_telemetry() if sample is None else sample))


def throttled_devices(
    sample: tuple[DeviceTelemetry, ...] | None = None,
) -> tuple[DeviceTelemetry, ...]:
    """Devices whose clocks the driver is currently clamping.

    A throttled device is the failure mode that looks like a performance regression: the job
    still completes and still returns the right answer, at a fraction of the rate, and nothing
    in the job's own timings says why.

    Args:
        sample: A reading already taken, to derive this view from instead of probing again.
            A report that wants the draw, the throttled set, and the raw readings otherwise
            sweeps every device three times — thirteen NVML calls per device per sweep — and
            gets three readings taken at three different instants, so its own numbers need not
            agree with each other. `None` keeps the pre-existing behavior of probing.

    Returns:
        The throttled subset of `device_telemetry`, empty when none or when unreadable.
    """
    return tuple(d for d in (device_telemetry() if sample is None else sample) if d.throttled)


def device_processes(index: int) -> tuple[tuple[int, int], ...]:
    """`(pid, bytes)` for every process holding memory on one device.

    The question `memory_used_bytes` cannot answer: that figure is the device's total, so a
    process reading it to decide what it may allocate is counting its own allocations against
    itself and everyone else's as though they were interchangeable. On a shared device — the
    normal case on a rented GPU, and the whole point of MPS and fractional scheduling — those
    are different numbers with different meanings.

    Args:
        index: NVML device index.

    Returns:
        One pair per compute process, in the order NVML lists them. Empty when NVML is
        unavailable, when the query is refused (common inside a container, which often cannot
        see processes in other PID namespaces), and genuinely when the device is idle. A caller
        that needs to tell those apart should compare against `DeviceTelemetry.memory_used_bytes`
        — memory resident with no process visible is exactly the containerized case.
    """
    nv = _nvml()
    if nv is None:
        return ()
    handle = _read(lambda: nv.nvmlDeviceGetHandleByIndex(index), None)
    if handle is None:
        return ()
    procs = _read(lambda: nv.nvmlDeviceGetComputeRunningProcesses(handle), None)
    if not procs:
        return ()
    out: list[tuple[int, int]] = []
    for proc in procs:
        used = getattr(proc, "usedGpuMemory", None)
        # NVML reports `None` for a process whose memory it cannot attribute — a MIG instance,
        # or a process in another namespace. Counting it as zero would under-report the device.
        out.append((int(getattr(proc, "pid", 0) or 0), int(used) if used else 0))
    return tuple(out)


@functools.lru_cache(maxsize=1)
def host_pid() -> int | None:
    """This process's PID as the **host** kernel numbers it, or `None` when it cannot be read.

    The identifier every NVML process query has to be compared against, and the one nothing
    was comparing against. NVML runs in the driver and reports PIDs from the initial PID
    namespace; a Ray worker on Kubernetes — or in any Docker container that did not ask for
    `hostPID` — reads its own `os.getpid()` from the *container's* namespace, where it is
    usually a small number like `7`. The two never match, so every per-process attribution in
    the engine silently failed on exactly the fleets it was written for, and each failure was
    worse than no answer at all:

    * `own_device_memory` returned `0` rather than `None`, because NVML *did* list processes —
      so the VRAM pool charged this worker's own allocations to a phantom co-tenant, concluded
      the device was full, and planned no allocator at all. The worker then ran the whole query
      on the synchronizing driver allocator that `carbonite.accel.allocator` exists to replace.
    * `device_shared_with_others` saw only PIDs unequal to its own and reported every
      exclusively-held device as contended, which is the reading that tells autobatching not to
      trust its own utilization measurements.

    Read from `/proc/self/sched`, whose first line carries `task->pid` — the *global* PID —
    rather than from `/proc/self/status`, whose `Pid:`/`NSpid:` fields are relative to the
    namespace that mounted the procfs and so report the container-local number from inside a
    container. Memoized: a process does not get renumbered.

    Returns:
        The host-namespace PID, or `None` on a platform with no procfs or a kernel whose
        `sched` file does not carry it — where the caller must treat attribution as
        unavailable rather than as zero.
    """
    try:
        with open("/proc/self/sched", encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return None  # not Linux, or procfs not mounted
    # `comm (pid, #threads: n)` — and `comm` may itself contain parentheses, so the opening
    # bracket is found from the right.
    start = first.rfind("(")
    if start < 0:
        return None
    token = first[start + 1 :].split(",", 1)[0].strip()
    return int(token) if token.isdigit() else None


def own_process_ids() -> tuple[int, ...]:
    """Every PID this process may be listed under by the driver, most authoritative first.

    Both are tried because neither is always right. `host_pid` is the correct comparison inside
    a container and is unavailable off Linux; `os.getpid` is correct whenever the process shares
    the host's PID namespace — a bare-metal worker, or a pod with `hostPID: true` — and is the
    only one available when procfs cannot be read. They are identical outside a namespace, which
    is why the bug this exists for was invisible on a development box.

    Returns:
        One or two distinct PIDs. Never empty.
    """
    import os

    local = os.getpid()
    host = host_pid()
    return (host, local) if host is not None and host != local else (local,)


def own_device_memory(index: int) -> int | None:
    """Device memory this process itself holds, in bytes, or `None` when unattributable.

    The correction a shared device needs. A pool sizing itself against the device's *total*
    resident memory is counting its own allocations as a competitor's, and the usual fix —
    subtracting what the pool believes it reserved — is accounting rather than measurement:
    the framework allocates, the pool only admits, and the two diverge by the allocator's own
    pool, the CUDA context, and every buffer nobody reserved.

    Args:
        index: NVML device index.

    Returns:
        Bytes attributed to this process, or `None` when NVML cannot attribute per process —
        inside a container that cannot see other PID namespaces, and on a MIG instance. `None`
        is distinct from `0`: a caller keeps its previous accounting rather than concluding it
        holds nothing.
    """
    procs = device_processes(index)
    if not procs:
        return None
    mine = set(own_process_ids())
    listed = {pid for pid, _ in procs}
    if not (mine & listed):
        # NVML listed processes and none of them is this one. On a device this process has
        # genuinely not allocated on, `0` would be the honest answer — but so would it be when
        # the PID namespaces simply do not line up, and the two are indistinguishable from here.
        # `None` is the safe reading of that ambiguity: it leaves the caller on its own
        # accounting instead of charging this worker's memory to an imaginary neighbour.
        return None
    return sum(used for pid, used in procs if pid in mine)
