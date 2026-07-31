"""Occupancy, tensor-core activity, and DRAM activity — the figures NVML structurally cannot give.

`nvml.DeviceTelemetry.sm_utilization` says so in its own docstring: it is a *duty cycle*, the
fraction of the sample period during which at least one kernel was resident. A kernel occupying
one SM of 132 reads as 100% busy. So does a kernel occupying all of them. Every tuning decision
that reads that number as "how much of the device am I using" is reading a number that cannot
answer the question, and the gap is not small — a badly shaped kernel and a perfectly shaped one
are indistinguishable there, and they differ by two orders of magnitude in throughput.

DCGM's profiling fields are the answer, because they come from the hardware performance counters
rather than from the driver's scheduler:

* **SM active** — fraction of SMs with at least one warp resident, averaged over the interval.
  This is the figure people believe `sm_utilization` is.
* **SM occupancy** — resident warps as a fraction of the maximum. Low occupancy with high SM
  activity is a register or shared-memory limited kernel, which is a code fix, not a batch-size
  one.
* **Tensor pipe active** — fraction of cycles the tensor cores issued. A half-precision inference
  stage reading near zero here is not using the hardware it selected the dtype for, which is the
  single most common silent loss on an inference node.
* **DRAM active** — fraction of cycles the memory interface was busy. Against the part's peak
  bandwidth this is the roofline position, and it is what separates "memory bound" from "badly
  blocked" — two diagnoses with opposite fixes.
* **PCIe and NVLink bytes** — counters rather than the sampled rates NVML gives, so a total over
  a stage is exact rather than a mean of snapshots.

**DCGM is usually absent, and that is fine.** It ships as a separate daemon and a separate set of
Python bindings that are not on PyPI; a runtime container will not have it. Every entry point
here degrades to empty, and the callers above it are written to prefer these fields when present
and to fall back to the NVML duty cycle when not — never to require them.

**Embedded mode, never the daemon's.** The bindings can attach to a running `nv-hostengine` or
run the engine in-process. This module runs it embedded and read-only: attaching to a shared
daemon would let one query's field-watch configuration change what every other tenant on the
node is collecting, which is a side effect a telemetry reader has no business having.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

__all__ = [
    "PROFILING_FIELDS",
    "DcgmProfile",
    "dcgm_available",
    "device_profiles",
    "reset_dcgm_probe",
    "tensor_cores_idle",
]

#: The profiling fields worth collecting, as `(dcgm_fields attribute name, record field)`. Read
#: by attribute name off the installed bindings rather than by numeric id: the ids are stable in
#: practice and the names are the documented interface, and a wrong id silently returns a
#: different metric rather than failing.
PROFILING_FIELDS: tuple[tuple[str, str], ...] = (
    ("DCGM_FI_PROF_SM_ACTIVE", "sm_active"),
    ("DCGM_FI_PROF_SM_OCCUPANCY", "sm_occupancy"),
    ("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "tensor_active"),
    ("DCGM_FI_PROF_PIPE_FP32_ACTIVE", "fp32_active"),
    ("DCGM_FI_PROF_PIPE_FP64_ACTIVE", "fp64_active"),
    ("DCGM_FI_PROF_DRAM_ACTIVE", "dram_active"),
    ("DCGM_FI_PROF_PCIE_TX_BYTES", "pcie_tx_bytes"),
    ("DCGM_FI_PROF_PCIE_RX_BYTES", "pcie_rx_bytes"),
    ("DCGM_FI_PROF_NVLINK_TX_BYTES", "nvlink_tx_bytes"),
    ("DCGM_FI_PROF_NVLINK_RX_BYTES", "nvlink_rx_bytes"),
)

#: How often the engine samples the watched fields, in microseconds. One second: the profiling
#: counters are multiplexed across metric groups by the hardware, so asking faster does not get
#: more resolution, it gets the same resolution with more overhead on every process using the
#: device.
_UPDATE_US = 1_000_000

#: Tensor-pipe activity below which a half-precision stage is treated as not using the tensor
#: cores at all. Deliberately very low: any real tensor-core kernel is orders of magnitude above
#: this, so the threshold separates "not using them" from "using them poorly" rather than
#: grading how well.
_TENSOR_IDLE = 0.01


@dataclass(frozen=True, slots=True)
class DcgmProfile:
    """One device's hardware performance counters over the last DCGM sample interval.

    Every activity field is a fraction of cycles in [0, 1]; the byte fields are totals over the
    interval. A field the part or the DCGM version does not implement reads `0.0`, which is why
    `fields` records what actually answered.

    Attributes:
        index: DCGM GPU id, which matches the NVML index on every configuration Batcher runs on.
        sm_active: Fraction of SMs with at least one resident warp. The figure `sm_utilization`
            is mistaken for.
        sm_occupancy: Resident warps as a fraction of the hardware maximum.
        tensor_active: Fraction of cycles the tensor pipes issued.
        fp32_active: Fraction of cycles the FP32 pipes issued.
        fp64_active: Fraction of cycles the FP64 pipes issued.
        dram_active: Fraction of cycles the memory interface was busy.
        pcie_tx_bytes: Bytes sent across the host link during the interval.
        pcie_rx_bytes: Bytes received across the host link during the interval.
        nvlink_tx_bytes: Bytes sent across the peer fabric during the interval.
        nvlink_rx_bytes: Bytes received across the peer fabric during the interval.
        fields: Record-field names that DCGM actually returned a value for.
    """

    index: int
    sm_active: float = 0.0
    sm_occupancy: float = 0.0
    tensor_active: float = 0.0
    fp32_active: float = 0.0
    fp64_active: float = 0.0
    dram_active: float = 0.0
    pcie_tx_bytes: int = 0
    pcie_rx_bytes: int = 0
    nvlink_tx_bytes: int = 0
    nvlink_rx_bytes: int = 0
    fields: tuple[str, ...] = ()

    @property
    def readable(self) -> bool:
        """Whether DCGM returned any field for this device."""
        return bool(self.fields)

    @property
    def occupancy_limited(self) -> bool:
        """Whether the SMs are busy while holding far fewer warps than they could.

        The signature of a kernel limited by registers or shared memory rather than by work: the
        scheduler has nothing more it can place on an SM that is nominally busy. It is a code
        finding — a smaller block, fewer registers, less shared memory — and no amount of larger
        batches or more devices addresses it, which is why it is worth naming separately from
        every other kind of "the GPU is busy".
        """
        return self.sm_active > 0.5 and self.sm_occupancy < 0.3

    @property
    def compute_pipe_active(self) -> float:
        """The busiest arithmetic pipe's activity, in [0, 1].

        Taken as the maximum rather than the sum because the pipes are alternative issue paths
        for the same scheduler: a kernel is FP32 or tensor, and adding the two would report a
        device above 100% for doing exactly one thing.
        """
        return max(self.tensor_active, self.fp32_active, self.fp64_active)


@functools.lru_cache(maxsize=1)
def _dcgm():
    """`(pydcgm, dcgm_fields, dcgm_structs)` when the bindings are usable, else `None`.

    Memoized because the import walks a path that is usually not there, and because starting the
    embedded engine is a handshake with the same character as `nvmlInit`: an engine that would
    not start at first call will not start later in the same process.
    """
    try:
        import dcgm_fields
        import dcgm_structs
        import pydcgm
    except Exception:
        return None
    return (pydcgm, dcgm_fields, dcgm_structs)


@functools.lru_cache(maxsize=1)
def _watched():
    """`(group, field_group, record fields in id order)` for the watched fields, or `None`.

    Sets up the embedded engine, a group over every device, and a field group over the
    profiling fields the installed bindings actually define — an older DCGM is missing several,
    and asking for one it does not define fails the whole watch rather than that field.
    """
    resolved = _dcgm()
    if resolved is None:
        return None
    pydcgm, dcgm_fields, dcgm_structs = resolved
    try:
        ids: list[int] = []
        names: list[str] = []
        for attr, field in PROFILING_FIELDS:
            value = getattr(dcgm_fields, attr, None)
            if value is None:
                continue
            ids.append(int(value))
            names.append(field)
        if not ids:
            return None
        handle = pydcgm.DcgmHandle(opMode=dcgm_structs.DCGM_OPERATION_MODE_MANUAL)
        group = pydcgm.DcgmGroup(
            handle, groupName="batcher-telemetry", groupType=dcgm_structs.DCGM_GROUP_DEFAULT
        )
        field_group = pydcgm.DcgmFieldGroup(handle, "batcher-profiling", ids)
        group.samples.WatchFields(field_group, _UPDATE_US, 3600.0, 0)
        handle.GetSystem().UpdateAllFields(1)
        return (handle, group, field_group, tuple(names), tuple(ids))
    except Exception:
        return None


def dcgm_available() -> bool:
    """Whether hardware performance counters can be read on this host.

    Returns:
        True when the DCGM bindings imported and the embedded engine started. False in every
        runtime container that did not install DCGM, which is the common case and not a fault.
    """
    return _watched() is not None


def reset_dcgm_probe() -> None:
    """Forget the DCGM handshake and field watch, so the next call re-establishes them."""
    _watched.cache_clear()
    _dcgm.cache_clear()


def _value(entry) -> float | None:
    """One DCGM field value as a float, or `None` when the sample is blank or an error.

    DCGM signals "not collected" through blank values with a distinguished sentinel rather than
    through an error, and the sentinels are large negative integers. Reading one as a number
    would report a device at minus nine quintillion percent occupancy, which is at least
    obviously wrong; reading a *blank* one as zero would report an idle device, which is not.
    """
    if entry is None:
        return None
    if getattr(entry, "isBlank", False):
        return None
    value = getattr(entry, "value", None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number < 0 else number


def device_profiles() -> tuple[DcgmProfile, ...]:
    """Hardware performance counters for every device DCGM can see, in device order.

    Costs one engine update and one sample fetch. The update is what makes the reading current,
    and it is why this is a per-stage rather than a per-batch call.

    Returns:
        One record per device, empty when DCGM is unavailable — which is the normal case, and
        is why every caller must treat these as an enrichment of the NVML readings rather than
        as a replacement for them.
    """
    watched = _watched()
    if watched is None:
        return ()
    handle, group, field_group, names, ids = watched
    try:
        handle.GetSystem().UpdateAllFields(1)
        latest = group.samples.GetLatest(field_group).values
    except Exception:
        return ()
    out: list[DcgmProfile] = []
    for gpu_id in sorted(latest):
        per_field = latest[gpu_id]
        values: dict[str, float] = {}
        present: list[str] = []
        for field_id, name in zip(ids, names, strict=False):
            entries = per_field.get(field_id)
            # DCGM returns a list of samples per field; the last is the most recent, and the
            # earlier ones describe intervals this call was not asked about.
            entry = entries[-1] if entries else None
            reading = _value(entry)
            if reading is None:
                continue
            values[name] = reading
            present.append(name)
        if not present:
            continue
        out.append(
            DcgmProfile(
                index=int(gpu_id),
                sm_active=min(1.0, values.get("sm_active", 0.0)),
                sm_occupancy=min(1.0, values.get("sm_occupancy", 0.0)),
                tensor_active=min(1.0, values.get("tensor_active", 0.0)),
                fp32_active=min(1.0, values.get("fp32_active", 0.0)),
                fp64_active=min(1.0, values.get("fp64_active", 0.0)),
                dram_active=min(1.0, values.get("dram_active", 0.0)),
                pcie_tx_bytes=int(values.get("pcie_tx_bytes", 0)),
                pcie_rx_bytes=int(values.get("pcie_rx_bytes", 0)),
                nvlink_tx_bytes=int(values.get("nvlink_tx_bytes", 0)),
                nvlink_rx_bytes=int(values.get("nvlink_rx_bytes", 0)),
                fields=tuple(present),
            )
        )
    return tuple(out)


def tensor_cores_idle(
    profiles: tuple[DcgmProfile, ...] | None = None,
) -> tuple[DcgmProfile, ...]:
    """Devices doing arithmetic without touching their tensor cores, in device order.

    The check to run once after selecting a half-precision dtype, because selecting one and
    getting one are different events. A model loaded in BF16 whose kernels never reach the
    tensor pipes is paying the precision cost of half and getting none of the throughput, and
    nothing else reports it: utilization is high, memory is right, and the run is simply slower
    than it should be by a factor no timing explains.

    Args:
        profiles: Records to inspect, or `None` to read them live.

    Returns:
        Devices with meaningful SM activity and effectively no tensor-pipe activity. Empty when
        DCGM is unavailable, so this is evidence of a problem only where `dcgm_available` holds.
    """
    records = device_profiles() if profiles is None else profiles
    return tuple(
        p for p in records if p.readable and p.sm_active > 0.2 and p.tensor_active < _TENSOR_IDLE
    )
