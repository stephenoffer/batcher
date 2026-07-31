"""What each device *is*, read from the device rather than matched from its name.

`device_specs` holds a curated table keyed on the model name, because there are facts about an
accelerator no cluster reports. There are also facts every cluster reports and that the table is
the wrong source for, and this module is those: compute capability, core count, memory bus
width, architecture, and the firmware and board identity behind them.

Reading them rather than looking them up fixes three concrete failures the table cannot:

* **Per-device answers on a heterogeneous node.** Half-precision selection currently asks
  `torch.cuda.get_device_capability()`, which answers for device 0 and for device 0 only. A box
  with an L40S beside an A100 — or one MIG slice beside another — gets device 0's answer applied
  to all of them, and choosing BF16 on a part that emulates it is a silent throughput loss.
* **MIG slices.** A slice reports its own capability and its own core count, and neither matches
  the board's model name. Every sizing decision made off the name is made for hardware the
  process does not have.
* **Peak memory bandwidth without a table.** Memory clock times bus width is the roofline
  denominator, and both are published per device. A model the curated table has never heard of
  — a new part, an OEM variant, a slice — still gets a real figure instead of a fallback.

**Nothing here moves within a run**, which is the difference between this module and its
siblings: these readings are memoized, and the memoization is cleared through the package's
usual probe reset. That matters because compute capability is consulted on the dtype path,
which runs per stage, and it is a driver round trip per device otherwise.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from batcher._internal.hardware.nvml import _decode, _device_count, _nvml, _read

__all__ = [
    "ARCHITECTURES",
    "DeviceIdentity",
    "device_identity",
    "half_precision_dtype",
    "peak_memory_bandwidth_bytes",
    "reset_identity_probe",
]

#: `nvmlDeviceArchitecture` values to the architecture name Batcher uses in reports and in
#: learned-parameter keys. An unlisted value reports as `""` rather than guessing, because a
#: wrong architecture name is a wrong dtype and a wrong cost model.
ARCHITECTURES: dict[int, str] = {
    2: "kepler",
    3: "maxwell",
    4: "pascal",
    5: "volta",
    6: "turing",
    7: "ampere",
    8: "ada",
    9: "hopper",
    10: "blackwell",
}

#: Compute capability at which the tensor cores gain each precision natively. Below the BF16
#: threshold the type is emulated, which is slower than the FP16 it would have replaced — the
#: precise case a name-matched table gets wrong on an unfamiliar part.
_NATIVE_FP8 = (8, 9)
_NATIVE_BF16 = (8, 0)
_FAST_FP16 = (7, 0)


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """One device's fixed hardware and firmware facts.

    Attributes:
        index: NVML device index on this host.
        uuid: Stable device UUID.
        name: Model name as the driver reports it.
        architecture: Architecture name from `ARCHITECTURES`, `""` when unrecognized.
        compute_capability: `(major, minor)`, `(0, 0)` when unreported. The authority on which
            precisions the tensor cores implement.
        cores: CUDA core count as the driver reports it, `0` when unreported. On a MIG slice
            this is the slice's share, which is the figure a sizing decision wants.
        memory_bus_width_bits: Memory interface width, `0` when unreported.
        memory_clock_max_mhz: Highest memory clock the part supports, `0` when unreported.
        minor_number: The `N` in `/dev/nvidiaN`, `-1` when unreported. The identifier a
            container runtime's device allowlist is written in, and the only one that joins a
            device to what the runtime actually granted.
        board_id: Board identifier, shared by the devices on one multi-GPU board.
        serial: Board serial number, `""` when unreported or refused.
        part_number: Board part number, `""` when unreported.
        vbios: VBIOS version string.
        driver_version: Host NVIDIA driver version.
        cuda_driver_version: CUDA version the driver supports, as `"12.4"`, `""` when
            unreported. The ceiling on what any CUDA library in this process can use, and a
            frequent explanation for a container's toolkit refusing to initialize.
        readable: Whether NVML answered any query.
    """

    index: int
    uuid: str = ""
    name: str = ""
    architecture: str = ""
    compute_capability: tuple[int, int] = (0, 0)
    cores: int = 0
    memory_bus_width_bits: int = 0
    memory_clock_max_mhz: int = 0
    minor_number: int = -1
    board_id: int = 0
    serial: str = ""
    part_number: str = ""
    vbios: str = ""
    driver_version: str = ""
    cuda_driver_version: str = ""
    readable: bool = False

    @property
    def native_bf16(self) -> bool:
        """Whether the tensor cores implement BF16 rather than emulating it."""
        return self.compute_capability >= _NATIVE_BF16

    @property
    def native_fp8(self) -> bool:
        """Whether the tensor cores implement FP8 rather than emulating it."""
        return self.compute_capability >= _NATIVE_FP8

    @property
    def fast_fp16(self) -> bool:
        """Whether half precision has a tensor-core path at all on this part.

        False below Volta, where FP16 is stored in half the space and computed no faster — so
        choosing it buys memory and costs precision, which is the wrong trade for inference.
        """
        return self.compute_capability >= _FAST_FP16

    @property
    def peak_memory_bandwidth_bytes(self) -> float:
        """Theoretical peak device memory bandwidth in bytes per second, `0.0` when unknown.

        Clock times bus width times two, because every part Batcher runs on uses a
        double-data-rate memory interface: GDDR and HBM both transfer on each clock edge. This
        is the denominator a roofline needs, and it is a *ceiling* — a real kernel reaching 80%
        of it is doing well, so a measured figure above it means the calculation is wrong
        rather than the kernel is fast.
        """
        if self.memory_bus_width_bits <= 0 or self.memory_clock_max_mhz <= 0:
            return 0.0
        return self.memory_clock_max_mhz * 1e6 * (self.memory_bus_width_bits / 8.0) * 2.0

    @property
    def mig_slice(self) -> bool:
        """Whether this handle is a MIG instance rather than a whole board.

        NVML names an instance with a `MIG` prefix, which is the only identification a process
        holding one gets: its memory, core count, and capability all describe the slice, while
        the model name in every log line describes the board it was cut from.
        """
        return self.name.startswith("MIG ")


def _compute_capability(nv, handle) -> tuple[int, int]:
    """`(major, minor)` compute capability, `(0, 0)` when the query is refused."""
    fn = getattr(nv, "nvmlDeviceGetCudaComputeCapability", None)
    if fn is None:
        return (0, 0)
    value = _read(lambda: fn(handle), None)
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return (0, 0)
    return (int(value[0] or 0), int(value[1] or 0))


def _int_query(nv, handle, getter: str, default: int = 0) -> int:
    """One integer NVML getter by name, `default` when absent or refused."""
    fn = getattr(nv, getter, None)
    if fn is None:
        return default
    value = _read(lambda: fn(handle), None)
    return default if value is None else int(value)


def _str_query(nv, handle, getter: str) -> str:
    """One string NVML getter by name, `""` when absent or refused."""
    fn = getattr(nv, getter, None)
    if fn is None:
        return ""
    return _decode(_read(lambda: fn(handle), ""))


def _cuda_driver_version(nv) -> str:
    """The CUDA version the driver supports, as `"12.4"`, `""` when unreported.

    NVML encodes it as one integer, `major * 1000 + minor * 10`, which is unreadable in a report
    and is compared wrongly against a string version by every caller that tries.
    """
    for getter in ("nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"):
        fn = getattr(nv, getter, None)
        if fn is None:
            continue
        raw = _read(lambda f=fn: f(), 0)
        if raw:
            return f"{int(raw) // 1000}.{(int(raw) % 1000) // 10}"
    return ""


@functools.lru_cache(maxsize=1)
def device_identity() -> tuple[DeviceIdentity, ...]:
    """Fixed hardware facts for every local device, in NVML index order.

    Memoized, unlike every other probe in this package: none of these readings can change
    within a process, and the compute-capability field is consulted on a per-stage path where a
    driver round trip per device is real cost. `reset_identity_probe` is the hook a test faking
    the driver needs.

    Returns:
        One record per device, empty when NVML is unavailable.
    """
    nv = _nvml()
    if nv is None:
        return ()
    driver = ""
    fn = getattr(nv, "nvmlSystemGetDriverVersion", None)
    if fn is not None:
        driver = _decode(_read(lambda: fn(), ""))
    cuda = _cuda_driver_version(nv)
    out: list[DeviceIdentity] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        capability = _compute_capability(nv, handle)
        name = _decode(_read(lambda h=handle: nv.nvmlDeviceGetName(h), ""))
        out.append(
            DeviceIdentity(
                index=index,
                uuid=_decode(_read(lambda h=handle: nv.nvmlDeviceGetUUID(h), "")),
                name=name,
                architecture=ARCHITECTURES.get(
                    _int_query(nv, handle, "nvmlDeviceGetArchitecture", -1), ""
                ),
                compute_capability=capability,
                cores=_int_query(nv, handle, "nvmlDeviceGetNumGpuCores"),
                memory_bus_width_bits=_int_query(nv, handle, "nvmlDeviceGetMemoryBusWidth"),
                # `nvmlClockType` 2 is memory; the maximum rather than the current clock,
                # because peak bandwidth is a ceiling and the current clock is a reading.
                memory_clock_max_mhz=int(
                    _read(lambda h=handle: nv.nvmlDeviceGetMaxClockInfo(h, 2), 0) or 0
                ),
                minor_number=_int_query(nv, handle, "nvmlDeviceGetMinorNumber", -1),
                board_id=_int_query(nv, handle, "nvmlDeviceGetBoardId"),
                serial=_str_query(nv, handle, "nvmlDeviceGetSerial"),
                part_number=_str_query(nv, handle, "nvmlDeviceGetBoardPartNumber"),
                vbios=_str_query(nv, handle, "nvmlDeviceGetVbiosVersion"),
                driver_version=driver,
                cuda_driver_version=cuda,
                readable=bool(name) or capability != (0, 0),
            )
        )
    return tuple(out)


def reset_identity_probe() -> None:
    """Forget the memoized device identities so the next call re-reads the driver."""
    device_identity.cache_clear()


def half_precision_dtype(index: int = 0) -> str | None:
    """The safe half-precision dtype for one device, or `None` to keep FP32.

    The per-device answer `torch.cuda.get_device_capability()` cannot give, and it needs no
    torch: on a heterogeneous node or a MIG-partitioned board, device 0's capability is not
    every device's capability, and applying it to all of them chooses an emulated type on the
    parts that do not implement it.

    Args:
        index: NVML device index.

    Returns:
        `"bfloat16"` on Ampere and later, `"float16"` on Volta and Turing, and `None` below
        that or when the device did not report a capability — never a guess, because a
        half-precision default nobody measured is exactly the fast-wrong-answer trade this
        engine refuses elsewhere.
    """
    record = next((d for d in device_identity() if d.index == index), None)
    if record is None or record.compute_capability == (0, 0):
        return None
    if record.native_bf16:
        return "bfloat16"
    return "float16" if record.fast_fp16 else None


def peak_memory_bandwidth_bytes(index: int = 0) -> float:
    """Theoretical peak device memory bandwidth for one device, in bytes per second.

    Args:
        index: NVML device index.

    Returns:
        Bytes per second, `0.0` when the device did not report a bus width or memory clock.
    """
    record = next((d for d in device_identity() if d.index == index), None)
    return 0.0 if record is None else record.peak_memory_bandwidth_bytes
