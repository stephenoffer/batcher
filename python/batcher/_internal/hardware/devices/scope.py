"""Which devices this process may actually use, and how much is left on each of them.

Every VRAM decision in the engine starts by answering "which GPU am I talking about", and
until now each caller answered it with `0`. On the single-GPU box that is developed on, `0`
is right. On a node with eight devices and one actor pinned to the sixth, it is a different
board: the packing math sizes against a device the process cannot touch, the OOM guard reads
a stranger's free memory, and both are wrong in the unsafe direction whenever the neighbour
happens to be emptier.

This module is the one place that answers it, and it answers three questions the scattered
`device 0` reads could not:

* **Which physical devices are visible.** A scheduler pins an actor by setting
  ``CUDA_VISIBLE_DEVICES``, and it does *not* always set it to ordinals. Ray writes indices,
  but the Kubernetes device plugin writes **UUIDs** (``GPU-4f2a...``) and MIG writes
  partition handles (``MIG-GPU-.../1/0``). Index-only parsing treats those as unparseable and
  falls back to *every device on the node* — so on precisely the fleets that pin hardest,
  every pinned actor silently measured the whole node and averaged a busy device with seven
  idle ones.
* **How much is really free.** A process's own allocator knows what *it* holds. It cannot see
  the co-tenant that is holding 60 GiB of the 80, which on a shared device is the entire
  binding constraint. The driver reports it, and nothing was asking.
* **Whether the devices differ.** Packing that sizes to the mean of a mixed node
  over-subscribes the small card. Anything sizing one number for a whole node must use the
  *smallest* visible device, and that requires knowing there is more than one size.

Reads only; nothing here allocates device memory. Every entry point degrades to `None`/empty
rather than raising, so a CPU-only host, a container with no driver mounted, and a build
without `pynvml` all leave callers on whatever default they had.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from batcher._internal.hardware.nvml import device_telemetry

__all__ = [
    "DEVICE_ORDER_ENV",
    "PCI_BUS_ORDER",
    "VISIBLE_DEVICE_ENVS",
    "DeviceScope",
    "current_ordinal",
    "current_physical_index",
    "device_free_bytes",
    "device_order_env",
    "device_scope",
    "min_visible_capacity_bytes",
    "visible_device_indices",
    "visible_device_telemetry",
]

#: The environment variables a scheduler pins a process's *visible* devices through, in
#: priority order. NVIDIA and HIP both honor ``CUDA_VISIBLE_DEVICES``; AMD ROCm adds its own
#: two. A host runs one vendor, so consulting all three is safe and the first one set wins.
VISIBLE_DEVICE_ENVS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

#: The variable that decides how the CUDA runtime *numbers* devices.
DEVICE_ORDER_ENV = "CUDA_DEVICE_ORDER"

#: The one ordering under which a CUDA ordinal and an NVML index mean the same board.
PCI_BUS_ORDER = "PCI_BUS_ID"


def device_order_env(process_env: dict[str, str] | None = None) -> dict[str, str]:
    """The device-ordering variable a worker needs set, or `{}` when it is already decided.

    This whole module translates between two numberings — the ordinals a framework uses and the
    indices NVML reports — and every translation in it assumes the two enumerate in the same
    order. **Unset, they do not.** The CUDA runtime defaults to ``FASTEST_FIRST``, which sorts
    devices by capability, while NVML always enumerates by PCI bus address. On a node whose
    boards are identical the two orders coincide and nothing is wrong; on a mixed node — an L4
    beside an A100, a fleet part-way through an upgrade, a box with a display adapter in it —
    they do not, and then:

    * ``CUDA_VISIBLE_DEVICES=1`` selects a different board than `visible_device_indices` reports,
      so the pool is sized against one device's capacity and reserved on another;
    * `current_physical_index` names the wrong board, so its telemetry, its NUMA node, and its
      health verdict all belong to a device this process is not using;
    * `feeder_cpus_for_device` binds the worker's decode threads to the wrong socket.

    None of it raises, and none of it reproduces on a homogeneous node, which is every node
    anybody develops on. Pinning the order to PCI is what makes the assumption true rather than
    merely usual, and it costs nothing where it was already true.

    Args:
        process_env: The environment to inspect, or `None` for this process's own.

    Returns:
        ``{"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}``, or an empty dict when the deployment has
        already set the variable — including to something else, which is a decision with a
        reason no probe can see.
    """
    ambient = os.environ if process_env is None else process_env
    if ambient.get(DEVICE_ORDER_ENV, "").strip():
        return {}
    return {DEVICE_ORDER_ENV: PCI_BUS_ORDER}


@dataclass(frozen=True, slots=True)
class DeviceScope:
    """The devices this process may use, and what the driver says about each.

    Attributes:
        indices: Physical driver indices this process can see, in the order the visibility
            environment named them — so position *i* is what the framework calls device *i*.
            Empty when no accelerator is present.
        capacities: Total memory per physical index, for the visible devices only.
        used: Memory resident on each visible device across *every* process, not just this
            one. The figure a co-tenant is invisible in.
        pinned: Whether a visibility variable was set. `False` means the process sees the
            whole node, which is the right reading for a driver or a monitor and the wrong
            one to conclude a device is idle from.
    """

    indices: tuple[int, ...] = ()
    capacities: dict[int, int] | None = None
    used: dict[int, int] | None = None
    pinned: bool = False

    @property
    def count(self) -> int:
        """How many devices this process may use."""
        return len(self.indices)

    def capacity_of(self, index: int) -> int | None:
        """Total memory of one physical device, or `None` when unreported."""
        return (self.capacities or {}).get(index)

    def free_of(self, index: int) -> int | None:
        """Memory not resident to any process on one device, or `None` when unreported.

        This is the number a co-tenant is visible in, and it is the one an allocation must be
        checked against. A framework's own allocator statistics cannot produce it: they report
        what *this* process reserved, so a device with 4 GiB left and 60 GiB held by a
        neighbour looks identical to an empty one.
        """
        total = self.capacity_of(index)
        held = (self.used or {}).get(index)
        if total is None or held is None:
            return None
        return max(0, total - held)

    @property
    def min_capacity_bytes(self) -> int | None:
        """The smallest visible device's total memory, or `None` when nothing reported.

        The figure any single node-wide sizing decision must use. A mixed node — an A100
        beside an L4, or a fleet part-way through an upgrade — has no single capacity, and
        sizing to the mean or to device 0 over-subscribes the small card into an OOM that
        only reproduces on the nodes that happen to be mixed.
        """
        values = [v for v in (self.capacities or {}).values() if v > 0]
        return min(values) if values else None

    @property
    def heterogeneous(self) -> bool:
        """Whether the visible devices differ in capacity.

        Worth knowing on its own: a heterogeneous scope means every "how many actors fit"
        answer is per-device rather than per-node, and a caller that can only produce one
        number should say so rather than quietly pick a device's worth.
        """
        values = {v for v in (self.capacities or {}).values() if v > 0}
        return len(values) > 1

    @property
    def emptiest(self) -> int | None:
        """The visible device with the most free memory, or `None` when free is unknown.

        Ties break on the lowest index so repeated placement is reproducible rather than
        drifting with dictionary order.
        """
        free = {i: f for i in self.indices if (f := self.free_of(i)) is not None}
        if not free:
            return None
        return min(free, key=lambda i: (-free[i], i))


def visible_device_indices() -> tuple[int, ...]:
    """Physical driver indices this process may use, honoring the visibility environment.

    Parses all three forms a scheduler writes:

    * **Ordinals** (``"2,3"``) — Ray's form, taken as physical indices directly.
    * **UUIDs** (``"GPU-4f2a..."``) — the Kubernetes device plugin's form, resolved against
      the driver's reported UUIDs. Nothing resolved these before, so they fell through to
      "every device on the node".
    * **MIG handles** (``"MIG-GPU-<uuid>/1/0"``) — resolved to the *parent* device, which is
      the board whose memory and telemetry the partition draws from.

    Returns:
        The visible physical indices in the order they were named, or every device on the
        host when nothing is pinned or the value cannot be resolved. An empty tuple means no
        accelerator was detected at all.
    """
    return device_scope().indices


def current_physical_index() -> int | None:
    """The physical index of the device the framework is currently computing on.

    A process pinned to two devices still runs on *one* at a time, and which one is a torch
    fact rather than an environment fact — ``torch.cuda.set_device`` moves it. Reading the
    driver's device 0 instead is right only by coincidence.

    Returns:
        The physical index, or `None` when there is no accelerator or torch is absent.
    """
    indices = visible_device_indices()
    if not indices:
        return None
    ordinal = current_ordinal()
    if ordinal is None or ordinal >= len(indices):
        return indices[0]
    return indices[ordinal]


def device_free_bytes(index: int | None = None) -> int | None:
    """Memory free on one device across every process, or `None` when unreadable.

    Prefers the driver's own accounting (which sees co-tenants) and falls back to torch's
    ``mem_get_info``, which reports the same figure for the current device. Note what is
    *not* used as a fallback: this process's allocator statistics, which cannot see a
    neighbour and would report an over-subscribed device as empty.

    Args:
        index: Physical device index, or `None` for the one currently in use.

    Returns:
        Free bytes, or `None` when neither source reported.
    """
    target = current_physical_index() if index is None else index
    if target is None:
        return None
    free = device_scope().free_of(target)
    if free is not None:
        return free
    return _torch_free_bytes(target)


def min_visible_capacity_bytes() -> int | None:
    """The smallest visible device's total memory — see `DeviceScope.min_capacity_bytes`."""
    return device_scope().min_capacity_bytes


def visible_device_telemetry():
    """Live readings for the devices **this process may use**, in visibility order.

    `nvml.device_telemetry` reports every device on the host, which is the right answer for a
    monitor and the wrong one for a worker deciding anything about itself. A pinned actor on an
    eight-device node that averages, minimizes, or maximizes over all eight is reading seven
    boards it cannot touch: its utilization advice is diluted by idle neighbours, and its memory
    sizing is bounded by whichever stranger happens to be busiest. Both readings look plausible
    and neither is about this process.

    Falls back to the full host list when nothing is pinned — an unpinned process really can
    use every device — and when the visibility value cannot be resolved, which is the same
    conservative reading `visible_device_indices` already takes.

    Returns:
        The visible subset of `nvml.device_telemetry()`, in the order visibility named the
        devices. Empty when telemetry is unavailable.
    """
    # Resolved through the module at call time rather than through this file's own bound name.
    # The readings are the one thing every test of a caller has to fake, and they all fake it
    # by patching `nvml.device_telemetry` — which a name bound at import silently ignores,
    # leaving the test passing against the real (empty) host instead of the fixture it wrote.
    from batcher._internal.hardware import nvml

    readings = nvml.device_telemetry()
    if not readings:
        return ()
    by_index = {r.index: r for r in readings}
    visible = [by_index[i] for i in visible_device_indices() if i in by_index]
    return tuple(visible) if visible else readings


def device_scope() -> DeviceScope:
    """Resolve the visible devices and their live memory in one read.

    Not memoized. Capacity is stable, but `used` is a live figure and the whole reason to ask
    is that a co-tenant's footprint changes; a cached scope would report an emptied device as
    still full and, worse, a filled one as still empty. Callers that only want the index list
    on a hot path should hold the result rather than re-resolving.

    Returns:
        The scope, empty when no accelerator is detectable on this host.

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware.devices import device_scope
            >>> device_scope().count >= 0
            True
    """
    telemetry = device_telemetry()
    if not telemetry:
        return DeviceScope()
    raw, pinned = _visibility_env()
    indices = _resolve(raw, telemetry) if pinned else tuple(t.index for t in telemetry)
    by_index = {t.index: t for t in telemetry}
    return DeviceScope(
        indices=indices,
        capacities={i: by_index[i].memory_total_bytes for i in indices if i in by_index},
        used={i: by_index[i].memory_used_bytes for i in indices if i in by_index},
        pinned=pinned,
    )


def _visibility_env() -> tuple[str, bool]:
    """The first visibility variable that is set, and whether one was.

    An *empty* value is meaningful and distinct from unset: ``CUDA_VISIBLE_DEVICES=""`` is how
    a scheduler says "no devices at all", and treating it as unset would hand the process the
    whole node.
    """
    for env in VISIBLE_DEVICE_ENVS:
        raw = os.environ.get(env)
        if raw is not None:
            return raw.strip(), True
    return "", False


def _resolve(raw: str, telemetry) -> tuple[int, ...]:
    """Map one visibility value onto physical indices, as the CUDA runtime itself would.

    **The list is truncated at the first entry that names no live device, not filtered.** That
    is what the runtime does, and the difference is a mis-mapping rather than a missing device:
    given ``CUDA_VISIBLE_DEVICES=0,9,1`` on a four-device node, CUDA exposes exactly one device
    and calls it ordinal 0. Skipping the bad entry instead yields `(0, 1)`, so ordinal 1 maps
    to physical device 1 — a board this process cannot address — and every reading taken
    against it (its free memory, its NUMA node, its link) belongs to a device the framework
    will never place work on.

    A value where *nothing* resolves is a different case and still falls back to the whole
    host: that is a value this code does not understand rather than a device list it disagrees
    with, and the conservative reading of "I cannot tell which device is mine" is the one every
    caller already handles.
    """
    if not raw:
        return ()
    by_uuid = {t.uuid: t.index for t in telemetry if t.uuid}
    count = len(telemetry)
    out: list[int] = []
    for token in (t.strip() for t in raw.split(",")):
        index = _resolve_token(token, by_uuid, count)
        if index is None:
            break
        if index not in out:
            out.append(index)
    return tuple(out) if out else tuple(t.index for t in telemetry)


def _resolve_token(token: str, by_uuid: dict[str, int], count: int) -> int | None:
    """One visibility entry as a physical index, or `None` when it names no live device."""
    if not token:
        return None
    if token.isdigit():
        index = int(token)
        return index if index < count else None
    # A MIG handle is `MIG-GPU-<parent-uuid>/<gi>/<ci>`, and on some driver versions simply
    # `MIG-<uuid>`. The partition draws its memory from the parent board, so the parent is the
    # device whose capacity and residency actually bound this process.
    if token.startswith("MIG-"):
        token = token[4:].split("/", 1)[0]
    if token in by_uuid:
        return by_uuid[token]
    # Drivers report the UUID with a `GPU-` prefix; a scheduler may or may not include it.
    for candidate in (f"GPU-{token}", token.removeprefix("GPU-")):
        if candidate in by_uuid:
            return by_uuid[candidate]
    return None


def current_ordinal() -> int | None:
    """The device ordinal this process is computing on, as CUDA numbers it, or `None`.

    "Ordinal" is the framework-visible index — `0` for the first device
    ``CUDA_VISIBLE_DEVICES`` names, whatever physical board that is. `current_physical_index`
    translates it; this is the untranslated half, and it is what RMM's per-device resource
    table is keyed by.

    Asked of whichever accelerator library is **already loaded**, never by importing one: this
    is on the path of a decision about placement, and importing torch to answer it would cost
    more than the decision saves and would initialize a CUDA context as a side effect. A
    relational cuDF worker has no torch at all, which is why CuPy and Numba are consulted too —
    without them that worker reported "no framework" and every caller fell back to ordinal
    zero, which is right on a one-device node and silently wrong on the eight-device nodes this
    engine is meant to run on.

    Returns:
        The current ordinal, or `None` when no accelerator library in this process can say.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        for name in ("cuda", "xpu"):
            backend = getattr(torch, name, None)
            try:
                if backend is not None and backend.is_available():
                    return int(backend.current_device())
            except Exception:
                continue
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            return int(cupy.cuda.runtime.getDevice())
        except Exception:
            pass
    numba = sys.modules.get("numba")
    if numba is not None:
        try:
            from numba import cuda as numba_cuda

            return int(numba_cuda.get_current_device().id)
        except Exception:
            pass
    return None


def _torch_free_bytes(index: int) -> int | None:
    """Free bytes from ``torch.cuda.mem_get_info``, or `None` when it cannot answer.

    The driver query torch exposes, which — unlike ``memory_reserved`` — reports the device's
    real free memory including every other process on it. Only valid for a device this
    process can address, so it is attempted for the current device only.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is None or current_physical_index() != index:
        return None
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        free, _total = cuda.mem_get_info()
        return int(free)
    except Exception:
        return None
