"""Effective hardware detection — the CPU parallelism the process may actually use.

`os.cpu_count()` reports the *host's* logical cores, which over-counts inside a container:
a Kubernetes/Ray pod is throttled by a cgroup CPU quota (CFS bandwidth) or pinned to a
cpuset, so sizing thread pools and task fan-out to the host count over-subscribes — the
scheduler thrashes on context switches for cores the process will never get. This resolves
the real budget: the CPU-affinity mask (cpuset) capped by the CFS quota, falling back to the
host count when neither is discoverable. A neutral utility (any layer may import `_internal`).
"""

from __future__ import annotations

import contextlib
import functools
import glob
import os
import sys

from batcher._internal.mathx import ceil_div

__all__ = [
    "INFERENCE_INFLIGHT_DEPTH_MAX",
    "available_cpu_count",
    "cgroup_v2_dirs",
    "cpu_contention",
    "gpu_devices_absent",
    "gpu_inventory",
    "l3_cache_bytes",
    "machine_memory_bytes",
]

# Hard ceiling on how many partitions an inference actor keeps in flight at once (submit-ahead
# depth), and the mirror ceiling on the intra-worker autobatch pipeline. One home in a neutral
# layer both the ML autobatcher (`ml.gpu`, layer 6) and the distributed actor pool (`dist`,
# layer 4) import, rather than two copies that drift — `dist` cannot import `ml`, so a shared
# copy was the ONLY correct way to share this, and it used to be pasted instead. Past ~16
# in-flight the GPU is already saturated and deeper submit-ahead only grows resident memory.
INFERENCE_INFLIGHT_DEPTH_MAX = 16


def _affinity_count() -> int | None:
    """Cores in this process's scheduling-affinity mask (cpuset), or `None` if unavailable."""
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:  # not Linux (macOS/Windows expose no affinity mask)
        return None
    try:
        n = len(getaffinity(0))
    except OSError:
        return None
    return n if n > 0 else None


def _quota_cores(quota: int, period: int) -> int | None:
    """`ceil(quota / period)` cores, or `None` when either is non-positive (unlimited)."""
    if quota > 0 and period > 0:
        return max(1, ceil_div(quota, period))  # ceil-div
    return None


def _read_cgroup_v2_quota(base: str) -> int | None:
    """Cores the cgroup v2 ``<base>/cpu.max`` permits, or `None` when unlimited/absent.

    ``cpu.max`` is ``"<quota> <period>"``; a ``max`` quota means unlimited.
    """
    try:
        with open(os.path.join(base, "cpu.max")) as f:
            parts = f.read().split()
    except OSError:
        return None
    if len(parts) >= 1 and parts[0] != "max":
        try:
            period = int(parts[1]) if len(parts) > 1 else 100_000
            return _quota_cores(int(parts[0]), period)
        except ValueError:
            return None
    return None


def cgroup_v2_dirs() -> list[str]:
    """Every cgroup v2 dir whose ``cpu.max`` can bind this process: mount root through leaf.

    The leaf comes from the process's own ``/proc/self/cgroup``. Root and leaf coincide inside a
    K8s pod (a cgroup *namespace* maps the pod's cgroup to the mount root) but diverge in a
    *delegated* cgroup with no namespace (a Ray worker under a systemd slice). cgroup v2 enforces
    the CFS bandwidth quota at **every** level, so the effective limit is the tightest ``cpu.max``
    anywhere in the chain — checking only the ends would miss a quota set on a parent slice.
    """
    dirs = ["/sys/fs/cgroup"]
    sub = ""
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):  # the unified-hierarchy (v2) line
                    sub = line.rstrip().split("::", 1)[1]
                    break
    except OSError:
        return dirs
    parts = [p for p in sub.split("/") if p]
    # Leaf first (most specific) down to the root; `_cfs_quota_count` mins over all anyway.
    for i in range(len(parts), 0, -1):
        dirs.append("/sys/fs/cgroup/" + "/".join(parts[:i]))
    return dirs


def _cfs_quota_count() -> int | None:
    """Whole cores the cgroup CFS bandwidth quota permits, or `None` when unlimited/unavailable.

    cgroup v2 first (the tightest ``cpu.max`` across every dir in [`cgroup_v2_dirs`]), then
    v1 (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``).
    """
    v2 = [q for d in cgroup_v2_dirs() if (q := _read_cgroup_v2_quota(d)) is not None]
    if v2:
        return min(v2)  # the most restrictive limit in the hierarchy is the effective one
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:  # cgroup v1
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        return _quota_cores(quota, period)
    except (OSError, ValueError):
        pass  # no cgroup v1 quota files (or unparseable) -> fall through to the next probe
    return None


def available_cpu_count() -> int:
    """The number of CPUs this process may actually use — never fewer than 1.

    The minimum of the affinity-mask size (cpuset pin) and the CFS-quota core count (bandwidth
    throttle), floored by `os.cpu_count()` and finally 1. Prefer this over `os.cpu_count()`
    anywhere thread pools or task fan-out are sized, so a container throttled to N cores fans
    out to N — not to the host core count it will never receive (which over-subscribes and
    thrashes the scheduler).

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware import available_cpu_count
            >>> available_cpu_count() >= 1
            True
    """
    candidates = [c for c in (_affinity_count(), _cfs_quota_count()) if c is not None]
    host = os.cpu_count() or 1
    return max(1, min([host, *candidates]))


def _cgroup_throttled_ratio() -> float | None:
    """Share of CFS periods in which this cgroup was throttled, or `None` if unreadable.

    ``cpu.stat``'s ``nr_throttled``/``nr_periods`` are monotonic process-lifetime counters, so
    this is a lifetime average rather than a live reading — enough to answer "is the quota
    binding at all", which is the question that distinguishes a throttled container from an
    under-parallelized query.
    """
    for base in cgroup_v2_dirs():
        try:
            with open(os.path.join(base, "cpu.stat")) as f:
                stat = {
                    parts[0]: int(parts[1])
                    for line in f
                    if len(parts := line.split()) == 2 and parts[1].isdigit()
                }
        except (OSError, ValueError):
            continue
        periods = stat.get("nr_periods", 0)
        if periods > 0:
            return stat.get("nr_throttled", 0) / periods
    return None


@functools.lru_cache(maxsize=1)
def l3_cache_bytes() -> int:
    """Bytes of last-level (L3) cache in this core's cache domain, or `0` if undetectable.

    The physical quantity a broadcast-join threshold actually depends on: a broadcast builds
    one hash table probed from every core, so the strategy wins only while that table stays
    L3-resident. A fixed byte threshold is therefore wrong by the ratio of the real cache to
    the assumed one — ~1 MiB on a small ARM core to 32+ MiB per CCX on an EPYC, an 8x spread
    in both directions. Reading it lets the optimizer size the threshold to the machine.

    Reads ``cpu0``'s own cache hierarchy, so on a chiplet design it reports the L3 *shared by
    the cores that probe one bucket* (per-CCX), which is exactly the residency that matters,
    not the socket total. Linux-only (`/sys`); `0` elsewhere so the caller keeps its default.
    """
    best = 0
    for idx in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        try:
            with open(os.path.join(idx, "level")) as f:
                if f.read().strip() != "3":
                    continue
            with open(os.path.join(idx, "size")) as f:
                best = max(best, _parse_cache_size(f.read().strip()))
        except (OSError, ValueError):
            continue
    return best


def _parse_cache_size(raw: str) -> int:
    """Parse a `/sys` cache size like ``"16384K"`` / ``"32M"`` / ``"1G"`` into bytes."""
    raw = raw.strip()
    if not raw:
        return 0
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    mult = units.get(raw[-1].upper(), 1)
    return int(raw[:-1] if mult > 1 else raw) * mult


@functools.lru_cache(maxsize=1)
def machine_memory_bytes() -> int:
    """The memory ceiling this process runs under: `min(host RAM, cgroup limit)`, or `0`.

    The neutral hardware fact behind every memory-sizing decision. `min` because a container's
    cgroup cap — not the host's RAM — is the real ceiling: sizing to host RAM over-commits and
    gets the cgroup OOM-killed. Fixed for the process's lifetime, so memoized.

    (Carbonite's `pressure.total_memory_bytes` computes the same ceiling for its own live
    pressure sensing; this is the copy the layers that cannot import Carbonite — notably Kyber
    — read, so the planner can size to real memory without reaching across a subsystem boundary.)
    """
    host = 0
    try:
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        host = 0
    limits = [host] if host > 0 else []
    for base in cgroup_v2_dirs():
        cap = read_cgroup_bytes(os.path.join(base, "memory.max"))
        if cap is not None:
            limits.append(cap)
    v1 = read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None:
        limits.append(v1)
    return min(limits) if limits else 0


def read_cgroup_bytes(path: str) -> int | None:
    """A byte-valued cgroup file, or `None` when absent, unlimited (`max`), or a v1 sentinel.

    Public within `_internal` because Carbonite's pressure monitor reads the same files.
    The *policy* above it is deliberately duplicated per layer (see `machine_memory_bytes`),
    but this parser is pure file-format mechanics with no layer-specific meaning — one copy.
    """
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw in ("", "max"):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 reports a near-2^63 sentinel when unlimited; treat huge/non-positive as none.
    if value <= 0 or value >= (1 << 62):
        return None
    return value


def cpu_contention() -> dict[str, float]:
    """How much of this machine's CPU is being taken by *other* work, right now.

    Low CPU utilization during a query has two completely different causes with opposite
    fixes: the engine failed to parallelize (fix the plan), or the cores were never available
    because something else was using them (fix the placement, or stop trusting the timing).
    Nothing distinguishes them from inside the process — a query pinned to 1 of 15 cores by a
    noisy co-tenant looks exactly like a query that only asked for 1 core. This reports the
    outside view so a diagnosis can tell them apart.

    Keys: ``load_per_core`` (1-minute run-queue length over [`available_cpu_count`]; ``1.0``
    means the box is exactly saturated, ``>1.0`` oversubscribed) and ``throttled_ratio``
    (share of CFS periods the cgroup quota throttled us). Omits any key the platform cannot
    report, rather than substituting a zero that would read as "no contention".

    Deliberately does **not** try to net out this process's own contribution. Load average
    counts *runnable* tasks while a process can only cheaply report its total thread count,
    most of them asleep; subtracting one from the other would over-subtract an idle worker
    pool into a negative reading. The caller already knows its own CPU use — comparing that
    against `load_per_core` is the sound way to attribute the difference.

    Returns:
        The measurable contention signals, each omitted when unavailable.
    """
    out: dict[str, float] = {}
    with contextlib.suppress(OSError, AttributeError):  # not Linux, or /proc unavailable
        out["load_per_core"] = os.getloadavg()[0] / available_cpu_count()
    throttled = _cgroup_throttled_ratio()
    if throttled is not None:
        out["throttled_ratio"] = throttled
    return out


# Device nodes of accelerators that are not NVIDIA GPUs: Google TPU (`/dev/accel*`, and
# `/dev/vfio` for v5+), AWS Neuron / Trainium / Inferentia (`/dev/neuron*`), and Intel
# Gaudi (`/dev/hl*`). Used only to *refuse* the cheap negative — presence here means "ask
# properly", never "an accelerator is usable".
_ACCELERATOR_DEVICE_GLOBS = (
    "/dev/accel[0-9]*",
    "/dev/neuron[0-9]*",
    "/dev/hl[0-9]*",
    "/dev/vfio/[0-9]*",
)


@functools.lru_cache(maxsize=1)
def gpu_devices_absent() -> bool:
    """True when this host demonstrably has no GPU, decided without importing a framework.

    Answers only the *cheap negative*. Proving a GPU is present needs the vendor runtime, but
    proving one is absent usually does not, and that asymmetry is worth exploiting: the natural
    way to ask "is there a GPU" is `torch.cuda.is_available()`, which costs ~2 s of import on
    first call — paid on the first query of every GPU-less run, to learn there is nothing to
    accelerate. A `stat` of a device node answers the same question in microseconds.

    Deliberately conservative: it returns True only on Linux, where the vendor device nodes are
    authoritative, and only when every one of them is missing. Anywhere else — macOS, whose
    Metal devices have no node, or an unrecognized accelerator — it returns False, meaning "ask
    properly", so the cheap path can never produce a false negative on a machine that has a
    device. An explicitly empty ``CUDA_VISIBLE_DEVICES`` is honored as a definitive no.

    Returns:
        True only if a real probe is certain to find nothing.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return True  # the user masked every device; nothing to find
    if not sys.platform.startswith("linux"):
        return False  # no authoritative node to check — make the caller probe for real
    # Numbered *device* nodes, not the driver's control node: a GPU-less machine built from a
    # GPU-capable cloud image has `/dev/nvidiactl` (the driver is loaded) and no `/dev/nvidia0`
    # at all. Keying on the control node would call that host GPU-equipped and re-introduce the
    # very import this exists to avoid, on exactly the fleet where it is most common.
    if glob.glob("/dev/nvidia[0-9]*"):
        return False
    # Non-NVIDIA accelerators expose their own nodes. Without these, a TPU, Trainium, or
    # Gaudi host answered "definitely nothing here" — a *false* negative, which is the one
    # answer this function promises never to give, and it suppressed the real probe on the
    # machines that most need it. Globs, because these are numbered per device.
    if any(glob.glob(pattern) for pattern in _ACCELERATOR_DEVICE_GLOBS):
        return False
    return not any(os.path.exists(p) for p in ("/dev/kfd", "/dev/dxg"))


def accelerator_backend() -> str:
    """The accelerator this host can compute on: ``cuda``/``rocm``/``xpu``/``mps``/``tpu``/
    ``neuron``/``hpu``/``cpu``.

    A hardware *fact*, so it lives here (the executor picks a device without importing `ml`);
    `ml.gpu.detect_backend` re-exports it. Detected via torch where one exists (ROCm through
    the CUDA API with ``torch.version.hip``; Intel ``torch.xpu``; Apple MPS; Cloud TPU
    ``torch_xla``) and via device nodes for Trainium/Inferentia (``neuron``) and Gaudi
    (``hpu``), whose frameworks are expensive to import. Naming the specific backend rather
    than ``cpu`` lets such a host self-identify for diagnostics and `torch_device`.
    """
    if gpu_devices_absent() and not _tpu_available():
        return "cpu"  # cheap negative first: `torch.cuda.is_available()` costs a ~2 s import
    try:
        import torch

        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass  # no torch → fall through to the device-node accelerators
    from batcher._internal.accelerators import has_gaudi_device, has_neuron_device

    if _tpu_available():
        return "tpu"
    if has_neuron_device():
        return "neuron"
    return "hpu" if has_gaudi_device() else "cpu"


def _tpu_available() -> bool:
    """Whether a Cloud TPU is present, via `torch_xla` — import-gated and side-effect-free.

    `find_spec` avoids importing (and so initializing) the XLA runtime on the common
    no-TPU host; only when `torch_xla` is actually installed do we ask its runtime for the
    device type. Any failure (older API, no device) reads as "no TPU"."""
    import importlib.util

    if importlib.util.find_spec("torch_xla") is None:
        return False
    try:
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        return xr.device_type() == "TPU"
    except Exception:
        return False


def gpu_inventory() -> list[dict[str, object]]:
    """Visible GPUs as ``{"index", "name", "memory_bytes"}``, or `[]` when none/undetectable.

    **Inventory only, deliberately.** This answers "what devices are attached", which is a
    hardware fact and so belongs in this neutral layer where any package may read it. It does
    *not* decide how to use them — VRAM budgeting, actors-per-GPU, backend selection, and the
    autocast policy all live in `ml.gpu`, which is the executor-facing owner of those
    decisions. Keeping the split at inventory-vs-policy is what stops this from becoming a
    second, competing GPU module: there is one place that says what exists and one that says
    what to do about it.

    Best-effort and dependency-free at import: NVML first (accurate, no CUDA context, and it
    sees devices even when no framework is installed), then torch as a fallback. Both are
    optional, so a CPU-only or stripped-down install gets `[]` rather than an error.

    Returns:
        One dict per visible device, in device order.
    """
    return _nvml_inventory() or _torch_inventory() or _other_accelerator_inventory()


def _nvml_inventory() -> list[dict[str, object]]:
    """GPUs via NVML, or `[]`. Preferred: no CUDA context, and no framework required."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            out: list[dict[str, object]] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                out.append(
                    {
                        "index": index,
                        "name": name.decode() if isinstance(name, bytes) else str(name),
                        "memory_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    }
                )
            return out
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - NVML absent or no NVIDIA driver
        return []


def _torch_inventory() -> list[dict[str, object]]:
    """GPUs via torch, or `[]`. The fallback when NVML is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception:  # pragma: no cover - torch absent or CUDA unusable
        return []


def _other_accelerator_inventory() -> list[dict[str, object]]:
    """Non-NVIDIA accelerators, so diagnostics stop reporting `[]` on a machine that has one.

    NVML and `torch.cuda` between them cover NVIDIA and (via HIP) AMD; everything else fell
    through to an empty list, which the observability page then rendered as "no GPUs" on a
    TPU, Trainium, or Gaudi host. Reports what the device nodes say, since there is no
    portable cross-vendor API and no memory figure to be had without each vendor's runtime —
    an accurate name with unknown memory beats a confident, wrong "nothing here".
    """
    devices: list[dict[str, object]] = []
    for kind, pattern in (
        ("TPU", "/dev/accel[0-9]*"),
        ("Neuron", "/dev/neuron[0-9]*"),
        ("Gaudi", "/dev/hl[0-9]*"),
    ):
        for index, node in enumerate(sorted(glob.glob(pattern))):
            devices.append(
                {"index": index, "name": f"{kind} ({os.path.basename(node)})", "memory_bytes": 0}
            )
    return devices


def process_start_method_context():
    """A `multiprocessing` context using the best start method this platform offers.

    Preference order is ``forkserver`` (cheap children, no fork-safety hazards from the
    parent's threads), then ``fork``, then ``spawn``. Picking the *first available* rather
    than assuming a fallback matters on Windows, which offers only ``spawn``: the previous
    ``forkserver`` -> ``fork`` fallback raised `ValueError` there, so any pipeline reaching
    a process pool crashed on a platform the package otherwise supports.

    Returns:
        A `multiprocessing` context whose start method this platform actually provides.
    """
    import multiprocessing as mp

    available = mp.get_all_start_methods()
    for method in ("forkserver", "fork", "spawn"):
        if method in available:
            return mp.get_context(method)
    return mp.get_context()  # platform default; every platform provides one
