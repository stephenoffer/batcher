"""Accelerator detection + utilization feedback — the adaptive half of scheduling.

Declaring `map_batches(..., num_gpus=1)` gets a GPU inference stage *placed* on a
GPU worker. This module closes the loop: the actors measure how busy the device
actually was, that utilization is persisted to the MetadataHub keyed by the
pipeline, and the next run's `num_gpus` request adapts — packing more tasks onto a
fraction of a GPU when it sat idle, or asking for a whole GPU when it saturated.
This is "num_gpus based on utilization", measured and consumed, not guessed.

**Vendor-agnostic.** `detect_backend` / `torch_device` / `vram_context_overhead` cover
NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple (MPS), and Cloud TPU, with a CPU
fallback. Utilization feedback is available where the vendor exposes a counter —
NVIDIA (NVML), AMD (ROCm SMI), Intel (`torch.xpu.utilization`); Apple and TPU have no
stable per-process API, so their loop is a no-op (the declared `num_gpus` stands), but
MPS still drives VRAM-based packing via its unified-memory budget. Any measurement
failure (no driver, no SMI, no device) yields `None`. Recommendation/persistence are
pure.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.hardware import INFERENCE_INFLIGHT_DEPTH_MAX, available_cpu_count
from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.metadata.hardware_scope import scoped
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "SustainedUtilization",
    "actors_per_gpu_from_learned_vram",
    "autocast_call",
    "detect_backend",
    "gpu_aware_pool_default",
    "gpu_feedback_key",
    "gpu_vram_gb",
    "inference_mode_call",
    "inference_vram_multiplier",
    "load_gpu_peak_vram",
    "load_gpu_utilization",
    "max_actors_per_gpu",
    "recommend_gpu_fraction",
    "recommend_inference_dtype",
    "recommend_inflight_depth",
    "recommend_num_gpus",
    "recommend_quantization",
    "record_gpu_peak_vram",
    "record_gpu_utilization",
    "resolve_num_workers",
    "sample_gpu_utilization",
    "sample_gpu_vram_fraction",
    "torch_device",
    "vram_context_overhead",
]


def resolve_num_workers(num_workers: int | str, num_gpus: float) -> int:
    """Resolve ``num_workers="auto"`` (the ML default) to a concrete per-worker count.

    Auto means: a GPU stage keeps **one** model / CUDA context per worker (GPU scale-out
    is the distributed actor pool's job, not intra-worker threads), and a CPU stage fans
    the per-batch calls across **all local cores** — so inference is parallel by default
    instead of single-threaded (the Ray Data foot-gun). Threads only speed up a
    GIL-releasing `fn` (Arrow / NumPy / torch); a GIL-bound pure-Python `fn` should pass
    ``multiprocessing=True`` to use those cores across processes. An explicit int wins.
    """
    if num_workers != "auto":
        return max(1, _as_worker_count(num_workers))
    if num_gpus > 0:
        return 1
    return available_cpu_count()  # usable local cores (cgroup/affinity aware), not host count


def _as_worker_count(num_workers: object) -> int:
    """Coerce an explicit `num_workers` to an int, or say what it should have been.

    The only accepted values are ``"auto"`` and an integer, but anything `int()` could
    parse got through and anything it could not raised `int()`'s own message —
    ``invalid literal for int() with base 10: 'AUTO'`` — which names neither the parameter
    nor the two things it takes. ``"AUTO"`` is the mistake worth catching by name: it is a
    plausible spelling of the default, and it failed as a parse error rather than as an
    unrecognized option.
    """
    # A float was never a documented value, but `int()` accepted one, so `num_workers=4.0`
    # has always worked. Keep it working when it is integral and reject it only when
    # truncating would silently change the answer — rejecting 4.0 outright would break
    # callers for no benefit, and accepting 2.5 as 2 is the surprise worth stopping.
    if isinstance(num_workers, float) and not isinstance(num_workers, bool):
        if num_workers.is_integer():
            return int(num_workers)
        raise PlanError(
            f"num_workers must be 'auto' or a whole number, got {num_workers!r}; it would "
            f"otherwise be truncated to {int(num_workers)}."
        )
    if isinstance(num_workers, bool) or not isinstance(num_workers, (int, str)):
        raise PlanError(
            f"num_workers must be 'auto' or an integer, got {type(num_workers).__name__} "
            f"{num_workers!r}."
        )
    try:
        return int(num_workers)
    except ValueError:
        hint = " Did you mean 'auto'?" if str(num_workers).strip().lower() == "auto" else ""
        raise PlanError(
            f"num_workers must be 'auto' or an integer, got {num_workers!r}.{hint}"
        ) from None


def _accelerator_pool_size(resources: dict[str, float] | None, num_partitions: int) -> int | None:
    """Actor-pool size that fills a cluster's *custom* accelerators (TPU / Trainium / Gaudi).

    The GPU foot-gun the pool default avoids — one actor idling a whole cluster — is identical
    for a non-GPU accelerator, which Ray reports as a named resource rather than as `GPU`. A
    stage asking `{"TPU": 4}` should spawn ``cluster_TPU / 4`` replicas so every chip runs one.
    Bounded by the *most* constraining requested resource (a stage needing both `TPU` and a
    scarce custom resource fills the scarcer), and by the partition count. Returns `None` — so
    the caller keeps its worker-count fallback — when no accelerator is requested, the cluster
    advertises none of a requested one (nothing to fill), or Ray is unreadable.
    """
    if not resources:
        return None
    try:
        import ray

        avail = ray.cluster_resources()
    except Exception:
        return None
    replicas: int | None = None
    for name, per in resources.items():
        if per <= 0:
            continue
        total = float(avail.get(name, 0.0))
        if total <= 0:  # a requested accelerator the cluster doesn't have → don't guess
            return None
        fit = int(total / per)
        replicas = fit if replicas is None else min(replicas, fit)
    if not replicas or replicas <= 0:
        return None
    return max(1, min(num_partitions, replicas))


def gpu_aware_pool_default(
    num_gpus: float,
    fallback: int,
    num_partitions: int,
    accelerator_type: str | None = None,
    *,
    tensor_parallel_size: int = 1,
    resources: dict[str, float] | None = None,
) -> int:
    """Default distributed actor-pool size when `concurrency` is unset.

    For a GPU stage, size the pool to the cluster's GPUs so *every* GPU gets an actor
    (replicas = total_GPUs / per-actor `num_gpus`) — never one engine idling a multi-GPU
    cluster (the Ray Data ``concurrency=1`` foot-gun). A stage on a **custom** accelerator
    (TPU / Trainium / Inferentia / Gaudi, which carries `num_gpus == 0` plus a named
    `resources` request) is sized the same way against that resource's cluster total, so a
    TPU pod fills its chips rather than running one actor. For a plain CPU stage, keep the
    cluster worker count (`fallback`). Clamped to the partition count (no idle actors); falls
    back when Ray reports no matching accelerator.

    When the stage is pinned to an `accelerator_type` on a **heterogeneous** cluster
    (mixed GPU classes), size against *that class's* GPUs — Ray tags them as the
    ``accelerator_type:<NAME>`` resource — so a stage pinned to the 4 A100s never spawns
    actors for the 8 T4s it can't run on. Taken as a `min` with the total GPU count, so
    an absent or sentinel typed resource only ever sizes *down* (never over-subscribes).

    A **tensor-parallel** engine (vLLM ``tensor_parallel_size=N``) spawns N GPU workers per
    replica, so one actor consumes ``num_gpus * N`` GPUs even though it declares `num_gpus`.
    Sizing on `num_gpus` alone therefore spawns N times too many replicas and the pool never
    schedules. Pass `tensor_parallel_size` and the replica count becomes Ray's documented
    ``available_GPUs / tensor_parallel_size``. The default of 1 leaves every existing caller
    unchanged.
    """
    if num_gpus <= 0:
        pool = _accelerator_pool_size(resources, num_partitions)
        return pool if pool is not None else fallback
    per_replica = num_gpus * max(1, tensor_parallel_size)
    try:
        import ray

        resources = ray.cluster_resources()
        total = float(resources.get("GPU", 0.0))
        if accelerator_type:
            # The topology first, and NOT `resources["accelerator_type:<MODEL>"]`. That
            # resource reads like a device count and is a per-node constraint *marker*: Ray
            # sets it to exactly 1 per node (`_private/resource_and_label_spec.py`) and a task
            # requests 0.001 of it. On a four-node, sixteen-device T4 fleet it totals 4.0
            # against `GPU`'s 16.0, so `min` sized this pool to the number of GPU *nodes* and
            # left three quarters of the devices without an actor — silently, and only for
            # callers who pinned a model to be precise about which devices they get.
            from batcher.dist.executors.ray_runtime.fabric.topology import devices_of_class

            typed = float(devices_of_class(accelerator_type))
            if typed <= 0:
                # No readable topology (Ray down mid-call, a stubbed cluster, a fleet whose
                # nodes carry no accelerator label). The marker is not a device count, but it
                # is still a *bound* — one per node is never more than the devices on it — so
                # falling back to it keeps a pinned stage from being sized against the whole
                # cluster, which is the one direction that over-provisions.
                typed = float(resources.get(f"accelerator_type:{accelerator_type}", 0.0))
            if typed > 0:
                total = min(total, typed)
    except Exception:
        return fallback
    if total <= 0:
        return fallback
    return max(1, min(num_partitions, int(total / per_replica)))


_NAMESPACE = "ml.gpu"
# Below this measured utilization a whole-GPU task is wasting the device, so pack
# more tasks onto a fraction of it; above the saturation mark, give it a whole GPU.
_PACK_BELOW = 0.5
_SATURATED_ABOVE = 0.9
# Don't fragment a GPU finer than this (avoids requesting unschedulable slivers).
# Raised back from 0.25: the old value silently CAPPED packing at 4 actors/GPU, so a
# 0.1 GB embedding model that VRAM-fits 36 actors on an 80 GB card still got 4 and
# stranded ~90% of the device. `max_actors_per_gpu` has already proven the slice fits in
# VRAM, so the only job left for a floor is keeping the request schedulable; 0.05 (20
# actors/GPU) is the density past which per-actor CUDA contexts and SM time-slicing stop
# paying, and Ray schedules fractions this small without trouble.
_MIN_FRACTION = 0.05
# `_PACK_BELOW`/`recommend_num_gpus` keeps the coarser 0.25 floor: that path packs from
# *utilization*, which says nothing about whether the model's weights fit in the slice.
_UTIL_MIN_FRACTION = 0.25
# Bounds on the adaptive per-actor submit-ahead depth (how many partitions an inference
# actor keeps in flight). A starved GPU (low measured utilization) submits deeper to
# overlap the dispatch/gather round-trip; a saturated one keeps the shallow default. The
# ceiling is shared with the distributed actor pool via the neutral `_internal.hardware`
# so the two paths cannot drift (the alias keeps the short local name at the call sites).
_INFLIGHT_DEPTH_MAX = INFERENCE_INFLIGHT_DEPTH_MAX
# Above this measured peak-VRAM fraction the device has too little headroom to hold several
# partitions in flight, so the submit-ahead depth stays shallow regardless of utilization.
_VRAM_TIGHT = 0.8
# Per-vendor VRAM a process reserves for its runtime context before any model loads —
# the overhead that makes packing many tiny models less dense than naive math. MPS
# shares unified memory (no separate context reserve); TPU/CPU have none.
_CONTEXT_OVERHEAD_GB = {
    "cuda": 0.4,
    "rocm": 0.5,
    "xpu": 0.3,
    "mps": 0.0,
    "tpu": 0.0,
    # Neuron (Trainium/Inferentia) and Gaudi manage device memory through their own runtimes
    # rather than a CUDA-style context reserve, so no separate host-side overhead is budgeted.
    "neuron": 0.0,
    "hpu": 0.0,
    "cpu": 0.0,
}
# Budget peak inference VRAM at ~1.5x model size (activations + batch tensors). This is
# the multiplier at the REFERENCE workload below; `inference_vram_multiplier` scales it
# with the three things that actually drive peak activation memory.
_INFERENCE_VRAM_MULTIPLIER = 1.5
# The workload the flat 1.5x was implicitly calibrated for: 32 rows of 512-token sequences
# in fp16. At exactly this point the scaled multiplier still returns 1.5, so every caller
# that passes no workload information is unchanged.
_REF_BATCH_ROWS = 32
_REF_SEQ_LEN = 512
_REF_DTYPE_BYTES = 2
# Activations are the part of the 1.5x that scales; the remaining 1.0x is the weights,
# which do not depend on the batch at all.
_ACTIVATION_SHARE = _INFERENCE_VRAM_MULTIPLIER - 1.0
# Clamp the scaled multiplier. The low bound keeps a tiny batch from budgeting below the
# weights plus a working slice; the high bound stops a huge declared sequence length from
# collapsing packing to 1 actor on a model whose activations are checkpointed or paged.
_MIN_VRAM_MULTIPLIER = 1.1
_MAX_VRAM_MULTIPLIER = 4.0


def inference_vram_multiplier(
    *,
    batch_rows: int | None = None,
    seq_len: int | None = None,
    activation_dtype_bytes: int | None = None,
) -> float:
    """Peak-inference VRAM as a multiple of model size, scaled by the workload.

    Peak VRAM during inference is the weights plus the activation working set, and the
    activations scale with the batch size, the sequence length, and the width of the
    activation dtype — none of which a flat multiplier can see. A fixed 1.5x therefore
    over-packs a long-context fp32 job into an OOM and under-packs a short fp16 one.

    The scaling is anchored so that the reference workload (32 rows, 512 tokens, fp16)
    returns exactly 1.5, the value this was before. Omit an argument and that dimension
    contributes its reference value, so passing nothing is the old behavior.

    Args:
        batch_rows: rows per forward pass; activations scale linearly with it.
        seq_len: tokens per row for a sequence model; 1 for a fixed-shape vision model.
        activation_dtype_bytes: 2 for fp16/bf16, 4 for fp32.

    Returns:
        The multiplier, clamped to ``[1.1, 4.0]``.

    Examples:
        .. doctest::

            >>> from batcher.ml.gpu import inference_vram_multiplier
            >>> inference_vram_multiplier()
            1.5
            >>> inference_vram_multiplier(batch_rows=32, seq_len=512)
            1.5
            >>> inference_vram_multiplier(batch_rows=128, seq_len=2048) > 1.5
            True
    """
    rows = _REF_BATCH_ROWS if batch_rows is None or batch_rows <= 0 else batch_rows
    tokens = _REF_SEQ_LEN if seq_len is None or seq_len <= 0 else seq_len
    width = (
        _REF_DTYPE_BYTES
        if activation_dtype_bytes is None or activation_dtype_bytes <= 0
        else activation_dtype_bytes
    )
    scale = (rows / _REF_BATCH_ROWS) * (tokens / _REF_SEQ_LEN) * (width / _REF_DTYPE_BYTES)
    return min(_MAX_VRAM_MULTIPLIER, max(_MIN_VRAM_MULTIPLIER, 1.0 + _ACTIVATION_SHARE * scale))


def detect_backend() -> str:
    """The accelerator backend, one of ``cuda`` / ``rocm`` / ``xpu`` / ``mps`` / ``tpu`` /
    ``neuron`` / ``hpu`` / ``cpu``.

    Thin re-export of `_internal.hardware.accelerator_backend`. The detection itself is a
    hardware fact and lives in the neutral layer so `core` can use it to place the
    relational GPU kernels without importing this user-facing package; keeping one
    implementation is what stops the executor and the ML layer from disagreeing about
    which device is present.
    """
    from batcher._internal.hardware import accelerator_backend

    return accelerator_backend()


def torch_device(backend: str | None = None) -> str:
    """The torch device string for `backend` (default: detected) — what ``.to(...)`` wants.

    ROCm uses the ``cuda`` device string (HIP shims the CUDA API); Intel is ``xpu``,
    Apple ``mps``, a TPU and AWS Neuron (Trainium/Inferentia) are ``xla`` (both run through
    torch_xla), Intel Gaudi is ``hpu`` (habana_frameworks), and CPU ``cpu``.

    An unrecognized backend degrades to ``cpu`` rather than raising. This is a mapping of
    *names*, and the names come from places this function does not control — a caller
    naming an accelerator we have not taught it about, or a newer `detect_backend`. A
    `KeyError` from a device-string lookup would abort a job that could have run correctly,
    if slower, on the CPU; the whole point of the accelerator layer is that it is an
    optimization.
    """
    b = detect_backend() if backend in (None, "auto") else backend
    return {
        "cuda": "cuda",
        "rocm": "cuda",
        "xpu": "xpu",
        "mps": "mps",
        "tpu": "xla",
        "neuron": "xla",  # torch-neuronx runs through XLA, same as Cloud TPU
        "hpu": "hpu",  # Intel Gaudi via habana_frameworks
        "cpu": "cpu",
    }.get(b, "cpu")


def vram_context_overhead(backend: str | None = None) -> float:
    """Per-process runtime-context VRAM overhead (GB) for `backend` (default: detected)."""
    return _CONTEXT_OVERHEAD_GB.get(backend or detect_backend(), 0.4)


@functools.cache
def _nvml() -> Any | None:
    """The initialized NVML module (`pynvml`) for this process, or `None`.

    NVML is initialized **once** and held open for the process lifetime rather than
    `nvmlInit`/`nvmlShutdown` around every sample: VRAM and utilization are sampled
    per batch (the throughput autobatcher's live cap, the adaptive `num_gpus` loop),
    so a driver init/shutdown handshake per call is pure per-batch overhead. The
    session is refcounted and released at process exit. `None` when `pynvml` is
    absent, NVML won't initialize, or the host exposes no devices — a stable fact for
    a worker process, so caching the negative is correct."""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            return None
        return pynvml
    except Exception:
        return None


@functools.cache
def _nvml_handles() -> tuple[Any, ...]:
    """Per-device NVML handles for this process (empty when NVML is unavailable).

    Cached alongside the session: a handle lookup is cheap, but resolving it once keeps
    the per-sample path a bare counter read with no repeated device enumeration."""
    nvml = _nvml()
    if nvml is None:
        return ()
    try:
        return tuple(nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount()))
    except Exception:
        return ()


# The env vars a scheduler pins a GPU actor's *visible* devices through, in priority order:
# NVIDIA/HIP honor CUDA_VISIBLE_DEVICES; AMD ROCm adds HIP_ / ROCR_VISIBLE_DEVICES. The first
# one set names this process's devices — a host is one vendor, so checking all is safe.
_VISIBLE_DEVICE_ENVS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def _visible_device_indices(n_handles: int) -> tuple[int, ...]:
    """Physical device indices this process can actually see, honoring the vendor visibility env.

    Ray pins each GPU actor by setting a ``*_VISIBLE_DEVICES`` var to its assigned device(s), so
    a sample must attribute VRAM/utilization to *those* physical devices — not to physical 0 or
    to a node-wide mean that a co-located actor's idle or busy device would distort. Returns every
    in-range physical index named by the first env var set (a multi-GPU actor sees several), or
    **all** devices when none is set/parseable (an unpinned driver or monitor — the safe historical
    behavior).

    Resolution goes through Carbonite's device scope first, which understands the two forms this
    used to discard. Ray writes ordinals, but the **Kubernetes device plugin writes UUIDs** and
    **MIG writes partition handles**, and an ordinal-only parse treated both as unparseable and
    fell through to "every device on the node" — so on exactly the fleets that pin hardest, every
    pinned actor sampled the whole node and averaged its own busy device with its neighbours' idle
    ones. The ordinal parse remains as the fallback for a vendor whose devices the NVML-backed
    scope cannot enumerate (AMD), where it is the behavior this always had.
    """
    import os

    from batcher._internal.hardware.devices import visible_device_indices

    resolved = tuple(i for i in visible_device_indices() if i < n_handles)
    if resolved:
        return resolved
    for env in _VISIBLE_DEVICE_ENVS:
        raw = os.environ.get(env, "").strip()
        if not raw:
            continue
        toks = (t.strip() for t in raw.split(","))
        idxs = tuple(int(t) for t in toks if t.isdigit() and int(t) < n_handles)
        if idxs:
            return idxs
    return tuple(range(n_handles))


def _vram_handle() -> Any | None:
    """The NVML handle for this process's *visible* GPU 0, honoring ``CUDA_VISIBLE_DEVICES``.

    Falls back to physical device 0 when the pin resolves to nothing in range, so a bad env
    never breaks the sample — it only reverts to the old behavior. `None` when NVML exposes
    no devices.
    """
    handles = _nvml_handles()
    if not handles:
        return None
    return handles[_visible_device_indices(len(handles))[0]]


def _bound_ordinal(backend: Any) -> int:
    """The device ordinal `backend` is currently computing on, or `0` when it cannot say.

    A worker pinned to two boards runs on one at a time, and which one is a framework fact
    (`set_device` moves it) rather than an environment fact — so reading properties off
    ordinal 0 is right only by coincidence, and on a mixed node it is a different card's
    capacity. Tolerates a backend with no `current_device` (older torch builds, and the
    minimal fakes the XPU tests stand in for a driver), which is exactly the case where `0`
    is the only ordinal there is.
    """
    getter = getattr(backend, "current_device", None)
    if getter is None:
        return 0
    try:
        return int(getter())
    except Exception:
        return 0


def gpu_vram_gb() -> float | None:
    """Total VRAM (GB) of the **smallest** device this process can see, or `None` if unknown.

    Used to VRAM-pack inference actors. Tries the vendor SMI (NVML) first, then torch's
    device properties (covers CUDA/ROCm/XPU); returns `None` on a host with no
    accelerator (e.g. a GPU-less driver), where packing is simply skipped.

    The smallest rather than the first, because this figure sizes one actor-count for the whole
    stage and a node's devices are not always the same size — a fleet part-way through an
    upgrade, or a box with an A100 beside an L4. Packing to the larger card's capacity produces
    a replica count that fits on some devices and OOMs on the rest, which surfaces as a job
    that fails only on certain nodes. On the ordinary homogeneous node every device reports the
    same number and this is exactly what it always returned."""
    from batcher._internal.hardware.devices import min_visible_capacity_bytes

    smallest = min_visible_capacity_bytes()
    if smallest:
        return smallest / (1 << 30)
    handle = _vram_handle()  # this process's visible device, not physical 0
    if handle is not None:  # NVML reports total memory without allocating a CUDA context
        try:
            return _nvml().nvmlDeviceGetMemoryInfo(handle).total / (1 << 30)
        except Exception as exc:
            note_suppressed("ml", "read total VRAM from NVML", exc)
    try:
        import torch

        if torch.cuda.is_available():
            # `current_device()`, not `0`: a worker that has been `set_device`d onto its second
            # visible board reads a different card's capacity from physical 0, and on a mixed
            # node that is a different number.
            return torch.cuda.get_device_properties(_bound_ordinal(torch.cuda)).total_memory / (
                1 << 30
            )
        # Intel XPU: `cuda.is_available()` is False here, so without this branch the
        # docstring's XPU claim was empty and an Intel GPU packed nothing.
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return xpu.get_device_properties(_bound_ordinal(xpu)).total_memory / (1 << 30)
        # Apple MPS shares unified memory; `recommended_max_memory` is the working
        # budget torch will use before paging — the right number to pack against.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.mps.recommended_max_memory() / (1 << 30)
    except Exception as exc:
        note_suppressed("ml", "read total accelerator memory", exc)
    return None


def _own_budget_fraction(used_by_pid: float, total: float, n_processes: int) -> float | None:
    """This process's VRAM use as a fraction of its **fair share** of a shared device.

    The number an actor must react to when several inference actors are packed onto one
    GPU. Reading the device-wide ``used`` instead makes every co-tenant observe the
    *aggregate*, so a single actor growing its batch pushes all of them over the cap and
    they all shrink together — a synchronized oscillation down to the minimum batch size
    that no actor caused and none can escape. Dividing the device between its `n_processes`
    tenants gives each actor a private budget, so it responds only to its own allocations.

    Degenerates exactly to ``used / total`` for a sole tenant, so the single-actor path is
    unchanged.

    Args:
        used_by_pid: bytes this process has allocated on the device.
        total: the device's total VRAM in bytes.
        n_processes: compute processes resident on the device (clamped to >= 1).

    Returns:
        The fraction of this process's budget in use, or `None` when `total` is zero.
    """
    if total <= 0:
        return None
    budget = total / max(1, n_processes)
    return used_by_pid / budget if budget else None


def _nvml_own_vram_fraction(handle: Any) -> float | None:
    """`_own_budget_fraction` from NVML's per-process accounting, or `None` if unavailable.

    `None` (rather than a guess) whenever NVML cannot attribute memory per process — an
    older driver with no ``nvmlDeviceGetComputeRunningProcesses``, or a container without
    the PID namespace visibility it needs — so the caller falls back to the device-wide
    reading instead of silently reporting zero usage."""
    nvml = _nvml()
    query = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None)
    if query is None:
        return None
    try:
        import os

        procs = query(handle)
        total = nvml.nvmlDeviceGetMemoryInfo(handle).total
        pid = os.getpid()
        # `usedGpuMemory` is None when the driver won't attribute it (permission, MIG).
        # Such an entry still proves a tenant exists, so it counts toward the divisor.
        accounted = [p for p in procs if getattr(p, "usedGpuMemory", None) is not None]
        if not accounted:
            return None
        used = sum(float(p.usedGpuMemory) for p in accounted if getattr(p, "pid", None) == pid)
        return _own_budget_fraction(used, float(total), len(procs))
    except Exception:
        return None


def sample_gpu_vram_fraction() -> float | None:
    """Fraction (0..1) of this process's VRAM budget in use, or `None` without a GPU.

    Feeds the throughput autobatcher's VRAM cap so it shrinks (or refuses to grow) the
    batch *before* an out-of-memory rather than catching one after the fact. Prefers NVML's
    **per-process** accounting scaled to this process's share of the device
    (`_own_budget_fraction`), so packed inference actors don't all react to each other's
    allocations; falls back to the device-wide reading, then to torch's reserved memory.
    Returns `None` on a GPU-less host, where the guard is simply inert."""
    handle = _vram_handle()
    if handle is not None:
        own = _nvml_own_vram_fraction(handle)
        if own is not None:
            return own
        try:
            info = _nvml().nvmlDeviceGetMemoryInfo(handle)
            return info.used / info.total if info.total else None
        except Exception as exc:
            note_suppressed("ml", "read used VRAM from NVML", exc)
    try:
        import torch

        if torch.cuda.is_available():
            # The *current* device throughout, not physical 0. A worker pinned to its second
            # visible board was dividing its own reserved bytes by another card's capacity —
            # a ratio of two different devices, which on a mixed node is not even a fraction
            # of anything. On the single-GPU host these develop on, the two are the same.
            device = _bound_ordinal(torch.cuda)
            total = torch.cuda.get_device_properties(device).total_memory
            # torch's reserved bytes are already this process's own allocator, so this
            # path needs no per-process attribution.
            return torch.cuda.memory_reserved(device) / total if total else None
        # Intel XPU: without this branch the predictive VRAM cap was inert on Intel GPUs,
        # so the throughput hill-climb grew until a hard OOM — the failure the cap prevents.
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            device = _bound_ordinal(xpu)  # the bound board, not physical 0 — see the CUDA case
            total = xpu.get_device_properties(device).total_memory
            return xpu.memory_reserved(device) / total if total else None
        # MPS unified memory: current allocation against the recommended budget.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            total = torch.mps.recommended_max_memory()
            return torch.mps.current_allocated_memory() / total if total else None
    except Exception as exc:
        note_suppressed("ml", "read accelerator memory in use", exc)
    return None


def max_actors_per_gpu(
    model_vram_gb: float,
    gpu_vram_gb: float,
    *,
    headroom: float = 0.2,
    context_overhead_gb: float | None = None,
    inference_multiplier: float | None = None,
    batch_rows: int | None = None,
    seq_len: int | None = None,
    activation_dtype_bytes: int | None = None,
    respect_co_tenants: bool = True,
) -> int:
    """How many inference actors fit on one GPU, VRAM-budgeted.

    Each actor needs ``model_vram_gb * inference_multiplier + context_overhead_gb``;
    usable VRAM leaves `headroom` free for batch data and runtime spikes. At least 1
    (a model that doesn't fit the budget still gets a whole GPU, where it may swap).
    `context_overhead_gb` defaults to the detected vendor's process-context overhead
    (NVIDIA 0.4, AMD 0.5, Intel 0.3, Apple 0.0). This packs a small model and refuses
    to over-subscribe a large one into an OOM.

    `inference_multiplier` defaults to `inference_vram_multiplier` evaluated on
    `batch_rows` / `seq_len` / `activation_dtype_bytes` — the three drivers of peak
    activation memory. Passing none of them reproduces the flat 1.5x this used before.
    An explicit `inference_multiplier` overrides the workload scaling entirely.

    With `respect_co_tenants` (the default) the budget is the device's *free* memory when the
    driver reports less than its capacity, rather than the capacity itself. Packing against
    capacity is correct only on a device this process has to itself, and that is the one case
    where the two figures are equal anyway — everywhere else it counts memory another tenant
    is already holding, which is the single most common way a fractional-GPU stage that
    "obviously fits" OOMs on landing. Pass `False` to size a device that is expected to be
    empty by the time the stage runs.
    """
    if model_vram_gb <= 0 or gpu_vram_gb <= 0:
        return 1
    if inference_multiplier is None:
        inference_multiplier = inference_vram_multiplier(
            batch_rows=batch_rows, seq_len=seq_len, activation_dtype_bytes=activation_dtype_bytes
        )
    overhead = vram_context_overhead() if context_overhead_gb is None else context_overhead_gb
    budget_gb = _free_vram_gb(gpu_vram_gb) if respect_co_tenants else gpu_vram_gb
    usable = budget_gb * (1.0 - headroom)
    per_actor = model_vram_gb * inference_multiplier + overhead
    return max(1, int(usable // per_actor))


def _free_vram_gb(capacity_gb: float) -> float:
    """The device's free memory in GB, or `capacity_gb` when the driver will not say.

    Never larger than the declared capacity: a caller that passed a deliberately reduced
    figure (a MIG slice, a hand-set budget) must not have it widened by a driver reading of
    the whole board.
    """
    from batcher._internal.hardware.devices import device_free_bytes

    free = device_free_bytes()
    if free is None or free <= 0:
        return capacity_gb
    return min(capacity_gb, free / (1 << 30))


def recommend_gpu_fraction(model_vram_gb: float, gpu_vram_gb: float, **kwargs: Any) -> float:
    """The per-actor ``num_gpus`` fraction so several actors share a GPU when the
    model is small, floored at `_MIN_FRACTION` to avoid unschedulable slivers; 1.0
    when only one actor fits. The static counterpart to the measured-utilization
    `recommend_num_gpus` (use this to size a cold start, that to adapt across runs).

    Use *this* for the scheduler's ``num_gpus``, not `max_actors_per_gpu` directly: the
    actors Ray actually packs onto one GPU is ``floor(1 / fraction)``, so the fraction has
    to be the reciprocal of the VRAM-derived count rather than the count itself. The floor
    only guards schedulability — `max_actors_per_gpu` has already proven the slice holds
    the model — so it no longer caps density at 4 actors/GPU and a small model can now use
    the whole device. `kwargs` forward to `max_actors_per_gpu` (including the
    `batch_rows` / `seq_len` / `activation_dtype_bytes` workload scaling)."""
    n = max_actors_per_gpu(model_vram_gb, gpu_vram_gb, **kwargs)
    if n <= 1:
        return 1.0
    return max(_MIN_FRACTION, round(1.0 / n, 2))


def sample_gpu_utilization(backend: str | None = None) -> float | None:
    """Mean accelerator utilization now as a fraction in [0, 1], or `None` if unavailable.

    Dispatches to the vendor's metrics via the `_UTILIZATION` registry: NVML for
    NVIDIA, ROCm SMI for AMD, ``torch.xpu.utilization`` for Intel. Apple MPS and Cloud
    TPU expose no stable per-process utilization API, so they (and CPU) return `None` —
    the loop is then a no-op (the declared `num_gpus` stands). Any failure (no driver,
    no SMI library, no device) also yields `None`."""
    probe = _UTILIZATION.get(backend or detect_backend())
    return probe() if probe is not None else None


def _nvml_utilization() -> float | None:
    """Mean NVIDIA GPU utilization via NVML (`pynvml`); `None` on any failure.

    Reads through the process-wide cached NVML session (`_nvml`) instead of an
    init/shutdown per sample — the adaptive `num_gpus` loop polls this repeatedly."""
    handles = _nvml_handles()
    if not handles:
        return None
    try:
        nvml = _nvml()
        # Average only over the devices this process can see (a pinned actor's own GPU[s]),
        # not every physical device — a co-located idle GPU must not dilute a busy one.
        visible = [handles[i] for i in _visible_device_indices(len(handles))]
        total = sum(nvml.nvmlDeviceGetUtilizationRates(h).gpu for h in visible)
        return max(0.0, min(1.0, total / len(visible) / 100.0))
    except Exception:
        return None


def _rocm_utilization() -> float | None:
    """Mean AMD GPU utilization via the ROCm SMI (`amdsmi`); `None` on any failure."""
    try:
        import amdsmi  # type: ignore[import-not-found]

        amdsmi.amdsmi_init()
        try:
            handles = amdsmi.amdsmi_get_processor_handles()
            if not handles:
                return None
            # Average only the devices this actor can see (HIP_/ROCR_/CUDA_VISIBLE_DEVICES),
            # not every physical GPU — a co-located idle device must not dilute a busy one.
            visible = [handles[i] for i in _visible_device_indices(len(handles))]
            total = sum(amdsmi.amdsmi_get_gpu_activity(h)["gfx_activity"] for h in visible)
            return max(0.0, min(1.0, total / len(visible) / 100.0))
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception:
        return None


def _xpu_utilization() -> float | None:
    """Mean Intel GPU utilization via ``torch.xpu.utilization`` (a percent); `None` if the
    torch build doesn't expose it (older builds lack the sysman/Level-Zero counter)."""
    try:
        import torch

        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return None
        # The bound device, not physical 0: an actor pinned to its second visible board was
        # adapting `num_gpus` from a neighbour's load.
        util = xpu.utilization(_bound_ordinal(xpu))  # percent, newer torch w/ Level-Zero sysman
        return max(0.0, min(1.0, float(util) / 100.0))
    except Exception:
        return None


# Per-backend utilization probe. NVIDIA/AMD/Intel expose a counter; Apple MPS and
# Cloud TPU have no stable per-process API (absent here → loop is a no-op).
_UTILIZATION = {
    "cuda": _nvml_utilization,
    "rocm": _rocm_utilization,
    "xpu": _xpu_utilization,
}

# NVIDIA compute capability with native FP8 tensor cores: Ada (8.9, L4/L40S) and
# Hopper (9.0+, H100). At/above this, FP8 halves weight + KV-cache memory at <1%
# quality loss; below it (Ampere A100/A10G 8.x, Turing 7.5, Volta 7.0) FP8 is a
# software emulation, so it is not a safe zero-config default.
_NATIVE_FP8_CAPABILITY = (8, 9)


def recommend_quantization(backend: str | None = None) -> str | None:
    """A safe default vLLM `quantization` for the current GPU, or `None` for native
    precision (BF16/FP16).

    Returns ``"fp8"`` only on GPUs with **native** FP8 tensor cores — NVIDIA Ada
    (L4/L40S, compute 8.9) and Hopper (H100, 9.0) — where FP8 halves weight/KV-cache
    memory at <1% quality loss. Older NVIDIA (Ampere A100/A10G, Turing, Volta), non-CUDA
    backends, and any probe failure return `None`, so the model keeps its native
    precision rather than a risky software-emulated FP8. The zero-config win that Ray
    Data users otherwise select by hand, per GPU."""
    if (backend or detect_backend()) != "cuda":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        capability = torch.cuda.get_device_capability()
        return "fp8" if tuple(capability) >= _NATIVE_FP8_CAPABILITY else None
    except Exception:
        return None


# NVIDIA compute capability with native BF16 tensor cores: Ampere (8.0+, A100/A10G)
# and up. BF16 keeps FP32's exponent range, so it is the numerically-safe half-precision
# default there. Turing/Volta (7.x, T4/V100) have fast FP16 tensor cores but emulate BF16,
# so they take FP16. Below 7.0 (Pascal) half-precision has no tensor-core speedup — keep FP32.
_NATIVE_BF16_CAPABILITY = (8, 0)
_FAST_FP16_CAPABILITY = (7, 0)


def recommend_inference_dtype(backend: str | None = None) -> str | None:
    """A safe half-precision dtype name for model inference on the current GPU, or `None`.

    Inference is numerically forgiving (no gradients to accumulate error), so half
    precision roughly doubles compute-bound throughput at negligible quality loss — the
    single lever that turns a compute-bound job from parity into a win. Returns
    ``"bfloat16"`` on Ampere+ (native BF16, FP32 exponent range — the safe default),
    ``"float16"`` on Turing/Volta and Apple MPS (fast FP16, no native BF16), and `None`
    (keep FP32) on older/CPU/probe-failure so the model never silently loses precision
    where half gives no speedup. The per-GPU default Ray Data users otherwise set by hand.

    TPU and Intel XPU deliberately still return `None`, even though both have native BF16
    and a TPU's MXU multiplies in it. Returning ``"bfloat16"`` there would set the model's
    *storage* dtype, which is a stronger change than the FP32→BF16 rewrite XLA already
    applies to matmuls on its own — so it is a numerics change, not just a speed one, and
    nobody has measured it on the hardware. Characterize it on a real device before
    changing this; an unvalidated half-precision default is exactly the "fast wrong
    answer" this function exists to avoid.
    """
    b = backend or detect_backend()
    try:
        import torch

        if b in ("cuda", "rocm"):
            if not torch.cuda.is_available():
                return None
            capability = tuple(torch.cuda.get_device_capability())
            if capability >= _NATIVE_BF16_CAPABILITY:
                return "bfloat16"
            if capability >= _FAST_FP16_CAPABILITY:
                return "float16"
            return None
        if b == "mps":
            return "float16"
    except Exception:
        return None
    return None


def recommend_num_gpus(util_fraction: float | None, requested: float) -> float:
    """Adapt a per-task `num_gpus` request from measured utilization.

    * `None` utilization (no measurement) or no GPU requested → keep `requested`.
    * Under-utilized whole GPU → request a fraction (≈ the measured load, floored at
      `_MIN_FRACTION`) so several tasks share one device.
    * Saturated fractional request → grow toward a whole GPU.
    * Otherwise keep the current request.
    """
    if util_fraction is None or requested <= 0.0:
        return requested
    if requested >= 1.0 and util_fraction < _PACK_BELOW:
        frac = max(_UTIL_MIN_FRACTION, round(util_fraction, 2))
        return min(1.0, frac)
    if requested < 1.0 and util_fraction > _SATURATED_ABOVE:
        return 1.0
    return requested


def recommend_inflight_depth(
    util_fraction: float | None, default: int, peak_vram_fraction: float | None = None
) -> int:
    """Adapt an inference actor's per-actor submit-ahead depth from measured utilization.

    A shallow pipeline leaves a GPU idle across the dispatch/gather round-trip; submitting
    several partitions ahead keeps it fed (Ray Data's `max_tasks_in_flight` lever). The
    depth is raised only from a *prior* low-utilization measurement, so a first run (no
    measurement) is unchanged:

    * `None` utilization or util ``>= _SATURATED_ABOVE`` → keep `default` (GPU already fed).
    * util ``< _PACK_BELOW`` (starved) → ``default * 4``, capped at `_INFLIGHT_DEPTH_MAX`.
    * otherwise (partly fed) → ``default * 2``, capped.

    `peak_vram_fraction` is the learned peak VRAM the pipeline used. A VRAM-tight pipeline
    (``>= _VRAM_TIGHT``) keeps the shallow `default` regardless of utilization: each in-flight
    partition holds its own activations/output, so submitting several ahead into a near-full
    device would OOM. Safety-only — this can lower the depth but never raises it above what
    utilization alone would grant.

    Always at least `default` (never shrinks below the configured floor).
    """
    base = max(1, default)
    if peak_vram_fraction is not None and peak_vram_fraction >= _VRAM_TIGHT:
        return base  # too little VRAM headroom to hold several partitions in flight
    if util_fraction is None or util_fraction >= _SATURATED_ABOVE:
        return base
    factor = 4 if util_fraction < _PACK_BELOW else 2
    return max(base, min(_INFLIGHT_DEPTH_MAX, base * factor))


def gpu_feedback_key(plan: LogicalPlan) -> str:
    """A stable key for a map/inference pipeline's GPU utilization.

    Built from each `map_batches` stage's UDF identity (not the rows it processed), so
    the same pipeline matches across runs while distinct models stay separate. A stage
    pinned to an `accelerator_type` carries it in the key (``@A100``), so utilization
    learned on one device class isn't replayed onto another (an A100's load means
    nothing for a T4); an unpinned stage keeps its bare identity (unchanged key)."""
    from batcher.plan.logical import MapBatches

    parts: list[str] = []
    node: Any = plan
    while node is not None:
        if isinstance(node, MapBatches):
            fn = node.fn
            name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
            mod = getattr(fn, "__module__", "")
            atype = getattr(node, "accelerator_type", None)
            suffix = f"@{atype}" if atype else ""
            parts.append(f"{mod}.{name}{suffix}")
        node = getattr(node, "input", None)
    return "|".join(parts) if parts else "map"


# Both learned GPU figures go through `metadata.smoothed`, the neutral one-scalar-per-key
# helper, rather than the whole-blob read-modify-write they used to do. Three things came
# with the move, and none of them is cosmetic:
#
# * **The cold run stops anchoring the estimate.** The blend weight was the static
#   `learning_smoothing_alpha` (0.5), under which the *first* observation still holds an
#   eighth of the value after four runs and never fully washes out. `smoothed` uses
#   `max(floor, 1/(n+1))` — a running mean while evidence is thin — so the first profiling
#   run of a model, which is the one most likely to be unrepresentative, is averaged away.
# * **Concurrent writers stop clobbering each other.** Loading the namespace's whole blob,
#   editing one key, and writing it all back is a lost update whenever two inference
#   pipelines record at once, and an autoscaled fleet records constantly. Per-key writes
#   touch only their own entry.
# * **Reads stop re-parsing the fleet.** One blob held every pipeline's figure, so reading
#   one model's utilization parsed them all.
#
# Both namespaces stay `scoped` to the hardware fingerprint: a fraction of an A100 says
# nothing about a T4. An existing whole-blob store keeps answering — the per-key view merges
# a legacy single-blob value underneath its own entries, so nothing learned is lost.
def load_gpu_utilization(hub: MetadataHub | None, key: str) -> float | None:
    """The smoothed utilization recorded for `key`, or `None` if unseen."""
    return load_scalar(hub, scoped(_NAMESPACE), key)


def record_gpu_utilization(hub: MetadataHub | None, key: str, util_fraction: float | None) -> None:
    """Record a measured utilization for `key`, exp-smoothed across runs. Best-effort."""
    if util_fraction is None:
        return
    record_smoothed_scalar(hub, scoped(_NAMESPACE), key, float(util_fraction))


_VRAM_NAMESPACE = "ml.gpu.peak_vram"


def load_gpu_peak_vram(hub: MetadataHub | None, key: str) -> float | None:
    """The smoothed peak-VRAM *fraction* (0..1) an actor of pipeline `key` used, or `None`.

    The memory twin of `load_gpu_utilization`: where utilization sizes `num_gpus`, the peak
    VRAM sizes how many inference actors safely pack onto one device (`actors_per_gpu_from_
    learned_vram`) from what a prior run actually consumed, rather than the declared model size."""
    return load_scalar(hub, scoped(_VRAM_NAMESPACE), key)


def record_gpu_peak_vram(hub: MetadataHub | None, key: str, vram_fraction: float | None) -> None:
    """Record a measured peak-VRAM fraction for `key`, exp-smoothed across runs. Best-effort."""
    if vram_fraction is None:
        return
    record_smoothed_scalar(hub, scoped(_VRAM_NAMESPACE), key, float(vram_fraction))


def actors_per_gpu_from_learned_vram(
    peak_vram_fraction: float | None, *, headroom: float = 0.2
) -> int | None:
    """Inference actors that fit on one GPU from the *measured* peak-VRAM fraction one used.

    The learned counterpart to the declared-size `max_actors_per_gpu`: if a prior run's actor
    peaked at 30% of VRAM, ~2 fit within a `headroom`-reduced budget. At least 1; `None` (no
    measurement) leaves the caller on its declared-size estimate. Pure — it only sizes packing."""
    if peak_vram_fraction is None or peak_vram_fraction <= 0.0:
        return None
    usable = max(0.0, 1.0 - headroom)
    return max(1, int(usable / peak_vram_fraction))


# --- Sustained utilization: the number the packing loop actually needs -------------------

#: How often the background sampler reads the device. Fast enough to see the idle gaps
#: between one batch's forward and the next (tens of ms on a real inference stage), cheap
#: enough that the poll itself is noise — an NVML read is ~100 us.
_UTIL_SAMPLE_INTERVAL_S = 0.05


class SustainedUtilization:
    """Time-weighted mean accelerator utilization across an actor's working window.

    The adaptive loop asks "is this GPU being kept busy?", and the honest answer is a mean
    over time. Sampling the device *right after a forward pass* — which is where the engine
    used to take its one reading per batch — answers a different question: it samples at
    precisely the instant the device is busiest, so it reports near-saturation for a stage
    that is in fact idle most of the time.

    That is not a small bias. Measured on a four-T4 ResNet-50 inference stage: the
    post-forward reading peaked at **86%** while NVML sampled on a timer put the sustained
    figure at **13%**. `recommend_num_gpus` packs a stage onto a fraction of a device only
    below `_PACK_BELOW` (50%), so the peak reading kept every stage on a whole GPU each,
    three quarters idle, and `recommend_inflight_depth` likewise stayed shallow — the two
    levers that exist to fix a starved GPU were held shut by the measurement that was
    supposed to open them.

    So this samples on a daemon thread instead, and reports the mean over the window
    **between the first call's start and the last call's end** — excluding the model load
    before any work arrives and the idle tail after the last partition, neither of which the
    scheduler can do anything about. The idle *between* calls is deliberately included: that
    gap is the starvation this measurement exists to find.

    Best-effort throughout: a device that reports no utilization (Apple MPS, Cloud TPU, CPU)
    yields `None` and the caller keeps its declared request.
    """

    def __init__(self, interval_s: float = _UTIL_SAMPLE_INTERVAL_S) -> None:
        self._interval = interval_s
        self._sum = 0.0
        self._n = 0
        self._peak: float | None = None
        self._window_start: tuple[float, int] | None = None
        self._window_end: tuple[float, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval):
            util = sample_gpu_utilization()
            if util is None:
                continue
            self._sum += util
            self._n += 1
            self._peak = util if self._peak is None else max(self._peak, util)

    def begin_call(self) -> None:
        """Mark the start of a unit of work; starts sampling on the first one."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        if self._window_start is None:
            self._window_start = (self._sum, self._n)

    def end_call(self) -> None:
        """Mark the end of a unit of work, closing the window at this instant."""
        self._window_end = (self._sum, self._n)

    def mean(self) -> float | None:
        """Sustained utilization over the working window, or `None` if nothing was sampled."""
        if self._window_start is None or self._window_end is None:
            return None
        samples = self._window_end[1] - self._window_start[1]
        if samples <= 0:
            # The whole stage finished inside one sampling interval, so there is no window to
            # average. A single reading is still better than nothing for the packing decision.
            return self._peak
        return max(0.0, min(1.0, (self._window_end[0] - self._window_start[0]) / samples))

    def peak(self) -> float | None:
        """The highest single reading seen, or `None`."""
        return self._peak

    def close(self) -> None:
        """Stop sampling and wait for the sampler to exit. Idempotent.

        Joins rather than merely signalling, so a closed monitor cannot still be polling the
        driver a moment later — which for a caller that swapped the probe out (a test, a
        vendor fallback) means the swapped-in probe is what the thread was last using.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=max(1.0, self._interval * 4))


# --- Auto mixed-precision (the tensor-core ~2x hardware lever) ---------------------------

# Whether autocast measurably speeds a given model up — stable per callable, probed once.
_AUTOCAST_VERDICT: dict[str, bool] = {}
# Sample rows and the speedup a model must show for autocast (half precision) to be kept: it
# is not bit-identical, so it is only worth applying when it is a real tensor-core win.
_AUTOCAST_PROBE_ROWS = 64
_AUTOCAST_MIN_SPEEDUP = 1.15
# A UDF sets this attribute (on the function, or on a class UDF's instance/class) to False
# to decline the probe outright. The probe re-executes the model, which is free for a local
# tensor forward but is a *duplicated billed request* for a UDF that calls a hosted LLM, and
# is not idempotent for one that writes anything. Neither is detectable from the callable,
# so an author who knows their UDF has side effects opts out here.
_AUTOCAST_OPT_OUT_ATTR = "batcher_autocast"
# The same escape hatch for the autograd-off wrap. A UDF whose *output* is a gradient — a
# saliency map, an adversarial perturbation, an influence score — needs the backward graph
# the wrap removes, and nothing about the callable says so from outside.
_INFERENCE_MODE_OPT_OUT_ATTR = "batcher_inference_mode"


def inference_mode_call(call: Callable) -> Callable:
    """Wrap a per-batch model `call` so its forward runs under `torch.inference_mode()`.

    Autograd is on by default in PyTorch, so every forward inside a `map_batches` inference
    stage builds a backward graph nobody will ever use: version counters on each tensor, an
    autograd node per op, and the activations kept alive to feed a backward pass that never
    comes. `inference_mode` switches all of that off — strictly more than `no_grad`, which
    still tracks version counters and view metadata — which cuts host overhead per op and,
    more importantly, frees the activation memory that would otherwise cap the batch size.

    This is a pure resource win with no numerical effect: the forward computes the identical
    values either way. That is what makes it safe to apply unconditionally, unlike
    `autocast_call` — which changes precision and therefore has to prove it pays first.

    Ray Data users apply this by hand in every `__call__`, and forgetting it is common enough
    that the pattern catalog has an entry for it. Here it is applied by the engine, so an
    opaque `map_batches(model, num_gpus=1)` gets it whether or not the model's author
    remembered.

    A UDF that genuinely needs autograd (computing gradients as its *output* — a saliency map,
    an adversarial perturbation, an influence score) sets ``batcher_inference_mode = False`` on
    itself or its class to decline. The wrap is skipped entirely when torch is absent, so a
    non-torch UDF pays nothing.

    Returns `call` unchanged when it cannot apply. Wrapping is by exception, not by probe: no
    extra execution of `call`, so it is safe for a UDF that bills a request or writes.
    """
    if getattr(call, _INFERENCE_MODE_OPT_OUT_ATTR, None) is False:
        return call
    obj = getattr(call, "__self__", call)  # bound method -> its instance; else the callable
    for target in (obj, type(obj)):
        if getattr(target, _INFERENCE_MODE_OPT_OUT_ATTR, True) is False:
            return call

    @functools.wraps(call)
    def _no_autograd(batch):
        torch = _torch()
        if torch is None:
            return call(batch)
        # `inference_mode` landed in torch 1.9; `no_grad` is the equivalent-intent fallback
        # for anything older, and still removes the backward graph.
        guard = getattr(torch, "inference_mode", None) or torch.no_grad
        with guard():
            return call(batch)

    return _no_autograd


def _torch() -> Any | None:
    """The **already-imported** `torch` module, or `None`.

    Deliberately a `sys.modules` lookup rather than an import. A UDF that calls torch has
    imported it in this process by definition, so looking is sufficient — while importing
    would make a `num_gpus=1` stage that never touches torch pay a multi-second import on
    its first batch, charged to that stage in the profile, for a guard it cannot use.

    Checked per call rather than once, so a model that imports torch lazily inside its first
    `__call__` is still wrapped from the next batch onward.
    """
    import sys

    return sys.modules.get("torch")


def autocast_call(call: Callable) -> Callable:
    """Wrap a per-batch GPU model `call` so its forward runs under `torch.autocast` — but only
    when a probe shows autocast actually speeds this model up.

    The accelerator's fast half type (`recommend_inference_dtype` — BF16 on Ampere+, FP16 on
    Turing/Volta/MPS) casts the matmuls/convs onto the tensor cores (~1.5-2x) while autocast
    keeps reductions/softmax in FP32 for stability. Unlike `torch.compile` this needs no model
    object, so it optimizes an opaque `map_batches(model, num_gpus=...)` call.

    Half precision is not bit-identical, so it is applied ONLY where it pays off: on the first
    batch the model is timed FP32 vs autocast on a small slice, and autocast is kept only if it
    is meaningfully faster (a compute-bound conv/matmul forward — image/vision/embedding). A
    forward that autocast does not accelerate — an autoregressive generation loop is launch- and
    memory-bound, not tensor-core-bound — stays FP32, so its exact output is preserved. The
    verdict is cached per model. This keeps "correctness before speed": FP16 is used only when it
    is a genuine speedup, never as a silent output change with no benefit.

    The probe re-executes the model, which costs nothing for a local tensor forward but
    duplicates a *billed request* for a UDF that calls a hosted model, and is not idempotent
    for a UDF that writes anything. Two guards keep that from happening. A UDF may set
    ``batcher_autocast = False`` on itself (or on a class UDF's class) to decline outright,
    in which case `call` is returned untouched and never runs twice. Otherwise the probe
    stops after its FIRST execution unless that execution actually allocated accelerator
    memory, so an opaque UDF that never touches the local GPU costs one extra slice-sized
    run rather than the full timing sweep.

    Returns `call` unchanged when it can't apply — `distributed.autocast_inference` off, a CPU
    host, a GPU with no fast half type, torch absent, or the opt-out above. Config is read each
    call so a job can pin FP32 (bit-exact repro); the device/dtype probe is cached (stable per
    worker).
    """
    if not active_config().distributed.autocast_inference:
        return call
    if not _autocast_probe_allowed(call):
        return call
    ctx = _autocast_device_dtype()
    if ctx is None:
        return call
    device_type, dtype = ctx
    key = _autocast_key(call)

    @functools.wraps(call)
    def _cast(batch):
        import torch

        use = _AUTOCAST_VERDICT.get(key) if key is not None else None
        if use is None:
            use = _autocast_speeds_up(call, batch, device_type, dtype)
            if key is not None:
                _AUTOCAST_VERDICT[key] = use
        if not use:
            return call(batch)
        with torch.autocast(device_type=device_type, dtype=dtype):
            return call(batch)

    return _cast


def _autocast_probe_allowed(call: Callable) -> bool:
    """Whether `call` permits the re-executing autocast probe (``batcher_autocast`` opt-out)."""
    obj = getattr(call, "__self__", call)  # bound method -> its instance; else the callable
    for target in (call, obj, type(obj)):
        if getattr(target, _AUTOCAST_OPT_OUT_ATTR, True) is False:
            return False
    return True


def _allocated_accelerator_bytes(call: Callable, probe: Any, device_type: str) -> int | None:
    """Peak accelerator bytes `call(probe)` allocated, or `None` when it can't be measured.

    Runs `call` exactly once. This is the cheap half of the autocast probe: a UDF that
    allocates nothing on the local device is not a tensor forward autocast could speed up
    (it is a hosted-API call, a CPU transform, or a no-op), so the verdict is settled after
    one execution instead of the eight the full timing sweep costs."""
    try:
        import torch

        stats = getattr(torch, device_type, None)
        reset = getattr(stats, "reset_peak_memory_stats", None)
        peak = getattr(stats, "max_memory_allocated", None)
        if reset is None or peak is None:
            return None  # backend has no allocator stats (mps/xla) — fall through to timing
        reset()
        base = int(peak())
        call(probe)
        return max(0, int(peak()) - base)
    except Exception:
        return None


def _autocast_key(call: Callable) -> str | None:
    """A stable cache key for a model call's autocast verdict (function or callable instance)."""
    obj = getattr(call, "__self__", call)  # bound method -> its instance; else the callable
    target = obj if hasattr(obj, "__qualname__") else type(obj)  # a class UDF instance -> its type
    mod = getattr(target, "__module__", None)
    qual = getattr(target, "__qualname__", None)
    return f"{mod}.{qual}" if mod and qual else None


def _autocast_speeds_up(call: Callable, batch, device_type: str, dtype: Any) -> bool:
    """Time `call` FP32 vs autocast on a slice; True if autocast is >= the required speedup.

    Cheap-exits after ONE execution when that execution allocated no accelerator memory —
    the call is then not a local tensor forward, so autocast cannot help it and the
    remaining seven runs would be pure duplicated side effects (see `autocast_call`).

    On any failure (a model that errors under autocast, an odd batch type) returns False — the
    safe, output-preserving FP32 path. The probe's outputs are discarded (timing only)."""
    try:
        import torch

        rows = getattr(batch, "num_rows", 0)
        probe = batch.slice(0, min(rows, _AUTOCAST_PROBE_ROWS)) if rows else batch

        allocated = _allocated_accelerator_bytes(call, probe, device_type)
        if allocated is not None and allocated <= 0:
            return False

        def _fp32() -> None:
            call(probe)

        def _fp16() -> None:
            with torch.autocast(device_type=device_type, dtype=dtype):
                call(probe)

        if allocated is None:
            _fp32()  # warm (weights resident, cudnn/cublas kernels selected)
        # else: the allocation probe above already ran `call` once, which is the warm-up.
        _fp16()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t_fp32 = _best_time(_fp32, device_type)
        t_fp16 = _best_time(_fp16, device_type)
        return t_fp16 > 0 and (t_fp32 / t_fp16) >= _AUTOCAST_MIN_SPEEDUP
    except Exception:
        return False


def _best_time(fn: Callable[[], None], device_type: str) -> float:
    """Best-of-3 wall time for `fn`, syncing CUDA so GPU work is actually measured."""
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        fn()
        if device_type == "cuda":
            import torch

            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


@functools.cache
def _autocast_device_dtype() -> tuple[str, Any] | None:
    """The `(device_type, torch.dtype)` for autocast on this worker, or None if inapplicable."""
    try:
        import torch

        backend = detect_backend()
        if backend == "cpu":
            return None
        dtype_name = recommend_inference_dtype(backend)
        if dtype_name is None:
            return None
        device_type = torch_device(backend).split(":")[0]  # 'cuda' / 'xpu' / 'mps'
        return device_type, getattr(torch, dtype_name)
    except Exception:
        return None
