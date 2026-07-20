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
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher._internal.hardware import available_cpu_count
from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "actors_per_gpu_from_learned_vram",
    "autocast_call",
    "detect_backend",
    "gpu_aware_pool_default",
    "gpu_feedback_key",
    "gpu_vram_gb",
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
        return max(1, int(num_workers))  # type: ignore[arg-type]
    if num_gpus > 0:
        return 1
    return available_cpu_count()  # usable local cores (cgroup/affinity aware), not host count


def gpu_aware_pool_default(
    num_gpus: float,
    fallback: int,
    num_partitions: int,
    accelerator_type: str | None = None,
) -> int:
    """Default distributed actor-pool size when `concurrency` is unset.

    For a GPU stage, size the pool to the cluster's GPUs so *every* GPU gets an actor
    (replicas = total_GPUs / per-actor `num_gpus`) — never one engine idling a multi-GPU
    cluster (the Ray Data ``concurrency=1`` foot-gun). For a CPU stage, keep the cluster
    worker count (`fallback`). Clamped to the partition count (no idle actors); falls back
    when Ray reports no GPUs.

    When the stage is pinned to an `accelerator_type` on a **heterogeneous** cluster
    (mixed GPU classes), size against *that class's* GPUs — Ray tags them as the
    ``accelerator_type:<NAME>`` resource — so a stage pinned to the 4 A100s never spawns
    actors for the 8 T4s it can't run on. Taken as a `min` with the total GPU count, so
    an absent or sentinel typed resource only ever sizes *down* (never over-subscribes).
    """
    if num_gpus <= 0:
        return fallback
    try:
        import ray

        resources = ray.cluster_resources()
        total = float(resources.get("GPU", 0.0))
        if accelerator_type:
            typed = float(resources.get(f"accelerator_type:{accelerator_type}", 0.0))
            if typed > 0:
                total = min(total, typed)
    except Exception:
        return fallback
    if total <= 0:
        return fallback
    return max(1, min(num_partitions, int(total / num_gpus)))


_NAMESPACE = "ml.gpu"
# Below this measured utilization a whole-GPU task is wasting the device, so pack
# more tasks onto a fraction of it; above the saturation mark, give it a whole GPU.
_PACK_BELOW = 0.5
_SATURATED_ABOVE = 0.9
# Don't fragment a GPU finer than this (avoids requesting unschedulable slivers).
_MIN_FRACTION = 0.25
# Bounds on the adaptive per-actor submit-ahead depth (how many partitions an inference
# actor keeps in flight). A starved GPU (low measured utilization) submits deeper to
# overlap the dispatch/gather round-trip; a saturated one keeps the shallow default.
_INFLIGHT_DEPTH_MAX = 16
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
    "cpu": 0.0,
}
# Budget peak inference VRAM at ~1.5x model size (activations + batch tensors).
_INFERENCE_VRAM_MULTIPLIER = 1.5


def detect_backend() -> str:
    """The accelerator backend: ``cuda`` / ``rocm`` / ``xpu`` / ``mps`` / ``tpu`` / ``cpu``.

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
    Apple ``mps``, a TPU is ``xla`` (torch_xla), and CPU ``cpu``.

    An unrecognized backend degrades to ``cpu`` rather than raising. This is a mapping of
    *names*, and the names come from places this function does not control — a caller
    naming an accelerator we have not taught it about (``"hpu"``, ``"neuron"``), or a newer
    `detect_backend`. A `KeyError` from a device-string lookup would abort a job that could
    have run correctly, if slower, on the CPU; the whole point of the accelerator layer is
    that it is an optimization.
    """
    b = backend or detect_backend()
    return {
        "cuda": "cuda",
        "rocm": "cuda",
        "xpu": "xpu",
        "mps": "mps",
        "tpu": "xla",
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
    behavior). A UUID-form entry that can't be indexed by ordinal is skipped.
    """
    import os

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


def gpu_vram_gb() -> float | None:
    """Total VRAM (GB) of accelerator 0, or `None` when it can't be determined.

    Used to VRAM-pack inference actors. Tries the vendor SMI (NVML) first, then torch's
    device properties (covers CUDA/ROCm/XPU); returns `None` on a host with no
    accelerator (e.g. a GPU-less driver), where packing is simply skipped."""
    handle = _vram_handle()  # this process's visible device, not physical 0
    if handle is not None:  # NVML reports total memory without allocating a CUDA context
        try:
            return _nvml().nvmlDeviceGetMemoryInfo(handle).total / (1 << 30)
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1 << 30)
        # Apple MPS shares unified memory; `recommended_max_memory` is the working
        # budget torch will use before paging — the right number to pack against.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.mps.recommended_max_memory() / (1 << 30)
    except Exception:
        pass
    return None


def sample_gpu_vram_fraction() -> float | None:
    """Fraction (0..1) of accelerator-0 VRAM in use, or `None` without a GPU.

    Feeds the throughput autobatcher's VRAM cap so it shrinks (or refuses to grow) the
    batch *before* an out-of-memory rather than catching one after the fact. Tries the
    vendor SMI (NVML — counts every process on the device) then torch's reserved
    memory; returns `None` on a GPU-less host, where the guard is simply inert."""
    handle = _vram_handle()
    if handle is not None:
        try:
            info = _nvml().nvmlDeviceGetMemoryInfo(handle)
            return info.used / info.total if info.total else None
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return torch.cuda.memory_reserved(0) / total if total else None
        # MPS unified memory: current allocation against the recommended budget.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            total = torch.mps.recommended_max_memory()
            return torch.mps.current_allocated_memory() / total if total else None
    except Exception:
        pass
    return None


def max_actors_per_gpu(
    model_vram_gb: float,
    gpu_vram_gb: float,
    *,
    headroom: float = 0.2,
    context_overhead_gb: float | None = None,
    inference_multiplier: float = _INFERENCE_VRAM_MULTIPLIER,
) -> int:
    """How many inference actors fit on one GPU, VRAM-budgeted.

    Each actor needs ``model_vram_gb * inference_multiplier + context_overhead_gb``;
    usable VRAM leaves `headroom` free for batch data and runtime spikes. At least 1
    (a model that doesn't fit the budget still gets a whole GPU, where it may swap).
    `context_overhead_gb` defaults to the detected vendor's process-context overhead
    (NVIDIA 0.4, AMD 0.5, Intel 0.3, Apple 0.0). This packs a small model and refuses
    to over-subscribe a large one into an OOM.
    """
    if model_vram_gb <= 0 or gpu_vram_gb <= 0:
        return 1
    overhead = vram_context_overhead() if context_overhead_gb is None else context_overhead_gb
    usable = gpu_vram_gb * (1.0 - headroom)
    per_actor = model_vram_gb * inference_multiplier + overhead
    return max(1, int(usable // per_actor))


def recommend_gpu_fraction(model_vram_gb: float, gpu_vram_gb: float, **kwargs: float) -> float:
    """The per-actor ``num_gpus`` fraction so several actors share a GPU when the
    model is small, floored at `_MIN_FRACTION` to avoid unschedulable slivers; 1.0
    when only one actor fits. The static counterpart to the measured-utilization
    `recommend_num_gpus` (use this to size a cold start, that to adapt across runs).

    Use *this* for the scheduler's ``num_gpus``, not `max_actors_per_gpu` directly:
    the actors Ray actually packs onto one GPU is ``floor(1 / fraction)``, which is
    ``min(max_actors_per_gpu(...), 4)`` — the 0.25 floor caps packing density at 4/GPU
    even when more would fit by VRAM. So this is always schedule-safe (never over-
    subscribes a GPU), at the cost of leaving VRAM unused for very small models."""
    n = max_actors_per_gpu(model_vram_gb, gpu_vram_gb, **kwargs)  # type: ignore[arg-type]
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
        util = xpu.utilization(0)  # percent, newer torch with Level-Zero sysman
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
        frac = max(_MIN_FRACTION, round(util_fraction, 2))
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


def load_gpu_utilization(hub: MetadataHub | None, key: str) -> float | None:
    """The smoothed utilization recorded for `key`, or `None` if unseen."""
    if hub is None:
        return None
    try:
        return hub.load_params(_NAMESPACE).get(key)
    except Exception:  # pragma: no cover - feedback must never break execution
        return None


def record_gpu_utilization(hub: MetadataHub | None, key: str, util_fraction: float | None) -> None:
    """Record a measured utilization for `key`, exp-smoothed across runs. Best-effort."""
    if hub is None or util_fraction is None:
        return
    try:
        stats = hub.load_params(_NAMESPACE)
        alpha = active_config().optimizer.learning_smoothing_alpha
        prior = stats.get(key)
        stats[key] = (
            float(util_fraction)
            if prior is None
            else alpha * float(util_fraction) + (1.0 - alpha) * float(prior)
        )
        hub.save_params(_NAMESPACE, stats)
    except Exception:  # pragma: no cover - feedback must never break execution
        pass


_VRAM_NAMESPACE = "ml.gpu.peak_vram"


def load_gpu_peak_vram(hub: MetadataHub | None, key: str) -> float | None:
    """The smoothed peak-VRAM *fraction* (0..1) an actor of pipeline `key` used, or `None`.

    The memory twin of `load_gpu_utilization`: where utilization sizes `num_gpus`, the peak
    VRAM sizes how many inference actors safely pack onto one device (`actors_per_gpu_from_
    learned_vram`) from what a prior run actually consumed, rather than the declared model size."""
    if hub is None:
        return None
    try:
        return hub.load_params(_VRAM_NAMESPACE).get(key)
    except Exception:  # pragma: no cover - a learned read must never break execution
        return None


def record_gpu_peak_vram(hub: MetadataHub | None, key: str, vram_fraction: float | None) -> None:
    """Record a measured peak-VRAM fraction for `key`, exp-smoothed across runs. Best-effort."""
    if hub is None or vram_fraction is None:
        return
    try:
        stats = hub.load_params(_VRAM_NAMESPACE)
        alpha = active_config().optimizer.learning_smoothing_alpha
        prior = stats.get(key)
        stats[key] = (
            float(vram_fraction)
            if prior is None
            else alpha * float(vram_fraction) + (1.0 - alpha) * float(prior)
        )
        hub.save_params(_VRAM_NAMESPACE, stats)
    except Exception:  # pragma: no cover - feedback must never break execution
        pass


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


# --- Auto mixed-precision (the tensor-core ~2x hardware lever) ---------------------------

# Whether autocast measurably speeds a given model up — stable per callable, probed once.
_AUTOCAST_VERDICT: dict[str, bool] = {}
# Sample rows and the speedup a model must show for autocast (half precision) to be kept: it
# is not bit-identical, so it is only worth applying when it is a real tensor-core win.
_AUTOCAST_PROBE_ROWS = 64
_AUTOCAST_MIN_SPEEDUP = 1.15


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

    Returns `call` unchanged when it can't apply — `distributed.autocast_inference` off, a CPU
    host, a GPU with no fast half type, or torch absent. Config is read each call so a job can
    pin FP32 (bit-exact repro); the device/dtype probe is cached (stable per worker).
    """
    if not active_config().distributed.autocast_inference:
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


def _autocast_key(call: Callable) -> str | None:
    """A stable cache key for a model call's autocast verdict (function or callable instance)."""
    obj = getattr(call, "__self__", call)  # bound method -> its instance; else the callable
    target = obj if hasattr(obj, "__qualname__") else type(obj)  # a class UDF instance -> its type
    mod = getattr(target, "__module__", None)
    qual = getattr(target, "__qualname__", None)
    return f"{mod}.{qual}" if mod and qual else None


def _autocast_speeds_up(call: Callable, batch, device_type: str, dtype: Any) -> bool:
    """Time `call` FP32 vs autocast on a slice; True if autocast is >= the required speedup.

    On any failure (a model that errors under autocast, an odd batch type) returns False — the
    safe, output-preserving FP32 path. The probe's outputs are discarded (timing only)."""
    try:
        import torch

        rows = getattr(batch, "num_rows", 0)
        probe = batch.slice(0, min(rows, _AUTOCAST_PROBE_ROWS)) if rows else batch

        def _fp32() -> None:
            call(probe)

        def _fp16() -> None:
            with torch.autocast(device_type=device_type, dtype=dtype):
                call(probe)

        _fp32()  # warm (weights resident, cudnn/cublas kernels selected)
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
