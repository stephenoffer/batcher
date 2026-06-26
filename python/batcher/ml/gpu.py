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

from typing import TYPE_CHECKING, Any

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "detect_backend",
    "gpu_aware_pool_default",
    "gpu_feedback_key",
    "gpu_vram_gb",
    "load_gpu_utilization",
    "max_actors_per_gpu",
    "recommend_gpu_fraction",
    "recommend_num_gpus",
    "recommend_quantization",
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
    import os

    if num_workers != "auto":
        return max(1, int(num_workers))  # type: ignore[arg-type]
    if num_gpus > 0:
        return 1
    return max(1, os.cpu_count() or 1)


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

    Detected via torch (the most portable probe): ROCm reports through the CUDA API
    with ``torch.version.hip`` set; Intel GPUs via ``torch.xpu``; Apple via MPS; Cloud
    TPUs via ``torch_xla``. Falls back to ``cpu`` when torch is absent or no accelerator
    is present.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if _tpu_available():
        return "tpu"
    return "cpu"


def _tpu_available() -> bool:
    """Whether a Cloud TPU is present, via `torch_xla` — import-gated and side-effect-free.

    `find_spec` avoids importing (and so initializing) the XLA runtime on the common
    no-TPU host; only when `torch_xla` is actually installed do we ask its runtime for
    the device type. Any failure (older API, no device) reads as "no TPU"."""
    import importlib.util

    if importlib.util.find_spec("torch_xla") is None:
        return False
    try:
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        return xr.device_type() == "TPU"
    except Exception:
        return False


def torch_device(backend: str | None = None) -> str:
    """The torch device string for `backend` (default: detected) — what ``.to(...)`` wants.

    ROCm uses the ``cuda`` device string (HIP shims the CUDA API); Intel is ``xpu``,
    Apple ``mps``, a TPU is ``xla`` (torch_xla), and CPU ``cpu``.
    """
    b = backend or detect_backend()
    return {
        "cuda": "cuda",
        "rocm": "cuda",
        "xpu": "xpu",
        "mps": "mps",
        "tpu": "xla",
        "cpu": "cpu",
    }[b]


def vram_context_overhead(backend: str | None = None) -> float:
    """Per-process runtime-context VRAM overhead (GB) for `backend` (default: detected)."""
    return _CONTEXT_OVERHEAD_GB.get(backend or detect_backend(), 0.4)


def gpu_vram_gb() -> float | None:
    """Total VRAM (GB) of accelerator 0, or `None` when it can't be determined.

    Used to VRAM-pack inference actors. Tries the vendor SMI (NVML) first, then torch's
    device properties (covers CUDA/ROCm/XPU); returns `None` on a host with no
    accelerator (e.g. a GPU-less driver), where packing is simply skipped."""
    try:  # NVML reports total memory without allocating a CUDA context
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1 << 30)
        finally:
            pynvml.nvmlShutdown()
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
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return None
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            return info.used / info.total if info.total else None
        finally:
            pynvml.nvmlShutdown()
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
    """Mean NVIDIA GPU utilization via NVML (`pynvml`); `None` on any failure."""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                return None
            total = sum(
                pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu
                for i in range(count)
            )
            return max(0.0, min(1.0, total / count / 100.0))
        finally:
            pynvml.nvmlShutdown()
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
            total = sum(amdsmi.amdsmi_get_gpu_activity(h)["gfx_activity"] for h in handles)
            return max(0.0, min(1.0, total / len(handles) / 100.0))
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
