"""Accelerator detection + utilization feedback — the adaptive half of scheduling.

Declaring `map_batches(..., num_gpus=1)` gets a GPU inference stage *placed* on a
GPU worker. This module closes the loop: the actors measure how busy the device
actually was, that utilization is persisted to the MetadataHub keyed by the
pipeline, and the next run's `num_gpus` request adapts — packing more tasks onto a
fraction of a GPU when it sat idle, or asking for a whole GPU when it saturated.
This is "num_gpus based on utilization", measured and consumed, not guessed.

**Vendor-agnostic.** Detection and utilization cover NVIDIA (CUDA/NVML), AMD (ROCm),
Intel (XPU), and Apple (MPS), with a CPU fallback; `detect_backend` / `torch_device`
pick the right device and `vram_context_overhead` the right VRAM headroom per vendor.
Any measurement failure (no driver, no SMI, no device) yields `None` and the loop is
a no-op (the declared `num_gpus` stands). Recommendation/persistence are pure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "detect_backend",
    "gpu_feedback_key",
    "gpu_vram_gb",
    "load_gpu_utilization",
    "max_actors_per_gpu",
    "recommend_gpu_fraction",
    "recommend_num_gpus",
    "record_gpu_utilization",
    "sample_gpu_utilization",
    "torch_device",
    "vram_context_overhead",
]

_NAMESPACE = "ml.gpu"
# Below this measured utilization a whole-GPU task is wasting the device, so pack
# more tasks onto a fraction of it; above the saturation mark, give it a whole GPU.
_PACK_BELOW = 0.5
_SATURATED_ABOVE = 0.9
# Don't fragment a GPU finer than this (avoids requesting unschedulable slivers).
_MIN_FRACTION = 0.25
# Per-vendor VRAM a process reserves for its runtime context before any model loads —
# the overhead that makes packing many tiny models less dense than naive math. MPS
# shares unified memory (no separate context reserve); CPU has none.
_CONTEXT_OVERHEAD_GB = {"cuda": 0.4, "rocm": 0.5, "xpu": 0.3, "mps": 0.0, "cpu": 0.0}
# Budget peak inference VRAM at ~1.5x model size (activations + batch tensors).
_INFERENCE_VRAM_MULTIPLIER = 1.5


def detect_backend() -> str:
    """The available accelerator backend: ``cuda`` / ``rocm`` / ``xpu`` / ``mps`` / ``cpu``.

    Detected via torch (the most portable probe): ROCm reports through the CUDA API
    with ``torch.version.hip`` set; Intel GPUs via ``torch.xpu``; Apple via MPS. Falls
    back to ``cpu`` when torch is absent or no accelerator is present.
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
    return "cpu"


def torch_device(backend: str | None = None) -> str:
    """The torch device string for `backend` (default: detected) — what ``.to(...)`` wants.

    ROCm uses the ``cuda`` device string (HIP shims the CUDA API); Intel is ``xpu``,
    Apple ``mps``, and CPU ``cpu``.
    """
    b = backend or detect_backend()
    return {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu", "mps": "mps", "cpu": "cpu"}[b]


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

    Dispatches to the vendor's metrics: NVML for NVIDIA, ROCm SMI for AMD. Intel/Apple
    expose no stable per-process utilization API, so they (and CPU) return `None`. Any
    failure (no driver, no SMI library, no device) yields `None`, so callers treat
    utilization as simply unavailable on this host and the feedback loop is a no-op."""
    b = backend or detect_backend()
    if b == "cuda":
        return _nvml_utilization()
    if b == "rocm":
        return _rocm_utilization()
    return None  # xpu/mps/cpu: no stable utilization API


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

    Built from each `map_batches` stage's UDF identity (not the rows it processed),
    so the same pipeline matches across runs while distinct models stay separate."""
    from batcher.plan.logical import MapBatches

    parts: list[str] = []
    node: Any = plan
    while node is not None:
        if isinstance(node, MapBatches):
            fn = node.fn
            name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
            mod = getattr(fn, "__module__", "")
            parts.append(f"{mod}.{name}")
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
