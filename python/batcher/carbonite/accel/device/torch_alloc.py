"""The PyTorch caching allocator — the one that actually decides whether a GPU stage OOMs.

`accel.allocator` configures **RAPIDS/RMM**, which is what the relational GPU kernels allocate
through. Every inference stage in the engine allocates through a different one: PyTorch's
caching allocator, which RMM does not govern and which has its own failure mode.

That failure mode is *fragmentation*, and it is the reason a job dies at 60% VRAM with a
message saying there is plenty free. The allocator carves the device into fixed segments and
splits blocks out of them; a workload whose tensor sizes vary — a batch of images at mixed
resolutions, a batch of sequences at mixed lengths, which is every real inference workload —
leaves each segment holding a live block too small to reuse and too scattered to coalesce. The
reported free bytes are real and unusable at the size being asked for. `empty_cache()` is the
usual response and is close to the worst one: it hands the whole cache back to the driver, so
the next allocation pays a synchronizing `cudaMalloc`, and the fragmentation returns.

Three levers, all applied before the first tensor is allocated because none of them can be
changed afterwards:

* **`expandable_segments`.** Backs each segment with virtual memory the allocator can grow in
  place instead of a fixed reservation. It is the single largest fragmentation fix PyTorch has
  shipped and it is off by default, so a workload gets it only if someone knew to set an
  environment variable before the process started — which is exactly the kind of tuning a data
  engine should be doing for its users rather than documenting at them.
* **A memory fraction.** A hard per-process cap the allocator enforces itself, so a stage that
  misjudges its footprint fails its own allocation instead of taking down the co-tenants
  sharing its device. On a node packing several actors per GPU this converts a node-wide
  cascade into one recoverable batch failure.
* **`garbage_collection_threshold`.** Lets the allocator reclaim cached blocks under pressure
  rather than only at an OOM, which keeps the steady state from creeping toward the cliff.

`configure` is a no-op wherever torch is absent, the device is not CUDA, or the allocator has
already allocated — never an error, because a worker that computes slowly is worth more than
one that will not start.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from batcher._internal.hardware.devices import (
    ALLOC_CONF_ENV,
    allocator_initialized,
    fragmentation_ratio,
    set_alloc_conf,
    set_memory_fraction,
)
from batcher._internal.logging import note_suppressed

__all__ = [
    "TorchAllocatorPlan",
    "configure_torch_allocator",
    "plan_torch_allocator",
    "reset_torch_allocator_state",
    "torch_allocator_state",
]

_applied: TorchAllocatorPlan | None = None


@dataclass(frozen=True, slots=True)
class TorchAllocatorPlan:
    """What a GPU worker should configure PyTorch's caching allocator with.

    Attributes:
        expandable_segments: Back segments with growable virtual reservations, so a workload
            with varying tensor sizes stops stranding memory in unusable holes.
        memory_fraction: Share of the device this process may allocate, or `None` for no cap.
            The allocator enforces it itself, turning a co-tenant-killing device exhaustion
            into this process's own recoverable failure.
        gc_threshold: Share of the cap past which the allocator reclaims cached blocks
            proactively, or `None` to leave it reclaiming only at an OOM.
    """

    expandable_segments: bool = False
    memory_fraction: float | None = None
    gc_threshold: float | None = None

    @property
    def is_inert(self) -> bool:
        """Whether applying this plan would change nothing about the process."""
        return (
            not self.expandable_segments
            and self.memory_fraction is None
            and self.gc_threshold is None
        )

    def alloc_conf(self) -> str:
        """The `PYTORCH_CUDA_ALLOC_CONF` value this plan implies (empty when it implies none).

        Returns:
            A comma-separated settings string in the form PyTorch parses.
        """
        parts: list[str] = []
        if self.expandable_segments:
            parts.append("expandable_segments:True")
        if self.gc_threshold is not None:
            parts.append(f"garbage_collection_threshold:{self.gc_threshold:g}")
        return ",".join(parts)


def plan_torch_allocator(cfg, *, tenants: int = 1) -> TorchAllocatorPlan:
    """Size a PyTorch allocator plan from the accelerator config and this device's tenancy.

    `tenants` is how many processes are packed onto the one device — the fractional
    `num_gpus` denominator, or the MPS client count. It is what turns the VRAM headroom into a
    *per-process* cap: four actors sharing a device may each have a quarter of it, and a plan
    that let each of them address the whole device would be no cap at all. Sizing the fraction
    against the same `vram_headroom` the admission pool holds back is what keeps the allocator
    and the admission check agreeing about how much of the device is this process's.

    Args:
        cfg: The `AcceleratorConfig` section to plan from.
        tenants: Processes sharing the device; clamped to at least 1.

    Returns:
        The plan, inert when the config asks for nothing.

    Examples:
        .. doctest::

            >>> from batcher.config import AcceleratorConfig
            >>> from batcher.carbonite.accel.device import plan_torch_allocator
            >>> plan_torch_allocator(AcceleratorConfig()).expandable_segments
            True
            >>> plan_torch_allocator(AcceleratorConfig(), tenants=4).memory_fraction
            0.21
    """
    memory = cfg.memory
    if not memory.torch_expandable_segments and not memory.torch_memory_fraction:
        return TorchAllocatorPlan()
    fraction: float | None = None
    if memory.torch_memory_fraction:
        share = (1.0 - min(0.9, max(0.0, cfg.vram_headroom))) / max(1, tenants)
        # Two decimals: the allocator compares against a byte count derived from this, and a
        # long binary fraction makes the cap a figure nobody reading a log can reconcile with
        # the device size they know.
        fraction = max(0.01, round(share, 2))
    return TorchAllocatorPlan(
        expandable_segments=bool(memory.torch_expandable_segments),
        memory_fraction=fraction,
        gc_threshold=memory.torch_gc_threshold or None,
    )


def configure_torch_allocator(plan: TorchAllocatorPlan) -> bool:
    """Apply `plan` to this process, once, before the caching allocator initializes.

    Idempotent: every GPU task body may call it and a Ray worker runs many, so the second call
    onward reports the first one's answer rather than re-applying settings that PyTorch has
    already parsed.

    The environment half must land before torch's allocator first initializes. This checks and
    reports rather than silently doing nothing: an allocator that is already up means the
    variable was set too late, which is a real misconfiguration of the worker's startup order
    and is invisible otherwise — the job simply keeps the fragmentation it was meant to avoid.

    Args:
        plan: The plan to apply, from `plan_torch_allocator`.

    Returns:
        True when this process now runs on the planned settings; False when the plan was
        inert, torch is absent, there is no CUDA device, or the allocator was already up.
    """
    global _applied
    if plan.is_inert or not _has_device():
        return False
    if _applied is not None:
        return _applied == plan
    conf = plan.alloc_conf()
    applied = False
    if conf and _set_alloc_conf(conf):
        applied = True
    if plan.memory_fraction is not None and _set_memory_fraction(plan.memory_fraction):
        applied = True
    if applied:
        _applied = plan
    return applied


def _has_device() -> bool:
    """Whether this host has an accelerator worth configuring an allocator for.

    Asked of the **driver**, not of torch, because the settings have to be in place before
    torch initializes CUDA and asking torch is what would initialize it. Without this gate a
    CPU-only host reports a configured allocator and exports `PYTORCH_CUDA_ALLOC_CONF` into
    every process it spawns — a claim about a device it does not have, and a variable that
    outlives the check that set it.
    """
    from batcher._internal.hardware.devices import device_scope

    return device_scope().count > 0


def _set_alloc_conf(conf: str) -> bool:
    """Install the settings string, reporting whether it can still take effect.

    Checks and reports rather than silently doing nothing when the allocator is already up:
    that means the variable was set too late, which is a real misconfiguration of the worker's
    startup order and is invisible otherwise — the job simply keeps the fragmentation it was
    meant to avoid.
    """
    if allocator_initialized():
        note_suppressed(
            "carbonite",
            "configure the PyTorch allocator",
            RuntimeError(
                "the caching allocator was already initialized, so "
                f"{ALLOC_CONF_ENV} cannot take effect in this process; configure it before "
                "the first CUDA allocation"
            ),
        )
        return False
    return set_alloc_conf(conf)


def _set_memory_fraction(fraction: float) -> bool:
    """Cap this process's share of its device, reporting a refusal rather than raising.

    A refusal is routine rather than exceptional: the driver reports a device (`_has_device`
    passed) but this particular worker has not imported torch, which is every relational GPU
    stage. The note is what distinguishes that from a cap that was asked for and silently did
    not apply.
    """
    if set_memory_fraction(fraction):
        return True
    note_suppressed(
        "carbonite",
        "cap this process's device memory share",
        RuntimeError("torch is not loaded in this worker, or exposes no usable device"),
    )
    return False


def torch_allocator_state() -> dict[str, float | bool | str]:
    """What this process's PyTorch allocator is configured with and how fragmented it is.

    Returns:
        `alloc_conf` (the settings string in force, empty when none), `memory_fraction` (the
        applied cap, `0.0` when uncapped), and `fragmentation` (the live ratio, `0.0` when
        unmeasurable — never a guess).

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel.device import torch_allocator_state
            >>> torch_allocator_state()["memory_fraction"]
            0.0
    """
    applied = _applied
    return {
        "alloc_conf": os.environ.get(ALLOC_CONF_ENV, ""),
        "memory_fraction": (applied.memory_fraction or 0.0) if applied else 0.0,
        "expandable_segments": bool(applied and applied.expandable_segments),
        "fragmentation": fragmentation_ratio() or 0.0,
    }


def reset_torch_allocator_state() -> None:
    """Forget the applied plan so the next `configure_torch_allocator` acts again.

    For tests, and for a worker deliberately re-pointed between stages. It does not undo the
    settings: PyTorch parsed them at allocator startup and there is no supported way back.
    """
    global _applied
    _applied = None
