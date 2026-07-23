"""Cluster-wide accelerator facts, for callers that would otherwise probe the driver.

Every "how big is a GPU?" question in the control plane used to be answered by probing the
*local* process — `gpu_inventory()`, `torch.cuda`, `gpu_vram_gb()`. That is right for a
single-node run and wrong for every distributed one, because the driver is routinely a
CPU-only head node scheduling GPU workers: the probe finds no device and the caller falls
back to a hardcoded constant, so an A100 fleet gets planned and packed as a 12 GB T4.

This module is the cluster-scoped answer, derived from the live topology via
`cluster_hardware_profile()`. It reports `None` rather than a guess whenever the topology
cannot answer, so a caller keeps whatever local default it already had instead of acting on
a fabricated figure.
"""

from __future__ import annotations

__all__ = [
    "cluster_accelerator_type",
    "cluster_gpu_memory_gb",
    "recommend_accelerator_type",
]

#: Usable fraction of a device's nameplate VRAM when deciding whether a model fits — the same
#: headroom (~15%) the packing math leaves for the CUDA context, activations, and fragmentation.
_USABLE_VRAM = 0.85


def recommend_accelerator_type(model_memory_gb: float) -> str | None:
    """The smallest GPU class in a mixed cluster whose VRAM fits `model_memory_gb`, else `None`.

    On a heterogeneous GPU cluster (a mix of, say, T4s and A100s) an inference stage that leaves
    `accelerator_type` unset can be scheduled onto *any* GPU — including one too small to hold the
    model, an OOM the moment it loads. Pinning the stage to the smallest device class that fits
    prevents that while wasting the least VRAM. `None` means "don't pin", returned whenever pinning
    would not help or could not be decided:

    * a homogeneous cluster (one GPU class) — nothing to choose between;
    * the smallest device already fits — every device is safe, so a pin only constrains placement;
    * no device fits — pinning to nothing is not an option, and the sizing path shards instead;
    * unreadable topology or unlabelled GPU nodes.

    Args:
        model_memory_gb: The stage's declared model footprint; `<= 0` reports `None` (unknown).

    Returns:
        A `ray.util.accelerators` model name to pin to, or `None` to leave placement unpinned.
    """
    if model_memory_gb <= 0:
        return None
    try:
        import ray

        if not ray.is_initialized():
            return None
        from batcher._internal.accelerators import accelerator_memory_bytes
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        classes = node_classes()
    except Exception:
        return None
    # Distinct GPU classes with a known VRAM, as (usable_gb, name).
    seen: dict[str, float] = {}
    for c in classes:
        name = c.get("accelerator_type")
        if c["gpus"] > 0 and name:
            seen[name] = accelerator_memory_bytes(name) / (1 << 30) * _USABLE_VRAM
    usable = {n: gb for n, gb in seen.items() if gb > 0}
    if len(usable) < 2:
        return None  # homogeneous (or unknowable) → no class choice to make
    if min(usable.values()) >= model_memory_gb:
        return None  # every device already fits → a pin would only constrain placement
    fitting = {n: gb for n, gb in usable.items() if gb >= model_memory_gb}
    if not fitting:
        return None  # nothing fits one device → let the sizing path shard instead
    return min(fitting, key=lambda n: fitting[n])  # the smallest device that fits (least waste)


def cluster_accelerator_type() -> str | None:
    """The device model every GPU node shares, or `None` when they don't all share one.

    Identifies the fleet for anything learned *per device* — a measured GPU/CPU crossover from
    an H100 fleet says nothing about a T4 fleet, and folding both into one regression converges
    on an average right for neither.

    `None` on a **mixed** fleet is deliberate rather than a limitation worked around. There is
    no single honest answer to "which device did this run on" when the models differ, and
    inventing one (the binding device, say) would attach an H100's timing to a T4's bucket. A
    mixed fleet therefore keeps the single pooled bucket it has always used — no worse than
    before, and without manufacturing a precision the topology cannot support.

    Returns:
        The shared `ray.io/accelerator-type` name, or `None` when the fleet is mixed, has no
        GPU nodes, has unlabelled ones, or the topology is unreadable.
    """
    try:
        import ray

        if not ray.is_initialized():
            return None
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        names = {c.get("accelerator_type") for c in node_classes() if c["gpus"] > 0}
    except Exception:
        return None
    if len(names) != 1:
        return None
    only = names.pop()
    return only if only else None


def cluster_gpu_memory_gb() -> float | None:
    """VRAM of the cluster's **smallest** GPU in GB, or `None` when it can't be determined.

    The minimum is the binding figure: a model packed against the largest device in a mixed
    fleet OOMs every smaller one it lands on, which is the failure this exists to prevent.

    Returns:
        Usable VRAM in GB of the binding device, or `None` when Ray is down, the topology is
        unreadable, or any GPU node's device model is unrecognized.
    """
    try:
        import ray

        if not ray.is_initialized():
            return None
        from batcher.dist.executors.ray_runtime.scaling import cluster_hardware_profile

        hw = cluster_hardware_profile()
    except Exception:
        return None
    if hw is None or hw.gpu_memory_bytes <= 0:
        return None
    return hw.gpu_memory_bytes / (1 << 30)
