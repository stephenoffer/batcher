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
    "cluster_gpu_memory_bytes",
    "cluster_gpu_memory_gb",
    "recommend_accelerator_type",
]

from batcher._internal.device_share import device_headroom


def recommend_accelerator_type(model_memory_gb: float) -> str | None:
    """The GPU class in a mixed cluster a stage should be pinned to, or `None` to leave it free.

    On a heterogeneous fleet a stage that leaves `accelerator_type` unset can be scheduled onto
    *any* GPU, including one too small to hold the model — an OOM the moment it loads. Pinning
    prevents that.

    The *choice* is Kyber's (`select_device_class`), because "which device class" is an
    optimization decision and it depends on policy this layer has no business knowing: the
    smallest device that fits wastes the least VRAM, the most efficient one that fits wastes
    the least power, and on a fleet with measured history the one that actually delivered most
    per joule beats both. This function's job is the half Kyber cannot do — reading the live
    topology to find out which classes exist.

    `None` means "don't pin", returned whenever pinning would not help or could not be
    decided: a homogeneous cluster, every device already fits, nothing fits (the sizing path
    shards instead), or an unreadable topology.

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
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        classes = node_classes()
    except Exception:
        return None
    candidates = sorted({c.get("accelerator_type") or "" for c in classes if c["gpus"] > 0})
    if not any(candidates):
        return None
    from batcher.kyber.gpu import select_device_class

    return select_device_class(
        [c for c in candidates if c],
        # GiB, which is the unit the device table's `memory_gib` is written in and the unit
        # `select_device_class` compares against. `model_memory_gb` is a decimal-GB figure the
        # user declared, and handing it over unconverted under-states the model by 7.4% —
        # enough to pin a stage to the device one size down and out-of-memory it at load,
        # which is the exact failure this pinning exists to prevent.
        model_memory_gb * 1e9 / (1 << 30),
        headroom=device_headroom(),
        hub=_learned_hub(),
    )


def _learned_hub():
    """The metadata hub, so the choice can prefer what this fleet measured, or `None`.

    Best-effort: a fleet with no learned history, or a metadata backend that cannot be opened,
    simply falls back to the datasheet ordering.
    """
    try:
        from batcher.core.runtime import default_hub

        return default_hub()
    except Exception:
        return None


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


def cluster_gpu_memory_bytes() -> int | None:
    """Total VRAM of the cluster's **smallest** GPU in bytes, or `None` when undeterminable.

    The minimum is the binding figure: a model packed against the largest device in a mixed
    fleet OOMs every smaller one it lands on, which is the failure this exists to prevent.

    **Nameplate, and in bytes.** One meaning for "how big is one device", stated in the unit
    the driver reports it in, because the two figures this replaces were neither. The GB
    spelling below divides by `1 << 30` — a *gibibyte* — while every consumer compared it
    against a working set built as `rows x width / 1e9`, a decimal gigabyte, so an 80 GiB
    board was routed as though it were 80 GB and the device was over-stated by 7%. Callers
    that want a budget rather than a capacity subtract `device_headroom()` themselves, once.

    Returns:
        Bytes of the binding device, or `None` when Ray is down, the topology is unreadable,
        or any GPU node's device model is unrecognized.
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
    return int(hw.gpu_memory_bytes)


def cluster_gpu_memory_gb() -> float | None:
    """Total VRAM of the cluster's smallest GPU in **decimal** GB, or `None` when unknown.

    The unit matters and used to be wrong. Kyber sizes a working set as `rows x width / 1e9`
    — decimal gigabytes — and compared it against this, which divided by `1 << 30`. An 80 GiB
    A100 therefore presented as "80" against a working set measured in GB, over-stating the
    device by 7.4% in the direction that dispatches a query the board cannot hold.

    Returns:
        Decimal GB of the binding device's total memory, or `None` when undeterminable. This
        is a *capacity*, not a budget: subtract `device_headroom()` to get what a claimant may
        actually plan against.
    """
    total = cluster_gpu_memory_bytes()
    return None if total is None else total / 1e9
