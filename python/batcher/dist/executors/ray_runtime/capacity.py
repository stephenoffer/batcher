"""How many workers a cluster can actually *place*, as opposed to afford.

Fan-out sizing reads the cluster as totals — `floor(total_cpus / num_cpus)`. On a homogeneous
cluster that is the same number a per-node count would give, which is why the difference went
unnoticed. On a heterogeneous one it is not: four 8-core nodes total 32 cores, so the sum says
two `num_cpus=16` workers fit, while every individual node is too small to host even one. The
grant is uniform and placement is per node, so the sum is an upper bound that no arrangement
of nodes can reach.

That gap is not a rounding error. Ray schedules the fleet as a gang, so a fan-out sized above
what any node can host leaves the placement group permanently unsatisfiable — the job hangs
rather than failing, which is the failure mode `clamp_workers` exists to prevent.

Related: `dist.executor._cluster_fill_workers` slices nodes the same way when *choosing* a
fan-out. This module answers the complementary question — whether a fan-out already chosen can
be placed — and the two must keep using the same `floor(node_cores / num_cpus)` rule.
"""

from __future__ import annotations

__all__ = ["placeable_workers"]


def placeable_workers(num_cpus: float, num_gpus: float = 0.0) -> int | None:
    """Workers that fit when each is placed on a single node, or `None` if unknown.

    Sums each node's own capacity rather than dividing the cluster's total, so a node too
    small to host one worker contributes zero instead of a fraction of one.

    Args:
        num_cpus: CPU shares each worker requests. Values at or below zero report unknown.
        num_gpus: GPUs each worker requests; `0` for a CPU-only fleet, which then bounds by
            cores alone.

    Returns:
        The number of placeable workers, or `None` when the topology is unreadable — the
        caller then keeps its total-based estimate rather than clamping on a guess.
    """
    if num_cpus <= 0:
        return None
    try:
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        nodes = node_classes()
    except Exception:
        return None
    if not nodes:
        return None
    total = 0
    for node in nodes:
        fits = int(float(node["cpus"]) // num_cpus)
        if num_gpus > 0:
            fits = min(fits, int(float(node["gpus"]) // num_gpus))
        total += max(0, fits)
    return total
