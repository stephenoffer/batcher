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

from batcher._internal.logging import get_logger, note_suppressed

__all__ = ["placeable_workers", "warn_once_if_allocation_is_wider_than_ray"]


def placeable_workers(
    num_cpus: float,
    num_gpus: float = 0.0,
    *,
    memory_bytes: int = 0,
    cpu_only: bool = False,
) -> int | None:
    """Workers that fit when each is placed on a single node, or `None` if unknown.

    Sums each node's own capacity rather than dividing the cluster's total, so a node too
    small to host one worker contributes zero instead of a fraction of one.

    The count must be taken over the nodes the fleet may actually land on, and against
    every resource its bundle reserves. Both were previously ignored, and each produces the
    same failure — a fan-out above what any arrangement of nodes can host, which leaves the
    gang-scheduling placement group permanently unsatisfiable and hangs the job:

    * `cpu_only` mirrors the node-class restriction (`scaling.node_class_selector`). When a
      relational fleet is held off accelerator nodes, those nodes' cores cannot host it, so
      counting them overstates capacity by exactly the GPU half of a mixed cluster.
    * `memory_bytes` is reserved by the bundle (`scheduling._bundle`), so a node with spare
      cores but no spare RAM hosts zero workers however many cores it has. On the memory
      -heavy shuffles this grant exists for, RAM binds well before cores do.

    Args:
        num_cpus: CPU shares each worker requests. Values at or below zero report unknown.
        num_gpus: GPUs each worker requests; `0` for a CPU-only fleet, which then bounds by
            cores alone.
        memory_bytes: Heap bytes each worker's bundle reserves; `0` skips the memory bound
            (the caller had no memory grant to place).
        cpu_only: Count only non-accelerator nodes, matching a fleet restricted to them.

    Returns:
        The number of placeable workers, or `None` when the topology is unreadable — the
        caller then keeps its total-based estimate rather than clamping on a guess.
    """
    if num_cpus <= 0:
        return None
    try:
        from batcher._internal.accelerators import is_accelerator_node
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        nodes = node_classes()
    except Exception:
        return None
    if not nodes:
        return None
    if cpu_only:
        nodes = [n for n in nodes if not is_accelerator_node(n)]
        if not nodes:
            # The restriction is only ever applied when CPU-only nodes can host the fleet
            # (`cpu_only_can_host`), so an empty list here means the topology changed under
            # us. Report unknown rather than zero: zero would clamp the fan-out to one.
            return None
    total = 0
    for node in nodes:
        fits = int(float(node["cpus"]) // num_cpus)
        if num_gpus > 0:
            fits = min(fits, int(float(node["gpus"]) // num_gpus))
        node_memory = float(node.get("memory", 0.0))
        # Only bound by memory where the node actually reports it. A node advertising no
        # `memory` resource is not a node with no memory — Ray simply is not tracking it —
        # and reading the absent value as zero would place zero workers everywhere and
        # collapse the whole fan-out to one.
        if memory_bytes > 0 and node_memory > 0:
            fits = min(fits, int(node_memory // memory_bytes))
        total += max(0, fits)
    return total


# Set once the allocation-width notice has been given. The shape of a job does not change
# under it, and repeating the notice per query trains the reader to skip it.
_ALLOCATION_WARNED = False


def warn_once_if_allocation_is_wider_than_ray() -> None:
    """Say so, once, when a batch allocation spans more nodes than Ray is using.

    The silent failure this exists for: `srun -N 4 python job.py` gives the job four nodes,
    and a bare `ray.init()` starts a *local* single-node Ray on whichever one the script
    happens to be on. The job runs, returns the right answer, and uses a quarter of the
    hardware it was billed for — with nothing anywhere to say so, because from Batcher's side
    a one-node cluster is a perfectly ordinary cluster.

    Batcher cannot fix it from here: bringing up Ray across a Slurm allocation is the
    launcher's job (`ray start` on each node, then `RAY_ADDRESS`). So this reports rather
    than acts, and reports only where the two figures actually disagree — a single-node
    allocation, an unscheduled process, and a Ray cluster that already spans the allocation
    all say nothing.
    """
    global _ALLOCATION_WARNED
    if _ALLOCATION_WARNED:
        return
    from batcher._internal.site import scheduler_job

    job = scheduler_job()
    if not job.multi_node:
        return
    try:
        import ray

        if not ray.is_initialized():
            return
        ray_nodes = len([n for n in ray.nodes() if n.get("Alive", True)])
    except Exception as exc:
        note_suppressed("dist", "compare the Ray cluster against the allocation", exc)
        return
    if ray_nodes >= len(job.nodes):
        return
    _ALLOCATION_WARNED = True
    get_logger("dist").warning(
        "this %s job holds %d nodes but Ray sees %d: the run will use one node's worth of "
        "the allocation. Start Ray across the allocation (`ray start --head` on one node, "
        "`ray start --address` on the rest) and point %s at it.",
        job.kind,
        len(job.nodes),
        ray_nodes,
        "RAY_ADDRESS",
    )
