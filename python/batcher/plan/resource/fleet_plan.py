"""How a fleet of unequal machines is cut into worker slots — the pure sizing, no Ray.

Every distributed sizing decision Batcher made before this file existed collapsed the cluster
to its **smallest** node. That is the right instinct for a *uniform* grant, because a grant no
node can host is a placement group that never forms, and it is invisible on the homogeneous
clusters the sizing was written against. On a mixed fleet it is a large, silent loss. Measured
on the 27-node cluster this was written for -- one 96-core/206 GB node, two 48-core/103 GB,
seven 16-core/34 GB and sixteen 4-core/8.6 GB, 384 cores in total:

* the per-worker **core** grant came out at the smallest node's 4, so the 96-core node ran 24
  separate 4-core actors instead of a few fat ones, and the shuffle carried 96x96 streams
  rather than 32x32;
* the per-worker **memory** budget came out at ``min_node_ram * soft_limit / max_workers_on_
  any_node`` -- **228 MB**, handed to every worker in the fleet including the four sitting on
  the 206 GB node, so the big machines spilled a join they could have held entirely in RAM.

Neither figure is wrong for the node it was derived from. The mistake is that one figure was
derived at all: a fleet of unequal machines has no single right answer, and forcing one makes
the whole cluster behave like its weakest member.

This module states the alternative. A node is cut into slots sized for *that node*, so a
worker's grant describes the machine it actually lands on. The output is a list of
`WorkerSlot`s -- deliberately a list and not a count-plus-grant, because the count and the
grant are exactly what stop agreeing once the nodes differ.

**Nothing here can change a result.** A slot's core count sets how many threads compute a
partial and its memory budget sets when that partial spills; both are the scheduling half of
the mergeable algebra, which is associative and commutative over any partitioning. What they
change is how much of the cluster the query uses.

It lives in `plan` for the reason `ClusterShape` does: `carbonite` sizes envelopes and `dist`
places fleets, those layers cannot import each other, and two statements of this arithmetic is
the one way the grant a bundle reserves and the budget a worker enforces could disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WorkerSlot", "is_heterogeneous", "plan_worker_slots"]


@dataclass(frozen=True, slots=True)
class WorkerSlot:
    """One worker's share of one node.

    Attributes:
        cpus: Cores this worker is granted -- what its bundle reserves.
        compute_cpus: Cores this worker may size its thread pool to. Equal to `cpus` unless
            `node_reserve_cores` held a core back, in which case the reserve is a *scheduling*
            concession and must not also cost the worker a thread -- see the field on
            `plan.resource.SchedulingEnvelope` that carries this to the data plane.
        memory_bytes: Heap bytes this worker may hold before it spills, `0` when the node's
            RAM was not reported (the caller then keeps whatever budget it already had).
        node_index: Position in the node list this slot was cut from, so a caller can group a
            fleet back into the machines that host it.
    """

    cpus: float
    memory_bytes: int
    node_index: int
    compute_cpus: float = 0.0


def is_heterogeneous(node_cores: list[float]) -> bool:
    """Whether the fleet's nodes differ in size enough that one grant cannot describe them.

    A single distinct core count is the homogeneous case, where a uniform grant is exactly
    right and every path that predates this module already produces it. Callers use this to
    stay on that path rather than re-deriving an identical answer through a different
    function.

    Args:
        node_cores: Nameplate cores of each worker-eligible node.

    Returns:
        True when the nodes report more than one distinct core count.
    """
    return len({float(c) for c in node_cores if c > 0}) > 1


def _slices(cores: float, target: float, min_cores: float, domains: int) -> int:
    """How many workers one node of `cores` cores should be cut into, at least one.

    Three constraints, in the order they bind:

    * **The target.** `round(cores / target)` is the pipeline figure -- a worker alternates
      between gathering a shuffle bucket and computing it, so a node wants enough workers for
      those phases to overlap. This is `dist.executor._numa_sliced`'s measured optimum, and it
      is a *per-node* question that was previously asked once for the whole cluster.
    * **The memory domains.** Never coarser, so a worker never spans two NUMA domains and
      makes half its loads remote.
    * **The minimum worth a process.** Never finer than `cores / min_cores`: a worker carries
      a Flight server, its own hash tables and its own share of every shuffle bucket, so a
      two-core worker pays that fixed cost for parallelism the cores cannot deliver. Slicing
      is *declined* rather than reduced when it would breach this, which is what leaves a
      small node hosting exactly one worker.

    Args:
        cores: The node's nameplate cores.
        target: Cores one worker should aim to hold.
        min_cores: Cores below which a further cut is refused.
        domains: Memory domains on the node, floor for the slice count.

    Returns:
        The slice count, at least 1 for any node with cores, 0 for a node with none.
    """
    if cores <= 0:
        return 0
    want = max(max(1, domains), round(cores / max(1.0, target)))
    affordable = max(1, int(cores // max(1.0, min_cores)))
    return max(1, min(want, affordable))


def plan_worker_slots(
    node_cores: list[float],
    node_memory: list[int],
    *,
    target_cores: float,
    min_slice_cores: float,
    domains: int = 1,
    memory_share: float = 1.0,
    node_free_cores: list[float] | None = None,
    node_reserve_cores: float = 0.0,
) -> list[WorkerSlot]:
    """Cut every node into worker slots sized for that node.

    Each node is sliced independently (`_slices`), its cores are dealt out evenly among its
    slices, and its RAM is divided the same way. A node whose cores do not divide evenly gives
    its remainder to the earliest slots one core at a time, so the slots differ by at most one
    core and none is left short -- the cluster's cores are fully accounted for, which is the
    whole point of not tiling by the smallest node.

    Memory is divided by the *hosting node's* slice count, never by the fleet's busiest node's.
    Those are the same number only on a homogeneous cluster, and the gap between them is the
    228 MB figure in this module's docstring.

    A node reporting no memory contributes `0` to its slots, which every caller must read as
    "keep the budget you had" rather than as a budget of nothing.

    `node_reserve_cores` is why a node is not tiled exactly, and it thins the *reservation*
    only -- `WorkerSlot.compute_cpus` still carries the node's full usable share, so the
    headroom costs a scheduling slot and not a worker thread.

    `node_reserve_cores` is why a node is not tiled exactly. A fleet spans every node, so a
    plan that claims each node's whole core count claims **100% of the cluster's schedulable
    CPU** -- and a distributed query does not run entirely inside its fleet. The map UDF task
    and the hardware probe are plain Ray tasks submitted outside the reservation, so they have
    nowhere to go, and the fleet then waits on capacity only it could release. This is the
    failure `dist.executor._headroom_grant` prevents on the uniform path, stated per node
    instead of per cluster, and it is not hypothetical here: the 27-node fleet's slots summed
    to 384 cores against a cluster of exactly 384.

    `node_free_cores` separates the two questions a core count is asked. The **shape** -- how
    many workers a node hosts -- comes from the nameplate, because a node whose cores are
    momentarily all held is still a node the fleet will run on, and sizing the shape from free
    capacity makes a busy cluster look like a small one. The **grant** each of those workers
    reserves is capped at what is actually free, because a gang is all-or-nothing and a bundle
    the free capacity cannot hold leaves the placement group pending until it times out. This
    is `dist.executor._placeable_grant`'s contract, asked once per node instead of once for a
    cluster that has no single answer. Omit it, or pass an idle cluster's figures, and every
    grant is the nameplate one.

    Args:
        node_cores: Nameplate cores per worker-eligible node.
        node_memory: Host RAM per node, parallel to `node_cores`. Missing or short entries are
            read as unreported.
        target_cores: Cores one worker should aim to hold.
        min_slice_cores: Cores below which a node is not cut further.
        domains: Memory domains per node, the floor on any node's slice count.
        memory_share: Fraction of a node's RAM the fleet may budget (the soft limit).
        node_free_cores: Unreserved cores per node, parallel to `node_cores`. Caps the grant
            without moving the slice count; `None` reads every node as idle.
        node_reserve_cores: Cores held back on every node that hosts a worker, so the fleet
            never reserves the whole machine. `0.0` tiles each node exactly.

    Returns:
        One `WorkerSlot` per worker, grouped by node in the order the nodes were given. Empty
        when no node reports cores, which the caller must treat as an unreadable topology.
    """
    slots: list[WorkerSlot] = []
    for index, cores in enumerate(node_cores):
        count = _slices(float(cores), target_cores, min_slice_cores, domains)
        if count <= 0:
            continue
        usable = int(cores)
        if node_free_cores is not None and index < len(node_free_cores):
            usable = min(usable, max(count, int(node_free_cores[index])))
        # Never below one core per slice: a node too small to spare the reserve keeps its
        # workers rather than losing them, which is the safe direction for a headroom.
        grantable = max(count, usable - int(node_reserve_cores))
        whole, remainder = grantable // count, grantable - (grantable // count) * count
        # The reserve is scheduling headroom, not a core the node stopped having, so the
        # thread width is dealt from `usable` — the cores another tenant is not already
        # holding. Identical to the grant whenever nothing was reserved.
        c_whole, c_remainder = usable // count, usable - (usable // count) * count
        memory = node_memory[index] if index < len(node_memory) else 0
        per_memory = int(max(0, memory) * memory_share / count)
        for slice_index in range(count):
            cpus = float(whole + (1 if slice_index < remainder else 0))
            compute = float(c_whole + (1 if slice_index < c_remainder else 0))
            slots.append(
                WorkerSlot(
                    cpus=max(1.0, cpus),
                    memory_bytes=per_memory,
                    node_index=index,
                    compute_cpus=max(1.0, compute),
                )
            )
    return slots
