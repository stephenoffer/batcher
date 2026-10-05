"""How many aligned units a node may run at once, from its memory as well as its cores.

`run._unit_slots` sizes unit tasks by cores alone: two 8-core tasks per 16-core node. A unit
task holds the unit it computes on, the next one it prefetches, and the join and aggregate
state the engine builds over the first, and units are cut from whole files, so on a source
with few large files a unit stays large whatever its target. Sized by cores, TPC-H q9 at
SF1000 on 64 GB nodes ran two tasks per node at 28 GB each over ~6 GB units, and Ray's OOM
killer took them in a loop until the query timed out.

So the fit takes the larger of the two constraints: the cores a task may use, and the share
of a node's memory its largest unit needs. It also hands each task its share as the engine's
memory budget, so a join that would still outgrow it spills (`execute_plan` re-runs a plan
whose streamed breaker overruns the budget on the materializing executor, which spills)
rather than taking the node down.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["UnitFit", "fit_units", "fit_units_to_cluster"]

#: A unit task's peak memory, as a multiple of its largest unit's projected input bytes: the
#: unit it computes on, the one it prefetches, and the engine state over the first. Read off
#: TPC-H q9 at SF1000: 28 GB per task over units of ~6 GB at 4.5; re-measured 2026-10-05 at
#: 30-35 GB of resident memory per task over the same units, so 6. An under-count here is an
#: OOM kill (two tasks per 64 GB node at 30+ GB each); an over-count is a node running one
#: task where it could have run two, which the engine's headroom guard now makes safe to risk.
UNIT_FOOTPRINT = 6.0

#: The share of a node's Ray-schedulable memory unit tasks may plan on. Ray reports the whole
#: node (64 GB on an m6id.4xlarge), and the object store, the raylet, the driver's broadcasts and
#: the OS live in it too: TPC-H q9 at SF1000 was fitted two tasks per node at 32 GB each, the
#: entire node, and Ray's memory monitor killed them at 95% of it.
USABLE_FRACTION = 0.85


@dataclass(frozen=True)
class UnitFit:
    """Cores per unit task, tasks across the cluster, tasks per node, and each task's budget."""

    unit_cpus: int
    slots: int
    per_node: int
    memory_bytes: int | None


def fit_units(
    nodes: list[tuple[int, int]], unit_cpus: int, slots: int, largest_unit: int
) -> UnitFit:
    """Fit unit tasks to `nodes` (`(cpus, memory bytes)` each) for units up to `largest_unit`.

    Args:
        nodes: Each schedulable node's CPU count and Ray-schedulable memory in bytes.
        unit_cpus: Cores per task as sized by cores alone.
        slots: Tasks across the cluster as sized by cores alone.
        largest_unit: Projected input bytes of the largest unit.

    Returns:
        The fit. Unchanged, with no budget, when no node reports both cores and memory: an
        unmeasured node is no evidence for running fewer tasks.
    """
    usable = [(c, int(m * USABLE_FRACTION)) for c, m in nodes if c >= unit_cpus and m > 0]
    if not usable or largest_unit <= 0:
        return UnitFit(unit_cpus, slots, slots, None)
    need = int(largest_unit * UNIT_FOOTPRINT)
    by_cores = min(c // unit_cpus for c, _ in usable)
    by_memory = min(max(1, m // need) for _, m in usable)
    per_node = max(1, min(by_cores, by_memory))
    cpus = max(unit_cpus, min(c for c, _ in usable) // per_node)
    budget = min(m for _, m in usable) // per_node
    # Every task is handed `budget`, so a node runs no more than its memory holds of them: on
    # a mixed cluster a node with more cores than the tightest one but no more memory would
    # otherwise take a task per `cpus` cores and overrun its memory by the core ratio.
    slots = sum(min(c // cpus, max(1, m // budget)) for c, m in usable)
    return UnitFit(cpus, slots, per_node, budget)


def fit_units_to_cluster(unit_cpus: int, slots: int, largest_unit: int) -> UnitFit:
    """`fit_units` over the live Ray cluster's nodes; the core-only sizing if it is unreadable."""
    try:
        import ray

        nodes = [
            (int(r.get("CPU", 0)), int(r.get("memory", 0)))
            for n in ray.nodes()
            if n.get("Alive")
            for r in [n.get("Resources", {})]
        ]
    except Exception as exc:
        from batcher._internal.logging import note_suppressed

        note_suppressed("dist", "read cluster memory for aligned units", exc)
        nodes = []
    return fit_units(nodes, unit_cpus, slots, largest_unit)
