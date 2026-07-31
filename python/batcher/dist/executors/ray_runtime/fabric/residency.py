"""Residency as a placement filter — the point where a sovereignty rule reaches the scheduler.

`governance.residency` states which regions a dataset may be *computed* in. That statement is
inert until something consults it before choosing a node, which is what this module does: it
reads the regions the fleet's nodes are labelled with, asks the catalog which of them every
input permits, and hands back the nodes that survive.

Two properties keep this safe to leave on. **An unlabelled node is never filtered out**: a
worker whose region Batcher cannot see is not evidence of a violation, and dropping it would
take a cluster offline the day a label was missed. And **an unregistered dataset restricts
nothing**, so a fleet with no residency rules gets exactly the placement it had.

The layering: `dist` may read `governance`, never the reverse. Governance decides what is
allowed; scheduling is what acts on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.dist.executors.ray_runtime.fabric.topology import GpuNodeTopology, gpu_node_topology

if TYPE_CHECKING:
    from batcher.governance.residency import ResidencyCatalog

__all__ = ["fleet_regions", "permitted_nodes", "residency_report"]


def fleet_regions(nodes: tuple[GpuNodeTopology, ...] | None = None) -> tuple[str, ...]:
    """Regions the accelerator fleet spans, in stable order.

    Args:
        nodes: Topology records, or `None` to read them live.

    Returns:
        The distinct labelled regions, sorted. Unlabelled nodes contribute nothing, so an
        entirely unlabelled fleet reports an empty tuple rather than a placeholder region.
    """
    records = gpu_node_topology() if nodes is None else nodes
    return tuple(sorted({n.region for n in records if n.region}))


def permitted_nodes(
    catalog: ResidencyCatalog,
    datasets: list[str] | tuple[str, ...],
    nodes: tuple[GpuNodeTopology, ...] | None = None,
) -> tuple[GpuNodeTopology, ...]:
    """The accelerator nodes every named dataset may be processed on.

    Args:
        catalog: The residency rules in force.
        datasets: Dataset names or paths the stage reads.
        nodes: Topology records, or `None` to read them live.

    Returns:
        The permitted subset, order preserved. Nodes with no region label are kept, because an
        unreadable label is not evidence of a violation. Every node is returned when the
        catalog's mode is `off` or no input is registered.
    """
    records = gpu_node_topology() if nodes is None else nodes
    permitted = catalog.permitted_regions(list(datasets))
    if permitted is None:
        return tuple(records)
    return tuple(n for n in records if not n.region or n.region in permitted)


def residency_report(
    catalog: ResidencyCatalog,
    datasets: list[str] | tuple[str, ...],
    nodes: tuple[GpuNodeTopology, ...] | None = None,
) -> dict:
    """What residency does to this stage's placement, for the decision log.

    A placement that silently loses two thirds of a fleet to a compliance rule looks
    indistinguishable from a cluster that is simply busy, so the reduction is reported rather
    than left to be inferred from a queue time.

    Args:
        catalog: The residency rules in force.
        datasets: Dataset names or paths the stage reads.
        nodes: Topology records, or `None` to read them live.

    Returns:
        The mode, the permitted regions (`None` when unrestricted), the regions the fleet
        spans, the device counts before and after filtering, and whether any node was excluded.
    """
    records = gpu_node_topology() if nodes is None else nodes
    allowed = permitted_nodes(catalog, datasets, records)
    permitted = catalog.permitted_regions(list(datasets))
    return {
        "mode": catalog.mode,
        "permitted_regions": None if permitted is None else sorted(permitted),
        "fleet_regions": list(fleet_regions(records)),
        "gpus_total": sum(n.gpus for n in records),
        "gpus_permitted": sum(n.gpus for n in allowed),
        "excluded_nodes": len(records) - len(allowed),
    }
