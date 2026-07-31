"""Placing accelerator work on the fleet: gang bundles, power zones, and efficiency order.

`topology` says what the fleet looks like; this decides where a stage goes. Three decisions,
each of which a topology-blind scheduler gets wrong in a way that shows up as a performance
mystery rather than an error:

* **A collective must not span a fabric.** A tensor-parallel stage of eight devices placed
  four-and-four across hosts runs its all-reduce over the network at a fraction of the NVLink
  rate. The fix is a strict-pack bundle inside one domain, and a deliberate, *reported*
  decision when the world size is wider than any domain the fleet has.
* **A power zone has a budget.** Filling every slot in a rack whose busway cannot power them
  either trips a breaker or, more commonly, causes every device in the rack to be clamped —
  which reads as the whole rack getting slower for no visible reason.
* **A heterogeneous fleet has an order.** When several device models can host a stage, the
  most efficient one that fits should get it, because on a power-constrained fleet throughput
  per watt is what converts to throughput per rack.

Every function degrades to the pre-existing behavior on an unreadable or unlabelled topology:
no bundles, no preference, no cap. A placement hint that fires on missing data is worse than
no hint, because it moves work for a reason that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.dist.executors.ray_runtime.fabric.topology import (
    LINK_CLASSES,
    GpuNodeTopology,
    domain_groups,
    gpu_node_topology,
    interconnect_class,
    largest_local_domain,
)

__all__ = [
    "CollectivePlacement",
    "devices_within_power_budget",
    "plan_collective",
    "power_zone_load",
    "rank_nodes_by_efficiency",
]


@dataclass(frozen=True, slots=True)
class CollectivePlacement:
    """Where a multi-device collective stage should run.

    Attributes:
        world_size: Devices the stage needs.
        bundles: Ray placement-group bundles, one per node, as `{"GPU": n, "CPU": c}`.
        strategy: Ray placement strategy — `STRICT_PACK` when the collective fits one node,
            `PACK` when it must span nodes but should stay as close as possible.
        spans_fabric: Whether the collective is wider than any single coherent domain, so its
            all-reduce leaves the *vendor fabric*. On a PCIe-attached fleet (T4, L4, A10G,
            L40S — devices whose domain is one) this is True for every collective of two or
            more, which is why it must not be read as "leaves the machine": see `spans_nodes`.
        spans_nodes: Whether the bundles land on more than one host, so the all-reduce crosses
            the network. This is the expensive boundary, and the one distinct from
            `spans_fabric` on any PCIe-only fleet: four devices on one PCIe host exchange at
            host-link rates, four devices split across two hosts at NIC rates.
        link_class: The slowest tier any pair of ranks communicates over, from `LINK_CLASSES`.
            The single field a caller should rank placements by.
        node_ids: Nodes the bundles target, empty when the topology could not be read.
        reason: One line for the decision log.
    """

    world_size: int
    bundles: tuple[dict[str, float], ...] = ()
    strategy: str = "PACK"
    spans_fabric: bool = False
    spans_nodes: bool = False
    link_class: str = ""
    node_ids: tuple[str, ...] = ()
    reason: str = ""


def plan_collective(
    world_size: int,
    nodes: tuple[GpuNodeTopology, ...] | None = None,
    *,
    cpus_per_device: float = 1.0,
    datasets: list[str] | tuple[str, ...] = (),
    zone_budget_watts: float = 0.0,
) -> CollectivePlacement:
    """Lay a collective of `world_size` devices out, staying inside one fabric where it fits.

    Three constraints are applied before the fabric preference, because each of them can
    remove a node that the fabric would otherwise have picked: a dataset's residency rules,
    the power a zone can still supply, and — when `accelerator.efficiency_first_placement` is
    on — the order that fills the most efficient hardware first.

    Args:
        world_size: Devices the collective needs.
        nodes: Topology records, or `None` to read them live.
        cpus_per_device: Host cores to co-schedule per device, for the feeding pipeline.
        datasets: Dataset names or paths the stage reads. Given these, nodes whose region any
            input forbids are removed before placement rather than after, which is the
            difference between a compliance control and a compliance report.
        zone_budget_watts: Watts one power zone may supply, `0.0` for unbudgeted. A zone that
            cannot power more devices is skipped, because filling it clamps every device in it.

    Returns:
        The placement. With an unreadable topology the bundles are empty and the strategy is
        the existing `PACK` default, so the caller schedules exactly as it did before.
    """
    want = max(1, world_size)
    fleet = gpu_node_topology() if nodes is None else nodes
    records = _eligible(fleet, datasets, zone_budget_watts)
    if not records:
        # An empty fleet and a fleet filtered to nothing are different failures with different
        # fixes, and reporting both as "unreadable" sends the reader looking for a label
        # problem when the answer is a policy they wrote.
        reason = (
            "topology unreadable: scheduling without a placement hint"
            if not fleet
            else (
                f"no eligible node of {len(fleet)}: residency or the power budget excluded "
                "every one, so this stage cannot be placed as configured"
            )
        )
        return CollectivePlacement(world_size=want, reason=reason)

    # Prefer a single node whose coherent domain already holds the whole collective.
    for node in _preferred_order(records):
        if node.local_domain >= want:
            return CollectivePlacement(
                world_size=want,
                bundles=({"GPU": float(want), "CPU": cpus_per_device * want},),
                strategy="STRICT_PACK",
                spans_fabric=False,
                spans_nodes=False,
                link_class="nvlink" if node.local_domain > 1 else "intra-node",
                node_ids=(node.node_id,),
                reason=(
                    f"fits one {node.accelerator_type or 'GPU'} fabric domain "
                    f"of {node.local_domain}"
                ),
            )

    # Failing that, prefer a single node that has the devices even though no vendor fabric
    # joins them. This tier is the whole PCIe-attached fleet — T4, L4, A10G, L40S, and every
    # workstation part — where the coherent domain is one device and the test above can never
    # pass. Skipping it sent a four-way collective on a four-device host down the spanning
    # path, reported as leaving the fast path, when in truth it never left the machine: an
    # intra-host exchange runs at host-link rates and a spanning one at NIC rates, and
    # treating them alike is the difference the placement exists to see.
    for node in _preferred_order(records, by_capacity=True):
        if node.gpus >= want:
            return CollectivePlacement(
                world_size=want,
                bundles=({"GPU": float(want), "CPU": cpus_per_device * want},),
                strategy="STRICT_PACK",
                spans_fabric=True,
                spans_nodes=False,
                link_class="intra-node",
                node_ids=(node.node_id,),
                reason=(
                    f"fits one node's {node.gpus} {node.accelerator_type or 'GPU'} devices; "
                    f"no vendor fabric joins them (domain of {node.local_domain}), so the "
                    "all-reduce crosses the host link but stays off the network"
                ),
            )

    # Otherwise fill the largest fabric group first, so the split is across the fewest tiers.
    bundles: list[dict[str, float]] = []
    ids: list[str] = []
    remaining = want
    for group in domain_groups(records).values():
        for node in _preferred_order(group, by_capacity=True):
            if remaining <= 0:
                break
            take = min(node.gpus, remaining)
            bundles.append({"GPU": float(take), "CPU": cpus_per_device * take})
            ids.append(node.node_id)
            remaining -= take
        if remaining <= 0:
            break
    widest = largest_local_domain(records)
    chosen = tuple(n for n in records if n.node_id in set(ids))
    worst = _worst_link_class(chosen)
    if remaining > 0:
        return CollectivePlacement(
            world_size=want,
            bundles=tuple(bundles),
            strategy="PACK",
            spans_fabric=True,
            spans_nodes=len(ids) > 1,
            link_class=worst,
            node_ids=tuple(ids),
            reason=(
                f"fleet has {want - remaining} of {want} devices: "
                "the collective will wait on capacity"
            ),
        )
    return CollectivePlacement(
        world_size=want,
        bundles=tuple(bundles),
        strategy="PACK",
        spans_fabric=True,
        spans_nodes=len(ids) > 1,
        link_class=worst,
        node_ids=tuple(ids),
        reason=(
            f"world size {want} exceeds the widest fabric domain ({widest}): "
            f"the collective spans {len(bundles)} nodes over {worst} and its all-reduce "
            "leaves the fast path"
        ),
    )


def _worst_link_class(chosen: tuple[GpuNodeTopology, ...]) -> str:
    """The slowest tier any pair among `chosen` communicates over.

    A collective runs at the rate of its slowest hop, so the ranking field names that hop
    rather than the best one available.
    """
    if not chosen:
        return ""
    if len(chosen) == 1:
        node = chosen[0]
        return "nvlink" if node.local_domain > 1 else "intra-node"
    worst = 0
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            worst = max(worst, LINK_CLASSES.index(interconnect_class(a, b)))
    return LINK_CLASSES[worst]


def _eligible(
    records: tuple[GpuNodeTopology, ...],
    datasets: list[str] | tuple[str, ...],
    zone_budget_watts: float,
) -> tuple[GpuNodeTopology, ...]:
    """Nodes a collective may actually use: healthy, residency-permitted, and with power left.

    The residency and power filters are no-ops by default — no datasets named, no budget
    configured — so a caller that passes neither gets exactly the fleet it passed in, minus
    any node the fleet health probe has condemned.
    """
    out = _without_unhealthy(records)
    if datasets:
        from batcher.dist.executors.ray_runtime.fabric.residency import permitted_nodes
        from batcher.governance.residency import active_residency

        out = permitted_nodes(active_residency(), datasets, out)
    if zone_budget_watts > 0:
        drawn = power_zone_load(out)
        out = tuple(
            n
            for n in out
            if devices_within_power_budget(
                zone_budget_watts, n.accelerator_type, drawn.get(n.power_zone, 0.0)
            )
            != 0
        )
    return out


def _without_unhealthy(records: tuple[GpuNodeTopology, ...]) -> tuple[GpuNodeTopology, ...]:
    """Drop nodes holding a device that should not be scheduled on.

    A gang-scheduled collective is the placement least tolerant of a bad node: every rank
    waits on the slowest, so one device with its NVLink down or its memory condemned sets the
    rate for the whole group — and a strict-pack bundle has no way to route around it once
    placed. This is the one filter worth paying a fleet probe for.

    Only *quarantined* devices exclude a node. A degraded one is still contributing, and a
    fleet under a thermal or power clamp is often every node at once, so excluding those would
    empty the cluster over a condition the clamp is there to manage.

    Args:
        records: The candidate nodes.

    Returns:
        The nodes with nothing quarantined on them. Returns `records` unchanged when the probe
        answered for no node — an unreadable fleet is not an unhealthy one — and when every
        node is condemned, because refusing to place work at all is worse than placing it on a
        fleet the operator has already been told about.
    """
    from batcher.config import active_config

    if not records or not active_config().accelerator.health.enabled:
        return records
    from batcher.dist.executors.ray_runtime.hardware_probe import cluster_device_health

    condemned = {r["node_id"] for r in cluster_device_health() if r.get("quarantined")}
    if not condemned:
        return records
    healthy = tuple(n for n in records if n.node_id not in condemned)
    return healthy or records


def _preferred_order(
    records: tuple[GpuNodeTopology, ...] | list[GpuNodeTopology],
    *,
    by_capacity: bool = False,
) -> list[GpuNodeTopology]:
    """Nodes in the order a collective should fill them.

    Widest coherent domain first (or largest device count, when filling a group), because a
    collective that fits one fabric is worth more than any other property. Under
    `accelerator.efficiency_first_placement` the throughput-per-watt order breaks ties, which
    is what fills the efficient hardware first on a fleet that is power-bound rather than
    slot-bound.
    """
    from batcher.config import active_config

    primary = (lambda n: -n.gpus) if by_capacity else (lambda n: -n.local_domain)
    if active_config().accelerator.efficiency_first_placement:
        ranked = {n.node_id: i for i, n in enumerate(rank_nodes_by_efficiency(tuple(records)))}
        return sorted(records, key=lambda n: (primary(n), ranked.get(n.node_id, 0), n.node_id))
    return sorted(records, key=lambda n: (primary(n), n.node_id))


def power_zone_load(
    nodes: tuple[GpuNodeTopology, ...] | None = None,
    utilization: float = 1.0,
) -> dict[str, float]:
    """Estimated full-load draw per power zone, in watts.

    What a rack-level budget is checked against. Nodes with no `batcher.io/power-zone` label are
    grouped under `""`, which a caller should read as "unattributed" rather than as one zone.

    Args:
        nodes: Topology records, or `None` to read them live.
        utilization: Utilization the devices are assumed to run at.

    Returns:
        Power zone to watts, including each device's host share. Zones whose device models are
        unrecognized contribute `0.0`, making every figure a lower bound.
    """
    from batcher.plan.energy.power import device_power_watts

    records = gpu_node_topology() if nodes is None else nodes
    out: dict[str, float] = {}
    for node in records:
        watts = device_power_watts(node.accelerator_type, utilization, include_host=True)
        out[node.power_zone] = out.get(node.power_zone, 0.0) + watts * node.gpus
    return out


def devices_within_power_budget(
    budget_watts: float,
    accelerator_type: str | None,
    already_drawn_watts: float = 0.0,
    *,
    utilization: float = 1.0,
) -> int:
    """Devices of one model a zone can still power, given what it is already drawing.

    Args:
        budget_watts: The zone's budget; `<= 0` means unbudgeted.
        accelerator_type: Device model to add.
        already_drawn_watts: Draw the zone is already committed to.
        utilization: Utilization the new devices are assumed to run at.

    Returns:
        Devices that fit, `-1` for "no opinion" when the budget is unset or the device model is
        unrecognized — the same unbounded sentinel `plan.energy` uses, so a caller cannot
        mistake "no limit" for "no room".
    """
    from batcher.plan.energy.power import max_concurrent_devices

    if budget_watts <= 0:
        return -1
    return max_concurrent_devices(
        budget_watts - max(0.0, already_drawn_watts), accelerator_type, utilization
    )


def rank_nodes_by_efficiency(
    nodes: tuple[GpuNodeTopology, ...] | None = None,
) -> tuple[GpuNodeTopology, ...]:
    """Accelerator nodes ordered by throughput per watt, most efficient first.

    Nodes whose device model has no published efficiency figure keep their relative order at
    the end of the list rather than being dropped: unlike a ranking of device *models*, a
    ranking of nodes must stay total, because every node is still a placement candidate.

    Args:
        nodes: Topology records, or `None` to read them live.

    Returns:
        The same nodes, reordered.
    """
    from batcher._internal.device_specs import device_tflops_per_watt

    records = gpu_node_topology() if nodes is None else nodes
    return tuple(
        sorted(
            records,
            key=lambda n: (-device_tflops_per_watt(n.accelerator_type), n.node_id),
        )
    )
