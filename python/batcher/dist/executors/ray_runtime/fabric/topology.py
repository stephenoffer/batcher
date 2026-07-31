"""Where the accelerators actually are — NVLink domains, nodes, racks, and power zones.

Ray's topology answers "how many GPUs are there". A GPU datacenter needs "how many GPUs are
there *that can talk to each other quickly*", and the two differ by more than an order of
magnitude in bandwidth. Eight devices inside one NVLink domain exchange at 900 GB/s; the same
eight split across two hosts exchange over the network, and a collective that was an on-package
copy becomes the stage's entire runtime. Placement that cannot see the difference produces
correct results at a fraction of the rate, with nothing in the logs to say why.

This module reads the live cluster into that shape: per-node device counts and models, plus the
rack, zone, and fabric each node sits in, taken from node labels. The label vocabulary is
deliberately conventional rather than invented — Kubernetes' own `topology.kubernetes.io/zone`
and `topology.kubernetes.io/region` are read first, because on a managed fleet they are already
set, and Batcher's own `batcher.io/rack` and `batcher.io/fabric` fill the two gaps Kubernetes
has no vocabulary for. An unlabelled cluster reports `""` for those fields and every
topology-aware decision degrades to the node-level answer it made before.

**Bandwidth is classified, not fabricated.** `interconnect_class` names the tier a pair of
devices communicates over, and only the intra-domain figure has a number attached, because that
one comes from the device's own datasheet. A node's NIC speed is a property of a cluster this
process cannot probe, so nothing here invents one: a caller that needs a number takes it from
configuration, and a caller that needs only "is this pair cheap or expensive" uses the class.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FABRIC_LABEL",
    "LINK_CLASSES",
    "POWER_ZONE_LABEL",
    "RACK_LABEL",
    "GpuNodeTopology",
    "domain_groups",
    "fits_one_domain",
    "gpu_node_topology",
    "interconnect_class",
    "largest_local_domain",
    "nvlink_domain_size",
    "topology_summary",
]

#: Batcher's own node labels, for the two facts no standard vocabulary covers. `rack` is the
#: physical enclosure (which bounds a rack-scale NVLink domain and a shared busway); `fabric`
#: is the RDMA partition — two nodes in different InfiniBand partitions cannot reach each other
#: over the fast path however close they are physically.
RACK_LABEL = "batcher.io/rack"
FABRIC_LABEL = "batcher.io/fabric"
#: The power domain a node draws from: the breaker, busway, or PDU whose budget its draw counts
#: against. Distinct from the rack, because one busway commonly feeds several.
POWER_ZONE_LABEL = "batcher.io/power-zone"

#: Zone and region labels, most specific first. The Kubernetes topology labels are read before
#: Ray's own because on a managed cluster they are already set by the cloud provider.
_ZONE_LABELS = ("topology.kubernetes.io/zone", "ray.io/availability-zone")
_REGION_LABELS = ("topology.kubernetes.io/region", "ray.io/region")

#: Interconnect tiers, fastest first. The order is the whole contract: a caller ranks candidate
#: placements by the index of their class, and the specific numbers behind each tier vary by
#: fleet in a way this process cannot see.
LINK_CLASSES = ("nvlink", "intra-node", "intra-rack", "intra-zone", "cross-zone")


@dataclass(frozen=True, slots=True)
class GpuNodeTopology:
    """One accelerator node's position in the cluster.

    Attributes:
        node_id: Ray node id.
        gpus: Devices the node advertises.
        accelerator_type: Device model, or `""` when the node is unlabelled.
        rack: Physical rack from `batcher.io/rack`, `""` when unlabelled.
        fabric: RDMA fabric partition from `batcher.io/fabric`, `""` when unlabelled.
        power_zone: Power domain from `batcher.io/power-zone`, `""` when unlabelled.
        zone: Availability zone, `""` when unlabelled.
        region: Region, `""` when unlabelled.
        cpus: CPU cores the node advertises, for co-scheduling the host side of a GPU stage.
    """

    node_id: str
    gpus: int
    accelerator_type: str = ""
    rack: str = ""
    fabric: str = ""
    power_zone: str = ""
    zone: str = ""
    region: str = ""
    cpus: float = 0.0

    @property
    def local_domain(self) -> int:
        """Devices on this node that share one coherent fabric.

        The device's own NVLink domain capped by how many devices the node actually has: an
        H100's domain is eight, but a two-GPU node has a domain of two whatever the datasheet
        says, and sizing a collective to eight there would span the network.
        """
        return nvlink_domain_size(self.accelerator_type, self.gpus)


def _label(labels: dict, names: tuple[str, ...]) -> str:
    """First non-empty value among `names`, or `""`."""
    for name in names:
        value = labels.get(name)
        if value:
            return str(value)
    return ""


def gpu_node_topology() -> tuple[GpuNodeTopology, ...]:
    """Every alive accelerator node with its position in the cluster.

    Read live on each call so it tracks autoscaler growth and shrink, the same contract
    `node_classes()` holds to.

    Returns:
        One record per GPU-bearing node, empty when Ray is down or the topology is unreadable.
    """
    try:
        import ray

        if not ray.is_initialized():
            return ()
        nodes = ray.nodes()
    except Exception:
        return ()
    out: list[GpuNodeTopology] = []
    for node in nodes:
        if not node.get("Alive", True):
            continue
        resources = node.get("Resources", {}) or {}
        gpus = int(float(resources.get("GPU", 0.0)))
        if gpus <= 0:
            continue
        labels = node.get("Labels", {}) or {}
        out.append(
            GpuNodeTopology(
                node_id=str(node.get("NodeID", "")),
                gpus=gpus,
                accelerator_type=str(labels.get("ray.io/accelerator-type") or ""),
                rack=_label(labels, (RACK_LABEL,)),
                fabric=_label(labels, (FABRIC_LABEL,)),
                power_zone=_label(labels, (POWER_ZONE_LABEL,)),
                zone=_label(labels, _ZONE_LABELS),
                region=_label(labels, _REGION_LABELS),
                cpus=float(resources.get("CPU", 0.0)),
            )
        )
    return tuple(out)


def nvlink_domain_size(accelerator_type: str | None, gpus_on_node: int) -> int:
    """Devices reachable over one coherent fabric on a node, at least 1.

    Args:
        accelerator_type: The node's device model.
        gpus_on_node: Devices the node advertises.

    Returns:
        The smaller of the device's fabric domain and the node's device count. An unrecognized
        device model reports the node's device count, which is the pre-existing assumption
        (everything on a node is local) rather than a fabricated fabric width.
    """
    from batcher._internal.device_specs import device_nvlink_domain

    node_devices = max(1, gpus_on_node)
    domain = device_nvlink_domain(accelerator_type)
    return node_devices if domain <= 0 else min(domain, node_devices)


def largest_local_domain(nodes: tuple[GpuNodeTopology, ...] | None = None) -> int:
    """The widest single-fabric device group anywhere in the cluster.

    The hard bound on a tensor-parallel degree that stays on the fast path.

    Args:
        nodes: Topology records, or `None` to read them live.

    Returns:
        Devices in the widest domain, `0` when there are no accelerator nodes.
    """
    records = gpu_node_topology() if nodes is None else nodes
    return max((n.local_domain for n in records), default=0)


def fits_one_domain(
    world_size: int,
    nodes: tuple[GpuNodeTopology, ...] | None = None,
) -> bool:
    """Whether a collective of `world_size` devices can stay inside one coherent fabric.

    Args:
        world_size: Devices the collective needs.
        nodes: Topology records, or `None` to read them live.

    Returns:
        True when some node's domain is wide enough. False on an unreadable topology, which is
        the conservative direction: the caller then plans for a collective that crosses hosts,
        which is correct everywhere and merely pessimistic on a fleet it could not see.
    """
    if world_size <= 1:
        return True
    return largest_local_domain(nodes) >= world_size


def interconnect_class(
    a: GpuNodeTopology,
    b: GpuNodeTopology,
    *,
    same_node_fabric: bool = True,
) -> str:
    """The tier a pair of devices communicates over, from `LINK_CLASSES`.

    Args:
        a: One node.
        b: The other node.
        same_node_fabric: Whether devices on one node share a coherent fabric. False models a
            node whose devices are PCIe-attached with no NVLink between them.

    Returns:
        The fastest tier that applies. Unlabelled nodes fall back to the coarsest tier that
        can be established from what *is* labelled, never to an optimistic guess.
    """
    if a.node_id and a.node_id == b.node_id:
        return "nvlink" if same_node_fabric and a.local_domain > 1 else "intra-node"
    if a.rack and a.rack == b.rack and (not a.fabric or a.fabric == b.fabric):
        return "intra-rack"
    if a.zone and a.zone == b.zone:
        return "intra-zone"
    return "cross-zone"


def domain_groups(
    nodes: tuple[GpuNodeTopology, ...] | None = None,
) -> dict[str, list[GpuNodeTopology]]:
    """Accelerator nodes grouped by the fabric they share, largest group first.

    The unit a gang-scheduled stage should be placed within: every node in a group can reach
    every other over the same fast path, so a collective inside one group never crosses a tier.

    Args:
        nodes: Topology records, or `None` to read them live.

    Returns:
        Group key (`"rack/fabric"`, or the node id when neither is labelled) to its nodes.
        Insertion order is by descending device count, so the first group is the largest.
    """
    records = gpu_node_topology() if nodes is None else nodes
    groups: dict[str, list[GpuNodeTopology]] = {}
    for node in records:
        key = f"{node.rack}/{node.fabric}" if (node.rack or node.fabric) else node.node_id
        groups.setdefault(key, []).append(node)
    ordered = sorted(groups.items(), key=lambda kv: (-sum(n.gpus for n in kv[1]), kv[0]))
    return dict(ordered)


def topology_summary(nodes: tuple[GpuNodeTopology, ...] | None = None) -> dict:
    """A flat description of the fleet's shape, for the decision log and the dashboard.

    Args:
        nodes: Topology records, or `None` to read them live.

    Returns:
        Device and node counts, the widest coherent domain, the device models present, and
        how many racks, power zones, and fabrics the fleet spans. An unlabelled fleet reports
        zero for the label-derived counts rather than pretending it is one rack.
    """
    records = gpu_node_topology() if nodes is None else nodes
    return {
        "gpu_nodes": len(records),
        "gpus": sum(n.gpus for n in records),
        "largest_domain": largest_local_domain(records),
        "device_models": sorted({n.accelerator_type for n in records if n.accelerator_type}),
        "racks": len({n.rack for n in records if n.rack}),
        "power_zones": len({n.power_zone for n in records if n.power_zone}),
        "fabrics": len({n.fabric for n in records if n.fabric}),
        "zones": len({n.zone for n in records if n.zone}),
    }
