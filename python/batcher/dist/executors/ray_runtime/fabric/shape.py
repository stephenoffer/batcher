"""The live cluster, rendered into the neutral shape Kyber plans against.

`topology` reads Ray into `GpuNodeTopology`, which is a `dist` type: rich, accelerator-only,
and unreachable from the optimizer, because `kyber` (layer 3) may not import `dist` (layer 4).
So every locality question the optimizer wanted to ask — is this exchange crossing a network or
a NVLink, does this fan-out fit inside one host, how many nodes is the fleet actually spread
over — had no way to be answered and was answered by assumption instead: one flat pool of
anonymous workers, every byte between any two of them priced the same.

This module is the one-way bridge. It projects the live topology onto `plan.resource.ClusterShape`
— a neutral, policy-free record `kyber` *can* read — and stops there. It classifies nothing and
decides nothing: what a byte costs on each tier is Kyber's call, and keeping that judgment out
of here is what stops the fleet description and the cost model from drifting into two opinions.

Every node the cluster reports is included, not only the accelerator-bearing ones, because a
relational shuffle runs on the CPU fleet and a fleet's CPU-only nodes are most of it.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.plan.resource import ClusterShape, NodeShape

__all__ = ["cluster_shape"]


def _node_records() -> list[dict]:
    """Every node that can host a worker, or `[]` when the topology is unreadable.

    Worker-eligible, not merely alive: the Ray head and anything Ray is draining are excluded,
    exactly as they are for every fan-out sizing in `scaling`. The shape describes the machines
    a plan will run *on*, and a node that will host no worker distorts every figure derived
    from it —

    * `total_cores` drives `exchange_width`, which sets how wide a shuffle is priced as being
      spread; head cores inflate it, and a wider exchange keeps less of itself local;
    * `binding_cpu_cores` and `binding_memory_bytes` take the **smallest** node, and a head is
      routinely the smallest node in the fleet, so it would bind the whole plan's per-worker
      sizing to a machine that runs none of the work;
    * an unlabelled head becomes a rack of its own in `locality_shares`, adding a tier crossing
      to an exchange that never touches it.

    Snapshot-aware, so inside a `topology_scope()` this is the same read every other sizing
    path uses and cannot disagree with them across an autoscale.
    """
    try:
        import ray

        if not ray.is_initialized():
            return []
        from batcher.dist.executors.ray_runtime.scaling import _TOPOLOGY, _worker_eligible

        snapshot = _TOPOLOGY.get()
        nodes = snapshot.alive_nodes if snapshot is not None else ray.nodes()
        return _worker_eligible([n for n in nodes if n.get("Alive", True)])
    except Exception as exc:  # pragma: no cover - Ray optional / cluster down
        note_suppressed("dist", "read the cluster's node list", exc)
        return []


def cluster_shape() -> ClusterShape:
    """The fleet's shape for the optimizer, empty when the topology is unreadable.

    Read live on every call so it tracks autoscaler growth and shrink, the same contract
    `node_classes()` and `gpu_node_topology()` hold to. The cost of that is a `ray.nodes()`
    round trip per optimize, which is the same call the sizing path already makes.

    Inside a `topology_scope()` it is memoized alongside `node_classes()`, for the reason
    given there: the scope fixes the topology, so a projection of it is fixed too, and
    building one `NodeShape` per node was 271 ms of a 50,000-node query's placement phase.
    Outside a scope nothing is cached, so the autoscale wait still sees the fleet grow.

    Each node contributes its cores, RAM, devices, device model, and the labels that place it
    physically. Two figures are *derived* rather than reported, and both are derived
    conservatively:

    * **VRAM per device** comes from the node's advertised model, because Ray never reports
      device memory. An unrecognized model contributes `0`, which reads as "unknown" and leaves
      every VRAM-sized decision on its existing default rather than on a guess.
    * **The coherent fabric width** comes from the model's specification capped at the node's
      own device count, so a two-device node has a domain of two whatever an eight-way part's
      datasheet says. An unrecognized model reports the node's whole device count — "no
      narrower than the node", the assumption in force before any of this existed — rather than
      a fabric width taken from a datasheet nothing matched.

    Returns:
        The shape, or an empty `ClusterShape` when Ray is down or reports nothing. An empty
        shape makes every locality-aware decision report the flat answer it gave before, so a
        caller never has to test for it.
    """
    from batcher._internal.accelerators import accelerator_memory_bytes
    from batcher.dist.executors.ray_runtime.fabric.topology import (
        nvlink_domain_size,
    )
    from batcher.dist.executors.ray_runtime.scaling import _TOPOLOGY, fleet_census

    snapshot = _TOPOLOGY.get()
    if snapshot is not None:
        cached = snapshot.derived.get("cluster_shape")
        if cached is not None:
            return cached  # type: ignore[return-value]

    # One pass, shared with the placement path: `scaling.fleet_census()` already walked the
    # fleet and classified it, so this groups that census down to the fields a plan is sized
    # against instead of walking every node a second time. Building the per-node form here
    # was 170 ms of a single query against a synthetic 100,000-node fleet, on top of the
    # 240 ms the placement path spent on its own pass over the same records.
    fabric = _declared_fabric_gbps()
    counts: dict[tuple, int] = {}
    for node, ids in fleet_census().items():
        cores = int(node.cpus)
        if cores <= 0:
            continue  # a node with no schedulable cores hosts no worker, so it holds no share
        key = (
            cores,
            int(node.memory),
            int(node.gpus),
            str(node.accelerator_type or ""),
            node.rack,
            node.shape_zone,
            node.power_zone,
            node.unhealthy_gpus,
        )
        counts[key] = counts.get(key, 0) + len(ids)

    census: dict[NodeShape, int] = {}
    for key, count in counts.items():
        cores, memory, gpus, model, rack, zone, power_zone, unhealthy = key
        census[
            NodeShape(
                node_id="",
                cpu_cores=cores,
                memory_bytes=memory,
                gpus=gpus,
                accelerator_type=model,
                gpu_memory_bytes=accelerator_memory_bytes(model) if gpus > 0 else 0,
                nvlink_domain=nvlink_domain_size(model, gpus) if gpus > 0 and model else 0,
                rack=rack,
                zone=zone,
                power_zone=power_zone,
                fabric_gbps=fabric,
                rails=0,
                unhealthy_gpus=unhealthy,
            )
        ] = count

    # Ordered canonically so two reads of an unchanged cluster produce an identical shape. The
    # shape reaches the plan cache key, and a set-ordered tuple would invalidate every memoized
    # plan on a cluster that had not changed at all. This used to sort by node id, which a
    # census entry no longer carries and which was itself an arbitrary tie-break, since Ray
    # node ids are random. Sorting by the fields instead makes the order a property of the
    # fleet rather than of which machines happened to register first.
    ordered = sorted(
        census.items(),
        # Densest first, then every remaining field, so the order is total and depends on
        # nothing but the fleet. `NodeShape` is frozen but not `order=True`, so the tie-break
        # spells the fields out rather than comparing the records.
        key=lambda item: (
            -item[0].cpu_cores,
            -item[0].gpus,
            item[0].zone,
            item[0].rack,
            item[0].power_zone,
            item[0].accelerator_type,
            -item[0].memory_bytes,
            -item[0].gpu_memory_bytes,
            -item[0].nvlink_domain,
            -item[0].fabric_gbps,
            -item[0].rails,
            item[0].unhealthy_gpus,
        ),
    )
    result = ClusterShape(
        nodes=tuple(shape for shape, _ in ordered),
        multiplicity=tuple(count for _, count in ordered),
    )
    if snapshot is not None:
        snapshot.derived["cluster_shape"] = result
    return result


#: Labels a node's availability zone can arrive under, most current first. The Kubernetes
#: `topology.kubernetes.io/zone` was the only one read, which covers a KubeRay cluster and
#: nothing else: a Ray cluster launched by the cluster launcher on EC2 or GCE carries the
#: cloud's own label instead, and a pre-1.17 Kubernetes carries the `failure-domain.beta` form.
#: On those fleets every node reported no zone, so a multi-AZ exchange was indistinguishable
#: from a single-rack one and the zone field — recorded, summarized, and now priced — was
#: uniformly empty.
_ZONE_LABELS = (
    "topology.kubernetes.io/zone",
    "failure-domain.beta.kubernetes.io/zone",
    "ray.io/availability-zone",
)


def _zone_label(labels: dict) -> str:
    """The node's availability zone under whichever label its provider wrote, `""` when none."""
    for name in _ZONE_LABELS:
        value = str(labels.get(name) or "").strip()
        if value:
            return value
    return ""


def _declared_fabric_gbps() -> float:
    """The operator's declared per-node fabric rate, `0.0` when unset.

    Declared rather than probed on purpose: this runs on the *driver*, and the driver's NIC is
    not the workers'. The one honest source for a worker's fabric rate from here is the
    operator stating it, which is exactly what `accelerator.fabric_gbps` exists for. `0.0`
    leaves the rate unknown, and every consumer then keeps the default it had.

    A per-node probe belongs on the worker that owns the NIC, which is a measurement Core would
    have to report back; until that exists, inventing a figure here would put a number nobody
    measured underneath a join order.
    """
    try:
        from batcher.config import active_config

        return max(0.0, float(active_config().accelerator.fabric_gbps))
    except Exception:  # pragma: no cover - config unavailable
        return 0.0
