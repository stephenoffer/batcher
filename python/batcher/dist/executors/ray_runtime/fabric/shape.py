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
        POWER_ZONE_LABEL,
        RACK_LABEL,
        nvlink_domain_size,
    )
    from batcher.dist.executors.ray_runtime.hardware_probe import unhealthy_gpus_by_node

    # Devices the fleet has taken out of rotation, from whatever health sample is already in
    # hand — never a fresh probe, which would put a fleet-wide fan-out on every optimize.
    # Without this the `unhealthy_gpus` field was declared, documented, derived into
    # `healthy_gpus`, summarized in `ClusterShape.summary()` and read by `exchange_width`, and
    # *never once set*: every health-aware sizing decision in the engine was inert, and a
    # device fan-out was sized onto boards the scheduler would refuse to place on.
    out_of_rotation = unhealthy_gpus_by_node()

    nodes: list[NodeShape] = []
    for node in _node_records():
        resources = node.get("Resources", {}) or {}
        cores = int(float(resources.get("CPU", 0.0)))
        if cores <= 0:
            continue  # a node with no schedulable cores hosts no worker, so it holds no share
        labels = node.get("Labels", {}) or {}
        gpus = int(float(resources.get("GPU", 0.0)))
        model = str(labels.get("ray.io/accelerator-type") or "")
        node_id = str(node.get("NodeID", ""))
        nodes.append(
            NodeShape(
                node_id=node_id,
                cpu_cores=cores,
                memory_bytes=int(float(resources.get("memory", 0.0))),
                gpus=gpus,
                accelerator_type=model,
                gpu_memory_bytes=accelerator_memory_bytes(model) if gpus > 0 else 0,
                nvlink_domain=nvlink_domain_size(model, gpus) if gpus > 0 and model else 0,
                rack=str(labels.get(RACK_LABEL) or ""),
                zone=_zone_label(labels),
                power_zone=str(labels.get(POWER_ZONE_LABEL) or ""),
                fabric_gbps=_declared_fabric_gbps(),
                rails=0,
                # Capped at the node's own device count: a stale health record naming devices
                # a resized node no longer has must never drive `healthy_gpus` negative.
                unhealthy_gpus=min(gpus, out_of_rotation.get(node_id, 0)),
            )
        )
    # Ordered by node id so two reads of an unchanged cluster produce an identical shape. The
    # shape reaches the plan cache key, and a set-ordered tuple would invalidate every memoized
    # plan on a cluster that had not changed at all.
    return ClusterShape(nodes=tuple(sorted(nodes, key=lambda n: n.node_id)))


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
