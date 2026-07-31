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
    """Every alive node as Ray reports it, or `[]` when the topology is unreadable."""
    try:
        import ray

        if not ray.is_initialized():
            return []
        return [n for n in ray.nodes() if n.get("Alive", True)]
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
      datasheet says. `0` for an unrecognized model, which `NodeShape.local_domain` reads as
      "no narrower than the node" — the assumption in force before any of this existed.

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

    nodes: list[NodeShape] = []
    for node in _node_records():
        resources = node.get("Resources", {}) or {}
        cores = int(float(resources.get("CPU", 0.0)))
        if cores <= 0:
            continue  # a node with no schedulable cores hosts no worker, so it holds no share
        labels = node.get("Labels", {}) or {}
        gpus = int(float(resources.get("GPU", 0.0)))
        model = str(labels.get("ray.io/accelerator-type") or "")
        nodes.append(
            NodeShape(
                node_id=str(node.get("NodeID", "")),
                cpu_cores=cores,
                memory_bytes=int(float(resources.get("memory", 0.0))),
                gpus=gpus,
                accelerator_type=model,
                gpu_memory_bytes=accelerator_memory_bytes(model) if gpus > 0 else 0,
                nvlink_domain=nvlink_domain_size(model, gpus) if gpus > 0 and model else 0,
                rack=str(labels.get(RACK_LABEL) or ""),
                zone=str(labels.get("topology.kubernetes.io/zone") or ""),
                power_zone=str(labels.get(POWER_ZONE_LABEL) or ""),
                fabric_gbps=_declared_fabric_gbps(),
                rails=0,
            )
        )
    # Ordered by node id so two reads of an unchanged cluster produce an identical shape. The
    # shape reaches the plan cache key, and a set-ordered tuple would invalidate every memoized
    # plan on a cluster that had not changed at all.
    return ClusterShape(nodes=tuple(sorted(nodes, key=lambda n: n.node_id)))


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
