"""One pass over the fleet, shared by everything that needs to know its shape.

Two things classify the cluster on every distributed query, and until this module existed
they each walked every node to do it: `scaling._class_index` (what the placement phase
reads — cores, free cores, devices, zone, market type) and `fabric.shape.cluster_shape`
(what the optimizer plans against — cores, RAM, devices, rack, power zone, device health).
Overlapping fields, the same iteration, twice.

At the scale this engine targets that is the whole cost. Measured against a synthetic
100,000-node fleet, the two passes were 240 ms and 170 ms of a single query's placement
phase; nothing else in it came close. So the extraction happens once, here, into a census
keyed by everything either consumer distinguishes nodes on, and each of them groups that
census down to the fields it actually cares about — which is O(classes), and a cluster of
a hundred thousand nodes is a handful of instance types across a handful of zones.

**The key deliberately carries two readings of the availability zone.** `topology.node_zone`
and `shape._zone_label` do not recognize the same labels — the latter also reads the
pre-1.17 Kubernetes `failure-domain.beta.kubernetes.io/zone`. That is a real inconsistency
and it is not this module's to settle: reconciling them changes which fleets get zone-aware
placement, on clusters no test here can run. Keeping both readings in the key means
grouping is at least as fine as either consumer needs, so neither one's behaviour moves.

Pure by construction: it takes the node records and the two side tables rather than reading
Ray, so the snapshot in `scaling` stays the single place the cluster is read from and this
stays testable without a cluster.
"""

from __future__ import annotations

from typing import NamedTuple

from batcher._internal.accelerators import accelerator_units
from batcher.dist.executors.ray_runtime.fabric.topology import (
    POWER_ZONE_LABEL,
    RACK_LABEL,
    SPOT,
    market_type,
    node_zone,
)

__all__ = ["FleetClass", "build_census"]


class FleetClass(NamedTuple):
    """Every field either consumer distinguishes nodes on — the census key.

    A `NamedTuple` rather than a dict so it hashes and compares as a tuple (the grouping is
    a dict lookup per node, on the hot pass) while still reading by name at the two places
    that unpack it.

    Attributes:
        cpus: Nameplate cores.
        free_cpus: Cores unreserved right now, or the nameplate when Ray will not say.
        gpus: Accelerator devices Ray counts as `GPU`.
        memory: Host RAM the node advertises, `0.0` when it advertises none.
        accelerators: Non-GPU accelerator units (TPU / Trainium / Gaudi / NPU).
        accelerator_type: Device model from `ray.io/accelerator-type`, `None` when unlabelled.
        zone_label: The label key the placement zone was read from, `""` when unlabelled.
        zone: The placement zone, under `topology.node_zone`'s vocabulary.
        shape_zone: The zone under `fabric.shape`'s wider vocabulary. See the module note.
        rack: Physical enclosure, `""` when unlabelled.
        power_zone: Power domain, `""` when unlabelled.
        market_label: The label key the node's purchase mode was read from, `""` when
            unlabelled. Carried for the same reason `zone_label` is: a fleet pinned to
            on-demand capacity must be selected on the key that actually holds it, and a
            KubeRay fleet labelled by Karpenter answers nothing to `ray.io/market-type`.
        market_type: `"spot"`, `"on-demand"`, or `""` when the node says nothing. The empty
            value is *no opinion* and never on-demand: most fleets carry no market label at
            all, and reading their silence as a purchase mode would pin every stage of every
            query on them.
        unhealthy_gpus: Devices quarantined or degraded out of rotation on this node.
    """

    cpus: float
    free_cpus: float
    gpus: float
    memory: float
    accelerators: float
    accelerator_type: str | None
    zone_label: str
    zone: str
    shape_zone: str
    rack: str
    power_zone: str
    market_label: str
    market_type: str
    unhealthy_gpus: int

    @property
    def preemptible(self) -> bool:
        """Whether this class of node is spot capacity that can be reclaimed mid-query.

        Derived rather than stored so there is one statement of what counts as spot. Reads
        False for an unlabelled node, which is what keeps an unlabelled fleet behaving as it
        always has.
        """
        return self.market_type == SPOT


def build_census(
    nodes: list[dict],
    free_cpus: dict[str, float] | None,
    unhealthy: dict[str, int],
    *,
    shape_zone: object,
) -> dict[FleetClass, list[str]]:
    """Group `nodes` into classes, remembering which node ids fell into each.

    The ids are kept because one consumer genuinely needs identity — the shuffle places a
    replica outside the spot failure domain — while every other one asks an aggregate
    question that a count answers.

    A node advertising no schedulable cores is dropped: it hosts no worker, so it holds no
    share of anything derived from this.

    Args:
        nodes: Worker-eligible alive node records, as Ray reports them.
        free_cpus: Node id -> unreserved cores, or `None` to assume nameplate.
        unhealthy: Node id -> devices out of rotation; missing means none.
        shape_zone: `fabric.shape._zone_label`, passed in rather than imported because
            `shape` reads the census and importing it back would close a cycle. The rack and
            power-zone labels have no such problem — they live in `topology`, below both.

    Returns:
        Class -> the ids of the nodes in it, in the order the fleet was walked.
    """
    # Grouped on a plain tuple, with the `FleetClass` record built once per class at the end.
    # A `NamedTuple`'s constructor is a Python-level call, and paying one per *node* to
    # discover that a few thousand of them are identical is the same waste the census exists
    # to remove — it was a fifth of this pass on a 100,000-node fleet.
    grouped: dict[tuple, list[str]] = {}
    for node in nodes:
        if not node.get("Alive", True):
            continue
        resources = node.get("Resources", {}) or {}
        cpus = float(resources.get("CPU", 0.0))
        if cpus <= 0:
            continue
        labels = node.get("Labels", {}) or {}
        node_id = node.get("NodeID", "")
        gpus = float(resources.get("GPU", 0.0))
        zone_label, zone = node_zone(labels)
        key = (
            cpus,
            cpus if free_cpus is None else min(cpus, free_cpus.get(node_id, cpus)),
            gpus,
            float(resources.get("memory", 0.0)),
            accelerator_units(resources),
            labels.get("ray.io/accelerator-type"),
            zone_label,
            zone,
            shape_zone(labels),
            str(labels.get(RACK_LABEL) or ""),
            str(labels.get(POWER_ZONE_LABEL) or ""),
            *market_type(labels),
            # Capped at the node's own device count: a stale health record naming devices a
            # resized node no longer has must never drive `healthy_gpus` negative.
            min(int(gpus), unhealthy.get(node_id, 0)),
        )
        grouped.setdefault(key, []).append(node_id)
    return {FleetClass(*key): ids for key, ids in grouped.items()}
