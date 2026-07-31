"""Fabric-aware placement: what the accelerator fleet looks like, and where work should go.

Ray schedules against device *counts*. A GPU datacenter is not uniform in the dimension that
decides throughput: eight devices in one NVLink domain exchange at 900 GB/s, the same eight
split across hosts exchange over the network, and a rack's busway caps what its slots may draw
however many are free.

* `topology` reads the live cluster into that shape — per-node devices and models, plus the
  rack, fabric, power zone, and availability zone each node sits in.
* `placement` turns it into decisions: gang bundles that keep a collective inside one fabric,
  per-power-zone budgets, and an efficiency order for a heterogeneous fleet.
* `residency` is where a sovereignty rule reaches the scheduler: the nodes whose region every
  input of a stage permits.
* `shape` projects the same live topology onto the neutral `plan.resource.ClusterShape`, which
  is how the optimizer — forbidden from importing `dist` — gets to see the fleet at all.

Both degrade to the pre-existing, topology-blind behavior when the cluster is unreadable or
unlabelled, because a placement hint that fires on missing data moves work for a reason that is
not there.
"""

from __future__ import annotations

from batcher.dist.executors.ray_runtime.fabric.placement import (
    CollectivePlacement,
    devices_within_power_budget,
    plan_collective,
    power_zone_load,
    rank_nodes_by_efficiency,
)
from batcher.dist.executors.ray_runtime.fabric.residency import (
    fleet_regions,
    permitted_nodes,
    residency_report,
)
from batcher.dist.executors.ray_runtime.fabric.shape import cluster_shape
from batcher.dist.executors.ray_runtime.fabric.topology import (
    FABRIC_LABEL,
    LINK_CLASSES,
    POWER_ZONE_LABEL,
    RACK_LABEL,
    GpuNodeTopology,
    domain_groups,
    fits_one_domain,
    gpu_node_topology,
    interconnect_class,
    largest_local_domain,
    nvlink_domain_size,
    topology_summary,
)

__all__ = [
    "FABRIC_LABEL",
    "LINK_CLASSES",
    "POWER_ZONE_LABEL",
    "RACK_LABEL",
    "CollectivePlacement",
    "GpuNodeTopology",
    "cluster_shape",
    "devices_within_power_budget",
    "domain_groups",
    "fits_one_domain",
    "fleet_regions",
    "gpu_node_topology",
    "interconnect_class",
    "largest_local_domain",
    "nvlink_domain_size",
    "permitted_nodes",
    "plan_collective",
    "power_zone_load",
    "rank_nodes_by_efficiency",
    "residency_report",
    "topology_summary",
]
