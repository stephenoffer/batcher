"""Scheduling a GPU stage against the wires, not just the device count.

Ray places work by counting devices. Two decisions it therefore leaves open decide what a
multi-device stage actually achieves, and both are made here:

* `placement` — *which* devices a stage gets (the tightest coherent group the node has, bounded
  to one fabric island when the stage exchanges) and how its shards are dealt across them
  (weighted by measured throughput, so a fast device does not wait on a slow one).
* `collective_env` — what the collective library is told about the node: which NIC each device
  is rail-aligned with, which interfaces carry the fabric, and whether peer-to-peer can help
  here, instead of leaving it to re-derive all three by probing.

Both degrade to the pre-existing behavior on a node whose topology cannot be read: no group, no
reweighting, an empty environment block. A placement hint that fires on missing data moves work
for a reason that is not there.

A subpackage rather than two more modules beside `tasks.py`: these are the *fabric* half of GPU
scheduling and they are read together, and `dist/gpu/` is at its file budget.
"""

from __future__ import annotations

from batcher.dist.gpu.fabric.collective_env import (
    COLLECTIVE_VARS,
    collective_env,
    gdr_level,
    ib_hca_list,
    merge_env,
    node_collective_env,
    p2p_disabled,
    socket_ifnames,
)
from batcher.dist.gpu.fabric.placement import (
    adaptive_shard_factor,
    device_shard_counts,
    local_device_group,
    placement_summary,
    shard_device_assignment,
)

__all__ = [
    "COLLECTIVE_VARS",
    "adaptive_shard_factor",
    "collective_env",
    "device_shard_counts",
    "gdr_level",
    "ib_hca_list",
    "local_device_group",
    "merge_env",
    "node_collective_env",
    "p2p_disabled",
    "placement_summary",
    "shard_device_assignment",
    "socket_ifnames",
]
