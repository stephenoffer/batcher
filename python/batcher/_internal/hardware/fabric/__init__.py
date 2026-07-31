"""The interconnect a node sits on — RDMA NICs, PCIe links, and NVLink.

`hardware.topology` describes the CPUs and `accelerators` describes the devices. Neither
describes the wires between them, and on a GPU datacenter the wires are what the run is
usually waiting on: a shuffle moving a terabyte between nodes is bounded by the NIC, a host
staging buffer feeding a device is bounded by the PCIe link it crosses, and a collective
inside a node is bounded by whether NVLink is actually up. All three are readable, none is
reported by a cluster manager, and each of them fails *quietly* — a link that renegotiated
to half width still works, it just halves the stage.

Organized by the wire each module reads:

* `rdma` — InfiniBand and RoCE NICs from `/sys/class/infiniband`: rate, state, partition.
* `pcie` — a device's PCIe link, its NUMA home, and how far apart two devices are on the bus.
* `device_links` — the join of the two: which host link each accelerator is *actually* on.
* `counters` — what an RDMA port has carried and what it got wrong doing it: the signal
  that predicts a cable failure before the port ever leaves `ACTIVE`.
* `nvlink` — per-device NVLink state and error counters through NVML.

Every entry point degrades to an empty or neutral answer off Linux, without the driver, or
inside a container that did not mount the relevant `/sys` tree. A caller that gets nothing
keeps whatever default it had, which is the behavior the engine had before this existed.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.hardware.fabric.counters import (
    ERROR_COUNTERS,
    PortCounters,
    fabric_error_total,
    port_counters,
    throughput_delta,
)
from batcher._internal.hardware.fabric.device_links import (
    degraded_device_links,
    device_link_efficiency,
    device_pcie_links,
    gpu_pci_addresses,
)
from batcher._internal.hardware.fabric.ethernet import (
    EthernetLink,
    ethernet_bandwidth_gbps,
    ethernet_links,
    ethernet_summary,
)
from batcher._internal.hardware.fabric.nvlink import (
    NvLinkStatus,
    nvlink_degraded_devices,
    nvlink_status,
    nvlink_summary,
    p2p_pairs,
)
from batcher._internal.hardware.fabric.pcie import (
    PCIE_CLASSES,
    PcieLink,
    degraded_pcie_links,
    device_numa_node,
    pcie_bandwidth_gbps,
    pcie_class,
    pcie_link,
)
from batcher._internal.hardware.fabric.rdma import (
    RdmaDevice,
    active_rdma_devices,
    fabric_bandwidth_gbps,
    fabric_interface_address,
    fabric_partition,
    rdma_available,
    rdma_devices,
    rdma_link_layers,
    rdma_net_interfaces,
    rdma_summary,
    reset_fabric_probes,
)

__all__ = [
    "ERROR_COUNTERS",
    "PCIE_CLASSES",
    "EthernetLink",
    "NvLinkStatus",
    "PcieLink",
    "PortCounters",
    "RdmaDevice",
    "active_rdma_devices",
    "degraded_device_links",
    "degraded_pcie_links",
    "device_link_efficiency",
    "device_numa_node",
    "device_pcie_links",
    "ethernet_bandwidth_gbps",
    "ethernet_links",
    "ethernet_summary",
    "fabric_bandwidth_gbps",
    "fabric_error_total",
    "fabric_interface_address",
    "fabric_partition",
    "gpu_pci_addresses",
    "nvlink_degraded_devices",
    "nvlink_status",
    "nvlink_summary",
    "p2p_pairs",
    "pcie_bandwidth_gbps",
    "pcie_class",
    "pcie_link",
    "port_counters",
    "rdma_available",
    "rdma_devices",
    "rdma_link_layers",
    "rdma_net_interfaces",
    "rdma_summary",
    "reset_fabric_probes",
    "throughput_delta",
]
