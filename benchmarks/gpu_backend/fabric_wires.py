"""What this node's wires are, and what a stage placed on them would get.

The decisions in `fabric.p2p`, `fabric.rails`, `carbonite.transfer.device_exchange` and
`kyber.gpu.exchange` are all pure functions of a topology, and on a CPU host every one of them
reports "no opinion". This is what runs them against a *real* multi-GPU node: the peer matrix
the traffic actually uses, the rail each device leaves through, and what an all-to-all across
the devices is predicted to cost against staging the same bytes through host memory.

Nothing here needs Ray, cuDF, or a model. It reads `/sys` and NVML, which is the point: the
first thing to establish on a new fleet is whether Batcher can see the fabric at all, and the
second is whether the picture it built matches `nvidia-smi topo -m`.

Run:
    python benchmarks/gpu_backend/fabric_wires.py
    BENCH_EXCHANGE_GB=32 python benchmarks/gpu_backend/fabric_wires.py
"""

from __future__ import annotations

import functools
import os

print = functools.partial(print, flush=True)


def _row(label: str, value: object) -> None:
    print(f"  {label:<28} {value}")


def main() -> None:
    from batcher._internal.device_specs import device_nvlink_gbps
    from batcher._internal.hardware import accelerator_backend
    from batcher._internal.hardware.fabric import device_pcie_links
    from batcher._internal.hardware.fabric.p2p import (
        bisection_gbps,
        host_staged_pairs,
        peer_islands,
        peer_matrix,
        peer_summary,
        tightest_peer_group,
    )
    from batcher._internal.hardware.fabric.rails import rail_summary, rails
    from batcher.carbonite.transfer.device_exchange import plan_exchange, worth_device_exchange
    from batcher.carbonite.transfer.staging import plan_staging
    from batcher.kyber.gpu.exchange import device_fabric, device_net_gbps, device_net_weight

    gb = float(os.environ.get("BENCH_EXCHANGE_GB", "8"))
    print(f"backend: {accelerator_backend()}")

    matrix = peer_matrix()
    if not matrix:
        print("no device topology on this host: nothing below would be measured, only defaulted")
        return

    print("\nPEER TOPOLOGY (the fabric overlaid on the bus)")
    for i, row in enumerate(matrix):
        print(f"  gpu{i}: {' '.join(row)}")
    summary = peer_summary(matrix)
    _row("islands", summary["islands"])
    _row("pairs on the fabric", summary["fabric_pairs"])
    _row("pairs staged via the host", summary["staged_pairs"])
    _row("worst pair class", summary["class"])
    _row("tightest group of 4", tightest_peer_group(4, matrix))

    print("\nRAILS (which NIC each device leaves through)")
    rail_records = rails()
    for rail in rail_records:
        _row(rail.nic, f"devices {list(rail.devices)} at {rail.rate_gbps:.0f} Gb/s")
    layout = rail_summary(rail_records)
    _row("imbalance", f"{layout['imbalance']:.0%}")
    _row("loaded rails", f"{layout['loaded_rails']} of {layout['rails']}")

    print("\nWHAT A DEVICE BYTE COSTS")
    wires = device_fabric(0)
    _row("rail share (Gb/s)", f"{wires.rail_gbps:.1f}")
    _row("host link (GB/s)", f"{wires.host_link_gbps:.1f}")
    _row("off-node rate (GB/s)", f"{device_net_gbps(wires):.1f}")
    _row("weight vs a local byte", device_net_weight(wires))

    print(f"\nAN ALL-TO-ALL OF {gb:.0f} GB ACROSS THE DEVICES")
    devices = list(range(len(matrix)))
    nvlink = device_nvlink_gbps(os.environ.get("BENCH_DEVICE_MODEL") or None)
    links = device_pcie_links()
    pcie = max((link.bandwidth_gbps / 8.0 for link in links), default=0.0)
    plan = plan_exchange(
        devices,
        int(gb * 1e9),
        matrix,
        nvlink_gbps=nvlink,
        pcie_gbps=pcie,
        host_gbps=wires.host_link_gbps,
    )
    _row("congestion-free rounds", plan.rounds)
    _row("ring order", plan.ring)
    _row("device path (s)", f"{plan.seconds:.3f}")
    _row("host path (s)", f"{plan.host_seconds:.3f}")
    _row("speedup", f"{plan.speedup:.2f}x")
    _row("worth the device path", worth_device_exchange(plan))
    cut = bisection_gbps(devices, matrix, nvlink_gbps=nvlink, pcie_gbps=pcie)
    _row("bisection (GB/s)", f"{cut:.0f}")
    _row("pairs needing a bounce", len(host_staged_pairs(matrix)))
    _row("islands", len(peer_islands(matrix)))

    staging = plan_staging(int(gb * 1e9), wires.host_link_gbps, streams=len(devices))
    print("\nHOST-SIDE STAGING THIS LINK WANTS")
    _row("chunk", f"{staging.chunk_bytes >> 10} KiB")
    _row("in flight", staging.depth)
    _row("pinned", staging.pinned)
    _row("host memory held", f"{staging.buffer_bytes / 1e6:.0f} MB")

    print(
        "\nCompare the peer matrix against `nvidia-smi topo -m` and the rail map against "
        "`NCCL_DEBUG=INFO`. A disagreement is a bug in the probe, not in the hardware."
    )


if __name__ == "__main__":
    main()
