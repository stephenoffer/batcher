"""Checking a GPU-cloud node before you trust its throughput.

On a rented GPU node the things that halve a job's rate do not fail it. A host link that
renegotiated to half width, a device whose memory has repaired itself as far as it can, an
NVLink fabric that is down, a spill landing on the container's 40 GB overlay instead of the
node's 7 TB of NVMe: every one of those returns correct results, slowly, with nothing in the
job's own timings to say why. This script prints all of them in one pass.

Run it once on each node shape you rent, before the first real job.

    python examples/operations/gpu_cloud.py
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.hardware.fabric import (
    degraded_device_links,
    fabric_bandwidth_gbps,
    nvlink_summary,
    rdma_summary,
)
from batcher._internal.hardware.faults import device_faults, faulted_devices, xid_readable
from batcher._internal.site import local_scratch_root, scheduler_job, site_summary


def main() -> None:
    # Where this process is: the platform, the instance type, and what scheduled it. Every
    # field is empty on a laptop, which is the honest answer rather than a guess.
    site = site_summary()
    print("site:", site)
    assert set(site) >= {"provider", "scheduler", "neocloud"}

    # The allocation, as its scheduler describes it. Under Slurm this is the node list and
    # the per-node device grant before Ray has started; under Kubernetes it is the node the
    # pod landed on.
    job = scheduler_job()
    print(f"job: {job.kind} across {len(job.nodes)} node(s), {job.gpus_per_node} device(s) each")
    assert job.total_gpus >= 0

    # The full accelerator report — devices, fleet, power, and (on a GPU cloud) the site and
    # fabric sections. This is the thing to paste into a bug report.
    bt.show_accelerators()
    report = bt.accelerators()
    assert "devices" in report

    # The fabric a cross-node shuffle actually runs on. Zero here means either no RDMA
    # hardware or a container that cannot see `/sys/class/infiniband` — set
    # `accelerator.fabric_gbps` for the second case, so the optimizer prices a shuffled byte
    # against the fabric the node really has.
    print("rdma:", rdma_summary())
    print("nvlink:", nvlink_summary())
    if fabric_bandwidth_gbps() == 0.0:
        print("no readable RDMA fabric: the cost model keeps its default network weight")

    # Everything wrong with this node in one list, which is the form a deployment check
    # wants: run it before the fleet takes work rather than reading a report after a job
    # came back slow. Empty on a healthy node *and* on one that could read nothing, so a
    # check that treats an empty list as proof of health is checking the wrong thing.
    from batcher.api.session.accelerators import accelerator_problems

    problems = accelerator_problems()
    print(f"problems: {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")
    assert isinstance(problems, list)

    # Silent hardware faults, by device. An empty list here is the answer you want.
    degraded = degraded_device_links()
    for link in degraded:
        print(f"DEGRADED LINK {link.address}: {link.degradation_ratio:.0%} of nameplate")
    for fault in faulted_devices():
        print(f"FAULTED DEVICE {fault.uuid}: reset={fault.needs_reset} rma={fault.remap_failure}")
    assert isinstance(degraded, tuple)

    # Whether the fault sources can be read at all. "No faults" and "no visibility" look
    # identical from the values alone, and only one of them is good news.
    print("kernel log readable:", xid_readable(), "| devices probed:", len(device_faults()))

    # Where an out-of-core query will spill. `None` means no fast local volume was found and
    # a system tempdir will be used — correct on a laptop, and worth fixing on a GPU node
    # whose NVMe is mounted somewhere this did not look (set `BATCHER_SCRATCH_DIR`).
    scratch = local_scratch_root()
    print("local scratch:", scratch or "<tempdir>")

    # Nothing above changes a result, so a query still runs the same on any of these nodes.
    total = bt.from_pydict({"v": [1, 2, 3]}).agg(bt.col("v").sum()).to_pydict()
    assert total == {"v": [6]}
    print("engine check:", total)


if __name__ == "__main__":
    main()
