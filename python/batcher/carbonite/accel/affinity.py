"""Putting a device's host-side work on the cores next to it, and knowing when it is shared.

A GPU worker is half host. It reads files, decodes, stages buffers, and hands them to a device
— and on a dense node those two halves can end up on opposite sockets. When they do, the input
crosses the inter-socket link on the way into the staging buffer and again on the way out to
the DMA engine, for work that had no reason to leave its own socket. The device reports normal
utilization the whole time, because it *is* busy: busy waiting.

Two policies, both of which do nothing at all where the facts are unavailable:

* **Bind the worker to its device's NUMA node.** The kernel publishes which CPUs are local to
  each PCI device; `hardware.fabric.device_links` reads it, and this decides whether to apply
  it. Binding is refused when it would leave the process fewer cores than it needs, because a
  worker pinned to one core to save a memory hop is a worker that has stopped decoding.
* **Notice a shared device.** Under the CUDA Multi-Process Service several processes submit to
  one device through one scheduler, which is what makes co-tenancy on a large device efficient
  — and it means this process is not alone on its device, so sizing a pool to the *whole*
  device is how every tenant gets an out-of-memory error at once.

Carbonite's lane: both are resource decisions, made against facts layer 0 measured. Nothing
here computes a result or rewrites a plan.
"""

from __future__ import annotations

import os

__all__ = [
    "MIN_BOUND_CPUS",
    "bind_host_threads_to_device",
    "device_affinity_summary",
    "feeder_cpus_for_device",
    "mps_active",
    "mps_client_share",
]

#: Fewer usable cores than this and binding is refused. A worker's host half is a decode
#: pipeline; pinning it into a corner to save a memory hop trades a bandwidth problem for a
#: throughput one, and the trade is bad at any node count. Four is the smallest set on which a
#: reader, a decoder, and a submitting thread still overlap.
MIN_BOUND_CPUS = 4

#: The environment variable the CUDA Multi-Process Service sets in every client process. Its
#: presence is what the CUDA runtime itself keys on, so it is the honest signal rather than an
#: inference from process listings.
_MPS_PIPE_VAR = "CUDA_MPS_PIPE_DIRECTORY"

#: How many clients an MPS daemon is serving, when the deployment publishes it. There is no
#: portable API for this — the daemon knows and the clients do not — so an operator running
#: co-tenanted devices sets it, and a deployment that does not simply gets the single-tenant
#: answer it had before.
_MPS_CLIENTS_VAR = "BATCHER_MPS_CLIENTS"


def feeder_cpus_for_device(ordinal: int = 0) -> tuple[int, ...]:
    """The CPUs a device's host-side feeder threads should run on.

    Args:
        ordinal: The device's index *as this process sees it* — CUDA's numbering, which starts
            at zero in every worker however many devices the host has. It is translated through
            `CUDA_VISIBLE_DEVICES` to the host's own index, because a worker handed the host's
            device 5 calls it device 0 and asking NVML about "device 0" would return a
            different board's NUMA node. On a node of identical devices that answer even looks
            right.

    Returns:
        CPU ids local to that device and usable by this process, ascending. Empty when the
        device is unknown, when the kernel publishes no mapping, or when the usable set is too
        small to bind into — all of which mean "leave the scheduler alone".
    """
    from batcher._internal.hardware.fabric.device_links import (
        device_cpu_affinity,
        gpu_pci_addresses,
        visible_device_indices,
    )

    visible = visible_device_indices()
    if ordinal < 0 or ordinal >= len(visible):
        return ()
    addresses = gpu_pci_addresses()
    index = visible[ordinal]
    # Positional: `gpu_pci_addresses` keeps one slot per NVML device and leaves the slot empty
    # where the driver refused, so this index is the device it names. An empty slot means the
    # device did not publish an address, which is a reason to leave the scheduler alone.
    if index >= len(addresses) or not addresses[index]:
        return ()
    cpus = device_cpu_affinity(addresses[index])
    return cpus if len(cpus) >= MIN_BOUND_CPUS else ()


def bind_host_threads_to_device(ordinal: int = 0) -> tuple[int, ...]:
    """Pin this process to the cores next to its device, where that is safe and possible.

    **Call this before anything sizes itself.** The mask decides how wide every thread pool
    in the process should be, and the probes that report it are memoized, so binding after
    they have been read leaves the pools sized for a machine this process no longer has. A
    change here therefore invalidates them.

    Idempotent and safe to call on every task: a process already bound to exactly the right
    set is left alone, and every failure path returns empty rather than raising. Turned off
    entirely by `accelerator.bind_host_to_device_numa`, for a deployment whose own scheduler
    already places threads and would fight this one. Affinity is a
    performance property, and a worker that refuses to start because it could not set one has
    turned an optimization into an outage.

    Args:
        ordinal: The device's index as this process sees it (CUDA's numbering).

    Returns:
        The CPU set now in force, or empty when nothing was changed — because the mapping was
        unreadable, because the local set was too small, or because the platform has no
        affinity control.
    """
    from batcher.config import active_config

    if not active_config().accelerator.bind_host_to_device_numa:
        return ()
    cpus = feeder_cpus_for_device(ordinal)
    if not cpus:
        return ()
    setaffinity = getattr(os, "sched_setaffinity", None)
    getaffinity = getattr(os, "sched_getaffinity", None)
    if setaffinity is None or getaffinity is None:
        return ()
    try:
        if set(getaffinity(0)) == set(cpus):
            return cpus  # already bound; the second task on this worker pays nothing
        setaffinity(0, set(cpus))
    except OSError:
        return ()
    # Narrowing the mask changes what `available_cpu_count` and every probe derived from it
    # should answer, and those are memoized for the process. Left stale, a worker bound to
    # half a node's cores would keep sizing its thread pools to the whole node — which is the
    # oversubscription those probes exist to prevent, reintroduced by the very call that was
    # supposed to place the work well. Both engines re-read the mask on the next probe.
    from batcher._internal.hardware import reset_hardware_probes

    reset_hardware_probes()
    return cpus


def mps_active() -> bool:
    """Whether this process submits work through the CUDA Multi-Process Service.

    Returns:
        True when an MPS pipe directory is set for this process. False elsewhere, including on
        a device shared *without* MPS, which this cannot see and which is a different problem:
        without MPS, co-tenants time-slice rather than share a scheduler.
    """
    return bool(os.environ.get(_MPS_PIPE_VAR, "").strip())


def mps_client_share() -> float:
    """The fraction of a shared device this process should size itself against.

    Args:
        None.

    Returns:
        `1.0` when the device is this process's alone or the tenancy is unpublished — the
        assumption every sizing decision already makes. Otherwise `1 / clients`, so four
        co-tenants each plan for a quarter of the device instead of four processes each
        planning for all of it and discovering the conflict as a simultaneous OOM.
    """
    if not mps_active():
        return 1.0
    try:
        clients = int(os.environ.get(_MPS_CLIENTS_VAR, "").strip())
    except ValueError:
        return 1.0
    return 1.0 / clients if clients > 1 else 1.0


def device_affinity_summary(ordinal: int = 0) -> dict:
    """What this worker's host half is bound to, for the decision log and the dashboard.

    Args:
        ordinal: The device's index as this process sees it (CUDA's numbering).

    Returns:
        The device's host index and NUMA node, the local CPU count, how many CPUs this process
        may use, and whether the device is shared through MPS. Every field is neutral on a node
        where none of it is readable.
    """
    from batcher._internal.hardware.fabric.device_links import (
        device_numa_nodes,
        visible_device_indices,
    )

    visible = visible_device_indices()
    index = visible[ordinal] if 0 <= ordinal < len(visible) else -1
    nodes = device_numa_nodes()
    getaffinity = getattr(os, "sched_getaffinity", None)
    usable = len(getaffinity(0)) if getaffinity is not None else 0
    return {
        "device_index": index,
        "numa_node": nodes[index] if 0 <= index < len(nodes) else -1,
        "local_cpus": len(feeder_cpus_for_device(ordinal)),
        "usable_cpus": usable,
        "mps": mps_active(),
        "device_share": mps_client_share(),
    }
