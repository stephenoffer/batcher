"""NUMA and SMT topology — which cores are really independent, and where memory is cheap.

A core count is not a description of a machine. Sixteen "cores" can be sixteen physical cores
on one memory controller, eight physical cores with SMT siblings that share an execution unit,
or sixteen physical cores split across two NUMA nodes where half the memory is roughly twice
as far away. Those three machines run the same plan at very different speeds, and a fan-out
sized to the logical count is right on only the first.

The distinctions matter concretely:

* **SMT siblings share a core.** They roughly double throughput on latency-bound work (a hash
  probe stalling on memory) and add nearly nothing to work already saturating the execution
  units (a tight compute kernel), while doubling the cache pressure either way. A CPU-bound
  operator sized to the logical count runs half its threads for no gain.
* **NUMA nodes do not share memory bandwidth.** A build-side hash table allocated on one node
  and probed from every core turns every remote probe into a cross-socket round trip. The
  same table partitioned per node keeps the traffic local.

Linux-only (`/sys/devices/system/node`, `/sys/devices/system/cpu`); everything degrades to a
single-node, no-SMT answer elsewhere, which is what the engine assumed before this existed.
"""

from __future__ import annotations

import functools
import glob
import os

__all__ = [
    "cpus_per_numa_node",
    "is_numa",
    "numa_node_count",
    "physical_core_count",
    "smt_threads_per_core",
]


def _parse_cpu_list(raw: str) -> set[int]:
    """Parse a Linux CPU list like ``"0-3,8,10-11"`` into the set of CPU ids it names."""
    out: set[int] = set()
    for part in raw.strip().split(","):
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        except ValueError:
            continue
    return out


def _read_cpu_list(path: str) -> set[int]:
    """The CPU set named by a `/sys` cpulist file, or empty when absent."""
    try:
        with open(path) as f:
            return _parse_cpu_list(f.read())
    except OSError:
        return set()


@functools.lru_cache(maxsize=1)
def numa_node_count() -> int:
    """How many NUMA nodes hold CPUs this process may run on — at least 1.

    Counts only nodes that own at least one CPU in this process's affinity mask, so a
    container pinned to one socket of a two-socket host correctly reports one node. A pinned
    process is *not* on a NUMA machine for any purpose the engine cares about: all its memory
    is local, and planning for a locality problem it cannot have would cost partitioning work
    for nothing.

    Returns:
        The number of NUMA nodes with usable CPUs, at least 1.
    """
    return max(1, len(cpus_per_numa_node()))


@functools.lru_cache(maxsize=1)
def cpus_per_numa_node() -> dict[int, int]:
    """Usable CPU count per NUMA node id, restricted to this process's affinity mask.

    The map a NUMA-aware partitioner needs: how many workers to place on each node, and
    therefore how to split a build side so each node probes its own copy. Empty on a machine
    with no NUMA information exposed, which callers read as "one node".

    Returns:
        NUMA node id to the count of usable CPUs on it, empty when NUMA is not exposed.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    allowed: set[int] | None = None
    if getaffinity is not None:
        try:
            allowed = set(getaffinity(0))
        except OSError:
            allowed = None
    out: dict[int, int] = {}
    for node_dir in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        try:
            node_id = int(os.path.basename(node_dir)[4:])
        except ValueError:
            continue
        cpus = _read_cpu_list(os.path.join(node_dir, "cpulist"))
        if allowed is not None:
            cpus &= allowed
        if cpus:
            out[node_id] = len(cpus)
    return out


def is_numa() -> bool:
    """Whether this process spans more than one NUMA node.

    The gate on every locality optimization: on a single-node machine they are pure overhead,
    so each one checks this first rather than paying to partition work that has nowhere else
    to go.

    Returns:
        `True` when usable CPUs live on two or more NUMA nodes.
    """
    return numa_node_count() > 1


@functools.lru_cache(maxsize=1)
def physical_core_count() -> int:
    """Physical cores backing this process's usable CPUs — never fewer than 1.

    Derived by collapsing each CPU's ``thread_siblings_list`` to one entry, so a 2-way SMT
    machine with 16 logical CPUs reports 8. This is the right denominator for compute-bound
    fan-out, where the second sibling of a saturated core contributes almost nothing while
    still halving the cache each thread sees.

    Falls back to the logical count when the sibling files are absent, which keeps the
    pre-existing behavior on any platform that does not publish them.

    Returns:
        The usable physical core count, at least 1.
    """
    from batcher._internal.hardware.cpu import available_cpu_count

    logical = available_cpu_count()
    getaffinity = getattr(os, "sched_getaffinity", None)
    allowed: set[int] | None = None
    if getaffinity is not None:
        try:
            allowed = set(getaffinity(0))
        except OSError:
            allowed = None
    cores: set[frozenset[int]] = set()
    for cpu_dir in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        try:
            cpu_id = int(os.path.basename(cpu_dir)[3:])
        except ValueError:
            continue
        if allowed is not None and cpu_id not in allowed:
            continue
        siblings = _read_cpu_list(os.path.join(cpu_dir, "topology", "thread_siblings_list"))
        if allowed is not None:
            siblings &= allowed
        cores.add(frozenset(siblings) if siblings else frozenset({cpu_id}))
    return max(1, len(cores)) if cores else logical


def smt_threads_per_core() -> float:
    """Logical CPUs per physical core — ``1.0`` with SMT off, ``2.0`` on typical hyperthreading.

    The correction factor between the two fan-out denominators. An operator whose measured
    per-core utilization is already near saturation gains nothing from the extra sibling
    threads and loses cache to them, so it should size to `physical_core_count`; a
    latency-bound operator should size to the logical count and use the stalls.

    Returns:
        Logical CPUs per physical core, at least 1.0.
    """
    from batcher._internal.hardware.cpu import available_cpu_count

    physical = physical_core_count()
    if physical <= 0:
        return 1.0
    return max(1.0, available_cpu_count() / physical)
