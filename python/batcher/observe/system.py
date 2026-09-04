"""The host and engine the queries are running on — the dashboard's hardware panel.

Answers "what machine is this, and how is the engine configured on it?", which is the
context every number elsewhere in the dashboard is relative to: a 400ms aggregate means
something different on 4 cores than on 96, and a spill verdict means nothing without the
memory budget it was measured against.

**Reports what it can actually see.** Every CPU and memory figure comes from
`_internal.hardware`, which resolves cgroup quotas and affinity masks rather than trusting
`os.cpu_count()` or the host's RAM — so a container limited to 2 cores and 8 GiB reports
those, not the host's 96 and 184. The core count always did; the memory total and the
physical-core count did not, and read `psutil` (or `SC_PHYS_PAGES`) directly, which report
the *host*. A panel pairing a cgroup-aware core count with a host memory total describes a
machine nobody is running on, and the memory budget is the very figure this module's own
spill verdict has to be read against.

GPU inventory still comes from an optional dependency; when it is absent the field is
`None` and the panel says "unknown" instead of guessing. A dashboard that invents a number
is worse than one that admits a gap, because the gap is at least actionable.

Sampled fresh on request rather than cached: the live memory figure is the point, and the
static fields cost nothing to re-read.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from batcher._internal.hardware import (
    available_cpu_count,
    gpu_inventory,
    hardware_profile,
    machine_memory_bytes,
    physical_core_count,
)
from batcher._internal.native import engine_or_none

__all__ = ["system_snapshot"]


def system_snapshot() -> dict[str, Any]:
    """Host, runtime, engine, and configuration facts, as one JSON-encodable dict.

    The ``hardware`` section is the full measured profile — NUMA nodes, SMT siblings, the
    cache hierarchy, vector width, page size, the scratch device's class — plus the
    ``fingerprint`` that names this machine class. That fingerprint is worth surfacing rather
    than hiding: it is the key every learned parameter is stored under, so it is the answer to
    "why did the optimizer start cold on this node?" (a different machine class) and to "will
    what this node learns help the next one?" (only if the fingerprints match).

    On a GPU cloud there is a ``site`` section too: the provider, what scheduled the process,
    the fabric it is on, and the local volume its spills will land on. Those are the facts that
    explain a default rather than a reading — why a spill went where it did, why the optimizer
    priced a shuffle the way it did — and they are omitted entirely on a machine that has none
    of them, so a laptop's snapshot is the size it always was.

    Returns:
        A dict with ``host``, ``hardware``, ``runtime``, ``engine``, ``config``, and
        ``cluster`` sections, plus ``site`` on a machine that reports one.
    """
    snapshot = {
        "host": _host(),
        "hardware": hardware_profile().to_dict(),
        "runtime": _runtime(),
        "engine": _engine(),
        "config": _config(),
        "cluster": _cluster(),
    }
    site = _site()
    if site:
        snapshot["site"] = site
    return snapshot


def _host() -> dict[str, Any]:
    """CPU, memory, and OS facts for the machine (or container) this process sees."""
    total, available = _memory()
    return {
        "cpus": available_cpu_count(),
        "cpus_physical": _physical_cpus(),
        "arch": platform.machine(),
        "platform": platform.system(),
        "hostname": platform.node(),
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "gpus": gpu_inventory(),
    }


def _memory() -> tuple[int | None, int | None]:
    """``(total, available)`` RAM in bytes, or ``(None, None)`` when unobservable.

    `total` is the **binding ceiling this process runs under** — `machine_memory_bytes`,
    which is `min(host RAM less reserved hugepages, memory.max, memory.high, a scheduler's
    grant, RLIMIT_AS)`. It used to be `psutil.virtual_memory().total`, which is the host's
    RAM: on an 8 GiB pod of a 184 GiB node the panel reported 184, beside a core count that
    correctly said 2.

    `available` stays live from `psutil`, because nothing at this layer publishes a
    cgroup-aware "available" (Carbonite's `memory.probe` does, and `observe` must not import
    a subsystem). It is clamped to `total` so the pair cannot report more free memory than
    the process is allowed to hold, which is what a host reading beside a cgroup ceiling
    would otherwise do.

    `psutil` is a declared dependency but documented as optional at runtime, so this
    degrades rather than raising — the dashboard must not be the thing that fails on a
    stripped-down install.
    """
    total = machine_memory_bytes() or None
    available: int | None = None
    try:
        import psutil

        virtual = psutil.virtual_memory()
        available = int(virtual.available)
        if total is None:
            total = int(virtual.total)
    except Exception:  # pragma: no cover - psutil absent or unreadable
        if total is None:
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                total = int(pages * page_size)
            except (ValueError, OSError, AttributeError):
                total = None
    if total is not None and available is not None:
        available = min(available, total)
    return total, available


def _physical_cpus() -> int | None:
    """Physical cores backing the CPUs this process may use, or `None` when unreadable.

    From `_internal.hardware`, for the same reason the logical count is: `psutil.cpu_count`
    counts the host's cores, so it ignored both a cpuset pin and a CFS bandwidth quota and
    reported 48 for a process budgeted 4.
    """
    return physical_core_count() or None


def _runtime() -> dict[str, Any]:
    """Python and process facts."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "pid": os.getpid(),
        "executable": sys.executable,
    }


def _engine() -> dict[str, Any]:
    """Batcher and native-engine versions, and whether the compiled engine loaded."""
    info: dict[str, Any] = {"version": None, "native": None, "native_loaded": False}
    # Read from installed metadata, not `import batcher`. Importing the root package pulls in
    # `api` and with it every subsystem, which would make this neutral module transitively
    # depend on all of them — the exact edge the `observe is a neutral sink layer` contract
    # exists to forbid, and one the import graph does catch.
    try:
        from importlib.metadata import PackageNotFoundError, version

        info["version"] = version("batcher-engine")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        pass
    # Through `_internal.native`, never `batcher._native` and never `api.session` — the
    # first would forge a phantom import cycle, the second would reach up two layers.
    native = engine_or_none()
    if native is not None:
        info["native"] = getattr(native, "__engine_version__", None)
        info["native_loaded"] = True
    return info


def _config() -> dict[str, Any]:
    """The tunables that explain the numbers elsewhere in the dashboard.

    A deliberately small selection. The full `Config` is large and mostly irrelevant to
    reading a run; these are the fields a person actually correlates against a timing —
    how work is sized, how much memory it may use, and whether spilling is even possible.
    """
    try:
        from batcher.config import active_config

        cfg = active_config()
    except Exception:  # pragma: no cover
        return {}
    return {
        "parallelism": cfg.execution.parallelism,
        "morsel_rows": cfg.execution.morsel_rows,
        "morsel_bytes": cfg.execution.morsel_bytes,
        "split_bytes": cfg.execution.split_bytes,
        "max_memory_bytes": cfg.memory.max_memory_bytes,
        "soft_limit": cfg.memory.soft_limit,
        "hard_limit": cfg.memory.hard_limit,
        "spill_enabled": cfg.memory.max_memory_bytes is not None,
        "spill_compression": cfg.memory.spill_compression,
        "verbosity": cfg.observability.verbosity,
        "log_level": cfg.observability.resolved_log_level,
        "adaptive_morsel_sizing": cfg.execution.adaptive_morsel_sizing,
    }


def _cluster() -> dict[str, Any]:
    """Ray cluster facts when one is attached, else ``{"attached": False}``.

    Never *starts* Ray — a dashboard that initialized a cluster as a side effect of being
    opened would be a genuinely harmful surprise. Reports only an already-running one.
    """
    try:
        import ray

        if not ray.is_initialized():
            return {"attached": False}
        resources = ray.cluster_resources()
        available = ray.available_resources()
        return {
            "attached": True,
            "nodes": len([n for n in ray.nodes() if n.get("Alive")]),
            "cpus": resources.get("CPU"),
            "gpus": resources.get("GPU"),
            "memory_bytes": resources.get("memory"),
            "cpus_available": available.get("CPU"),
        }
    except Exception:  # pragma: no cover - ray not installed or not initialized
        return {"attached": False}


def _site() -> dict[str, Any]:
    """Where this process is running, and on what wire — empty off a GPU cloud.

    Deliberately the facts that *explain a default* rather than the ones that measure a run.
    A reader looking at a slow query wants to know that the spill went to the container
    overlay because no local volume was found, or that the shuffle was priced against a fabric
    this container cannot see; neither is visible anywhere else in the snapshot.
    """
    from batcher._internal.hardware.fabric import fabric_bandwidth_gbps, rdma_summary
    from batcher._internal.site import local_scratch_root, site_profile, site_summary

    profile = site_profile()
    fabric = rdma_summary()
    scratch = local_scratch_root()
    # `site_summary` is the one description of the site, shared with the accelerator report.
    # Assembling a second one here is how the dashboard came to show a different set of facts
    # than `bt.accelerators()` about the same machine — and the shape of the job, which only
    # the summary carries, never reached the snapshot at all.
    out: dict[str, Any] = dict(site_summary())
    if not (profile.known or out["scheduler"] != "none" or fabric["ports"] or scratch):
        return {}
    out["scratch_dir"] = scratch or ""
    # Empty strings say nothing and cost a reader a line; the summary keeps them so its own
    # shape is stable, and this drops them because a snapshot is read by a person.
    for key in ("instance_type", "region", "node_name"):
        if not out.get(key):
            out.pop(key, None)
    if fabric["ports"]:
        out["fabric_ports"] = fabric["active_ports"]
        out["fabric_gbps"] = fabric_bandwidth_gbps()
    return out
