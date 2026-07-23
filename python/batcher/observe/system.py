"""The host and engine the queries are running on — the dashboard's hardware panel.

Answers "what machine is this, and how is the engine configured on it?", which is the
context every number elsewhere in the dashboard is relative to: a 400ms aggregate means
something different on 4 cores than on 96, and a spill verdict means nothing without the
memory budget it was measured against.

**Reports what it can actually see.** CPU count comes from `_internal.hardware`, which
already resolves cgroup quotas and affinity masks rather than trusting `os.cpu_count()` —
so a container limited to 2 cores reports 2, not the host's 96. Memory and GPU come from
optional dependencies; when they are absent the field is `None` and the panel says
"unknown" instead of guessing. A dashboard that invents a number is worse than one that
admits a gap, because the gap is at least actionable.

Sampled fresh on request rather than cached: the live memory figure is the point, and the
static fields cost nothing to re-read.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from batcher._internal.hardware import available_cpu_count, gpu_inventory
from batcher._internal.native import engine_or_none

__all__ = ["system_snapshot"]


def system_snapshot() -> dict[str, Any]:
    """Host, runtime, engine, and configuration facts, as one JSON-encodable dict.

    Returns:
        A dict with ``host``, ``runtime``, ``engine``, ``config``, and ``cluster`` sections.
    """
    return {
        "host": _host(),
        "runtime": _runtime(),
        "engine": _engine(),
        "config": _config(),
        "cluster": _cluster(),
    }


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

    `psutil` is a declared dependency but documented as optional at runtime, so this
    degrades rather than raising — the dashboard must not be the thing that fails on a
    stripped-down install.
    """
    try:
        import psutil

        virtual = psutil.virtual_memory()
        return int(virtual.total), int(virtual.available)
    except Exception:  # pragma: no cover - psutil absent or unreadable
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size), None
        except (ValueError, OSError, AttributeError):
            return None, None


def _physical_cpus() -> int | None:
    """Physical core count, or None. Distinguishes real cores from SMT siblings."""
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:  # pragma: no cover - psutil absent
        return None


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
        from importlib.metadata import version

        info["version"] = version("batcher-engine")
    except Exception:  # pragma: no cover - running from a source tree, not an install
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
