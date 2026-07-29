"""The one hook that clears every memoized hardware reading.

Each probe in this package is memoized, because it answers a question a running process cannot
see change — its cgroup ancestry and CPU quota, the machine's RAM, cache, and topology, the
attached accelerators — and the callers ask on every terminal op. That leaves exactly one
caller who needs them re-read: a test faking the underlying `/proc`, `/sys`, or device-node
state. This is its hook, and it lives in its own module so the package façade stays a pure
re-export list.
"""

from __future__ import annotations

from batcher._internal.accelerators import reset_accelerator_probes
from batcher._internal.hardware import cache, cgroup, isa, memory, profile, storage, topology

__all__ = ["reset_hardware_probes"]

# Every memoized probe in the package, by module. Listed explicitly rather than discovered by
# scanning module attributes: a scan would silently stop covering a probe the day someone
# renamed it, and the failure mode is a test that passes against a stale reading.
_MEMOIZED = (
    (cgroup, ("cgroup_v2_dirs", "cfs_quota_count", "_read_cgroup_v2_quota")),
    (cache, ("cache_hierarchy",)),
    (memory, ("machine_memory_bytes", "page_size_bytes", "hugepage_bytes", "swap_configured")),
    (isa, ("_cpuinfo_fields", "cpu_features", "cpu_vendor", "cpu_model_name")),
    (topology, ("numa_node_count", "cpus_per_numa_node", "physical_core_count")),
    (storage, ("device_class",)),
)


def reset_hardware_probes() -> None:
    """Forget every memoized hardware reading, so the next call re-probes the OS.

    The counterpart of `carbonite.memory.probe.reset_memory_sampling`. A name currently bound
    to a test stand-in has no cache to clear and is skipped, so patching a probe out and
    resetting in either order is safe.
    """
    for module, names in _MEMOIZED:
        for name in names:
            clear = getattr(getattr(module, name, None), "cache_clear", None)
            if clear is not None:
                clear()
    profile._reset_profile()
    reset_accelerator_probes()
