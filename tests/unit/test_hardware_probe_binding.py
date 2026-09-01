"""Three ways a hardware probe reported the *host* where it had to report this *process*.

Every probe in `_internal.hardware` exists because the host's own answer is wrong inside a
container. These are the cases where one of them still gave the host's answer, each one
reachable on hardware Batcher is routinely deployed to, and each one silent:

* `physical_core_count` counted every physical core in the affinity mask and never looked at
  the CFS quota, so a cgroup throttled to 4 cores reported the host's 48;
* `swap_configured` walked the cgroup ancestry outermost-first while documenting the opposite,
  so a leaf that denies swap was masked by any ancestor that permits it;
* `reset_hardware_probes` did not clear `nvml.host_pid`, so a test faking the host PID
  namespace kept reading the previous answer.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal import hardware
from batcher._internal.hardware import cgroup, cpu, memory, nvml, topology

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_probes():
    """Re-probe the OS around every test, since every reading here is memoized."""
    hardware.reset_hardware_probes()
    yield
    hardware.reset_hardware_probes()


# --------------------------------------------------------------------------------------
# physical_core_count must not exceed the CPU budget the process actually has
# --------------------------------------------------------------------------------------


def test_physical_cores_never_exceed_the_quota_budget(monkeypatch):
    """A CFS-throttled container reports its quota, not the host's physical core count.

    `/sys` is host-wide even inside a container, so the sibling walk sees every physical core
    on the box. The affinity mask narrows it; a *bandwidth* quota does not appear in the mask
    at all. A pod pinned to nothing but throttled to 4 cores therefore reported 48 — the
    figure compute-bound fan-out is sized from, and the one that goes into the machine
    fingerprint.
    """
    # 96 logical CPUs in the mask, 48 physical cores, throttled by CFS bandwidth to 4.
    monkeypatch.setattr(topology, "affinity_cpu_ids", lambda: set(range(96)))
    monkeypatch.setattr(os, "cpu_count", lambda: 96)
    monkeypatch.setattr(cpu, "_affinity_count", lambda: 96)
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: 4)
    monkeypatch.setattr(cpu, "_allocation_cpu_count", lambda: None)

    # Two-way SMT: cpu i and cpu i+48 share a core.
    def siblings(path: str) -> set[int]:
        if "thread_siblings_list" not in path:
            return set()
        cpu_id = int(os.path.basename(path.split("/topology")[0])[3:])
        return {cpu_id % 48, cpu_id % 48 + 48}

    monkeypatch.setattr(topology, "read_cpu_list", siblings)
    monkeypatch.setattr(
        topology.glob, "glob", lambda pat: [f"/sys/devices/system/cpu/cpu{i}" for i in range(96)]
    )

    assert cpu.available_cpu_count() == 4
    cores = topology.physical_core_count()
    assert cores <= cpu.available_cpu_count(), (
        f"physical_core_count reported {cores} on a process budgeted "
        f"{cpu.available_cpu_count()} CPUs"
    )
    assert cores == 4


def test_physical_cores_still_collapse_smt_when_no_quota_binds(monkeypatch):
    """The SMT collapse is preserved: the cap only ever lowers, never raises."""
    monkeypatch.setattr(topology, "affinity_cpu_ids", lambda: set(range(16)))
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(cpu, "_affinity_count", lambda: 16)
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: None)
    monkeypatch.setattr(cpu, "_allocation_cpu_count", lambda: None)

    def siblings(path: str) -> set[int]:
        if "thread_siblings_list" not in path:
            return set()
        cpu_id = int(os.path.basename(path.split("/topology")[0])[3:])
        return {cpu_id % 8, cpu_id % 8 + 8}

    monkeypatch.setattr(topology, "read_cpu_list", siblings)
    monkeypatch.setattr(
        topology.glob, "glob", lambda pat: [f"/sys/devices/system/cpu/cpu{i}" for i in range(16)]
    )
    # 16 logical, 8 physical, no quota: the SMT answer stands.
    assert topology.physical_core_count() == 8


def test_physical_cores_are_at_least_one(monkeypatch):
    """The floor holds even when nothing is readable."""
    monkeypatch.setattr(topology, "affinity_cpu_ids", lambda: None)
    monkeypatch.setattr(topology.glob, "glob", lambda pat: [])
    assert topology.physical_core_count() >= 1


# --------------------------------------------------------------------------------------
# swap_configured must read the leaf cgroup, which is what binds
# --------------------------------------------------------------------------------------


def test_swap_configured_reads_the_leaf_cgroup_not_an_ancestor(monkeypatch):
    """A leaf denying swap beats an ancestor permitting it.

    cgroup v2 publishes `memory.swap.max` at every level and the *tightest* one binds, which
    on a delegated hierarchy is the leaf. Reading an ancestor first and stopping at its
    ``max`` reported "swap available" for a container that had been denied swap — the reading
    that selects the *later*-spill policy on a node where overshooting is terminal.
    """
    dirs = (
        "/sys/fs/cgroup",
        "/sys/fs/cgroup/slice/leaf",  # leaf-most, as `cgroup_v2_dirs` orders them
        "/sys/fs/cgroup/slice",
    )
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: dirs)
    files = {
        "/sys/fs/cgroup/memory.swap.max": "max",
        "/sys/fs/cgroup/slice/memory.swap.max": "max",
        "/sys/fs/cgroup/slice/leaf/memory.swap.max": "0",  # Kubernetes denies swap here
        "/proc/swaps": "Filename\tType\tSize\tUsed\tPriority\n/dev/sda2\tpartition\t8\t0\t-2\n",
    }

    from io import StringIO

    def opener(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    monkeypatch.setattr("builtins.open", opener)
    assert memory.swap_configured() is False


def test_swap_configured_falls_through_to_the_host_when_the_leaf_permits_it(monkeypatch):
    """A leaf that permits swap defers to whether the host actually has any."""
    dirs = ("/sys/fs/cgroup", "/sys/fs/cgroup/slice/leaf", "/sys/fs/cgroup/slice")
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: dirs)
    files = {
        "/sys/fs/cgroup/slice/leaf/memory.swap.max": "max",
        "/proc/swaps": "Filename\tType\tSize\tUsed\tPriority\n/dev/sda2\tpartition\t8\t0\t-2\n",
    }

    from io import StringIO

    def opener(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    monkeypatch.setattr("builtins.open", opener)
    assert memory.swap_configured() is True


def test_swap_configured_lets_the_tightest_ancestor_bind(monkeypatch):
    """An ancestor denying swap beats a leaf granting it — the other direction of the same rule.

    cgroup v2 enforces `memory.swap.max` at every level, so a leaf with a gibibyte of
    allowance under a slice set to `0` gets nothing. Reading only the leaf would be as wrong
    as reading only the ancestor was; the minimum is what the kernel actually applies.
    """
    dirs = ("/sys/fs/cgroup", "/sys/fs/cgroup/slice/leaf", "/sys/fs/cgroup/slice")
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: dirs)
    files = {
        "/sys/fs/cgroup/slice/memory.swap.max": "0",
        "/sys/fs/cgroup/slice/leaf/memory.swap.max": str(1 << 30),
    }

    from io import StringIO

    def opener(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    monkeypatch.setattr("builtins.open", opener)
    assert memory.swap_configured() is False


def test_swap_configured_reports_a_positive_allowance_everywhere(monkeypatch):
    """Every level granting swap reports swap available."""
    dirs = ("/sys/fs/cgroup", "/sys/fs/cgroup/slice/leaf", "/sys/fs/cgroup/slice")
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: dirs)
    files = {
        "/sys/fs/cgroup/slice/memory.swap.max": str(4 << 30),
        "/sys/fs/cgroup/slice/leaf/memory.swap.max": str(1 << 30),
    }

    from io import StringIO

    def opener(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    monkeypatch.setattr("builtins.open", opener)
    assert memory.swap_configured() is True


def test_cgroup_v2_dirs_orders_leaf_before_its_ancestors(monkeypatch):
    """The ordering `swap_configured` depends on, pinned so it cannot drift again.

    The bug above was a docstring that said "leaf-most last" against code that puts the leaf
    second. A reader who trusted the prose wrote `reversed(...)` and got ancestors first.
    """
    from io import StringIO

    def opener(path, *a, **k):
        if path == "/proc/self/cgroup":
            return StringIO("0::/slice/pod/leaf\n")
        raise OSError

    monkeypatch.setattr("builtins.open", opener)
    cgroup.cgroup_v2_dirs.cache_clear()
    dirs = cgroup.cgroup_v2_dirs()
    assert dirs[0] == "/sys/fs/cgroup", "the mount root comes first"
    assert dirs[1] == "/sys/fs/cgroup/slice/pod/leaf", "the leaf comes immediately after it"
    assert dirs[-1] == "/sys/fs/cgroup/slice", "the shallowest named ancestor comes last"
    # The property every reader actually needs: descending depth after the mount root.
    depths = [d.count("/") for d in dirs[1:]]
    assert depths == sorted(depths, reverse=True)


# --------------------------------------------------------------------------------------
# reset_hardware_probes must clear every memoized reading in the package
# --------------------------------------------------------------------------------------


def test_reset_clears_the_host_pid_memo(monkeypatch):
    """`nvml.host_pid` is memoized and was not on the reset list.

    It answers from `/proc/self/sched`, so a test faking the host PID namespace — the whole
    reason the probe exists — kept reading whatever the first call returned.
    """
    from io import StringIO

    def opener(path, *a, **k):
        if path == "/proc/self/sched":
            return StringIO("python (4242, #threads: 1)\n")
        raise OSError

    monkeypatch.setattr("builtins.open", opener)
    nvml.host_pid.cache_clear()
    assert nvml.host_pid() == 4242

    def opener2(path, *a, **k):
        if path == "/proc/self/sched":
            return StringIO("python (9999, #threads: 1)\n")
        raise OSError

    monkeypatch.setattr("builtins.open", opener2)
    hardware.reset_hardware_probes()
    assert nvml.host_pid() == 9999, "reset_hardware_probes left host_pid memoized"


def test_reset_clears_every_memoized_probe_in_the_package():
    """Mechanical completeness: no `lru_cache` in the package survives a reset.

    `probes._MEMOIZED` is maintained by hand, deliberately — a scan would stop covering a
    probe the day someone renamed it. What a hand-maintained list cannot catch is a probe
    someone *added*, and the failure mode is a test that passes against a stale reading. This
    walks the package's own source for cached probes and holds the reset hook to all of them.
    """
    import ast
    import importlib
    import pathlib

    root = pathlib.Path(hardware.__file__).parent
    cached: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        module_name = (
            "batcher._internal.hardware."
            + str(path.relative_to(root).with_suffix("")).replace("/", ".")
        ).removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if "lru_cache" in ast.unparse(decorator):
                    cached.append((module_name, node.name))

    assert cached, "found no memoized probes at all — the walk is broken, not the package"

    primed: list[tuple[str, object]] = []
    for module_name, name in cached:
        fn = getattr(importlib.import_module(module_name), name, None)
        if fn is None or not hasattr(fn, "cache_info"):
            continue
        try:  # a probe that raises still populates nothing; one that answers primes its memo
            fn(*_sample_args(name))
        except Exception:  # priming is best-effort; the reset assertion is the point
            continue
        if fn.cache_info().currsize:
            primed.append((f"{module_name}.{name}", fn))

    hardware.reset_hardware_probes()
    stale = [name for name, fn in primed if fn.cache_info().currsize]
    assert not stale, f"reset_hardware_probes() left these memoized: {stale}"


def _sample_args(name: str) -> tuple:
    """Arguments that let a keyed probe be primed; `()` for the zero-argument majority."""
    if name in ("_read_cgroup_v2_quota",):
        return ("/sys/fs/cgroup",)
    if name in ("pcie_link", "device_numa_node"):
        return ("0000:00:00.0",)
    if name == "device_identity":
        return (0,)
    return ()
