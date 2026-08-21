"""The memory ceiling on a machine whose limits are not cgroups.

A container is confined by cgroups, and the engine has always read those. A *batch allocation*
is not, unless the site configured `task/cgroup` confinement, and plenty of HPC sites do not.
There a job granted 16 GiB on a 512 GiB node sees 512 GiB, sizes a hash table against it, and
is killed for exceeding a grant nothing in the process ever read. That is the memory half of
the core-count failure `available_cpu_count` already guards against.

`RLIMIT_AS` is the other one, and it binds harder than a cgroup: overshooting a cgroup gets the
process OOM-killed, while overshooting an address-space limit makes the allocator return NULL
and the query die of `MemoryError` inside a kernel that had no chance to spill instead. Grid
Engine's `h_vmem`, LSF's `-M`, PBS' `pvmem` and a plain `ulimit -v` all land there.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import memory as hwmem
from batcher._internal.site import container, scheduler

pytestmark = pytest.mark.unit

_GIB = 1 << 30
_MIB = 1 << 20


@pytest.fixture
def ceiling(monkeypatch, clean_site_env):
    """A 512 GiB host with no cgroup, no hugepages and no rlimit, so a test adds one bound."""
    monkeypatch.setattr(hwmem, "hugepage_bytes", lambda: 0)
    monkeypatch.setattr(hwmem, "cgroup_v2_dirs", lambda: ())
    monkeypatch.setattr(hwmem, "read_cgroup_bytes", lambda path: None)
    # `"SC_PHYS_PAGES"` also contains "PAGE", so the two are matched exactly.
    pages = 512 * _GIB // 4096
    monkeypatch.setattr(hwmem.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else pages)
    monkeypatch.setattr(container, "_rlimit", lambda name: 0)
    hwmem.machine_memory_bytes.cache_clear()
    yield
    hwmem.machine_memory_bytes.cache_clear()


def test_a_node_with_no_limits_reports_its_own_memory(ceiling):
    assert hwmem.machine_memory_bytes() == 512 * _GIB


def test_a_slurm_per_node_grant_binds(ceiling, monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")  # Slurm publishes mebibytes
    hwmem.machine_memory_bytes.cache_clear()
    assert hwmem.machine_memory_bytes() == 16 * _GIB


def test_a_per_cpu_grant_is_multiplied_by_the_core_grant(ceiling, monkeypatch):
    # `--mem-per-cpu=4G --cpus-per-task=8` is 32 GiB, and reading the per-CPU figure alone
    # would bound a whole task to one core's worth.
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    hwmem.machine_memory_bytes.cache_clear()
    assert hwmem.machine_memory_bytes() == 32 * _GIB


def test_a_per_node_grant_outranks_a_per_cpu_one(ceiling, monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    hwmem.machine_memory_bytes.cache_clear()
    assert hwmem.machine_memory_bytes() == 8 * _GIB


def test_a_per_cpu_grant_with_no_core_grant_claims_nothing(ceiling, monkeypatch):
    # Half a fact is not a bound: multiplying by a core count nobody published would invent
    # a ceiling, and the honest answer is the one that held before.
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    assert scheduler.scheduler_memory_bytes() is None


def test_an_address_space_limit_binds(ceiling, monkeypatch):
    monkeypatch.setattr(container, "_rlimit", lambda name: 24 * _GIB if name == "RLIMIT_AS" else 0)
    hwmem.machine_memory_bytes.cache_clear()
    assert hwmem.machine_memory_bytes() == 24 * _GIB


def test_the_tightest_of_every_bound_wins(ceiling, monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "65536")  # 64 GiB
    monkeypatch.setattr(container, "_rlimit", lambda name: 24 * _GIB if name == "RLIMIT_AS" else 0)
    hwmem.machine_memory_bytes.cache_clear()
    assert hwmem.machine_memory_bytes() == 24 * _GIB


def test_an_unlimited_rlimit_is_not_a_ceiling_of_zero(ceiling, monkeypatch):
    # `RLIM_INFINITY` reads as 0 through the shared rlimit helper, and 0 must mean "no bound"
    # rather than "no memory" — the second would collapse every budget on an ordinary host.
    import resource

    monkeypatch.setattr(
        resource, "getrlimit", lambda which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    assert container.address_space_limit_bytes() == 0
    hwmem.machine_memory_bytes.cache_clear()
    assert hwmem.machine_memory_bytes() == 512 * _GIB


def test_carbonite_admits_against_the_same_ceiling_the_planner_sized_to(ceiling, monkeypatch):
    # The two used to compute it separately and had drifted on reserved hugepages and on the
    # `memory.high` throttle threshold. The layer that was wrong is the one that decides
    # whether to spill, which is the worst place for the two views to differ.
    from batcher.carbonite.memory import probe

    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")
    probe.reset_memory_sampling()
    try:
        assert probe.total_memory_bytes() == hwmem.machine_memory_bytes() == 16 * _GIB
    finally:
        probe.reset_memory_sampling()


def test_an_undetectable_machine_falls_back_rather_than_reporting_no_memory(monkeypatch):
    from batcher.carbonite.memory import probe
    from batcher.config import active_config

    monkeypatch.setattr(hwmem, "machine_memory_bytes", lambda: 0)
    monkeypatch.setattr(probe, "machine_memory_bytes", lambda: 0)
    assert probe.total_memory_bytes() == active_config().memory.default_total_bytes


def test_an_address_space_limit_below_the_machine_is_reported(monkeypatch):
    # A query that dies of MemoryError with plenty of free RAM is a confusing failure. The
    # finding names the cause up front, and the knob each scheduler spells it with.
    monkeypatch.setattr(container, "_rlimit", lambda name: 8 * _GIB if name == "RLIMIT_AS" else 0)
    monkeypatch.setattr(container, "_host_memory_bytes", lambda: 512 * _GIB)
    monkeypatch.setattr(container, "_shm_stat", lambda: (0, 0))
    findings = container.container_findings()
    assert any("address space is limited to 8.0 GiB" in f for f in findings)
    assert any("h_vmem" in f for f in findings)


def test_an_address_space_limit_at_or_above_the_machine_is_not_a_finding(monkeypatch):
    # It changes no decision, and a line that changes no decision is one a reader learns to
    # skip -- which costs the findings that do.
    monkeypatch.setattr(container, "_rlimit", lambda name: 512 * _GIB if name == "RLIMIT_AS" else 0)
    monkeypatch.setattr(container, "_host_memory_bytes", lambda: 512 * _GIB)
    monkeypatch.setattr(container, "_shm_stat", lambda: (0, 0))
    assert not [f for f in container.container_findings() if "address space" in f]
