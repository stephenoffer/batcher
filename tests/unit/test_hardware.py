"""Effective hardware detection — affinity/cgroup aware, and the machine-class fingerprint.

Two things are under test. First, that every probe reports what this *process* really has
rather than what the host advertises, because inside a container those differ and sizing to
the host over-subscribes. Second, that the fingerprint splits machine classes that are
genuinely different and merges ones that are genuinely alike — the property every
hardware-scoped learned parameter depends on.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal import accelerators, hardware
from batcher._internal.hardware import cache, cgroup, cpu, isa, memory, profile, storage, topology

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_probes():
    """Re-probe the OS around every test here.

    The hardware readings are memoized for the process (they answer questions a running
    process cannot see change), so a test that fakes `/proc` or `/sys` has to invalidate
    them going in — and again coming out, so its fake state never leaks into the next test.
    """
    hardware.reset_hardware_probes()
    yield
    hardware.reset_hardware_probes()


def _fake_open(files: dict[str, str]):
    """An `open` stand-in serving `files` and raising `OSError` for anything else."""
    from io import StringIO

    def opener(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    return opener


def test_available_cpu_count_is_positive():
    assert hardware.available_cpu_count() >= 1


def test_available_cpu_count_takes_the_min_of_host_affinity_quota(monkeypatch):
    # A 64-core host, pinned to 8 by cpuset affinity, throttled to 4 by the CFS quota →
    # the process may really use 4. os.cpu_count() alone (64) would over-subscribe 16x.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(cpu, "_affinity_count", lambda: 8)
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: 4)
    assert hardware.available_cpu_count() == 4
    # Affinity tighter than quota → affinity wins.
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: 16)
    assert hardware.available_cpu_count() == 8
    # Neither discoverable (bare metal) → the host count stands.
    monkeypatch.setattr(cpu, "_affinity_count", lambda: None)
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: None)
    assert hardware.available_cpu_count() == 64
    # Never zero even if every source lies low.
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.setattr(cpu, "_affinity_count", lambda: 0 or None)
    assert hardware.available_cpu_count() == 1


def test_cfs_quota_count_parses_cgroup_v2(monkeypatch):
    # cgroup v2 `cpu.max` = "<quota> <period>"; 200000/100000 = 2 whole cores (ceil).
    import builtins

    # Namespaced pod: /proc/self/cgroup is "0::/", so the limit is read at the mount root.
    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/cpu.max": "200000 100000",  # 2 cores
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    assert cgroup.cfs_quota_count() == 2

    # An unlimited quota ("max") yields None (fall back to host/affinity). The quota is
    # memoized per process, so changing the simulated file means re-probing.
    files["/sys/fs/cgroup/cpu.max"] = "max 100000"
    hardware.reset_hardware_probes()
    assert cgroup.cfs_quota_count() is None


def test_cfs_quota_reads_non_namespaced_leaf_cgroup(monkeypatch):
    # A Ray worker in a delegated cgroup with NO namespace: the mount root reads unlimited,
    # but the real limit lives at the process's own leaf (from /proc/self/cgroup). Missing it
    # would over-subscribe. The tightest limit across root+leaf must win.
    import builtins

    files = {
        "/proc/self/cgroup": "0::/system.slice/ray-worker-7.scope\n",
        "/sys/fs/cgroup/cpu.max": "max 100000",  # root: unlimited
        "/sys/fs/cgroup/system.slice/ray-worker-7.scope/cpu.max": "400000 100000",  # leaf: 4 cores
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    assert cgroup.cfs_quota_count() == 4  # the leaf limit, not the root's "unlimited"


def test_cfs_quota_takes_tightest_across_the_hierarchy(monkeypatch):
    # cgroup v2 enforces the quota at every level: a limit set on a PARENT slice (not the
    # leaf, not the root) must still bind. The tightest cpu.max in the whole chain wins.
    import builtins

    files = {
        "/proc/self/cgroup": "0::/parent.slice/child.scope\n",
        "/sys/fs/cgroup/cpu.max": "max 100000",  # root: unlimited
        "/sys/fs/cgroup/parent.slice/cpu.max": "600000 100000",  # parent: 6 cores (the binder)
        "/sys/fs/cgroup/parent.slice/child.scope/cpu.max": "max 100000",  # leaf: unlimited
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    assert cgroup.cfs_quota_count() == 6  # the parent slice's limit, missed by root+leaf only


def test_cpu_contention_reports_load_relative_to_the_usable_cores(monkeypatch):
    # The denominator must be the cores this process may actually use, not the host's. Inside
    # a quota-throttled container those differ, and dividing by the host count would report a
    # saturated box as idle — the exact misreading that makes a diagnosis blame the plan.
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(os, "getloadavg", lambda: (8.0, 8.0, 8.0))
    monkeypatch.setattr(cpu, "cgroup_throttled_ratio", lambda: None)
    monkeypatch.setattr(cpu, "cgroup_pressure", dict)
    assert hardware.cpu_contention()["load_per_core"] == 2.0
    # A key the platform cannot answer is omitted, never defaulted to a reassuring zero.
    assert "throttled_ratio" not in hardware.cpu_contention()


def test_cpu_contention_survives_a_platform_without_loadavg(monkeypatch):
    def boom():
        raise OSError("no loadavg here")

    monkeypatch.setattr(os, "getloadavg", boom)
    monkeypatch.setattr(cpu, "cgroup_throttled_ratio", lambda: 0.25)
    monkeypatch.setattr(cpu, "cgroup_pressure", dict)
    out = hardware.cpu_contention()
    assert "load_per_core" not in out
    assert out["throttled_ratio"] == 0.25


def test_psi_stall_shares_are_normalized_to_a_fraction(monkeypatch):
    # PSI reports avg10 as a percentage; every other contention signal in the codebase is a
    # fraction. Mixing the two units would make a 12% stall read as a 12x oversubscription.
    import builtins

    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/cpu.pressure": "some avg10=12.50 avg60=1.00 avg300=0.00 total=1\n",
        "/sys/fs/cgroup/io.pressure": "some avg10=100.00 avg60=1.00 avg300=0.00 total=1\n",
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    out = cgroup.cgroup_pressure()
    assert out["cpu_stall"] == pytest.approx(0.125)
    assert out["io_stall"] == 1.0
    # memory.pressure is absent here, and an absent measurement is omitted rather than zeroed:
    # a zero would read as "memory was never short", which is a claim nothing measured.
    assert "memory_stall" not in out


def test_cpu_oversubscription_folds_queueing_and_stalling_together(monkeypatch):
    # A box with twice the runnable work as cores, half of whose wall time is stalled, is
    # worse than either signal alone says. Fan-out gating needs the combined figure.
    monkeypatch.setattr(cpu, "cpu_contention", lambda: {"load_per_core": 2.0, "cpu_stall": 0.5})
    assert cpu.cpu_oversubscription() == pytest.approx(4.0)
    # An idle, unstalled machine reports exactly 1.0 — never below, so the gate can only ever
    # hold fan-out back, never inflate it past the measured core budget.
    monkeypatch.setattr(cpu, "cpu_contention", lambda: {"load_per_core": 0.1})
    assert cpu.cpu_oversubscription() == 1.0
    # Nothing measurable → assume the core budget is real, which is the pre-existing behavior.
    monkeypatch.setattr(cpu, "cpu_contention", dict)
    assert cpu.cpu_oversubscription() == 1.0


def test_gpu_absence_keys_on_device_nodes_not_the_driver_control_node(monkeypatch):
    # A GPU-less host built from a GPU-capable cloud image has /dev/nvidiactl (driver loaded)
    # and no /dev/nvidia0. Keying on the control node would call that host GPU-equipped and
    # re-introduce the ~2s torch import this probe exists to avoid.
    # The probe lives in `accelerators` (and is re-exported by `hardware`), so the module
    # state it reads is patched there; the memoized answer is dropped between the two halves.
    hardware.reset_hardware_probes()
    monkeypatch.setattr(accelerators.sys, "platform", "linux")
    monkeypatch.setattr(accelerators.glob, "glob", lambda pat: [])
    monkeypatch.setattr(accelerators.os.path, "exists", lambda p: p == "/dev/nvidiactl")
    assert hardware.gpu_devices_absent() is True

    hardware.reset_hardware_probes()
    monkeypatch.setattr(accelerators.glob, "glob", lambda pat: ["/dev/nvidia0"])
    assert hardware.gpu_devices_absent() is False


def test_gpu_absence_never_false_negatives_off_linux(monkeypatch):
    # macOS Metal devices have no device node, so the cheap path must decline to answer there
    # rather than report "no GPU" on a machine that has one.
    hardware.reset_hardware_probes()
    monkeypatch.setattr(accelerators.sys, "platform", "darwin")
    monkeypatch.setattr(accelerators.os.path, "exists", lambda p: False)
    assert hardware.gpu_devices_absent() is False


def test_cache_size_is_parsed_from_sys():
    assert cache._parse_cache_size("16384K") == 16384 * 1024
    assert cache._parse_cache_size("32M") == 32 * 1024 * 1024
    assert cache._parse_cache_size("1G") == 1 << 30
    assert cache._parse_cache_size("512") == 512
    assert cache._parse_cache_size("") == 0
    # A malformed size must degrade to "unknown" rather than raise: this runs on the query
    # path, and a `/sys` file that a kernel spells differently must not fail a query.
    assert cache._parse_cache_size("bogusM") == 0


def test_cache_hierarchy_skips_instruction_caches(monkeypatch, tmp_path):
    # Only data-side caches bound a working set. Counting the i-cache as L1 would report a
    # residency budget that no data ever occupies.
    root = tmp_path / "cpu0" / "cache"
    for name, level, kind, size in (
        ("index0", "1", "Data", "32K"),
        ("index1", "1", "Instruction", "32K"),
        ("index2", "2", "Unified", "1024K"),
        ("index3", "3", "Unified", "32M"),
    ):
        d = root / name
        d.mkdir(parents=True)
        (d / "level").write_text(level)
        (d / "type").write_text(kind)
        (d / "size").write_text(size)
        (d / "coherency_line_size").write_text("64")
    monkeypatch.setattr(cache.glob, "glob", lambda pat: [str(p) for p in sorted(root.iterdir())])
    hardware.reset_hardware_probes()
    out = cache.cache_hierarchy()
    assert out == {"l1d": 32 << 10, "l2": 1 << 20, "l3": 32 << 20, "line": 64}
    assert cache.cache_line_bytes() == 64


def test_cache_line_falls_back_to_the_near_universal_size(monkeypatch):
    monkeypatch.setattr(cache, "cache_hierarchy", dict)
    # 64 rather than 0: every callsite is an alignment or padding decision that needs some
    # number, and 64 is correct on every architecture Batcher targets except Apple silicon.
    assert cache.cache_line_bytes() == 64


def test_per_core_cache_divides_the_shared_last_level(monkeypatch):
    # A threshold sized against the whole L3 is the standard way a single-threaded measurement
    # collapses under full parallelism, because every core then evicts every other core's lines.
    monkeypatch.setattr(cache, "l3_cache_bytes", lambda: 32 << 20)
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 8)
    assert cache.per_core_cache_bytes() == (32 << 20) // 8
    # Unknown cache size stays unknown; it is never turned into a fabricated share.
    monkeypatch.setattr(cache, "l3_cache_bytes", lambda: 0)
    assert cache.per_core_cache_bytes() == 0


def test_cpu_list_parsing_covers_ranges_and_singletons():
    assert topology._parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
    assert topology._parse_cpu_list("") == set()
    # A kernel that spells the list in a way we do not expect must yield nothing rather than
    # raise: topology is an optimization input, never a correctness one.
    assert topology._parse_cpu_list("not-a-list") == set()


def test_numa_node_count_counts_only_nodes_this_process_may_use(monkeypatch, tmp_path):
    # A container pinned to one socket of a two-socket host is not on a NUMA machine for any
    # purpose the engine cares about: all its memory is local. Reporting 2 would buy
    # partitioning work for a locality problem it cannot have.
    for node, cpus in (("node0", "0-3"), ("node1", "4-7")):
        d = tmp_path / node
        d.mkdir()
        (d / "cpulist").write_text(cpus)
    monkeypatch.setattr(
        topology.glob, "glob", lambda pat: [str(p) for p in sorted(tmp_path.iterdir())]
    )
    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    hardware.reset_hardware_probes()
    assert topology.numa_node_count() == 1
    assert topology.is_numa() is False

    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    hardware.reset_hardware_probes()
    assert topology.numa_node_count() == 2
    assert topology.cpus_per_numa_node() == {0: 4, 1: 4}
    assert topology.is_numa() is True


def test_physical_cores_collapse_smt_siblings(monkeypatch, tmp_path):
    # 8 logical CPUs on 4 physical cores. A compute-bound operator sized to 8 runs half its
    # threads for no gain while halving the cache each of them sees.
    for cpu_id in range(8):
        d = tmp_path / f"cpu{cpu_id}" / "topology"
        d.mkdir(parents=True)
        sibling = cpu_id ^ 1  # pair (0,1), (2,3), ...
        (d / "thread_siblings_list").write_text(f"{min(cpu_id, sibling)},{max(cpu_id, sibling)}")
    monkeypatch.setattr(
        topology.glob, "glob", lambda pat: [str(tmp_path / f"cpu{i}") for i in range(8)]
    )
    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 8)
    hardware.reset_hardware_probes()
    assert topology.physical_core_count() == 4
    assert topology.smt_threads_per_core() == 2.0


def test_physical_cores_fall_back_to_the_logical_count(monkeypatch):
    # No sibling files (not Linux, or a kernel that omits them) must keep the pre-existing
    # behavior of treating every logical CPU as its own core, not report zero cores.
    monkeypatch.setattr(topology.glob, "glob", lambda pat: [])
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 6)
    hardware.reset_hardware_probes()
    assert topology.physical_core_count() == 6
    assert topology.smt_threads_per_core() == 1.0


def test_machine_memory_takes_the_min_of_host_and_cgroup(monkeypatch):
    memory.machine_memory_bytes.cache_clear()
    # cgroup cap tighter than host RAM → the container limit binds.
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: ["/sys/fs/cgroup"])
    monkeypatch.setattr(memory, "read_cgroup_bytes", lambda p: 8 << 30)
    monkeypatch.setattr(
        os, "sysconf", lambda name: 4096 if "PHYS" not in name else (64 << 30) // 4096
    )
    assert memory.machine_memory_bytes() == 8 << 30
    memory.machine_memory_bytes.cache_clear()


def test_a_cgroup_sentinel_is_treated_as_unlimited(tmp_path):
    # cgroup v1 spells "unlimited" as a near-2^63 sentinel rather than the string "max".
    # Reading it literally would report an exabyte of memory and disable every memory bound.
    sentinel = tmp_path / "limit"
    sentinel.write_text(str((1 << 63) - 4096))
    assert cgroup.read_cgroup_bytes(str(sentinel)) is None
    unlimited = tmp_path / "max"
    unlimited.write_text("max")
    assert cgroup.read_cgroup_bytes(str(unlimited)) is None
    real = tmp_path / "real"
    real.write_text(str(8 << 30))
    assert cgroup.read_cgroup_bytes(str(real)) == 8 << 30


def test_simd_width_reports_the_widest_available_lane(monkeypatch):
    monkeypatch.setattr(isa, "_cpuinfo_fields", lambda: {"flags": "sse2 avx avx2 avx512f"})
    hardware.reset_hardware_probes()
    monkeypatch.setattr(isa, "_cpuinfo_fields", lambda: {"flags": "sse2 avx avx2 avx512f"})
    assert isa.simd_width_bits() == 512
    assert "avx2" in isa.cpu_features()
    # Unrecognized flags are dropped, so a microcode update that adds a flag the engine cannot
    # exploit does not change the fingerprint and discard everything learned on the machine.
    assert "fpu" not in isa.cpu_features()


def test_simd_width_floors_at_the_universal_baseline(monkeypatch):
    monkeypatch.setattr(isa, "_cpuinfo_fields", dict)
    hardware.reset_hardware_probes()
    monkeypatch.setattr(isa, "_cpuinfo_fields", dict)
    # 128, not 0: every 64-bit target Batcher supports has at least SSE2 or NEON.
    assert isa.simd_width_bits() == 128


def test_storage_class_is_coarse_enough_to_be_stable():
    # The class feeds the fingerprint, so it must be stable across reboots and identical
    # across instances of one shape — which a device name or serial number would not be.
    assert storage.device_class("/") in {
        "nvme",
        "ssd",
        "rotational",
        "network",
        "loopback",
        "raid",
        "mapped",
        "memory",
        "unknown",
    }
    # An unresolvable path must not raise on the query path.
    assert storage.device_class("/nonexistent-path-for-a-test") in {"memory", "unknown"}
    assert storage.is_rotational("/nonexistent-path-for-a-test") is None
    assert storage.filesystem_free_bytes("/nonexistent-path-for-a-test") == 0


def test_filesystem_free_bytes_excludes_the_root_reservation(tmp_path):
    # f_bavail, not f_bfree: a default ext4 reserves 5% for root, and spilling into it fails
    # late with a disk-full error having already written most of the data.
    free = storage.filesystem_free_bytes(str(tmp_path))
    assert free >= 0
    st = os.statvfs(str(tmp_path))
    assert free == st.f_bavail * st.f_frsize


def test_fingerprint_is_stable_and_short():
    first = hardware.fingerprint()
    assert len(first) == 12
    # Stable within a process, and stable across a re-probe of the same unchanged machine —
    # otherwise every restart would discard everything learned on this host.
    assert first == hardware.fingerprint()
    hardware.reset_hardware_probes()
    assert first == hardware.fingerprint()


def test_fingerprint_splits_machines_that_perform_differently():
    base = profile.HardwareProfile(
        logical_cpus=16,
        physical_cores=8,
        numa_nodes=1,
        memory_bytes=64 << 30,
        caches={"l2": 1 << 20, "l3": 32 << 20},
        simd_bits=256,
        vendor="GenuineIntel",
        model="Xeon",
        storage_class="nvme",
    )
    import dataclasses

    for changed in (
        {"logical_cpus": 64},
        {"physical_cores": 16},
        {"numa_nodes": 2},
        {"memory_bytes": 512 << 30},
        {"caches": {"l2": 1 << 20, "l3": 8 << 20}},
        {"simd_bits": 512},
        {"vendor": "AuthenticAMD"},
        {"storage_class": "rotational"},
        {"accelerators": ("A100",)},
    ):
        other = dataclasses.replace(base, **changed)
        assert base.fingerprint() != other.fingerprint(), f"{changed} must split the class"


def test_fingerprint_merges_near_identical_nodes_of_one_fleet():
    # Two nodes of the same instance type never report byte-identical memory: the kubelet and
    # the firmware each reserve a different amount. Splitting on that would give every node its
    # own model and destroy fleet-wide learning, which is the whole point of the key.
    import dataclasses

    node_a = profile.HardwareProfile(
        logical_cpus=16,
        physical_cores=8,
        memory_bytes=64 << 30,
        caches={"l3": 32 << 20},
        vendor="GenuineIntel",
        model="Xeon",
    )
    node_b = dataclasses.replace(node_a, memory_bytes=int(60.5 * (1 << 30)))
    assert node_a.fingerprint() == node_b.fingerprint()


def test_memory_bucketing_rounds_to_the_nearest_power_of_two():
    assert profile._nearest_power_of_two(0) == 0
    assert profile._nearest_power_of_two(60 << 30) == 64 << 30
    assert profile._nearest_power_of_two(40 << 30) == 32 << 30
    assert profile._nearest_power_of_two(64 << 30) == 64 << 30


def test_profile_reports_the_machine_it_actually_runs_on():
    p = hardware.hardware_profile()
    assert p.logical_cpus >= 1
    assert p.physical_cores >= 1
    assert p.numa_nodes >= 1
    assert p.page_bytes > 0
    assert p.simd_bits >= 128
    # The label is for humans and the fingerprint for keys; neither may be empty.
    assert p.label()
    assert p.to_dict()["fingerprint"] == p.fingerprint()


def test_memory_per_core_distinguishes_memory_rich_from_core_rich():
    # Two 64-core machines with 64 GiB and 1 TiB want different plans for the same query, and
    # the core count alone does not say so.
    lean = profile.HardwareProfile(logical_cpus=64, memory_bytes=64 << 30)
    rich = profile.HardwareProfile(logical_cpus=64, memory_bytes=1024 << 30)
    assert rich.memory_per_core_bytes == 16 * lean.memory_per_core_bytes
    # No cores reported → no ratio, rather than a division by zero on the query path.
    assert profile.HardwareProfile(logical_cpus=0, memory_bytes=1 << 30).memory_per_core_bytes == 0


def test_the_fingerprint_does_not_import_a_gpu_framework(monkeypatch):
    # `gpu_inventory` falls back to `torch.cuda` when NVML is absent, and importing torch costs
    # ~1.6 s. `fingerprint()` runs on the first feedback row in every process — on a Ray
    # cluster, every worker — so paying that to discover a CPU-only box has no GPU turned a
    # 5 ms probe into a 1.6 s one. `gpu_devices_absent` is the cheap device-node check that
    # exists for exactly this, and the profile must go through it.
    import batcher._internal.hardware.profile as profile_mod

    def explode():  # pragma: no cover - the point is that it is never called
        raise AssertionError("the profile probed the GPU inventory on a device-less machine")

    monkeypatch.setattr(profile_mod, "gpu_devices_absent", lambda: True)
    monkeypatch.setattr(profile_mod, "gpu_inventory", explode)
    hardware.reset_hardware_probes()
    assert profile_mod._accelerator_names() == ()
    assert len(hardware.fingerprint()) == 12


def test_a_machine_with_devices_still_gets_its_inventory(monkeypatch):
    # The guard must not become a blanket "assume no GPU". `gpu_devices_absent` returns False
    # both when devices exist and when the platform cannot tell (macOS Metal), and in either
    # case the real inventory is what the fingerprint needs — an A100 and a T4 are different
    # machine classes and must not collapse into one.
    import batcher._internal.hardware.profile as profile_mod

    monkeypatch.setattr(profile_mod, "gpu_devices_absent", lambda: False)
    monkeypatch.setattr(profile_mod, "gpu_inventory", lambda: [{"name": "T4"}, {"name": "A100"}])
    assert profile_mod._accelerator_names() == ("A100", "T4")
