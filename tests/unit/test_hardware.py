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


def test_psi_is_sampled_not_re_read_on_every_call(monkeypatch):
    # These files are read on every terminal op, and re-reading them per query cost about as
    # much as executing a small query. PSI publishes a 10-second rolling average, so a
    # quarter-second-old sample answers identically -- but only a call *count* can show the
    # re-read is gone, since the value is the same either way.
    import builtins

    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/cpu.pressure": "some avg10=12.50 avg60=1.00 avg300=0.00 total=1\n",
        "/sys/fs/cgroup/memory.pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        "/sys/fs/cgroup/io.pressure": "some avg10=100.00 avg60=1.00 avg300=0.00 total=1\n",
    }
    opened: list[str] = []
    inner = _fake_open(files)

    def counting_open(path, *a, **k):
        opened.append(str(path))
        return inner(path, *a, **k)

    monkeypatch.setattr(builtins, "open", counting_open)

    first = cgroup.cgroup_pressure()
    after_first = len(opened)
    for _ in range(20):
        assert cgroup.cgroup_pressure() == first
    assert len(opened) == after_first, "a sampled reading must not re-open the PSI files"

    # The sample is still invalidated by the one hook a test uses to fake `/sys`, so the
    # isolation every other test in this file relies on is unaffected.
    hardware.reset_hardware_probes()
    assert cgroup.cgroup_pressure() == first
    assert len(opened) > after_first


def test_the_machine_fingerprint_is_hashed_once_per_process(monkeypatch):
    # `fingerprint()` is the scoping key for every learned parameter, so the query path asks
    # for it repeatedly -- 11 times on a warm `collect()`. It joins fourteen fields and
    # SHA-256s them, for a profile that is assembled once and frozen, so every call after the
    # first re-derives a value that cannot have changed.
    calls: list[int] = []
    real = profile.HardwareProfile.fingerprint

    def counting(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(profile.HardwareProfile, "fingerprint", counting)
    hardware.reset_hardware_probes()

    first = profile.fingerprint()
    for _ in range(20):
        assert profile.fingerprint() == first
    assert len(calls) == 1, f"digest recomputed {len(calls)} times for one machine"

    # Still cleared by the one hook a test uses to fake the machine, so a profile faked after
    # a reset is not shadowed by the previous machine's key.
    hardware.reset_hardware_probes()
    assert profile.fingerprint() == first
    assert len(calls) == 2


def test_a_sampled_pressure_dict_cannot_be_mutated_by_a_caller(monkeypatch):
    # The memo holds one dict; handing it out directly would let any caller's edit become
    # every later caller's reading.
    import builtins

    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/cpu.pressure": "some avg10=12.50 avg60=1.00 avg300=0.00 total=1\n",
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    first = cgroup.cgroup_pressure()
    first["cpu_stall"] = 999.0
    assert cgroup.cgroup_pressure()["cpu_stall"] == pytest.approx(0.125)


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


def test_physical_cores_collapse_smt_siblings(monkeypatch, tmp_path):
    # 8 logical CPUs on 4 physical cores. The physical count is what the machine fingerprint
    # records, and two boxes with the same logical count but different SMT are different
    # hardware — an 8-core-no-SMT and a 4-core-2-way perform very differently on the same plan.
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


def test_physical_cores_fall_back_to_the_logical_count(monkeypatch):
    # No sibling files (not Linux, or a kernel that omits them) must keep the pre-existing
    # behavior of treating every logical CPU as its own core, not report zero cores.
    monkeypatch.setattr(topology.glob, "glob", lambda pat: [])
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 6)
    hardware.reset_hardware_probes()
    assert topology.physical_core_count() == 6


def test_storage_class_is_coarse_enough_to_be_stable():
    # The class feeds both the machine fingerprint and the spill cost term, so it must be
    # stable across reboots and identical across instances of one shape — which a device name
    # or serial number would not be.
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
    # An unresolvable path must not raise: this runs on the planning path.
    assert storage.device_class("/nonexistent-path-for-a-test") in {"memory", "unknown"}


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


def test_cpu_list_parsing_covers_ranges_and_singletons():
    assert topology.parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
    assert topology.parse_cpu_list("") == set()
    # A kernel that spells the list in a way we do not expect must yield nothing rather than
    # raise: topology is an optimization input, never a correctness one.
    assert topology.parse_cpu_list("not-a-list") == set()


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

    monkeypatch.setattr(topology.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    hardware.reset_hardware_probes()
    assert topology.numa_node_count() == 2
    assert topology.cpus_per_numa_node() == {0: 4, 1: 4}


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


# --- The contention signals a container can actually trust -----------------------------------


def test_oversubscription_reads_the_host_run_queue_not_the_container_slice(monkeypatch):
    # A 4-core container on a 128-core host that is HALF IDLE (load 64 of 128). The run queue
    # is a host-wide number and Linux publishes no per-cgroup equivalent, so dividing it by the
    # container's own 4 cores reported a sixteen-fold oversubscription — and the CPU-budget
    # policy then collapsed the fan-out to a sliver on a machine with abundant headroom. The
    # honest ratio divides the host-wide numerator by the host-wide denominator: 0.5.
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(os, "getloadavg", lambda: (64.0, 64.0, 64.0))
    monkeypatch.setattr(cpu, "cgroup_throttled_ratio", lambda: None)
    monkeypatch.setattr(cpu, "cgroup_pressure", dict)

    signals = hardware.cpu_contention()
    # Both are reported: the slice-relative figure is what a *diagnostic* asks for, and it is
    # still 16 — the box does have four times more runnable work than this process has cores.
    assert signals["load_per_core"] == 16.0
    assert signals["host_load_per_core"] == 0.5
    # But the verdict that gates fan-out reads the host-scoped one, so a half-idle host is
    # never reported as oversubscribed.
    assert cpu.cpu_oversubscription() == 1.0


def test_oversubscription_still_fires_when_the_host_really_is_oversubscribed(monkeypatch):
    # Same small container, host now at 2x its cores. That contention is real and reaches us.
    monkeypatch.setattr(cpu, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(os, "getloadavg", lambda: (256.0, 256.0, 256.0))
    monkeypatch.setattr(cpu, "cgroup_throttled_ratio", lambda: None)
    monkeypatch.setattr(cpu, "cgroup_pressure", dict)
    assert cpu.cpu_oversubscription() == pytest.approx(2.0)


def test_cgroup_v1_throttling_is_visible(monkeypatch):
    # cgroup v1 keeps `cpu.stat` under a per-controller mount, which the v2 directory walk
    # never reaches. Throttling was therefore invisible on every v1 host — so a container
    # pinned at its quota reported NO contention, and both the fan-out and the CPU-share loop
    # read the resulting idle cores as "this workload does not want them".
    import builtins

    files = {
        "/proc/self/cgroup": "1:cpu:/docker/abc\n",  # no `0::` line: this host is v1
        "/sys/fs/cgroup/cpu/cpu.stat": "nr_periods 1000\nnr_throttled 250\nthrottled_time 5\n",
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    assert cgroup.cgroup_throttled_ratio() == 0.25


def test_a_v2_reading_still_wins_over_the_v1_fallback(monkeypatch):
    import builtins

    files = {
        "/proc/self/cgroup": "0::/pod\n",
        "/sys/fs/cgroup/cpu.stat": "nr_periods 100\nnr_throttled 10\n",
        "/sys/fs/cgroup/pod/cpu.stat": "nr_periods 100\nnr_throttled 50\n",
        "/sys/fs/cgroup/cpu/cpu.stat": "nr_periods 100\nnr_throttled 90\n",  # v1: must not win
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    assert cgroup.cgroup_throttled_ratio() == pytest.approx(0.1)


def test_a_heterogeneous_slurm_allocation_still_bounds_the_fan_out(monkeypatch):
    # `SLURM_CPUS_ON_NODE` is a run-length list on a heterogeneous job. Any value carrying a
    # repeat count used to fall through entirely, leaving NO Slurm bound — so the job sized to
    # the affinity mask, which on an unconfined HPC node is every core on a shared machine.
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4(x2),8")
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cpu, "_affinity_count", lambda: 128)
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: None)
    # The smallest grant in the expansion binds: which entry is *this* node is not derivable
    # from the variable, and under-parallelizing costs throughput where over-parallelizing on
    # the node that got the small grant is what gets a job killed.
    assert hardware.available_cpu_count() == 4

    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")  # the plain-int case is unchanged
    assert hardware.available_cpu_count() == 16

    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "weird")  # unparseable → no bound, not a wrong one
    assert hardware.available_cpu_count() == 128


# --- The cache domain this process actually runs in ------------------------------------------


def _cache_files(cpu_id: int, *, l2: str, l3: str, shared: str) -> dict[str, str]:
    """The `/sys` cache tree for one CPU: an L1d, an L2, and an L3 shared with `shared`."""
    root = f"/sys/devices/system/cpu/cpu{cpu_id}/cache"
    return {
        f"{root}/index0/level": "1",
        f"{root}/index0/type": "Data",
        f"{root}/index0/size": "32K",
        f"{root}/index0/coherency_line_size": "64",
        f"{root}/index0/shared_cpu_list": str(cpu_id),
        f"{root}/index2/level": "2",
        f"{root}/index2/type": "Unified",
        f"{root}/index2/size": l2,
        f"{root}/index2/coherency_line_size": "64",
        f"{root}/index2/shared_cpu_list": str(cpu_id),
        f"{root}/index3/level": "3",
        f"{root}/index3/type": "Unified",
        f"{root}/index3/size": l3,
        f"{root}/index3/coherency_line_size": "64",
        f"{root}/index3/shared_cpu_list": shared,
    }


def _fake_cache_glob(files: dict[str, str]):
    """A `glob.glob` stand-in returning the `index*` dirs present in `files`."""

    def globber(pattern):
        prefix = pattern.removesuffix("index*")
        return sorted({p.rsplit("/", 1)[0] for p in files if p.startswith(prefix)})

    return globber


def test_the_cache_is_read_from_a_cpu_this_process_can_run_on(monkeypatch):
    # A container pinned by cpuset to cores 8-9 of a host whose cpu0 sits on a different,
    # larger cache domain. `/sys` is host-wide, so reading `cpu0` unconditionally described a
    # core this process can never be scheduled on — and a broadcast threshold sized to that
    # 32 MiB domain spills out of the 8 MiB domain the work actually runs in.
    import builtins

    files = {
        **_cache_files(0, l2="2M", l3="32M", shared="0-7"),
        **_cache_files(8, l2="512K", l3="8M", shared="8-9"),
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    monkeypatch.setattr(cache.glob, "glob", _fake_cache_glob(files))
    monkeypatch.setattr(cache, "affinity_cpu_ids", lambda: {8, 9})

    hierarchy = cache.cache_hierarchy()
    assert hierarchy["l3"] == 8 * (1 << 20)
    assert hierarchy["l2"] == 512 * (1 << 10)


def test_a_heterogeneous_socket_reports_its_binding_cache_domain(monkeypatch):
    # An AMD part with stacked cache: one CCD carries 96 MiB of L3 and its neighbour 32 MiB, a
    # threefold spread inside one socket. Whichever CCD happened to hold cpu0 used to decide
    # the figure. The smallest is the binding one — a table resident there is resident on both.
    import builtins

    files = {
        **_cache_files(0, l2="1M", l3="96M", shared="0-7"),
        **_cache_files(8, l2="1M", l3="32M", shared="8-15"),
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    monkeypatch.setattr(cache.glob, "glob", _fake_cache_glob(files))
    monkeypatch.setattr(cache, "affinity_cpu_ids", lambda: set(range(16)))
    assert cache.cache_hierarchy()["l3"] == 32 * (1 << 20)


def test_one_sys_read_per_cache_domain_not_per_core(monkeypatch):
    # Cost control: a core in an already-measured domain answers identically, so it is skipped.
    # A 16-core, 2-domain machine must read two CPUs, not sixteen.
    import builtins

    files = {
        **_cache_files(0, l2="1M", l3="32M", shared="0-7"),
        **_cache_files(8, l2="1M", l3="32M", shared="8-15"),
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    monkeypatch.setattr(cache, "affinity_cpu_ids", lambda: set(range(16)))
    probed: list[str] = []

    globber = _fake_cache_glob(files)

    def counting_glob(pattern):
        probed.append(pattern)
        return globber(pattern)

    monkeypatch.setattr(cache.glob, "glob", counting_glob)
    cache.cache_hierarchy()
    assert len(probed) == 2


def test_the_cache_line_is_the_widest_not_the_narrowest(monkeypatch):
    # Sizes bind the working set, so the smallest wins. The line size sizes false-sharing
    # padding, where the LARGEST is the safe one: padding to 64 bytes on a core with 128-byte
    # lines still false-shares.
    import builtins

    files = {
        **_cache_files(0, l2="1M", l3="32M", shared="0"),
        **_cache_files(1, l2="1M", l3="32M", shared="1"),
    }
    files["/sys/devices/system/cpu/cpu1/cache/index0/coherency_line_size"] = "128"
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    monkeypatch.setattr(cache.glob, "glob", _fake_cache_glob(files))
    monkeypatch.setattr(cache, "affinity_cpu_ids", lambda: {0, 1})
    assert cache.cache_hierarchy()["line"] == 128


# --- What the memory ceiling really is --------------------------------------------------------


def test_reserved_hugepages_come_off_the_memory_ceiling(monkeypatch):
    # Hugetlb pool memory exists and cannot be allocated by anything this engine does: it is
    # carved out of the general allocator at reservation time. Sizing to `SC_PHYS_PAGES` alone
    # therefore promised a hash table memory that structurally could not hold it.
    monkeypatch.setattr(memory, "hugepage_bytes", lambda: 8 * (1 << 30))
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(32 << 30))
    monkeypatch.setattr(memory, "cgroup_v2_dirs", tuple)
    monkeypatch.setattr(memory, "read_cgroup_bytes", lambda path: None)
    assert memory.machine_memory_bytes() == 24 * (1 << 30)


def test_a_hugepage_figure_larger_than_ram_cannot_make_memory_negative(monkeypatch):
    monkeypatch.setattr(memory, "hugepage_bytes", lambda: 999 << 30)
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(32 << 30))
    monkeypatch.setattr(memory, "cgroup_v2_dirs", tuple)
    monkeypatch.setattr(memory, "read_cgroup_bytes", lambda path: None)
    assert memory.machine_memory_bytes() == 0


def test_hugepage_pools_of_every_size_class_are_counted(monkeypatch):
    # A node commonly reserves 2 MiB pages for one tenant and 1 GiB pages for another; reading
    # only the default class would miss whichever one the operator actually used.
    import builtins

    pools = {
        "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages": "512",  # 1 GiB
        "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages": "4",  # 4 GiB
    }
    monkeypatch.setattr(builtins, "open", _fake_open(pools))
    monkeypatch.setattr(memory.glob, "glob", lambda p: [k.rsplit("/", 1)[0] for k in sorted(pools)])
    assert memory.hugepage_bytes() == 5 * (1 << 30)


def test_the_memory_high_throttle_binds_the_budget(monkeypatch):
    # `memory.high` is not a kill boundary, which is why it was skipped — but a cgroup above it
    # is put into synchronous reclaim and made to crawl. Budgeting to `memory.max` on such a
    # cgroup buys a thrashing query where spilling earlier buys a finishing one.
    caps = {
        "/sys/fs/cgroup/pod/memory.max": 16 << 30,
        "/sys/fs/cgroup/pod/memory.high": 12 << 30,
    }
    monkeypatch.setattr(memory, "hugepage_bytes", lambda: 0)
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(64 << 30))
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: ("/sys/fs/cgroup/pod",))
    monkeypatch.setattr(memory, "read_cgroup_bytes", caps.get)
    assert memory.machine_memory_bytes() == 12 << 30


def test_swap_is_reported_from_the_cgroup_before_the_host(monkeypatch):
    # A container with host swap can still be denied it (`memory.swap.max = 0`, which is what
    # Kubernetes writes), and there the host's swap partitions are present and irrelevant.
    # Swapless means overshooting the budget is terminal rather than merely slow, so the engine
    # must spill earlier — and nothing could tell the two apart before.
    import builtins

    files = {
        "/sys/fs/cgroup/pod/memory.swap.max": "0",
        "/proc/swaps": "Filename\tType\tSize\n/dev/sda2\tpartition\t8388604\n",
    }
    monkeypatch.setattr(builtins, "open", _fake_open(files))
    monkeypatch.setattr(memory, "cgroup_v2_dirs", lambda: ("/sys/fs/cgroup", "/sys/fs/cgroup/pod"))
    assert memory.swap_configured() is False

    files["/sys/fs/cgroup/pod/memory.swap.max"] = "max"  # unlimited → the host decides
    memory.swap_configured.cache_clear()
    assert memory.swap_configured() is True


def test_a_swapless_host_reports_no_swap(monkeypatch):
    import builtins

    monkeypatch.setattr(builtins, "open", _fake_open({"/proc/swaps": "Filename\tType\tSize\n"}))
    monkeypatch.setattr(memory, "cgroup_v2_dirs", tuple)
    assert memory.swap_configured() is False


# --- What the spill device really is ----------------------------------------------------------


def test_an_lvm_volume_is_priced_by_what_is_underneath_it(monkeypatch):
    # LVM over a network-attached volume is an ordinary cloud root-and-scratch layout. The
    # `dm-` prefix says nothing about speed, and reporting it as `mapped` carried the DEFAULT
    # cost factor — so a spilled byte on EBS was priced as local flash, understating it tenfold
    # in the one term that decides whether an out-of-core plan is acceptable at all.
    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "dm-0")
    monkeypatch.setattr(os, "listdir", lambda p: ["nbd0"] if p.endswith("/slaves") else [])
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/sys/block/nbd0")
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "network"
    assert storage.device_cost_factor("/scratch") == 10.0


def test_a_stripe_is_priced_at_its_slowest_member(monkeypatch):
    # A stripe finishes at the rate of its slowest member, and an external merge's concurrent
    # run reads touch every member.
    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "md0")
    monkeypatch.setattr(
        os, "listdir", lambda p: ["nvme0n1", "nbd3"] if p.endswith("/slaves") else []
    )
    monkeypatch.setattr(os.path, "exists", lambda p: p in ("/sys/block/nvme0n1", "/sys/block/nbd3"))
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "network"


def test_an_unresolvable_mapper_still_names_itself(monkeypatch):
    # No `slaves/` to read → fall back to the prefix, exactly as before. An unreadable `/sys`
    # must re-rank no plan.
    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "dm-7")
    monkeypatch.setattr(os, "listdir", _raise_oserror)
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "mapped"
    assert storage.device_cost_factor("/scratch") == 1.0


def test_nvme_over_fabrics_is_not_local_flash(monkeypatch):
    # An NVMe-oF namespace is named exactly like a local one, so the device name — the only
    # signal used before — reported the fastest class in the table for storage that crosses a
    # network. The driver publishes the transport; ask it.
    import builtins

    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "nvme0n1")
    monkeypatch.setattr(
        builtins, "open", _fake_open({"/sys/block/nvme0n1/device/transport": "tcp"})
    )
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "network"

    monkeypatch.setattr(
        builtins, "open", _fake_open({"/sys/block/nvme0n1/device/transport": "pcie"})
    )
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "nvme"


def test_a_kernel_that_does_not_publish_a_transport_assumes_local(monkeypatch):
    import builtins

    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "nvme0n1")
    monkeypatch.setattr(builtins, "open", _fake_open({}))
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "nvme"


def test_an_iscsi_lun_is_not_an_ssd(monkeypatch):
    # A remote LUN answers `rotational = 0` and was therefore classified `ssd`, at a tenth of
    # its real cost, on exactly the SAN-backed deployments where pricing an out-of-core plan
    # matters. Its place in the `/sys` device tree is a positive identification, not a guess.
    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "sdb")
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda p: "/sys/devices/platform/host6/session1/target6:0:0/6:0:0:0",
    )
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "network"


def test_a_local_sas_disk_is_still_classified_by_its_medium(monkeypatch):
    monkeypatch.setattr(storage, "_sys_block_name", lambda path: "sdb")
    monkeypatch.setattr(os.path, "realpath", lambda p: "/sys/devices/pci0000:00/0000:00:17.0/ata1")
    monkeypatch.setattr(storage, "read_optional_int", lambda p: 1)
    storage.device_class.cache_clear()
    assert storage.device_class("/scratch") == "rotational"


def _raise_oserror(*_a, **_k):
    raise OSError


def _fake_sysconf(total_bytes: int, page: int = 4096):
    """An `os.sysconf` stand-in reporting a host with `total_bytes` of RAM."""

    def sysconf(name):
        if name == "SC_PAGE_SIZE":
            return page
        if name == "SC_PHYS_PAGES":
            return total_bytes // page
        raise ValueError(name)

    return sysconf


# --- The vector width an ARM part actually has ------------------------------------------------


def test_sve_width_is_read_from_the_kernel_not_assumed(monkeypatch):
    """SVE has no architectural width — an implementation picks anything from 128 to 2048
    bits, and the server parts differ: Graviton3 is 256-bit, Graviton4 is 128, A64FX is 512.
    A flat 256 was right on one of the three and overstated Graviton4 twofold, in a figure that
    both scales a throughput estimate and keys every learned coefficient on the machine."""
    import builtins

    monkeypatch.setattr(isa, "cpu_features", lambda: frozenset({"asimd", "sve"}))
    monkeypatch.setattr(builtins, "open", _fake_open({isa._SVE_VECTOR_LENGTH_PATH: "16\n"}))
    isa.simd_width_bits.cache_clear()
    assert isa.simd_width_bits() == 128  # 16 bytes: Neoverse V2

    monkeypatch.setattr(builtins, "open", _fake_open({isa._SVE_VECTOR_LENGTH_PATH: "64\n"}))
    isa.simd_width_bits.cache_clear()
    assert isa.simd_width_bits() == 512  # 64 bytes: A64FX


def test_an_unreadable_sve_length_falls_back_to_the_neon_floor(monkeypatch):
    """Every aarch64 part has at least NEON, so 128 is a floor rather than a guess."""
    import builtins

    monkeypatch.setattr(isa, "cpu_features", lambda: frozenset({"asimd", "sve"}))
    monkeypatch.setattr(builtins, "open", _fake_open({}))
    isa.simd_width_bits.cache_clear()
    assert isa.simd_width_bits() == 128


def test_an_implausible_sve_length_is_discarded(monkeypatch):
    """A width outside the architectural range would have to come from a kernel reporting
    something this code does not understand, and it propagates into the machine fingerprint."""
    import builtins

    monkeypatch.setattr(isa, "cpu_features", lambda: frozenset({"asimd", "sve2"}))
    for bogus in ("7", "0", "9999"):
        monkeypatch.setattr(builtins, "open", _fake_open({isa._SVE_VECTOR_LENGTH_PATH: bogus}))
        isa.simd_width_bits.cache_clear()
        assert isa.simd_width_bits() == 128


def test_x86_widths_are_untouched_by_the_sve_path(monkeypatch):
    monkeypatch.setattr(isa, "cpu_features", lambda: frozenset({"sse2", "avx", "avx2"}))
    isa.simd_width_bits.cache_clear()
    assert isa.simd_width_bits() == 256


def test_an_arm_implementer_code_is_readable_in_a_label():
    """`0x41/64c/128GiB` in a log line tells nobody which fleet ran the query. The code stays
    the fingerprint material — remapping it would move every ARM machine's key and discard
    everything learned on it — and only the display is translated."""
    assert isa.vendor_display_name("0x41") == "ARM"
    assert isa.vendor_display_name("0xc0") == "Ampere"
    assert isa.vendor_display_name("GenuineIntel") == "GenuineIntel"
    assert isa.vendor_display_name("") == ""

    raw = profile.HardwareProfile(logical_cpus=64, memory_bytes=128 << 30, vendor="0xc0")
    assert raw.label().startswith("Ampere/64c/")
    # The key is built from the raw code and not from the display name, so translating for a
    # log line cannot move a machine's fingerprint out from under everything it has learned.
    translated = profile.HardwareProfile(logical_cpus=64, memory_bytes=128 << 30, vendor="Ampere")
    assert raw.fingerprint() != translated.fingerprint()


def test_the_reset_hook_forgets_every_vendor_probe(monkeypatch):
    """`reset_hardware_probes` promises to forget *every* memoized reading, and is what the
    whole suite's fixtures call. It never cleared AMD's device-identity memo, so an AMD test
    that faked `/sys/class/drm` and reset in the documented way kept reading the previous
    answer — a test that passes while testing nothing, which is the exact failure mode the
    explicit-list design of `probes` exists to prevent."""
    from batcher._internal.hardware.amd import devices as amd_devices

    seen: list[int] = []
    monkeypatch.setattr(amd_devices, "_probe", lambda: seen.append(1) or ())
    amd_devices._cached_identity.cache_clear()

    amd_devices.amd_present()
    amd_devices.amd_present()
    assert len(seen) == 1  # memoized, as intended

    hardware.reset_hardware_probes()
    amd_devices.amd_present()
    assert len(seen) == 2  # ...and the documented reset actually reaches it
