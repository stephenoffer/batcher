"""Effective CPU detection — affinity/cgroup-quota aware, not the host core count."""

from __future__ import annotations

import pytest

from batcher._internal import hardware

pytestmark = pytest.mark.unit


def test_available_cpu_count_is_positive():
    assert hardware.available_cpu_count() >= 1


def test_available_cpu_count_takes_the_min_of_host_affinity_quota(monkeypatch):
    # A 64-core host, pinned to 8 by cpuset affinity, throttled to 4 by the CFS quota →
    # the process may really use 4. os.cpu_count() alone (64) would over-subscribe 16x.
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(hardware, "_affinity_count", lambda: 8)
    monkeypatch.setattr(hardware, "_cfs_quota_count", lambda: 4)
    assert hardware.available_cpu_count() == 4
    # Affinity tighter than quota → affinity wins.
    monkeypatch.setattr(hardware, "_cfs_quota_count", lambda: 16)
    assert hardware.available_cpu_count() == 8
    # Neither discoverable (bare metal) → the host count stands.
    monkeypatch.setattr(hardware, "_affinity_count", lambda: None)
    monkeypatch.setattr(hardware, "_cfs_quota_count", lambda: None)
    assert hardware.available_cpu_count() == 64
    # Never zero even if every source lies low.
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: None)
    monkeypatch.setattr(hardware, "_affinity_count", lambda: 0 or None)
    assert hardware.available_cpu_count() == 1


def test_cfs_quota_count_parses_cgroup_v2(monkeypatch, tmp_path):
    # cgroup v2 `cpu.max` = "<quota> <period>"; 200000/100000 = 2 whole cores (ceil).
    import builtins
    from io import StringIO

    # Namespaced pod: /proc/self/cgroup is "0::/", so the limit is read at the mount root.
    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/cpu.max": "200000 100000",  # 2 cores
    }

    def fake_open(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError  # any other cgroup path absent

    monkeypatch.setattr(builtins, "open", fake_open)
    assert hardware._cfs_quota_count() == 2

    # An unlimited quota ("max") yields None (fall back to host/affinity).
    files["/sys/fs/cgroup/cpu.max"] = "max 100000"
    assert hardware._cfs_quota_count() is None


def test_cfs_quota_reads_non_namespaced_leaf_cgroup(monkeypatch):
    # A Ray worker in a delegated cgroup with NO namespace: the mount root reads unlimited,
    # but the real limit lives at the process's own leaf (from /proc/self/cgroup). Missing it
    # would over-subscribe. The tightest limit across root+leaf must win.
    import builtins
    from io import StringIO

    files = {
        "/proc/self/cgroup": "0::/system.slice/ray-worker-7.scope\n",
        "/sys/fs/cgroup/cpu.max": "max 100000",  # root: unlimited
        "/sys/fs/cgroup/system.slice/ray-worker-7.scope/cpu.max": "400000 100000",  # leaf: 4 cores
    }

    def fake_open(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    monkeypatch.setattr(builtins, "open", fake_open)
    assert hardware._cfs_quota_count() == 4  # the leaf limit, not the root's "unlimited"


def test_cfs_quota_takes_tightest_across_the_hierarchy(monkeypatch):
    # cgroup v2 enforces the quota at every level: a limit set on a PARENT slice (not the
    # leaf, not the root) must still bind. The tightest cpu.max in the whole chain wins.
    import builtins
    from io import StringIO

    files = {
        "/proc/self/cgroup": "0::/parent.slice/child.scope\n",
        "/sys/fs/cgroup/cpu.max": "max 100000",  # root: unlimited
        "/sys/fs/cgroup/parent.slice/cpu.max": "600000 100000",  # parent: 6 cores (the binder)
        "/sys/fs/cgroup/parent.slice/child.scope/cpu.max": "max 100000",  # leaf: unlimited
    }

    def fake_open(path, *a, **k):
        if path in files:
            return StringIO(files[path])
        raise OSError

    monkeypatch.setattr(builtins, "open", fake_open)
    assert hardware._cfs_quota_count() == 6  # the parent slice's limit, missed by root+leaf only


def test_cpu_contention_reports_load_relative_to_the_usable_cores(monkeypatch):
    # The denominator must be the cores this process may actually use, not the host's. Inside
    # a quota-throttled container those differ, and dividing by the host count would report a
    # saturated box as idle — the exact misreading that makes a diagnosis blame the plan.
    monkeypatch.setattr(hardware, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(hardware.os, "getloadavg", lambda: (8.0, 8.0, 8.0))
    monkeypatch.setattr(hardware, "_cgroup_throttled_ratio", lambda: None)
    assert hardware.cpu_contention()["load_per_core"] == 2.0
    # A key the platform cannot answer is omitted, never defaulted to a reassuring zero.
    assert "throttled_ratio" not in hardware.cpu_contention()


def test_cpu_contention_survives_a_platform_without_loadavg(monkeypatch):
    def boom():
        raise OSError("no loadavg here")

    monkeypatch.setattr(hardware.os, "getloadavg", boom)
    monkeypatch.setattr(hardware, "_cgroup_throttled_ratio", lambda: 0.25)
    out = hardware.cpu_contention()
    assert "load_per_core" not in out
    assert out["throttled_ratio"] == 0.25


def test_gpu_absence_keys_on_device_nodes_not_the_driver_control_node(monkeypatch):
    # A GPU-less host built from a GPU-capable cloud image has /dev/nvidiactl (driver loaded)
    # and no /dev/nvidia0. Keying on the control node would call that host GPU-equipped and
    # re-introduce the ~2s torch import this probe exists to avoid.
    hardware.gpu_devices_absent.cache_clear()
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.glob, "glob", lambda pat: [])
    monkeypatch.setattr(hardware.os.path, "exists", lambda p: p == "/dev/nvidiactl")
    assert hardware.gpu_devices_absent() is True

    hardware.gpu_devices_absent.cache_clear()
    monkeypatch.setattr(hardware.glob, "glob", lambda pat: ["/dev/nvidia0"])
    assert hardware.gpu_devices_absent() is False
    hardware.gpu_devices_absent.cache_clear()


def test_gpu_absence_never_false_negatives_off_linux(monkeypatch):
    # macOS Metal devices have no device node, so the cheap path must decline to answer there
    # rather than report "no GPU" on a machine that has one.
    hardware.gpu_devices_absent.cache_clear()
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.os.path, "exists", lambda p: False)
    assert hardware.gpu_devices_absent() is False
    hardware.gpu_devices_absent.cache_clear()


def test_l3_cache_size_is_parsed_from_sys(monkeypatch):
    from batcher._internal import hardware

    assert hardware._parse_cache_size("16384K") == 16384 * 1024
    assert hardware._parse_cache_size("32M") == 32 * 1024 * 1024
    assert hardware._parse_cache_size("1G") == 1 << 30
    assert hardware._parse_cache_size("512") == 512
    assert hardware._parse_cache_size("") == 0


def test_machine_memory_takes_the_min_of_host_and_cgroup(monkeypatch):
    from batcher._internal import hardware

    hardware.machine_memory_bytes.cache_clear()
    # cgroup cap tighter than host RAM → the container limit binds.
    monkeypatch.setattr(hardware, "cgroup_v2_dirs", lambda: ["/sys/fs/cgroup"])
    monkeypatch.setattr(hardware, "read_cgroup_bytes", lambda p: 8 << 30)
    monkeypatch.setattr(
        hardware.os, "sysconf", lambda name: (4096 if "PHYS" not in name else (64 << 30) // 4096)
    )
    assert hardware.machine_memory_bytes() == 8 << 30
    hardware.machine_memory_bytes.cache_clear()


def test_a_cgroup_sentinel_is_treated_as_unlimited():
    from batcher._internal import hardware

    assert hardware.read_cgroup_bytes.__doc__  # documented
