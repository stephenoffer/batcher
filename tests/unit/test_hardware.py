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
