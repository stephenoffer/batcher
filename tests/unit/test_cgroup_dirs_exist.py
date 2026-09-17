"""Every cgroup directory the probes read must actually exist.

`cgroup_v2_dirs` builds candidate paths by joining `/sys/fs/cgroup` with the sub-path in
`/proc/self/cgroup`. Under a cgroup **namespace** — Docker, Kubernetes, and the Anyscale
host this was found on — that sub-path is the one the *host* sees, while the mount shows the
same cgroup at the root. Every path built that way is then absent.

It reads as harmless, which is why it survived: a missing file yields `None` and the caller
moves on. It is not harmless, because `kernel._own_cgroup_dirs` treats a non-empty tuple as
proof that this workload has its own delegated slice and therefore never falls back to the
mount root. Measured on the host that prompted this: `/proc/self/cgroup` reported
`/anyscale/ctr_<id>/activities`, none of the three constructed directories existed,
`memory.current` read `None` — and `/sys/fs/cgroup/memory.current` read 19.57 GiB.

Two safety mechanisms failed open as a result, both of them in containers, which is where
they matter:

* the live-pressure signal `SpillAdvisor.spill_reason` calls "the one number here that
  cannot be wrong the way an estimate can";
* the OOM-kill history in `PressureMonitor._oom_history_factor`, the only *evidence*-based
  spill trigger — this cgroup reported **91 prior kills** the moment it could be read, and
  the 0.8 backoff it gates had never once applied.

The assertion is deliberately a property of the return value rather than a reconstruction of
the namespace, because the property is what every caller depends on and it holds on hosts
with no cgroups at all.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.hardware.cgroup import cgroup_v2_dirs

pytestmark = pytest.mark.unit


def test_every_returned_directory_exists():
    """A path that is not there can only contribute `None`, so it must not be returned."""
    missing = [d for d in cgroup_v2_dirs() if not os.path.isdir(d)]
    assert not missing, f"cgroup_v2_dirs returned directories that do not exist: {missing}"


@pytest.mark.skipif(not os.path.isdir("/sys/fs/cgroup"), reason="no cgroup v2 mount")
def test_the_mount_root_survives_when_it_is_the_only_real_cgroup():
    """Filtering must not empty the tuple on a namespaced host.

    The mount root is the entry every namespaced container actually has, and it is the one
    `_own_cgroup_dirs`'s fallback depends on. A filter that dropped it would replace a
    silent `None` with a different silent `None`.
    """
    assert "/sys/fs/cgroup" in cgroup_v2_dirs()


@pytest.mark.skipif(not os.path.exists("/sys/fs/cgroup/memory.current"), reason="no memory.current")
def test_the_live_footprint_is_actually_readable():
    """The consequence, asserted where a reader would look for it.

    This is the positive control the property test above cannot be: a `cgroup_v2_dirs` that
    returned an empty tuple would satisfy "every entry exists" and still leave the footprint
    unreadable. Skipped rather than failed where the kernel publishes no `memory.current`,
    since there the `None` is the honest answer.
    """
    from batcher.carbonite.memory.kernel import kernel_memory_state

    assert kernel_memory_state().current_bytes is not None


def test_a_host_with_no_cgroup_mount_reads_nothing_rather_than_crashing(monkeypatch):
    """No `/sys/fs/cgroup` at all empties the tuple, and the probes must answer with `None`.

    Found installing the musllinux wheel into an Alpine chroot with no `/sys` mounted: the
    usage fallback indexed `cgroup_v2_dirs()[0]` and the first query raised `IndexError`.
    gVisor and some minimal container runtimes present the same empty view. The docstring of
    `kernel_memory_state` already promises an all-`None` state outside a cgroup.
    """
    from batcher.carbonite.memory import kernel

    monkeypatch.setattr(kernel, "cgroup_v2_dirs", lambda: ())
    kernel.reset_kernel_sampling()
    try:
        state = kernel.kernel_memory_state()
        assert state.current_bytes is None
        assert state.oom_kills is None
    finally:
        kernel.reset_kernel_sampling()
