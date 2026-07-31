"""The container defaults that halve a GPU job, and the staging path that used to hit one.

A rented GPU node is reached inside a container the platform started, and a container runtime's
defaults were chosen for a web service. Three of them cost a data engine something without
naming themselves: a 64 MiB `/dev/shm`, a 64 KiB memlock ceiling, and a low descriptor limit.

The `/dev/shm` case is the one that was an actual bug rather than only invisible. Three staging
paths tested `os.path.isdir("/dev/shm")`, which is true inside a container with the default
allocation — so the directory exists, the write starts, and it fails with `ENOSPC` partway
through a batch group.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.site import container

pytestmark = pytest.mark.unit


def _fake_shm(monkeypatch, total_bytes, free_bytes=None):
    """Fake the shared-memory filesystem, without patching `os.statvfs` process-wide."""
    free = total_bytes if free_bytes is None else free_bytes
    monkeypatch.setattr(container, "_shm_stat", lambda: (total_bytes, free))


def _no_shm(monkeypatch):
    monkeypatch.setattr(container, "_shm_stat", lambda: (0, 0))


@pytest.fixture
def tempdir(tmp_path, monkeypatch):
    """Redirect `tempfile.gettempdir()`, which caches and so ignores a late `TMPDIR`."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


# --- Reading the limits ------------------------------------------------------------------


def test_a_default_docker_shm_is_recognized_as_too_small(monkeypatch):
    _fake_shm(monkeypatch, container.DOCKER_DEFAULT_SHM_BYTES)
    assert container.shm_bytes() == container.DOCKER_DEFAULT_SHM_BYTES
    assert container.usable_shm() is False
    (finding,) = [f for f in container.container_findings() if "/dev/shm" in f]
    assert "64 MiB" in finding
    assert "the container runtime's default" in finding, "worth saying nobody chose it"
    assert "--shm-size" in finding, "a finding without its fix is a complaint"


def test_a_deliberately_small_shm_is_reported_without_blaming_a_default(monkeypatch):
    _fake_shm(monkeypatch, 200 * (1 << 20))
    (finding,) = [f for f in container.container_findings() if "/dev/shm" in f]
    assert "200 MiB" in finding
    assert "default" not in finding


def test_a_real_shm_allocation_is_not_a_finding(monkeypatch):
    _fake_shm(monkeypatch, 64 * (1 << 30))
    assert container.usable_shm() is True
    assert not [f for f in container.container_findings() if "/dev/shm" in f]


def test_no_shm_at_all_is_unknown_rather_than_a_finding(monkeypatch):
    # Unknown is not a finding. A platform with no `/dev/shm` gets the temp-dir path and no
    # advice, rather than being told its zero-byte shared memory is too small.
    _no_shm(monkeypatch)
    assert container.shm_bytes() == 0
    assert container.usable_shm() is False
    assert not [f for f in container.container_findings() if "/dev/shm" in f]


def test_a_large_but_full_shm_will_not_take_the_write(monkeypatch):
    # An allocation that is large in general and full right now helps nobody, and the write
    # fails the same way as if it had never been large.
    _fake_shm(monkeypatch, 64 * (1 << 30), free_bytes=1 << 20)
    assert container.usable_shm() is True, "the allocation itself is fine"
    assert container.usable_shm(1 << 30) is False, "and there is no room in it right now"


def test_a_pinned_memory_ceiling_is_reported_with_its_flag(monkeypatch):
    monkeypatch.setattr(container, "_rlimit", lambda name: 65536 if "MEMLOCK" in name else 0)
    (finding,) = [f for f in container.container_findings() if "memlock" in f]
    assert "64 KiB" in finding
    assert "GPUDirect" in finding
    assert "--ulimit memlock=-1" in finding


def test_unlimited_memlock_reports_nothing_to_act_on(monkeypatch):
    # Unlimited and unreadable share the zero sentinel, because both mean "do nothing".
    monkeypatch.setattr(container, "_rlimit", lambda name: 0)
    assert container.memlock_limit_bytes() == 0
    assert container.container_findings() == () or all(
        "memlock" not in f for f in container.container_findings()
    )


def test_a_low_descriptor_limit_is_reported(monkeypatch):
    monkeypatch.setattr(container, "_rlimit", lambda name: 1024 if "NOFILE" in name else 0)
    (finding,) = [f for f in container.container_findings() if "descriptors" in f]
    assert "1024" in finding


def test_a_generous_descriptor_limit_is_not(monkeypatch):
    monkeypatch.setattr(container, "_rlimit", lambda name: 1_048_576 if "NOFILE" in name else 0)
    assert not [f for f in container.container_findings() if "descriptors" in f]


def test_this_host_answers_without_raising():
    assert container.shm_bytes() >= 0
    assert container.memlock_limit_bytes() >= 0
    assert container.open_files_limit() >= 0
    assert isinstance(container.in_container(), bool)
    assert isinstance(container.container_findings(), tuple)


# --- The staging path that used to fail ----------------------------------------------------


def test_a_small_shm_sends_staging_to_the_temp_dir_instead(monkeypatch, tempdir):
    # The bug: `isdir("/dev/shm")` is true in a container with 64 MiB, so a shard write starts
    # and dies with ENOSPC. A slower directory that works beats a fast one that does not.
    _fake_shm(monkeypatch, container.DOCKER_DEFAULT_SHM_BYTES)
    assert container.shm_root() == str(tempdir)


def test_a_real_shm_is_still_used(monkeypatch):
    _fake_shm(monkeypatch, 64 * (1 << 30))
    assert container.shm_root() == "/dev/shm"
    assert container.shm_root(1 << 30) == "/dev/shm"


def test_a_write_larger_than_the_free_space_goes_elsewhere(monkeypatch, tempdir):
    _fake_shm(monkeypatch, 64 * (1 << 30), free_bytes=100 * (1 << 20))
    assert container.shm_root(8 * (1 << 30)) == str(tempdir)


def test_the_udf_shard_directory_follows_the_same_rule(monkeypatch, tempdir):
    # The real call site. It builds a private 0700 directory under whichever root it is given,
    # and the point of the change is that the root is now a decision rather than an `isdir`.
    from batcher.core.udf.isolation import shard_directory

    _fake_shm(monkeypatch, container.DOCKER_DEFAULT_SHM_BYTES)
    path = shard_directory()
    assert path.startswith(str(tempdir)), path
    assert os.path.isdir(path)
    assert oct(os.stat(path).st_mode & 0o777) == "0o700", "still private, wherever it landed"


def test_the_container_findings_reach_the_problem_list(monkeypatch):
    # An operator running a deployment check gets the container's limits in the same list as
    # the hardware's, because both are reasons the node will underperform and neither is
    # visible from a job's own timings.
    from batcher.api.session.accelerators.report import accelerator_problems

    monkeypatch.setattr(container, "_rlimit", lambda name: 65536 if "MEMLOCK" in name else 0)
    _fake_shm(monkeypatch, container.DOCKER_DEFAULT_SHM_BYTES)
    problems = accelerator_problems()
    assert any(p.startswith("container: ") and "/dev/shm" in p for p in problems), problems
    assert any("memlock" in p for p in problems), problems


def test_a_healthy_host_contributes_no_container_problems(monkeypatch):
    from batcher.api.session.accelerators.report import accelerator_problems

    monkeypatch.setattr(container, "_rlimit", lambda name: 0)
    _fake_shm(monkeypatch, 64 * (1 << 30))
    assert not [p for p in accelerator_problems() if p.startswith("container: ")]
