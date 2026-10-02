"""Out-of-core scratch left by a killed process is reclaimed; a live process's is not.

`carbonite.spill.scratch.scratch_dir` allocates the working directory of every distributed
out-of-core breaker and of the result cache's disk tier. Its `rmtree`/`atexit` cleanup does
not run on `SIGKILL`, so a directory abandoned by an OOM-killed process used to stay on the
spill volume forever. The directory now carries its owner's pid, and the next allocation
under the same root sweeps the ones whose owner is gone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from batcher.carbonite.spill import scratch
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.unit


def _dead_pid() -> int:
    """The pid of a process that has already exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_dead_owners_scratch_is_swept_and_nothing_else_is(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scratch, "_SWEPT", set())
    prefix = "batcher_orphan_test_"
    dead = tmp_path / f"{prefix}{_dead_pid()}-abc"
    live = tmp_path / f"{prefix}{os.getppid()}-def"  # the test runner's parent: alive
    not_ours = tmp_path / f"{prefix}notapid"
    other_prefix = tmp_path / f"batcher_cache_{_dead_pid()}-ghi"
    for d in (dead, live, not_ours, other_prefix):
        d.mkdir()
        (d / "part-0.arrow").write_bytes(b"x")

    with config_context(Config().replace(memory=MemoryConfig(spill_dir=str(tmp_path)))):
        work_dir, owned = scratch.scratch_dir(None, prefix)

    assert owned
    assert Path(work_dir).name.startswith(f"{prefix}{os.getpid()}-")
    assert not dead.exists(), "a dead process's scratch must be reclaimed"
    assert live.exists(), "a live process's scratch must never be touched"
    assert not_ours.exists(), "a name this module did not create must never be touched"
    assert other_prefix.exists(), "a sweep only considers its own prefix"


def test_the_new_directory_is_never_swept_by_its_own_process(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scratch, "_SWEPT", set())
    with config_context(Config().replace(memory=MemoryConfig(spill_dir=str(tmp_path)))):
        first, _ = scratch.scratch_dir(None, "batcher_self_")
        monkeypatch.setattr(scratch, "_SWEPT", set())
        second, _ = scratch.scratch_dir(None, "batcher_self_")
    assert Path(first).exists() and Path(second).exists() and first != second


def test_an_explicit_directory_is_the_callers_and_is_not_swept(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scratch, "_SWEPT", set())
    orphan = tmp_path / f"batcher_x_{_dead_pid()}-zz"
    orphan.mkdir()
    work_dir, owned = scratch.scratch_dir(str(tmp_path), "batcher_x_")
    assert work_dir == str(tmp_path) and not owned
    assert orphan.exists(), "a caller-owned directory is not this module's to clean"
