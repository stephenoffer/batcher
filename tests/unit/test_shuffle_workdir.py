"""The distributed shuffle scratch dir must be reachable from every node.

The disk-transport shuffle passes only *paths* between Ray tasks, so on any cluster that
can span more than one node the dir has to live on a shared filesystem. `shared_scratch_root`
resolves it: an explicit `spill_dir`, else an auto-detected cluster-shared mount (e.g.
`/mnt/cluster_storage`), else `None` (a genuine single node → node-local temp). Crucially it
does NOT gate on the *current* node count — an autoscaling cluster is often single-node when
the dir is created and multi-node moments later, and a task on a freshly-joined node must
still find the files. These are pure, Ray-free tests.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from batcher.config import active_config, config_context
from batcher.dist import shuffle_io

pytestmark = pytest.mark.unit


def test_explicit_spill_dir_wins(monkeypatch):
    # A configured spill_dir is the operator's chosen shared scratch — honored first.
    monkeypatch.setattr(os.path, "isdir", lambda p: True)  # even with mounts present
    base = active_config()
    cfg = base.replace(memory=dataclasses.replace(base.memory, spill_dir="/data/scratch"))
    with config_context(cfg):
        assert shuffle_io.shared_scratch_root() == "/data/scratch"


def test_shared_mount_used_whenever_present(monkeypatch):
    # A shared mount exists → use it, WITHOUT consulting the live node count (an autoscaling
    # cluster may be single-node right now but gain workers mid-query).
    monkeypatch.setattr(os.path, "isdir", lambda p: p == "/mnt/cluster_storage")
    base = active_config()
    cfg = base.replace(memory=dataclasses.replace(base.memory, spill_dir=None))
    with config_context(cfg):
        assert shuffle_io.shared_scratch_root() == "/mnt/cluster_storage/batcher_shuffle"


def test_no_mount_falls_back_to_local(monkeypatch):
    # No shared mount → a genuine single node (laptop / CI). Node-local temp is correct.
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    base = active_config()
    cfg = base.replace(memory=dataclasses.replace(base.memory, spill_dir=None))
    with config_context(cfg):
        assert shuffle_io.shared_scratch_root() is None


def test_mount_precedence_is_narrowest_first(monkeypatch):
    # When several shared mounts exist, the cluster-scoped one is preferred.
    present = ("/mnt/user_storage", "/mnt/shared_storage")
    monkeypatch.setattr(os.path, "isdir", lambda p: p in present)
    base = active_config()
    cfg = base.replace(memory=dataclasses.replace(base.memory, spill_dir=None))
    with config_context(cfg):
        # cluster_storage absent → next in the ordered list (user_storage) wins.
        assert shuffle_io.shared_scratch_root() == "/mnt/user_storage/batcher_shuffle"


def test_distributed_work_dir_creates_under_root(monkeypatch, tmp_path):
    # `distributed_work_dir` makes a unique subdir under the resolved root and it exists.
    root = str(tmp_path / "shared")
    base = active_config()
    cfg = base.replace(memory=dataclasses.replace(base.memory, spill_dir=root))
    with config_context(cfg):
        wd = shuffle_io.distributed_work_dir("batcher_shuffle_")
    assert wd.startswith(root)
    assert os.path.isdir(wd)
