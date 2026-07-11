"""A disk-shuffle-only operator must refuse a multi-node cluster with no shared scratch.

`resolve_transport` steers every other operator onto Flight the moment the cluster spans
more than one node, precisely because the disk shuffle passes only *paths* between tasks.
`asof_join` has no Flight path, so nothing steers it — and a worker opening a driver-local
`/tmp` path either crashes or, if a same-named directory exists on its node, silently reads
nothing. This guard turns that into an actionable error.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ray", reason="ray not installed")

from batcher._internal.errors import PlanError
from batcher.dist import executor as EX


@pytest.fixture
def cluster(monkeypatch):
    def _set(nodes: int, shared: str | None):
        import batcher.dist.shuffle_io as sio

        monkeypatch.setattr(EX, "alive_node_count", lambda: nodes)
        monkeypatch.setattr(sio, "shared_scratch_root", lambda: shared)

    return _set


def test_multi_node_without_shared_scratch_is_refused(cluster):
    cluster(nodes=4, shared=None)
    with pytest.raises(PlanError, match="shared mount"):
        EX._require_shared_scratch("asof_join")


def test_multi_node_with_a_shared_mount_is_allowed(cluster):
    cluster(nodes=4, shared="/mnt/cluster_storage/batcher_shuffle")
    EX._require_shared_scratch("asof_join")  # must not raise


def test_single_node_is_always_allowed(cluster):
    """One node: a driver-local tempdir is reachable by every task, so disk is correct."""
    cluster(nodes=1, shared=None)
    EX._require_shared_scratch("asof_join")


def test_the_error_names_the_operator_and_the_fix(cluster):
    cluster(nodes=2, shared=None)
    with pytest.raises(PlanError) as ei:
        EX._require_shared_scratch("asof_join")
    msg = str(ei.value)
    assert "asof_join" in msg
    assert "spill_dir" in msg and "distributed=False" in msg
