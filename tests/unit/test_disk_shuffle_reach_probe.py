"""The disk shuffle proves its scratch is one directory on every node before using it.

The disk transport passes only *paths* between tasks, so it is correct only where a path
names the same file on every node. `distributed.shared_filesystem = True` declares that and
an explicit `transport="disk"` assumes it; nothing checked either, so a mount missing on one
node surfaced as a `FileNotFoundError` deep inside a reducer. `verify_shared_scratch` has the
driver write a random token and each other node read it back at the same path first.

The node reads are injected, so these tests exercise the decision on one machine. The Ray
half (`readiness._read_on_nodes`) is exercised by the distributed suite on a real cluster.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ConfigError
from batcher.dist import shuffle_io
from batcher.dist.shuffle_io import read_visibility_token, verify_shared_scratch

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(shuffle_io, "_VISIBLE_SCRATCH", set())
    monkeypatch.setattr(shuffle_io, "_scratch_base", lambda: str(tmp_path))


def _same_filesystem(node_ids, path):
    """Every node reads the file the driver wrote: the shared-mount case."""
    return {n: read_visibility_token(path) for n in node_ids}


def test_a_shared_mount_passes_and_is_probed_once():
    calls = []

    def reader(node_ids, path):
        calls.append(tuple(node_ids))
        return _same_filesystem(node_ids, path)

    verify_shared_scratch(["b", "a"], reader)
    verify_shared_scratch(["a", "b"], reader)
    assert calls == [("a", "b")], "a proven shape must not be probed again"


def test_a_node_without_the_mount_fails_with_the_node_named():
    def reader(node_ids, path):
        out = _same_filesystem(node_ids, path)
        out["worker-2"] = None  # this node cannot open the path at all
        return out

    with pytest.raises(ConfigError, match="worker-2 cannot read"):
        verify_shared_scratch(["worker-1", "worker-2"], reader)


def test_a_node_with_a_different_volume_at_the_same_path_fails():
    """The quieter failure: the path exists on that node but holds other bytes."""

    def reader(node_ids, path):
        out = _same_filesystem(node_ids, path)
        out["worker-1"] = "someone else's file"
        return out

    with pytest.raises(ConfigError, match="reads different bytes"):
        verify_shared_scratch(["worker-1"], reader)


def test_a_silent_node_is_logged_not_failed_and_not_reprobed():
    calls = []

    def reader(node_ids, path):
        calls.append(1)
        return {}  # nobody answered inside the timeout

    verify_shared_scratch(["worker-1"], reader)
    verify_shared_scratch(["worker-1"], reader)
    assert calls == [1]


def test_no_other_nodes_costs_nothing():
    def reader(node_ids, path):
        raise AssertionError("a single-node cluster must not probe")

    verify_shared_scratch([], reader)


def test_the_sentinel_is_removed(tmp_path):
    verify_shared_scratch(["a"], _same_filesystem)
    assert not list(tmp_path.glob(".batcher_visibility_*"))


def test_read_visibility_token_reports_a_missing_file(tmp_path):
    assert read_visibility_token(str(tmp_path / "absent")) is None
