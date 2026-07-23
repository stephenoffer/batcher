"""Per-worker memory must hold on the node packing the MOST workers, not an average one."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _sized(monkeypatch, node_cpus, workers, num_cpus, node_mem):
    from batcher.dist import executor as ex

    monkeypatch.setattr(ex, "_worker_node_cpus", lambda: node_cpus)
    monkeypatch.setattr(ex, "worker_node_memory_bytes", lambda: node_mem)
    monkeypatch.setattr(ex, "alive_node_count", lambda: len(node_cpus))
    return ex._size_worker_memory(None, workers, num_cpus)


def test_budget_holds_on_the_node_that_packs_the_most_workers(monkeypatch):
    """The regression: `_cluster_fill_workers` gives a 128-core node 4 workers beside three
    32-core nodes with 1 each. The average is ceil(7/4)=2, so each of those 4 workers was
    granted half the (smallest) node's RAM — twice what its node can honour."""
    gib = 1 << 30
    env = _sized(
        monkeypatch, [128.0, 32.0, 32.0, 32.0], workers=7, num_cpus=32.0, node_mem=64 * gib
    )
    from batcher.config import active_config

    soft = active_config().memory.soft_limit
    # 4 workers land on the 128-core node, so the divisor must be 4, not the average 2.
    assert env.memory_bytes == int(64 * gib * soft / 4)


def test_a_homogeneous_cluster_is_unchanged(monkeypatch):
    """One worker per node: max-per-node and the average agree, so nothing moves."""
    gib = 1 << 30
    env = _sized(monkeypatch, [32.0, 32.0, 32.0], workers=3, num_cpus=32.0, node_mem=64 * gib)
    from batcher.config import active_config

    assert env.memory_bytes == int(64 * gib * active_config().memory.soft_limit / 1)


def test_unreadable_topology_falls_back_to_the_average(monkeypatch):
    from batcher.dist import executor as ex

    gib = 1 << 30
    monkeypatch.setattr(ex, "_worker_node_cpus", lambda: [])
    monkeypatch.setattr(ex, "worker_node_memory_bytes", lambda: 64 * gib)
    monkeypatch.setattr(ex, "alive_node_count", lambda: 4)
    from batcher.config import active_config

    env = ex._size_worker_memory(None, 8, 32.0)
    assert env.memory_bytes == int(64 * gib * active_config().memory.soft_limit / 2)
