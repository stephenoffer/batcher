"""The distributed worker fan-out excludes the Ray head node.

Scheduling data operators on the head (GCS / dashboard / job supervisor) causes contention
and instability — the guides' "set num_cpus=0 on the head" rule. Batcher excludes the head
by its `node:__internal_head__` marker so it is correct on ANY cluster type, not only a
managed cluster that already gives the head 0 CPU. A single-node cluster keeps the head.

The subject here is *which nodes* host workers, so every assertion below is really about the
worker count. The per-worker grant is one core below the node size because `_headroom_grant`
keeps the fleet from reserving a node's last core — see `test_headroom_grant.py`; the fan-out
it is paired with is unchanged.
"""

from __future__ import annotations

import pytest

from batcher.dist import executor as ex

pytestmark = pytest.mark.unit


def _nodes(monkeypatch, specs):
    """specs: list of (cpu, is_head)."""

    class _Ray:
        @staticmethod
        def nodes():
            out = []
            for cpu, is_head in specs:
                res = {"CPU": cpu}
                if is_head:
                    res["node:__internal_head__"] = 1.0
                out.append({"Alive": True, "Resources": res})
            return out

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)


def test_head_with_cores_is_excluded(monkeypatch):
    # Raw Ray cluster: head has 16 cores too, plus 3 workers. The head must not count.
    _nodes(monkeypatch, [(16.0, True), (16.0, False), (16.0, False), (16.0, False)])
    assert ex._worker_node_cpus() == [16.0, 16.0, 16.0]
    assert ex._cluster_fill_workers() == (3, 15.0)  # 3 workers; grant leaves a spare core


def test_zero_cpu_head_already_excluded(monkeypatch):
    _nodes(monkeypatch, [(0.0, True), (16.0, False), (16.0, False)])
    assert ex._worker_node_cpus() == [16.0, 16.0]
    assert ex._cluster_fill_workers() == (2, 15.0)  # 2 workers; grant leaves a spare core


def test_single_node_head_only_keeps_the_head(monkeypatch):
    # Head-only cluster: it must run the work (there is nothing else).
    _nodes(monkeypatch, [(16.0, True)])
    assert ex._worker_node_cpus() == [16.0]
    assert ex._cluster_fill_workers() is None  # single node -> data-driven sizing


def test_even_cpu_share_ignores_head(monkeypatch):
    _nodes(monkeypatch, [(64.0, True), (16.0, False), (16.0, False)])
    # min over WORKER nodes is 16 (not the head's 64), and total worker CPU is 32.
    assert ex._even_cpu_share(2) == 16.0


def test_homogeneous_stays_one_worker_per_node(monkeypatch):
    # No change for a homogeneous cluster: one worker per node, granted the node's cores.
    _nodes(monkeypatch, [(32.0, False)] * 16)
    assert ex._cluster_fill_workers() == (16, 31.0)  # still one worker per node


def test_heterogeneous_fills_bigger_nodes(monkeypatch):
    # 8x32-core + 8x64-core: the 64-core nodes each host 2 min(=32)-core workers so their
    # extra cores are used, not stranded. 8*1 + 8*2 = 24 workers, each 32 cores.
    _nodes(monkeypatch, [(32.0, False)] * 8 + [(64.0, False)] * 8)
    assert ex._cluster_fill_workers() == (24, 31.0)  # still 8*1 + 8*2 = 24 workers


def test_heterogeneous_head_excluded_from_fill(monkeypatch):
    # The head (even with the most cores) never hosts workers, and never inflates the fill.
    _nodes(monkeypatch, [(128.0, True), (16.0, False), (48.0, False)])
    # min worker node = 16; 16->1 slot, 48->floor(48/16)=3 slots. 4 workers, 16 cores each.
    assert ex._cluster_fill_workers() == (4, 15.0)  # still 1 + 3 = 4 workers
