"""The distributed worker fan-out excludes the Ray head node.

Scheduling data operators on the head (GCS / dashboard / job supervisor) causes contention
and instability — the guides' "set num_cpus=0 on the head" rule. Batcher excludes the head
by its `node:__internal_head__` marker so it is correct on ANY cluster type, not only a
managed cluster that already gives the head 0 CPU. A single-node cluster keeps the head.
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
    assert ex._cluster_fill_workers() == (3, 16.0)


def test_zero_cpu_head_already_excluded(monkeypatch):
    _nodes(monkeypatch, [(0.0, True), (16.0, False), (16.0, False)])
    assert ex._worker_node_cpus() == [16.0, 16.0]
    assert ex._cluster_fill_workers() == (2, 16.0)


def test_single_node_head_only_keeps_the_head(monkeypatch):
    # Head-only cluster: it must run the work (there is nothing else).
    _nodes(monkeypatch, [(16.0, True)])
    assert ex._worker_node_cpus() == [16.0]
    assert ex._cluster_fill_workers() is None  # single node -> data-driven sizing


def test_even_cpu_share_ignores_head(monkeypatch):
    _nodes(monkeypatch, [(64.0, True), (16.0, False), (16.0, False)])
    # min over WORKER nodes is 16 (not the head's 64), and total worker CPU is 32.
    assert ex._even_cpu_share(2) == 16.0
