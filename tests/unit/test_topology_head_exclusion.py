"""The worker fan-out node count excludes the Ray head — matching worker placement.

`learned_num_workers` caps the worker count at `cluster_topology()["nodes"]`, and the SPREAD
placement bundle count comes from `alive_node_count()`. Worker actors are never placed on the
head (`node:__internal_head__`), so both counts MUST exclude it too — otherwise a data-heavy
shuffle requests one worker more than the schedulable node count and the un-placeable actor
hangs the fleet spawn (`ray.get` on its address never returns). A single-node cluster (head
only) keeps the head, since it has to run the work.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import scaling

pytestmark = pytest.mark.unit


def _patch_nodes(monkeypatch, specs):
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

        @staticmethod
        def cluster_resources():
            return {"CPU": sum(c for c, _ in specs)}

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr(scaling, "_TOPOLOGY", scaling._TOPOLOGY)  # ensure no active snapshot
    scaling._TOPOLOGY.set(None)


def test_head_excluded_from_node_count(monkeypatch):
    # 1 head + 16 workers: the fan-out cap and spread count must both be 16, not 17.
    _patch_nodes(monkeypatch, [(0.0, True)] + [(16.0, False)] * 16)
    assert scaling.alive_node_count() == 16
    assert scaling.cluster_topology()["nodes"] == 16
    assert scaling.cluster_topology()["cpus"] == 256.0  # 16 x 16, head's cores excluded


def test_head_with_cores_is_also_excluded(monkeypatch):
    # Raw Ray cluster: the head has cores too, but still must not host workers.
    _patch_nodes(monkeypatch, [(16.0, True), (16.0, False), (16.0, False)])
    assert scaling.alive_node_count() == 2
    assert scaling.cluster_topology()["nodes"] == 2


def test_single_node_cluster_keeps_head(monkeypatch):
    # Head-only cluster must keep the head — it has to run the work.
    _patch_nodes(monkeypatch, [(8.0, True)])
    assert scaling.alive_node_count() == 1
    assert scaling.cluster_topology()["nodes"] == 1
