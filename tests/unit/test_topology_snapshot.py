"""The per-query topology snapshot: one `ray.nodes()` read for the whole placement phase.

A distributed query reads the cluster shape from several scheduling helpers (transport
choice, placement strategy, node-class selector, spread heuristic). Each is an O(nodes)
`ray.nodes()` RPC, so at thousands of nodes the redundant reads add up. `topology_scope()`
snapshots the shape once; readers inside it share that snapshot. Outside a scope every
reader reads live (unchanged behavior). These assert both, with a call-counting fake Ray.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_counting_ray(monkeypatch, n_nodes: int = 500) -> dict:
    calls = {"nodes": 0, "resources": 0}

    def nodes():
        calls["nodes"] += 1
        cpu = {"Alive": True, "Resources": {"CPU": 16.0}, "Labels": {}}
        gpu = {"Alive": True, "Resources": {"CPU": 8.0, "GPU": 4.0}, "Labels": {}}
        return [cpu] * (n_nodes - 5) + [gpu] * 5

    def cluster_resources():
        calls["resources"] += 1
        return {"CPU": 16.0 * (n_nodes - 5) + 8.0 * 5, "GPU": 20.0}

    ray_mod = types.ModuleType("ray")
    ray_mod.nodes = nodes
    ray_mod.cluster_resources = cluster_resources
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    return calls


def test_scope_reads_topology_once(monkeypatch):
    from batcher.dist.executors.ray_runtime import scaling

    calls = _install_counting_ray(monkeypatch, n_nodes=500)
    with scaling.topology_scope():
        # A handful of reads that a placement phase makes, all inside the scope.
        t1 = scaling.cluster_topology()
        c1 = scaling.node_classes()
        n1 = scaling.alive_node_count()
        n2 = scaling.alive_node_count()
        t2 = scaling.cluster_topology()
    # One snapshot read for the whole scope, not one per call.
    assert calls["nodes"] == 1
    assert calls["resources"] == 1
    assert t1 == t2 and t1["nodes"] == 500
    assert n1 == n2 == 500
    assert len(c1) == 500 and sum(1 for c in c1 if c["gpus"] > 0) == 5


def test_no_scope_reads_live_each_time(monkeypatch):
    from batcher.dist.executors.ray_runtime import scaling

    calls = _install_counting_ray(monkeypatch, n_nodes=50)
    scaling.cluster_topology()
    scaling.alive_node_count()
    scaling.node_classes()
    # No scope active → each helper reads live (unchanged behavior).
    assert calls["nodes"] == 3


def test_scope_restores_live_reads_after_exit(monkeypatch):
    from batcher.dist.executors.ray_runtime import scaling

    calls = _install_counting_ray(monkeypatch, n_nodes=10)
    with scaling.topology_scope():
        scaling.alive_node_count()
    assert calls["nodes"] == 1
    scaling.alive_node_count()  # back to live
    assert calls["nodes"] == 2


def test_scope_falls_back_when_unreadable(monkeypatch):
    from batcher.dist.executors.ray_runtime import scaling

    ray_mod = types.ModuleType("ray")

    def _boom():
        raise RuntimeError("ray down")

    ray_mod.nodes = _boom
    ray_mod.cluster_resources = _boom
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    # A read failure entering the scope must not raise; readers just fall back to live
    # (and here that also fails, so alive_node_count returns 0 — treated as "unknown").
    with scaling.topology_scope():
        assert scaling.alive_node_count() == 0


def test_nested_scope_reuses_outer_snapshot(monkeypatch):
    from batcher.dist.executors.ray_runtime import scaling

    calls = _install_counting_ray(monkeypatch, n_nodes=20)
    with scaling.topology_scope():
        with scaling.topology_scope():
            scaling.alive_node_count()
        scaling.alive_node_count()
    assert calls["nodes"] == 1  # inner scope reused the outer snapshot


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
