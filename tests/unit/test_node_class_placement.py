"""WS1: the heterogeneous node-class placement seam.

Ray's built-in GPU-node avoidance is best-effort, so a CPU relational fleet can still
land on and hold an idle GPU node's cores. `node_class_selector` emits a HARD custom-
resource requirement pinning such a fleet to CPU-only nodes — but only when the cluster
opts in (`heterogeneous_node_isolation`) and its CPU-only nodes can host the fleet, so it
never makes a query unschedulable. `_resolve_placement_strategy` gang-schedules a
GPU-collective stage STRICT_PACK. Placement never changes results; these assert the
resolved Ray options against a fake topology.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from batcher.config import Config, config_context
from batcher.plan.resource import SchedulingEnvelope


def _fake_topology(monkeypatch, nodes: list[dict]) -> None:
    """Install a `ray` whose `nodes()` returns the given per-node Resources/Labels."""
    ray_mod = types.ModuleType("ray")
    ray_mod.nodes = lambda: nodes
    monkeypatch.setitem(sys.modules, "ray", ray_mod)


def _node(cpus: float, gpus: float = 0.0) -> dict:
    res = {"CPU": cpus}
    if gpus:
        res["GPU"] = gpus
    return {"Alive": True, "Resources": res, "Labels": {}}


def _isolated() -> Config:
    cfg = Config()
    return cfg.replace(
        distributed=dataclasses.replace(cfg.distributed, heterogeneous_node_isolation=True)
    )


def test_selector_empty_when_gate_off(monkeypatch):
    from batcher.dist.executors.ray_runtime import node_class_selector

    _fake_topology(monkeypatch, [_node(16), _node(16, gpus=4)])
    # Gate off (default) -> rely on Ray's soft avoidance; emit nothing.
    assert node_class_selector(True, workers=4, num_cpus=1.0) == {}


def test_selector_empty_on_homogeneous_cluster(monkeypatch):
    from batcher.dist.executors.ray_runtime import node_class_selector

    _fake_topology(monkeypatch, [_node(16), _node(16)])  # no GPU nodes -> nothing to avoid
    with config_context(_isolated()):
        assert node_class_selector(True, workers=4, num_cpus=1.0) == {}


def test_selector_pins_when_cpu_only_can_host(monkeypatch):
    from batcher.dist.executors.ray_runtime import node_class_selector

    _fake_topology(monkeypatch, [_node(16), _node(8, gpus=4)])  # 16 CPU-only cores
    with config_context(_isolated()):
        sel = node_class_selector(True, workers=4, num_cpus=1.0)  # needs 4 cores
        assert sel == {"resources": {"cpu_node": 0.001}}


def test_selector_empty_when_cpu_only_too_small(monkeypatch):
    from batcher.dist.executors.ray_runtime import node_class_selector

    _fake_topology(monkeypatch, [_node(2), _node(64, gpus=8)])  # only 2 CPU-only cores
    with config_context(_isolated()):
        # 8 workers x 1 core > 2 CPU-only cores -> don't make the fleet unschedulable.
        assert node_class_selector(True, workers=8, num_cpus=1.0) == {}


def test_selector_empty_when_not_preferred(monkeypatch):
    from batcher.dist.executors.ray_runtime import node_class_selector

    _fake_topology(monkeypatch, [_node(16), _node(8, gpus=4)])
    with config_context(_isolated()):
        assert node_class_selector(False, workers=4, num_cpus=1.0) == {}


def test_selector_empty_on_unreadable_topology(monkeypatch):
    from batcher.dist.executors.ray_runtime import node_class_selector

    ray_mod = types.ModuleType("ray")

    def _boom():
        raise RuntimeError("ray down")

    ray_mod.nodes = _boom
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    with config_context(_isolated()):
        assert node_class_selector(True, workers=4, num_cpus=1.0) == {}


def test_bundle_includes_cpu_node_when_isolated(monkeypatch):
    from batcher.dist.executors.ray_runtime import scheduling

    _fake_topology(monkeypatch, [_node(16), _node(8, gpus=4)])
    env = SchedulingEnvelope(num_cpus=1.0, n_tasks=4, prefer_cpu_only_nodes=True)
    with config_context(_isolated()):
        bundle = scheduling._bundle(env)
        assert bundle.get("cpu_node") == 0.001
    # Gate off -> no cpu_node key in the bundle.
    assert "cpu_node" not in scheduling._bundle(env)


def test_task_options_merges_selector(monkeypatch):
    from batcher.dist.executors.ray_runtime import scheduling

    _fake_topology(monkeypatch, [_node(16), _node(8, gpus=4)])
    env = SchedulingEnvelope(num_cpus=1.0, n_tasks=4, prefer_cpu_only_nodes=True)
    monkeypatch.setattr(scheduling, "worker_runtime_env", lambda: None)
    with config_context(_isolated()):
        opts = scheduling.task_options(env)
        assert opts["resources"] == {"cpu_node": 0.001}


def test_gpu_collective_forces_strict_pack(monkeypatch):
    from batcher.dist.executors.ray_runtime import scheduling

    _fake_topology(monkeypatch, [_node(16), _node(16), _node(16)])  # multi-node
    env = SchedulingEnvelope(gpu_collective=True, placement_strategy="SPREAD")
    # STRICT_PACK wins over the SPREAD preference even on a multi-node cluster.
    assert scheduling._resolve_placement_strategy(env) == "STRICT_PACK"


def test_no_collective_keeps_spread(monkeypatch):
    from batcher.dist.executors.ray_runtime import scheduling

    _fake_topology(monkeypatch, [_node(16), _node(16)])
    env = SchedulingEnvelope(placement_strategy="SPREAD")
    assert scheduling._resolve_placement_strategy(env) == "SPREAD"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
