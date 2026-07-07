"""WS2: adaptive stateless-map placement (SPREAD vs Ray's locality-aware DEFAULT).

Forcing SPREAD on every map task disables Ray's argument-locality scheduling and, past
~100 nodes, makes the scheduler itself the bottleneck. `_map_scheduling_options` resolves
SPREAD vs DEFAULT against the live cluster: SPREAD only where sub-core tasks would pack
onto one node and idle the rest. Placement never changes results, so these assert the
*strategy* choice against a fake topology and config.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from batcher.config import Config, config_context
from batcher.plan.resource import SchedulingEnvelope


def _fake_nodes(monkeypatch, n_alive: int) -> None:
    ray_mod = types.ModuleType("ray")
    ray_mod.nodes = lambda: [{"Alive": True} for _ in range(n_alive)]
    monkeypatch.setitem(sys.modules, "ray", ray_mod)


def _with_map_spread(**overrides):
    cfg = Config()
    return cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides))


def test_single_node_uses_default(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 1)
    # SPREAD == DEFAULT on one node — skip the spread bookkeeping.
    assert _map_scheduling_options(None, [0.1, 0.1, 0.1]) == {}


def test_small_shares_many_tasks_spreads(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 4)  # below the node cap
    # 8 sub-half-core tasks over 4 nodes -> DEFAULT would pack; keep SPREAD.
    shares = [0.125] * 8
    assert _map_scheduling_options(None, shares) == {"scheduling_strategy": "SPREAD"}


def test_near_whole_core_tasks_use_default(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 4)
    # ~1-core tasks fill nodes naturally under DEFAULT; locality wins.
    assert _map_scheduling_options(None, [1.0] * 8) == {}


def test_fewer_tasks_than_nodes_use_default(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 8)
    # 3 tasks, 8 nodes -> at most one per node under DEFAULT; no packing risk.
    assert _map_scheduling_options(None, [0.1, 0.1, 0.1]) == {}


def test_large_cluster_prefers_default(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 200)  # >= node cap: SPREAD's per-node scan is the bottleneck
    assert _map_scheduling_options(None, [0.1] * 500) == {}


def test_mode_always_forces_spread(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 4)
    with config_context(_with_map_spread(map_spread="always")):
        # Even near-whole-core tasks stay SPREAD — the historical unconditional behavior.
        assert _map_scheduling_options(None, [1.0] * 8) == {"scheduling_strategy": "SPREAD"}


def test_mode_never_forces_default(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 4)
    with config_context(_with_map_spread(map_spread="never")):
        assert _map_scheduling_options(None, [0.125] * 8) == {}


def test_strict_spread_envelope_always_spreads(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    _fake_nodes(monkeypatch, 200)  # even past the cap, an explicit STRICT_SPREAD wins
    env = SchedulingEnvelope(placement_strategy="STRICT_SPREAD")
    assert _map_scheduling_options(env, [1.0] * 3) == {"scheduling_strategy": "SPREAD"}


def test_unreadable_topology_falls_back_to_spread(monkeypatch):
    from batcher.dist.executors.map import _map_scheduling_options

    ray_mod = types.ModuleType("ray")

    def _boom():
        raise RuntimeError("ray down")

    ray_mod.nodes = _boom
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    # Conservative: keep the prior SPREAD behavior when the cluster shape can't be read.
    assert _map_scheduling_options(None, [0.5] * 4) == {"scheduling_strategy": "SPREAD"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
