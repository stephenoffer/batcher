"""The map route sizes a task from a node it can actually be placed on.

`_placeable_node_cores` is the ceiling on every map task's CPU grant: a share above it is a
bundle no node can host. It derived that from its own `ray.nodes()` read, which counted two
kinds of node a map task never runs on -- the Ray head, and any node Ray has marked for drain
-- so the narrowest machine in the cluster capped the fleet whether or not work could land on
it. A four-core head beside sixteen-core workers quartered every task's grant.

The same read also went to the GCS on every call, outside the `topology_scope()` snapshot and
the 50 ms window every other topology reader in `dist` shares, and the map route asks for it
several times per query (the fan-out cap, the fleet fill, the actor width).

Both are fixed by reading through `scaling.worker_node_cores` / `cluster_topology`, which
already answer exactly this question for memory. These tests pin the behaviour rather than
the routing: each one fails against a head-counting read.
"""

from __future__ import annotations

import sys

import pytest

from batcher.dist.executors import map as mapmod
from batcher.dist.executors.ray_runtime import scaling

# Import the real Ray *before* any test stubs it, for the reason spelled out at the top of
# `test_topology_drain_exclusion.py`: monkeypatch removes a `sys.modules` entry it did not
# find, which leaves the package half-imported for the next genuine `import ray`.
ray = pytest.importorskip("ray")

pytestmark = pytest.mark.unit


def _patch_cluster(monkeypatch, specs, draining=()) -> None:
    """specs: list of (node_id, cpu, is_head). `draining`: node ids Ray reports draining."""

    class _Ray:
        @staticmethod
        def nodes():
            return [
                {
                    "Alive": True,
                    "NodeID": node_id,
                    "Resources": {
                        "CPU": cpu,
                        "memory": cpu * 1e9,
                        **({"node:__internal_head__": 1.0} if is_head else {}),
                    },
                }
                for node_id, cpu, is_head in specs
            ]

        @staticmethod
        def cluster_resources():
            return {"CPU": sum(c for _, c, _ in specs)}

    monkeypatch.setitem(sys.modules, "ray", _Ray)
    monkeypatch.setattr(scaling, "_read_draining", lambda: frozenset(draining))
    # The live reads are windowed by `_LIVE_TTL_S` (50 ms) in module globals, and two tests
    # run well inside that, so without this the second stub is never consulted.
    scaling._reset_topology_cache()
    scaling._TOPOLOGY.set(None)


def test_a_narrow_head_does_not_cap_a_map_task(monkeypatch) -> None:
    """The defect: no map task is placed on the head, so its width is not a ceiling."""
    _patch_cluster(monkeypatch, [("head", 4.0, True)] + [(f"w{i}", 16.0, False) for i in range(8)])

    assert mapmod._placeable_node_cores() == 16.0


def test_a_draining_node_does_not_cap_a_map_task(monkeypatch) -> None:
    """Same argument, other exclusion: a node being reclaimed takes no new work."""
    _patch_cluster(
        monkeypatch,
        [("head", 0.0, True), ("a", 16.0, False), ("b", 16.0, False), ("small", 2.0, False)],
        draining=("small",),
    )

    assert mapmod._placeable_node_cores() == 16.0


def test_the_narrowest_worker_still_binds(monkeypatch) -> None:
    """The positive control for the two above: an exclusion, not a maximum."""
    _patch_cluster(
        monkeypatch,
        [("head", 64.0, True), ("a", 16.0, False), ("b", 8.0, False)],
    )

    assert mapmod._placeable_node_cores() == 8.0


def test_the_fleet_core_count_excludes_the_head(monkeypatch) -> None:
    """`_cluster_cores` bounds the fan-out, and the head runs none of it."""
    _patch_cluster(monkeypatch, [("head", 32.0, True)] + [(f"w{i}", 16.0, False) for i in range(4)])

    assert mapmod._cluster_cores() == 64.0


def test_a_single_node_run_keeps_the_head(monkeypatch) -> None:
    """Excluding the head must never leave nothing: a one-box cluster IS its head."""
    _patch_cluster(monkeypatch, [("head", 12.0, True)])

    assert mapmod._placeable_node_cores() == 12.0
    assert mapmod._cluster_cores() == 12.0


def test_an_unreadable_topology_falls_back_to_this_box(monkeypatch) -> None:
    """Ray down is not zero cores; it is a local run."""

    class _Broken:
        @staticmethod
        def nodes():
            raise RuntimeError("gcs unreachable")

        @staticmethod
        def cluster_resources():
            raise RuntimeError("gcs unreachable")

    monkeypatch.setitem(sys.modules, "ray", _Broken)
    scaling._reset_topology_cache()
    scaling._TOPOLOGY.set(None)

    from batcher._internal.hardware import available_cpu_count

    assert mapmod._placeable_node_cores() == float(available_cpu_count())
    assert mapmod._cluster_cores() == float(available_cpu_count())
