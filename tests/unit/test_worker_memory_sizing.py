"""Hardware metadata → decision: the distributed memory budget is sized from the WORKER
node's RAM, not the driver's.

`cluster_topology()` now captures per-node memory (the hardware fact), and the distributed
executor caps each worker's spill budget at the smallest worker node's RAM × soft_limit. A
large driver (a 197 GiB head) therefore never hands a 34 GiB worker a budget it can't hold,
and an unbounded grant becomes a machine-sized one — the "Carbonite protects" invariant made
hardware-aware. Carbonite's tighter data-driven estimate still wins.
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.dist import executor as ex
from batcher.dist.executors.ray_runtime import scaling
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit

_GB = 1024**3


def _patch_nodes(monkeypatch, specs):
    """specs: list of (cpu, mem_bytes, is_head)."""

    class _Ray:
        @staticmethod
        def nodes():
            out = []
            for cpu, mem, is_head in specs:
                res = {"CPU": cpu, "memory": mem}
                if is_head:
                    res["node:__internal_head__"] = 1.0
                out.append({"Alive": True, "Resources": res})
            return out

        @staticmethod
        def cluster_resources():
            return {"CPU": sum(c for c, _, _ in specs)}

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    scaling._TOPOLOGY.set(None)


def test_topology_captures_node_memory(monkeypatch):
    # 1 big head (excluded) + 16 x 34 GiB workers.
    _patch_nodes(monkeypatch, [(96.0, 197 * _GB, True)] + [(16.0, 34 * _GB, False)] * 16)
    topo = scaling.cluster_topology()
    assert topo["memory"] == 16 * 34 * _GB  # head's 197 GiB excluded
    assert topo["min_node_memory"] == 34 * _GB
    assert scaling.worker_node_memory_bytes() == 34 * _GB


def test_budget_sized_from_worker_not_driver(monkeypatch):
    # 16 workers, one per node → per-node-workers = 1 → budget = 34 GiB * soft_limit.
    _patch_nodes(monkeypatch, [(0.0, 197 * _GB, True)] + [(16.0, 34 * _GB, False)] * 16)
    soft = active_config().memory.soft_limit
    env = SchedulingEnvelope(num_cpus=16, n_tasks=16, memory_bytes=0)  # unbounded
    out = ex._size_worker_memory(env, workers=16, num_cpus=16)
    assert out is not env
    assert out.memory_bytes == int(34 * _GB * soft)  # sized to the worker node, not 197 GiB


def test_carbonite_tighter_estimate_wins(monkeypatch):
    _patch_nodes(monkeypatch, [(16.0, 34 * _GB, False)] * 4)
    tight = 2 * _GB
    env = SchedulingEnvelope(num_cpus=16, n_tasks=4, memory_bytes=tight)
    out = ex._size_worker_memory(env, workers=4, num_cpus=16)
    assert out.memory_bytes == tight  # a tighter data-driven estimate is kept (min wins)


def test_driver_oversized_grant_clamped_to_worker(monkeypatch):
    _patch_nodes(monkeypatch, [(16.0, 34 * _GB, False)] * 4)
    soft = active_config().memory.soft_limit
    huge = 100 * _GB  # e.g. sized from a 197 GiB driver
    env = SchedulingEnvelope(num_cpus=16, n_tasks=4, memory_bytes=huge)
    out = ex._size_worker_memory(env, workers=4, num_cpus=16)
    assert out.memory_bytes == int(34 * _GB * soft)  # clamped down to the worker machine


def test_multiple_workers_per_node_split_the_ram(monkeypatch):
    # 8 workers over 4 nodes → 2 per node → each gets half the node's soft budget.
    _patch_nodes(monkeypatch, [(16.0, 34 * _GB, False)] * 4)
    soft = active_config().memory.soft_limit
    out = ex._size_worker_memory(None, workers=8, num_cpus=8)
    assert out.memory_bytes == int(34 * _GB * soft / 2)


def test_no_node_memory_leaves_grant_untouched(monkeypatch):
    # A cluster that doesn't advertise a memory resource → grant unchanged (today's behavior).
    _patch_nodes(monkeypatch, [(16.0, 0, False)] * 4)
    env = SchedulingEnvelope(num_cpus=16, n_tasks=4, memory_bytes=0)
    assert ex._size_worker_memory(env, workers=4, num_cpus=16) is env
