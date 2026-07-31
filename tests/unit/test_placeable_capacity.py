"""A fan-out must not exceed what some single node can actually host.

Ray gang-schedules the worker fleet, so a fan-out sized above what any arrangement of
nodes can hold leaves the placement group permanently unsatisfiable — the job hangs rather
than failing, which is the failure `placeable_workers` exists to prevent. It gets that
right for cores. It ignored two other things the bundle reserves, each of which overstates
capacity in exactly the hanging direction:

the node-class restriction, so a relational fleet held off accelerator nodes was still
counted as able to use their cores; and the per-worker memory grant, which on the
memory-heavy shuffles the grant exists for binds well before cores do.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import capacity, scaling

pytestmark = pytest.mark.unit

_GIB = 1024**3


def _nodes(monkeypatch, specs):
    """specs: list of (cpus, gpus, memory_bytes)."""
    classes = [
        {
            "cpus": float(c),
            "gpus": float(g),
            "memory": float(m),
            "accelerators": 0.0,
            "accelerator_type": None,
        }
        for c, g, m in specs
    ]
    monkeypatch.setattr(scaling, "node_classes", lambda: classes)


def test_counts_per_node_not_cluster_total(monkeypatch):
    """The original contract: four 8-core nodes cannot host a 16-core worker, however
    many cores they total."""
    _nodes(monkeypatch, [(8, 0, 64 * _GIB)] * 4)
    assert capacity.placeable_workers(16.0) == 0


def test_unreadable_topology_reports_unknown(monkeypatch):
    monkeypatch.setattr(scaling, "node_classes", lambda: [])
    assert capacity.placeable_workers(4.0) is None


class TestCpuOnlyRestriction:
    """A fleet held to CPU-only nodes cannot use an accelerator node's cores."""

    def test_gpu_node_cores_are_not_counted(self, monkeypatch):
        _nodes(monkeypatch, [(16, 0, 64 * _GIB), (64, 8, 512 * _GIB)])
        assert capacity.placeable_workers(16.0) == 5  # 1 + 4, unrestricted
        assert capacity.placeable_workers(16.0, cpu_only=True) == 1  # the CPU node alone

    def test_unrestricted_fleet_is_unchanged(self, monkeypatch):
        """The default path must count every node exactly as before."""
        _nodes(monkeypatch, [(16, 0, 64 * _GIB), (64, 8, 512 * _GIB)])
        assert capacity.placeable_workers(16.0, cpu_only=False) == 5

    def test_a_homogeneous_cpu_cluster_is_unaffected(self, monkeypatch):
        _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 3)
        assert capacity.placeable_workers(16.0, cpu_only=True) == 3

    def test_no_cpu_only_nodes_reports_unknown_not_zero(self, monkeypatch):
        """Zero would clamp the fan-out to a single worker. The restriction is only applied
        when CPU-only nodes can host the fleet, so an all-accelerator cluster here means the
        topology moved under us — report unknown and keep the total-based estimate."""
        _nodes(monkeypatch, [(64, 8, 512 * _GIB)] * 2)
        assert capacity.placeable_workers(16.0, cpu_only=True) is None


class TestMemoryBound:
    """A node with spare cores but no spare RAM hosts zero workers."""

    def test_memory_binds_before_cores(self, monkeypatch):
        # 64 cores would host 16 four-core workers, but only 8 fit in 64 GiB at 8 GiB each.
        _nodes(monkeypatch, [(64, 0, 64 * _GIB)])
        assert capacity.placeable_workers(4.0) == 16
        assert capacity.placeable_workers(4.0, memory_bytes=8 * _GIB) == 8

    def test_zero_memory_grant_skips_the_bound(self, monkeypatch):
        _nodes(monkeypatch, [(64, 0, 64 * _GIB)])
        assert capacity.placeable_workers(4.0, memory_bytes=0) == 16

    def test_a_node_reporting_no_memory_is_not_read_as_having_none(self, monkeypatch):
        """Ray not tracking memory on a node must not collapse the whole fan-out to one."""
        _nodes(monkeypatch, [(64, 0, 0)])
        assert capacity.placeable_workers(4.0, memory_bytes=8 * _GIB) == 16

    def test_a_grant_larger_than_any_node_hosts_nothing(self, monkeypatch):
        _nodes(monkeypatch, [(64, 0, 32 * _GIB)] * 4)
        assert capacity.placeable_workers(4.0, memory_bytes=64 * _GIB) == 0


def test_bounds_compose(monkeypatch):
    """Cores, GPUs, memory and node class all narrow the same count."""
    _nodes(monkeypatch, [(64, 0, 32 * _GIB), (64, 8, 512 * _GIB)])
    # CPU-only node: 64 cores / 8 = 8 by cores, 32 GiB / 16 GiB = 2 by memory.
    assert capacity.placeable_workers(8.0, memory_bytes=16 * _GIB, cpu_only=True) == 2
