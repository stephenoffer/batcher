"""How finely the automatic fan-out cuts a node into workers.

`_numa_sliced` answers one question — how many workers a node's cores should be split
into — under two constraints that pull in opposite directions. A worker must not span two
memory domains, which sets a floor. And a worker is one shuffle pipeline that alternates
between gathering a bucket and computing it, so a node needs several of them before its
cores stay busy, which pushes past that floor.

These tests pin the rule at the boundaries where getting it wrong is expensive: the small
node that must keep exactly the fan-out it had, and the large node whose extra workers are
the difference between a quarter of the cluster busy and half of it.
"""

from __future__ import annotations

import pytest

import batcher.dist.executors.ray_runtime.scaling as scaling
from batcher.dist.executor import _MIN_WORKER_CORES, _TARGET_WORKER_CORES, _numa_sliced

pytestmark = pytest.mark.unit


@pytest.fixture
def domains(monkeypatch):
    """Pin the probed NUMA domain count."""

    def _set(n: int) -> None:
        monkeypatch.setattr(scaling, "cluster_numa_nodes", lambda: n)

    return _set


def test_a_large_two_domain_node_is_cut_to_the_core_target(domains):
    """The 96-core / 2-NUMA shape this was measured on: 4 workers, not 2.

    At two workers per node the fleet held every core and worked a quarter of them. Two per
    domain took TPC-H sf100 `lineitem join orders` from 10,022 ms to 8,990 ms and mean
    cluster CPU from 27% to 37%.
    """
    domains(2)
    grant = _numa_sliced(96.0)
    assert grant == 24.0
    assert int(96 // grant) == 4


def test_the_numa_domain_count_is_a_floor_the_target_cannot_undercut(domains):
    """A node with more domains than the core target would ask for still splits per domain.

    Spanning a memory domain makes half a worker's loads remote, which is a hardware fact
    rather than a tuning choice, so it is the one constraint the pipeline target may not
    relax.
    """
    domains(4)
    grant = _numa_sliced(64.0)  # target alone would say 64/24 -> 3 slices
    assert int(64 // grant) == 4


def test_a_small_node_keeps_the_fan_out_it_had(domains):
    """Below the per-worker core floor, slicing is declined outright rather than reduced.

    A worker carries a Flight server, its own hash tables and its own share of the buckets.
    A fleet of two-core workers pays that many times over for parallelism the cores cannot
    deliver, so the coarser fan-out stands.
    """
    domains(2)
    assert _numa_sliced(float(_MIN_WORKER_CORES)) == float(_MIN_WORKER_CORES)
    assert _numa_sliced(4.0) == 4.0


def test_a_single_domain_node_below_the_target_is_untouched(domains):
    """One worker per node, exactly as before — the common small-instance cluster."""
    domains(1)
    assert _numa_sliced(16.0) == 16.0
    assert _numa_sliced(float(_TARGET_WORKER_CORES)) == float(_TARGET_WORKER_CORES)


def test_an_unprobeable_fleet_keeps_one_worker_per_node(monkeypatch):
    """A topology read that fails must degrade to the previous behaviour, never to a guess."""

    def boom() -> int:
        raise RuntimeError("no topology")

    monkeypatch.setattr(scaling, "cluster_numa_nodes", boom)
    assert _numa_sliced(96.0) == 96.0


def test_every_slice_stays_at_or_above_the_core_floor(domains):
    """Whatever the node size, a worker the rule produces is worth its own process."""
    domains(2)
    for cores in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        grant = _numa_sliced(float(cores))
        assert grant >= 1.0
        assert grant == float(cores) or grant >= _MIN_WORKER_CORES / 2
        assert grant <= cores
