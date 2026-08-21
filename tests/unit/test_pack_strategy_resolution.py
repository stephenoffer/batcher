"""The PACK preference is reconciled with the cluster, not just honored.

Carbonite decides PACK against `cpu_budget`, which is the *driver's* core count because
Carbonite has no live topology. On a cluster whose nodes are smaller than the driver that
is a request to co-locate a gang no node can hold. Ray's PACK is best-effort so it does not
hang, but it spends the attempt and then lands the fleet unevenly, piling bundles onto
whichever nodes fit until they do not.

STRICT_PACK is deliberately not downgraded: it is asked for only by a GPU collective, whose
actors must be co-located to run their NCCL ring at all, so spreading it silently would
produce a fleet that cannot do its job.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import scaling
from batcher.dist.executors.ray_runtime.scheduling import _resolve_placement_strategy
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit

_GIB = 1024**3


def _nodes(monkeypatch, specs, alive=None):
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
    monkeypatch.setattr(scaling, "alive_node_count", lambda: alive or len(classes))
    # The SPREAD-to-PACK degrade asks how many machines exist, head included — a placement
    # bundle carries no head-excluding resource, so the head is one of them. `node_classes`
    # here describes worker-eligible nodes, so the cluster is one larger unless the caller
    # pins `alive` to say otherwise.
    monkeypatch.setattr(scaling, "cluster_node_count", lambda: alive or len(classes))


def test_pack_survives_when_a_node_can_hold_the_gang(monkeypatch):
    _nodes(monkeypatch, [(64, 0, 256 * _GIB), (16, 0, 64 * _GIB)])
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="PACK")
    assert _resolve_placement_strategy(env) == "PACK"


def test_pack_downgrades_when_no_node_can_hold_the_gang(monkeypatch):
    """Eight 4-core workers need 32 cores on one node; the widest here has 16."""
    _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 4)
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="PACK")
    assert _resolve_placement_strategy(env) == "SPREAD"


def test_memory_can_be_what_stops_the_pack(monkeypatch):
    """Cores fit but RAM does not, which is the common shape for a shuffle grant."""
    _nodes(monkeypatch, [(64, 0, 16 * _GIB)])
    env = SchedulingEnvelope(
        num_cpus=4.0, n_tasks=8, memory_bytes=8 * _GIB, placement_strategy="PACK"
    )
    assert _resolve_placement_strategy(env) == "SPREAD"


def test_strict_pack_is_never_downgraded(monkeypatch):
    """A GPU collective spread across nodes cannot run its ring, so silently spreading it
    would be worse than the placement failing loudly."""
    _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 4)
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="STRICT_PACK")
    assert _resolve_placement_strategy(env) == "STRICT_PACK"


def test_gpu_collective_still_forces_strict_pack(monkeypatch):
    _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 4)
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, gpu_collective=True)
    assert _resolve_placement_strategy(env) == "STRICT_PACK"


def test_spread_still_degrades_to_pack_on_one_node(monkeypatch):
    """The pre-existing direction must keep working."""
    _nodes(monkeypatch, [(64, 0, 256 * _GIB)], alive=1)
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="SPREAD")
    assert _resolve_placement_strategy(env) == "PACK"


def test_spread_stays_spread_on_a_real_cluster(monkeypatch):
    _nodes(monkeypatch, [(64, 0, 256 * _GIB)] * 3)
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="SPREAD")
    assert _resolve_placement_strategy(env) == "SPREAD"


def test_unreadable_topology_keeps_the_preference(monkeypatch):
    """Second-guessing a preference on no evidence is worse than honoring it."""
    monkeypatch.setattr(scaling, "node_classes", lambda: [])
    monkeypatch.setattr(scaling, "alive_node_count", lambda: 0)
    monkeypatch.setattr(scaling, "cluster_node_count", lambda: 0)
    env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="PACK")
    assert _resolve_placement_strategy(env) == "PACK"


def test_no_envelope_defaults_to_spread(monkeypatch):
    _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 4)
    assert _resolve_placement_strategy(None) == "SPREAD"


class TestGangSizeComesFromTheBundleCount:
    """The decision is about the gang actually being reserved, not the envelope's `n_tasks`.

    `create_worker_placement(workers, env)` builds `workers` bundles, and the fleet path
    spawns a worker count of its own — a reused warm fleet, or one `clamp_workers` reduced —
    against whatever envelope is ambient. Reading the gang size off the envelope therefore
    tests a fleet that is not the one being placed, and gets it wrong in both directions.
    """

    def test_a_stale_larger_n_tasks_does_not_spread_a_gang_that_fits(self, monkeypatch):
        _nodes(monkeypatch, [(64, 0, 256 * _GIB)])
        # The envelope still says 64 tasks; only 4 bundles are actually being reserved.
        env = SchedulingEnvelope(num_cpus=4.0, n_tasks=64, placement_strategy="PACK")
        assert _resolve_placement_strategy(env, workers=4) == "PACK"

    def test_a_stale_smaller_n_tasks_does_not_pack_a_gang_that_will_not_fit(self, monkeypatch):
        _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 4)
        # The envelope says 2 tasks; 8 bundles are actually being reserved, needing 32 cores.
        env = SchedulingEnvelope(num_cpus=4.0, n_tasks=2, placement_strategy="PACK")
        assert _resolve_placement_strategy(env, workers=8) == "SPREAD"

    def test_it_falls_back_to_n_tasks_when_the_caller_does_not_know(self, monkeypatch):
        _nodes(monkeypatch, [(16, 0, 64 * _GIB)] * 4)
        env = SchedulingEnvelope(num_cpus=4.0, n_tasks=8, placement_strategy="PACK")
        assert _resolve_placement_strategy(env) == "SPREAD"
