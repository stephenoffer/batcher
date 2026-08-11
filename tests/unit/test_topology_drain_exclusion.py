"""Fan-out sizing excludes nodes Ray has already marked for drain.

A draining node is alive, advertises its full resources, and is going away: the autoscaler
is scaling it in, KubeRay is evicting its pod, or a spot reclamation notice reached the node
provider. Ray keeps reporting it schedulable so work already there can finish, but sizing a
*new* fleet onto it reserves bundles on a node being removed, and the shuffle pays a
recompute for output that was never going to survive.

The drain list is the only signal that separates "alive" from "alive and staying", and
unlike a failed fetch it arrives before the loss. These tests pin that it narrows the fleet,
that it never narrows it to nothing, and that it costs one GCS read per scheduling phase
rather than one per worker.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import scaling

# Import the real Ray **before** any test can stub it. Several tests below put a fake into
# `sys.modules` under `ray` or `ray._private.state`; monkeypatch's teardown then *removes*
# the entry rather than restoring one, because there was nothing there to restore. That
# leaves the package half-present, and the next genuine `import ray` re-enters
# `ray/__init__` while `sys.modules["ray"]` already exists — Ray's own initialization then
# fails on itself ("partially initialized module 'ray' has no attribute '_private'").
#
# Importing it here makes every stub below a temporary override of a fully-imported module,
# which is what monkeypatch's restore semantics assume. The symptom without this was a
# class that passed in isolation and failed inside its own file.
ray = pytest.importorskip("ray")

pytestmark = pytest.mark.unit


def _patch_cluster(monkeypatch, specs, draining=(), *, count_reads=None):
    """specs: list of (node_id, cpu, is_head). `draining`: node ids Ray reports draining."""

    class _Ray:
        @staticmethod
        def nodes():
            out = []
            for node_id, cpu, is_head in specs:
                res = {"CPU": cpu, "memory": cpu * 1e9}
                if is_head:
                    res["node:__internal_head__"] = 1.0
                out.append({"Alive": True, "NodeID": node_id, "Resources": res})
            return out

        @staticmethod
        def cluster_resources():
            return {"CPU": sum(c for _, c, _ in specs)}

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)

    def _fake_draining():
        if count_reads is not None:
            count_reads.append(1)
        return frozenset(draining)

    monkeypatch.setattr(scaling, "_read_draining", _fake_draining)
    scaling._TOPOLOGY.set(None)


def test_draining_node_is_not_counted(monkeypatch):
    """The whole point: a fleet sized here must not include capacity being reclaimed."""
    _patch_cluster(
        monkeypatch,
        [("head", 0.0, True), ("a", 16.0, False), ("b", 16.0, False), ("c", 16.0, False)],
        draining=("c",),
    )
    assert scaling.alive_node_count() == 2
    assert scaling.cluster_topology()["nodes"] == 2
    assert scaling.cluster_topology()["cpus"] == 32.0  # c's 16 cores are leaving


def test_no_drain_leaves_the_topology_untouched(monkeypatch):
    """A stable cluster must read exactly as it did before."""
    _patch_cluster(
        monkeypatch,
        [("head", 0.0, True), ("a", 16.0, False), ("b", 16.0, False)],
        draining=(),
    )
    assert scaling.alive_node_count() == 2
    assert scaling.cluster_topology()["cpus"] == 32.0


def test_a_fully_draining_cluster_still_schedules(monkeypatch):
    """Running on capacity that is going away beats not running at all — the recovery path
    exists for exactly that case. Narrowing to zero nodes would hang the fleet spawn."""
    _patch_cluster(
        monkeypatch,
        [("a", 16.0, False), ("b", 16.0, False)],
        draining=("a", "b"),
    )
    assert scaling.alive_node_count() == 2
    assert scaling.cluster_topology()["nodes"] == 2


def test_drain_applies_after_head_exclusion(monkeypatch):
    """Head-only-plus-draining must not collapse to zero: the head still has to run it."""
    _patch_cluster(monkeypatch, [("head", 8.0, True), ("a", 16.0, False)], draining=("a",))
    assert scaling.alive_node_count() == 1
    assert scaling.cluster_topology()["nodes"] == 1


def test_a_broken_drain_accessor_reads_as_no_drain(monkeypatch):
    """A Ray version without the private accessor, or a GCS that will not answer, must
    degrade to "nothing is draining" rather than fail a query over a scheduling refinement.

    Exercises the real `_read_draining` — the guarded boundary — not the stub the other
    tests install.
    """

    class _BrokenState:
        class state:  # mirrors the lowercase ray._private.state.state singleton
            @staticmethod
            def get_draining_nodes():
                raise RuntimeError("gcs unavailable")

    monkeypatch.setitem(__import__("sys").modules, "ray._private.state", _BrokenState)
    assert scaling._read_draining() == frozenset()


class TestSnapshotSharesOneRead:
    """The drain list rides in the topology snapshot, so a W-worker fleet costs one GCS
    round trip rather than W — the same reason the node list is snapshotted."""

    def test_scope_reads_the_drain_list_once(self, monkeypatch):
        reads: list[int] = []
        _patch_cluster(
            monkeypatch,
            [("a", 16.0, False), ("b", 16.0, False), ("c", 16.0, False)],
            draining=("c",),
            count_reads=reads,
        )
        with scaling.topology_scope():
            for _ in range(10):
                scaling.alive_node_count()
                scaling.cluster_topology()
        assert len(reads) == 1, f"expected one drain read per scope, got {len(reads)}"

    def test_the_snapshot_still_excludes_the_draining_node(self, monkeypatch):
        _patch_cluster(
            monkeypatch,
            [("a", 16.0, False), ("b", 16.0, False), ("c", 16.0, False)],
            draining=("c",),
        )
        with scaling.topology_scope():
            assert scaling.alive_node_count() == 2


class TestDrainReadsAreCached:
    """The drain answer is polled from the shuffle barrier twice a second.

    It began as a stage-boundary question where cost did not matter. Uncached, the barrier
    turns it into a GCS round trip twice a second for the barrier's whole duration — and,
    once something *is* draining, `workers` actor RPCs per poll on top, i.e. hundreds a
    second on a large fleet, spent hardest exactly when a node is going away and the cluster
    is already under stress.
    """

    def _count_reads(self, monkeypatch) -> list[int]:
        """Count real GCS calls. Patches the accessor *object*, not `sys.modules`: the
        module under test does `import ray._private.state as ray_state`, which binds the
        parent package's real attribute once `ray._private` is imported, so a sys.modules
        entry is ignored. The module-level `import ray` is what makes that safe here —
        see the note at the top of this file."""
        import ray._private.state as ray_state

        reads: list[int] = []

        def _counting():
            reads.append(1)
            return {}

        monkeypatch.setattr(ray_state.state, "get_draining_nodes", _counting)
        return reads

    def test_the_cluster_read_is_reused_within_the_ttl(self, monkeypatch):
        reads = self._count_reads(monkeypatch)
        scaling._reset_drain_cache()
        for _ in range(20):
            scaling._read_draining()
        assert len(reads) == 1, f"expected one GCS read inside the TTL, got {len(reads)}"

    def test_resetting_forces_a_fresh_read(self, monkeypatch):
        reads = self._count_reads(monkeypatch)
        scaling._reset_drain_cache()
        scaling._read_draining()
        scaling._reset_drain_cache()
        scaling._read_draining()
        assert len(reads) == 2

    def test_worker_node_ids_are_read_once_per_fleet(self, monkeypatch):
        """An actor does not migrate, so its node is immutable data behind a remote call."""
        from batcher.dist.executors.ray_runtime.policies import _drain

        calls: list[int] = []

        class _Handle:
            def __init__(self, node):
                self._node = node
                self.node_id = self

            def remote(self):
                calls.append(1)
                return self._node

        class _Ray:
            @staticmethod
            def get(refs):
                return list(refs)

        monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
        _drain._reset_drain_caches()
        actors = [_Handle("n1"), _Handle("n2")]
        for _ in range(10):
            _drain._worker_node_ids(actors, 2)
        assert len(calls) == 2, "one call per actor, once — not once per poll"

    def test_a_rebuilt_fleet_is_a_cache_miss(self, monkeypatch):
        """Keyed by the actor handles, so a new fleet must not read a previous fleet's nodes."""
        from batcher.dist.executors.ray_runtime.policies import _drain

        class _Handle:
            def __init__(self, node):
                self._node = node
                self.node_id = self

            def remote(self):
                return self._node

        class _Ray:
            @staticmethod
            def get(refs):
                return list(refs)

        monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
        _drain._reset_drain_caches()
        assert _drain._worker_node_ids([_Handle("n1")], 1) == ["n1"]
        assert _drain._worker_node_ids([_Handle("n2")], 1) == ["n2"]
