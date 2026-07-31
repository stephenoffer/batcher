"""What the fault ledger changes about *where* work and copies are placed.

Learning that a node is bad is only half of it. The other half is that every placement
decision has to act on what was learned, and two of them are easy to miss because they are
not about running a task at all:

* A **replica** exists to die independently of its primary. Putting the only spare copy of a
  shuffle output on a worker that has been failing every task defeats the entire point — that
  copy is the one least likely to be there when it is needed.
* A **drain** is normally about a node that is leaving. A quarantined worker is not leaving;
  it is staying, and failing. Its shuffle output sits on a host whose next fetch is the one
  most likely to fail, and migrating it at a stage boundary costs one copy against a full
  recovery round.

Both deprioritize rather than exclude, because a fleet where most workers are suspect must
still be able to place something. No replica at all is worse than a replica on a shaky host.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience.replication import assign_replica_hosts

pytestmark = pytest.mark.unit

#: Four workers, one per node, so every replica choice is free of the off-node constraint and
#: the only thing deciding it is load — which is where the suspect ranking lives.
_NODES = ["n0", "n1", "n2", "n3"]


def test_a_replica_avoids_a_worker_the_ledger_has_condemned():
    plain = assign_replica_hosts({0: 0}, _NODES, factor=2)
    assert plain[0] == [1]
    avoided = assign_replica_hosts({0: 0}, _NODES, factor=2, suspect=frozenset({1}))
    assert avoided[0] and avoided[0] != [1]


def test_a_suspect_worker_is_still_used_when_nothing_else_is_left():
    # A fleet where every survivor is suspect is exactly the fleet where a replica matters
    # most. Excluding them outright would place no copy at all.
    every = frozenset({0, 1, 2, 3})
    out = assign_replica_hosts({0: 0}, _NODES, factor=2, suspect=every)
    assert len(out[0]) == 1


def test_the_dead_are_still_excluded_outright():
    # Deprioritizing is for a host that is failing; a host that is gone cannot hold anything.
    out = assign_replica_hosts({0: 0}, _NODES, factor=2, dead=frozenset({1, 2}))
    assert out[0] == [3]


def test_suspicion_does_not_disturb_a_clean_placement():
    # The pre-existing behavior has to survive the addition, not be replaced by it.
    assert assign_replica_hosts({0: 0, 1: 1}, ["a", "a", "b", "b"], factor=2) == {0: [2], 1: [3]}


def test_a_quarantined_worker_is_migrated_off_at_a_stage_boundary(monkeypatch):
    from batcher.carbonite.resilience import FaultLedger, QuarantinePolicy
    from batcher.dist.executors.ray_runtime.policies import _drain, _faults

    ledger = FaultLedger(QuarantinePolicy(failure_threshold=1.0, min_targets=1), label="worker")
    ledger.observe([str(i) for i in range(4)])
    ledger.record_failure("2", "storage")
    monkeypatch.setattr(_faults, "node_ledger", lambda: ledger)
    monkeypatch.setattr(_drain, "_nodes_draining", lambda actors, workers: set())
    _drain._reset_drain_caches()

    assert _drain.draining_workers([object()] * 4, 4) == {2}


def test_a_ledger_key_outside_the_fleet_never_migrates_a_worker(monkeypatch):
    # The ledger is keyed by string and outlives a fleet. A stale key must not be read as an
    # index into the current one, or an unrelated worker gets migrated for someone else's sins.
    from batcher.carbonite.resilience import FaultLedger, QuarantinePolicy
    from batcher.dist.executors.ray_runtime.policies import _drain, _faults

    ledger = FaultLedger(QuarantinePolicy(failure_threshold=1.0, min_targets=1), label="worker")
    ledger.observe([str(i) for i in range(16)])
    ledger.record_failure("11", "storage")
    monkeypatch.setattr(_faults, "node_ledger", lambda: ledger)
    monkeypatch.setattr(_drain, "_nodes_draining", lambda actors, workers: set())
    _drain._reset_drain_caches()

    assert _drain.draining_workers([object()] * 4, 4) == set()


def test_a_ledger_that_raises_never_fails_a_query(monkeypatch):
    from batcher.dist.executors.ray_runtime.policies import _drain, _faults

    def _boom():
        raise RuntimeError("the ledger is broken")

    monkeypatch.setattr(_faults, "node_ledger", _boom)
    monkeypatch.setattr(_drain, "_nodes_draining", lambda actors, workers: set())
    _drain._reset_drain_caches()

    # Proactive migration is an optimization over the reactive recompute path, so a broken
    # read has to degrade to that path rather than take the query with it.
    assert _drain.draining_workers([object()] * 4, 4) == set()
