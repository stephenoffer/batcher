"""Replica placement, straggler detection, backoff, and the preemption monitor's lifecycle.

Resilience code only runs when something has already gone wrong, which is why its bugs
survive: the happy path never reaches them, and the unhappy path is rare enough that a
wrong answer there reads as "the cluster was flaky".
"""

from __future__ import annotations

import threading

import pytest

from batcher.carbonite.resilience.preemption import PreemptionMonitor
from batcher.carbonite.resilience.recovery import _MAX_BACKOFF_S, RecoveryPolicy, ShuffleRecovery
from batcher.carbonite.resilience.replication import assign_replica_hosts
from batcher.carbonite.resilience.speculative import SpeculationPolicy, stragglers_to_backup
from batcher.carbonite.transfer.server import ShuffleTicket

pytestmark = pytest.mark.unit


# --- replica placement -------------------------------------------------------


def test_replicas_of_one_source_land_in_distinct_failure_domains() -> None:
    """Both copies on one node means one node loss still forces a lineage recompute."""
    nodes = ["n1", "n1", "n2", "n2", "n3", "n3"]
    out = assign_replica_hosts({0: 0}, nodes, factor=3)
    hosts = out[0]
    assert len(hosts) == 2
    assert len({nodes[w] for w in hosts}) == 2, "the two replicas share a node"
    assert "n1" not in {nodes[w] for w in hosts}, "a replica must not sit with the primary"


def test_replication_degrades_rather_than_failing_on_a_small_cluster() -> None:
    """Fewer nodes than copies is an optimization limit, not an error."""
    nodes = ["n1", "n2"]
    out = assign_replica_hosts({0: 0}, nodes, factor=4)
    assert out[0] == [1]  # only one other worker exists at all


def test_replica_placement_is_deterministic() -> None:
    nodes = ["n1", "n1", "n2", "n2"]
    primaries = {0: 0, 1: 1, 2: 2, 3: 3}
    assert assign_replica_hosts(primaries, nodes, factor=2) == assign_replica_hosts(
        primaries, nodes, factor=2
    )


def test_dead_workers_never_hold_a_replica() -> None:
    nodes = ["n1", "n2", "n3"]
    out = assign_replica_hosts({0: 0}, nodes, factor=3, dead={1})
    assert 1 not in out[0]


# --- straggler detection -----------------------------------------------------


def _policy(**kw) -> SpeculationPolicy:
    return SpeculationPolicy(max_backups=2, **kw)


def test_a_fast_stage_does_not_speculate_on_millisecond_differences() -> None:
    """Relative-only detection duplicates a task to save 3 ms, which it cannot even save."""
    finished = dict.fromkeys(range(8), 0.005)  # 5 ms median
    elapsed = {8: 0.010}  # 2x the median, and utterly not worth a backup
    assert stragglers_to_backup(9, finished, elapsed, _policy()) == []


def test_a_genuine_straggler_is_still_backed_up() -> None:
    finished = dict.fromkeys(range(8), 4.0)
    elapsed = {8: 20.0}
    assert stragglers_to_backup(9, finished, elapsed, _policy()) == [8]


def test_speculation_waits_for_a_meaningful_median() -> None:
    finished = {0: 4.0}
    elapsed = dict.fromkeys(range(1, 10), 20.0)
    assert stragglers_to_backup(10, finished, elapsed, _policy()) == []


def test_speculation_is_off_by_default() -> None:
    finished = dict.fromkeys(range(8), 4.0)
    assert stragglers_to_backup(9, finished, {8: 100.0}, SpeculationPolicy()) == []


def test_backups_are_capped_and_slowest_first() -> None:
    finished = dict.fromkeys(range(9), 2.0)  # 9 of 12 finished == the 0.75 threshold
    elapsed = {9: 10.0, 10: 30.0, 11: 20.0}
    assert stragglers_to_backup(12, finished, elapsed, _policy()) == [10, 11]


# --- recovery backoff --------------------------------------------------------


def test_the_backoff_is_capped(monkeypatch) -> None:
    """`base * 2**round` is unbounded; a reducer must not out-sleep its own query."""
    slept: list[float] = []
    monkeypatch.setattr("batcher.carbonite.resilience.recovery.time.sleep", slept.append)
    monkeypatch.setattr("batcher.carbonite.resilience.recovery.random.uniform", lambda a, b: b)

    rec = ShuffleRecovery(RecoveryPolicy(max_attempts=12, backoff_base_s=5.0), label="agg")
    with pytest.raises(Exception, match="did not recover"):
        rec.run(attempt=lambda: (None, [1]), recompute=lambda _f: None)

    assert slept, "a policy with a backoff must actually back off"
    assert max(slept) <= _MAX_BACKOFF_S


def test_a_clean_run_never_recomputes_or_sleeps() -> None:
    rec = ShuffleRecovery(RecoveryPolicy(max_attempts=3, backoff_base_s=99.0))
    assert rec.run(attempt=lambda: ("ok", []), recompute=lambda _f: None) == "ok"
    assert rec.recomputes == 0


# --- the preemption monitor --------------------------------------------------


def test_the_monitor_can_be_restarted_after_its_loop_exits() -> None:
    """A loop that ended on its own left `_thread` set, so `start()` became a no-op."""
    mon = PreemptionMonitor(probe=lambda: True, poll_interval_s=0.01)
    mon.start()
    for _ in range(500):
        if mon.is_draining() and mon._thread is None:
            break
        threading.Event().wait(0.005)
    assert mon.is_draining()
    assert mon._thread is None, "the poll thread must release its own slot on exit"

    mon._probe = lambda: False  # so the restarted loop stays alive to be observed
    mon.start()  # must actually start again rather than silently doing nothing
    assert mon._thread is not None
    mon.stop()


def test_a_drain_callback_may_call_stop_without_deadlocking() -> None:
    mon = PreemptionMonitor(probe=lambda: True, poll_interval_s=0.01)
    mon.on_drain(mon.stop)
    mon.start()
    for _ in range(500):
        if mon.is_draining():
            break
        threading.Event().wait(0.005)
    assert mon.is_draining()
    mon.stop()


def test_draining_is_sticky_and_callbacks_run_once() -> None:
    calls: list[int] = []
    mon = PreemptionMonitor(probe=lambda: False)
    mon.on_drain(lambda: calls.append(1))
    mon.trigger()
    mon.trigger()
    assert calls == [1]
    assert mon.is_draining()
    # A callback registered after the fact fires immediately.
    mon.on_drain(lambda: calls.append(2))
    assert calls == [1, 2]


# --- the shuffle ticket ------------------------------------------------------


def test_the_ticket_wire_form_is_built_once_and_stays_out_of_identity() -> None:
    a = ShuffleTicket(1, 2, 3, 4, 5)
    b = ShuffleTicket(1, 2, 3, 4, 5)
    assert str(a) == "1/2/3/4/5"
    assert a == b and hash(a) == hash(b)
    assert "1/2/3/4/5" not in repr(a), "the cached rendering must not leak into repr"
    assert str(a) is str(a), "the wire form is cached, not rebuilt per call"
