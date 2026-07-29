"""Fan-out is sized to the cores the machine delivers, not the cores it permits.

`available_cpu_count()` answers what the cgroup allows. Nothing answered what the scheduler
actually hands over, and the two diverge whenever a node is shared: two Ray workers on one
box, a co-tenant taking half the machine, a CFS quota that keeps binding. Fanning out to the
permitted count then adds context switches and cache pressure without adding throughput, so
the query gets slower the harder the engine tries — and reads as "the engine is slow here".

Two properties are load-bearing and tested below. The policy must only ever *reduce*, so a
quiet machine is byte-for-byte unaffected. And it must not be able to serialize a query on a
bad reading, because these signals are noisy by nature.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import cpu as hw_cpu
from batcher.carbonite.policies import cpu_budget

pytestmark = pytest.mark.unit


@pytest.fixture
def machine(monkeypatch):
    """Pin the permitted core count and the measured contention for a test."""

    def configure(*, permitted: int, oversubscription: float):
        monkeypatch.setattr(cpu_budget, "available_cpu_count", lambda: permitted)
        monkeypatch.setattr(cpu_budget, "cpu_oversubscription", lambda: oversubscription)

    return configure


def test_a_quiet_machine_keeps_every_permitted_core(machine):
    # The common case must be untouched: no contention, no reduction, no behavior change.
    machine(permitted=16, oversubscription=1.0)
    assert cpu_budget.effective_core_budget() == 16
    assert cpu_budget.oversubscription_note() == ""


def test_mild_contention_is_inside_the_deadband(machine):
    # An idle box still reports a small load average and occasional PSI stalls. Acting on
    # those would make fan-out jitter query to query for no reason.
    machine(permitted=16, oversubscription=1.2)
    assert cpu_budget.effective_core_budget() == 16
    assert cpu_budget.oversubscription_note() == ""


def test_real_contention_cuts_the_budget(machine):
    # Two workers sharing a 16-core node: each measures 2x oversubscription and should ask
    # for 8, which is what it will actually get. Asking for 16 makes both run slower.
    machine(permitted=16, oversubscription=2.0)
    assert cpu_budget.effective_core_budget() == 8
    note = cpu_budget.oversubscription_note()
    assert "16 -> 8" in note and "2.0x" in note, note


def test_a_pathological_reading_cannot_serialize_a_query(machine):
    # A load spike, a PSI file reporting a neighbour's stall, or a load average carrying a
    # finished job's tail must not be able to collapse the engine to one thread.
    machine(permitted=16, oversubscription=1000.0)
    budget = cpu_budget.effective_core_budget()
    assert budget == int(16 * cpu_budget.MIN_BUDGET_FRACTION) == 4
    assert budget >= 1


def test_the_budget_never_exceeds_what_is_permitted(machine):
    # Only ever a reduction. An engine that fanned out *past* its cgroup quota would be
    # throttled by the kernel — the exact failure this policy exists to avoid.
    for pressure in (0.1, 0.5, 1.0, 1.25, 3.0, 50.0):
        machine(permitted=8, oversubscription=pressure)
        assert 1 <= cpu_budget.effective_core_budget() <= 8


def test_an_explicit_setting_is_an_instruction_not_an_estimate(machine):
    # `execution.parallelism` is a user decision. Silently overriding it under load would make
    # the knob a lie, and a user who set it has context this process does not.
    machine(permitted=16, oversubscription=8.0)
    assert cpu_budget.effective_core_budget(configured=12) == 12


def test_the_manager_leaves_a_quiet_machine_alone(monkeypatch):
    # `recommended_config` returning a config on every query would defeat its own fast path
    # and reship an engine config for no reason.
    from batcher.carbonite import manager as mgr

    monkeypatch.setattr(mgr, "effective_core_budget", lambda: mgr.available_cpu_count())
    assert mgr.ResourceManager().recommend_parallelism() is None


def test_the_manager_passes_a_reduced_budget_to_the_engine(monkeypatch):
    # The whole point: the reduction has to reach `execution.parallelism`, which is what the
    # engine sizes its rayon pool from. A policy nobody consults changes nothing.
    from batcher.carbonite import manager as mgr

    monkeypatch.setattr(mgr, "effective_core_budget", lambda: 3)
    monkeypatch.setattr(mgr, "available_cpu_count", lambda: 16)
    manager = mgr.ResourceManager()
    assert manager.recommend_parallelism() == 3
    recommended = manager.recommended_config()
    assert recommended is not None
    assert recommended.execution.parallelism == 3


def test_oversubscription_folds_queueing_and_stalling(monkeypatch):
    # The signal behind the policy. Both terms matter: a long run queue says work is waiting,
    # and a stall share says wall time bought no progress. Either alone under-reports.
    monkeypatch.setattr(hw_cpu, "cpu_contention", lambda: {"load_per_core": 3.0})
    assert hw_cpu.cpu_oversubscription() == pytest.approx(3.0)
    monkeypatch.setattr(hw_cpu, "cpu_contention", lambda: {"throttled_ratio": 0.5})
    assert hw_cpu.cpu_oversubscription() == pytest.approx(2.0)
    monkeypatch.setattr(
        hw_cpu, "cpu_contention", lambda: {"load_per_core": 3.0, "throttled_ratio": 0.5}
    )
    assert hw_cpu.cpu_oversubscription() == pytest.approx(6.0)


def test_a_parallelism_only_recommendation_preserves_the_morsel_target(monkeypatch):
    # `recommended_config` used to fire on one lever (the morsel target) and now folds in a
    # second (fan-out). The merge is hand-rolled, so this pins the case that a hand-rolled
    # merge gets wrong: changing only the new lever must leave the old one exactly as
    # configured, not reset it to a default the engine would then run with.
    from batcher.carbonite import manager as mgr

    monkeypatch.setattr(mgr, "effective_core_budget", lambda: 3)
    monkeypatch.setattr(mgr, "available_cpu_count", lambda: 16)
    manager = mgr.ResourceManager()
    monkeypatch.setattr(manager, "recommend_morsel_target", lambda families=None: None)

    recommended = manager.recommended_config()
    assert recommended is not None
    configured = mgr.active_config().execution
    assert recommended.execution.parallelism == 3
    assert recommended.execution.morsel_rows == configured.morsel_rows
    assert recommended.execution.morsel_bytes == configured.morsel_bytes


def test_both_levers_apply_together(monkeypatch):
    # And when both move, neither silently wins: a pressured, contended machine needs the
    # tighter morsel *and* the narrower fan-out.
    from batcher.carbonite import manager as mgr

    monkeypatch.setattr(mgr, "effective_core_budget", lambda: 4)
    monkeypatch.setattr(mgr, "available_cpu_count", lambda: 32)
    manager = mgr.ResourceManager()
    monkeypatch.setattr(manager, "recommend_morsel_target", lambda families=None: (2048, 65536))

    recommended = manager.recommended_config()
    assert recommended is not None
    assert recommended.execution.parallelism == 4
    assert recommended.execution.morsel_rows == 2048
    assert recommended.execution.morsel_bytes == 65536


def test_an_unpressured_uncontended_machine_still_gets_no_config(monkeypatch):
    # The fast path must survive gaining a second lever: neither moving means no config is
    # built and no engine config is reshipped.
    from batcher.carbonite import manager as mgr

    monkeypatch.setattr(mgr, "effective_core_budget", lambda: mgr.available_cpu_count())
    manager = mgr.ResourceManager()
    monkeypatch.setattr(manager, "recommend_morsel_target", lambda families=None: None)
    assert manager.recommended_config() is None
