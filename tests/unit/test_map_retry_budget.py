"""The map barrier under a job-wide retry budget and a cross-stage fault ledger.

Two gaps these close, both of which only appear at fleet scale.

The per-partition attempt limit bounds a partition and bounds nothing about a stage. At a
hundred thousand partitions, `max_attempts=3` authorizes three hundred thousand retries, and a
fleet broken in a way no probe catches — a bad image, an expired credential, a driver that
stopped matching its runtime — will use every one of them. The run then takes hours at a
fraction of its rate and fails with whatever error happened to be last, long after the first
one said exactly what was wrong.

And the barrier's `dead` set is per-stage, so a worker that failed every source of the last
shuffle is rediscovered from scratch at the next one, at the cost of one attempt per barrier
for the whole query. The ledger is what remembers.
"""

from __future__ import annotations

import collections

import pytest

from _fake_ray import install_fake_ray
from batcher.carbonite.resilience import FaultLedger, QuarantinePolicy, RecoveryPolicy, RetryBudget

pytestmark = pytest.mark.unit


def _raise(exc: BaseException):
    raise exc


def test_the_budget_stops_a_stage_from_spending_itself_on_retries(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = install_fake_ray(monkeypatch)
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        return lambda: _raise(RayError("this whole fleet is broken"))

    # Generous per-partition attempts, a budget of two. Without the budget this would be
    # 4 partitions x 20 attempts before anything surfaced.
    budget = RetryBudget(fraction=0.0, floor=2)
    with pytest.raises(RayError):
        gather_map_results(submit, 4, RecoveryPolicy(max_attempts=20), budget=budget)
    assert budget.state().remaining == 0
    # Four initial submissions plus exactly the two retries the budget paid for.
    assert sum(calls.values()) <= 6


def test_a_healthy_stage_never_touches_the_budget(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    install_fake_ray(monkeypatch)
    budget = RetryBudget(fraction=0.1, floor=8)

    def submit(idx):
        return lambda i=idx: [f"r{i}"]

    assert gather_map_results(submit, 5, budget=budget) == [[f"r{i}"] for i in range(5)]
    assert budget.state().spent == 0


def test_one_transient_failure_still_retries_within_the_budget(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = install_fake_ray(monkeypatch)
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        if idx == 1 and calls[idx] == 1:
            return lambda: _raise(RayError("preempted"))
        return lambda i=idx: [f"r{i}"]

    budget = RetryBudget(fraction=0.5, floor=1)
    out = gather_map_results(submit, 3, RecoveryPolicy(max_attempts=3), budget=budget)
    assert out == [["r0"], ["r1"], ["r2"]]
    assert budget.state().spent == 1


def test_a_corrupting_device_fault_is_never_retried_past(monkeypatch):
    from batcher._internal.errors import ExecutionError
    from batcher.dist.executors.ray_runtime import gather_map_results

    _, RayTaskError = install_fake_ray(monkeypatch)
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        return lambda: _raise(RayTaskError("NVRM: Xid 95: uncontained ECC error on GPU 3"))

    # The failure classifies as retryable — it is a hardware condition, not a bug — and that
    # is exactly the trap. Retrying would finish the job and write out numbers the device
    # already returned wrong, which is worse than the crash it avoided.
    with pytest.raises(ExecutionError, match="cannot be trusted"):
        gather_map_results(submit, 2, RecoveryPolicy(max_attempts=10))
    assert max(calls.values()) == 1


def test_a_deployment_may_opt_out_of_the_untrusted_results_guard(monkeypatch):
    from batcher.config import Config, FaultToleranceConfig, config_context
    from batcher.dist.executors.ray_runtime import gather_map_results

    _, RayTaskError = install_fake_ray(monkeypatch)
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        if calls[idx] == 1:
            return lambda: _raise(RayTaskError("Xid 95: uncontained ECC error"))
        return lambda i=idx: [f"r{i}"]

    relaxed = Config().replace(
        fault_tolerance=FaultToleranceConfig(fail_on_untrusted_results=False)
    )
    with config_context(relaxed):
        assert gather_map_results(submit, 1, RecoveryPolicy(max_attempts=3)) == [["r0"]]


def test_the_barrier_avoids_a_worker_the_ledger_has_condemned(monkeypatch):
    from batcher.dist.executors.ray_runtime.policies import _barrier

    RayError, _ = install_fake_ray(monkeypatch)
    ledger = FaultLedger(QuarantinePolicy(failure_threshold=1.0, min_targets=1), label="worker")
    ledger.observe([str(i) for i in range(4)])
    ledger.record_failure("2", "storage")  # condemned by an earlier stage
    monkeypatch.setattr(_barrier, "node_ledger", lambda: ledger)

    hosts: list[int] = []

    def launch(host, src):
        hosts.append(host)
        if host == 0 and src == 0:
            return lambda: _raise(RayError("worker 0 is gone"))
        return lambda h=host: f"addr-{h}"

    results, dead = _barrier.map_barrier(4, launch, RecoveryPolicy(max_attempts=3))
    assert 0 in dead
    assert len(results) == 4
    # The first four launches are the initial one-source-per-worker pass; anything after is a
    # relocation. Source 0 had to move, and it must not have moved onto worker 2 — which the
    # previous stage already learned was bad, and which the rotation would otherwise have
    # picked, since 2 sorts between the other two survivors.
    relocations = hosts[4:]
    assert relocations, "source 0 should have been relocated after its host died"
    assert 2 not in relocations


def test_a_completed_source_clears_its_workers_record(monkeypatch):
    from batcher.dist.executors.ray_runtime.policies import _barrier

    install_fake_ray(monkeypatch)
    ledger = FaultLedger(QuarantinePolicy(failure_threshold=2.0, min_targets=1), label="worker")
    monkeypatch.setattr(_barrier, "node_ledger", lambda: ledger)

    _barrier.map_barrier(3, lambda host, src: lambda h=host: f"addr-{h}")
    # Successes are the only evidence that clears a quarantine. A ledger that records only
    # failures describes a fleet that can only ever shrink.
    assert all(ledger.health(str(i)).successes == 1 for i in range(3))
    assert ledger.blocked_keys() == ()
