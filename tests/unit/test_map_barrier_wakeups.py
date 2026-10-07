"""The map barrier's driver loop: what one wakeup costs, and when a stall stops waiting.

Both properties are about the *driver*, which no result can show: a barrier that walks its
whole in-flight list once per completion and one that walks it once per wakeup return the
same rows, and a barrier that waits forever on a task no node can host returns nothing at
all. So these count `ray.wait` calls against a fake Ray instead of comparing results.
"""

from __future__ import annotations

import sys

import pytest

from _fake_ray import install_fake_ray
from batcher.carbonite.resilience import RecoveryPolicy

pytestmark = pytest.mark.unit


def _counting_wait(monkeypatch, *, ready: bool, limit: int = 10_000) -> list[tuple[int, int]]:
    """Replace the fake `ray.wait` with one that records `(len(refs), num_returns)`.

    `ready=True` models a stage whose tasks have all finished by the time the driver looks;
    `ready=False` one where nothing ever finishes. `limit` turns a barrier that would poll
    forever into a test failure instead of a hung suite.
    """
    calls: list[tuple[int, int]] = []

    def wait(refs, num_returns=1, timeout=None):
        calls.append((len(refs), num_returns))
        if len(calls) > limit:
            raise AssertionError(f"ray.wait called {len(calls)} times: the barrier never left")
        if not ready:
            return [], list(refs)
        return list(refs[:num_returns]), list(refs[num_returns:])

    monkeypatch.setattr(sys.modules["ray"], "wait", wait)
    return calls


def test_one_wakeup_drains_every_ready_partition(monkeypatch):
    """BT-074: a wakeup takes every finished task, not one, so the loop is not O(n x window).

    Before the fix the barrier asked for one ref per wait and passed the whole in-flight list
    each time — 2,000 waits over a shrinking list of up to 2,000 refs here, two million ref
    marshals on the driver for a stage that had already finished.
    """
    from batcher.dist.executors.ray_runtime import gather_map_results

    install_fake_ray(monkeypatch)
    calls = _counting_wait(monkeypatch, ready=True)
    n = 2_000

    out = gather_map_results(
        lambda i: lambda i=i: i, n, RecoveryPolicy(max_attempts=1), max_pending=n
    )

    assert out == list(range(n))  # index-addressed: batching never reorders the result
    # One blocking wait for the first completion, one non-blocking drain for the rest.
    assert len(calls) == 2, calls[:5]
    assert calls[1] == (n - 1, n - 1)


def test_a_retry_inside_a_drained_batch_is_still_resubmitted(monkeypatch):
    """Batching the wakeup must keep the per-partition recovery the one-at-a-time loop had."""
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = install_fake_ray(monkeypatch)
    _counting_wait(monkeypatch, ready=True)
    seen: dict[int, int] = {}

    def submit(idx):
        seen[idx] = seen.get(idx, 0) + 1
        if idx == 3 and seen[idx] == 1:

            def lost():
                raise RayError("preempted")

            return lost
        return lambda idx=idx: idx

    out = gather_map_results(submit, 8, RecoveryPolicy(max_attempts=2))

    assert out == list(range(8))
    assert seen[3] == 2


def _stalled_barrier(monkeypatch, diagnosis: str, autoscale_wait_s: float):
    """A barrier whose tasks never finish, with `diagnosis` as the cluster's verdict on why.

    The stall-report cadence is set to zero so the first empty wait already counts as a
    stall; a real barrier reaches the same branch after two minutes.
    """
    import dataclasses

    from batcher.config import Config, config_context
    from batcher.dist.executors.ray_runtime import gather_map_results
    from batcher.dist.executors.ray_runtime.policies import _barrier

    install_fake_ray(monkeypatch)
    cancelled: list = []
    monkeypatch.setattr(sys.modules["ray"], "cancel", cancelled.append, raising=False)
    calls = _counting_wait(monkeypatch, ready=False, limit=50)
    monkeypatch.setattr("batcher.carbonite.resilience.STALL_WARN_AFTER_S", 0.0)
    monkeypatch.setattr("batcher.carbonite.resilience.warn_barrier_stalled", lambda *a: None)
    monkeypatch.setattr(_barrier, "_stall_diagnosis", lambda *_a: diagnosis)
    monkeypatch.setattr(_barrier, "_relieve_stall", lambda *_a, **_k: False)
    base = Config()
    cfg = base.replace(
        distributed=dataclasses.replace(base.distributed, autoscale_wait_s=autoscale_wait_s)
    )

    def run():
        with config_context(cfg):
            return gather_map_results(lambda i: object(), 4, RecoveryPolicy(max_attempts=1))

    return run, calls, cancelled


_UNSATISFIABLE = (
    "no node advertises accelerator_type:TPU: this stage asks for 1 TPU per task and the "
    "cluster has none of that resource, so waiting cannot schedule it"
)


def test_a_stage_no_node_can_host_fails_instead_of_polling_forever(monkeypatch):
    """BT-068: the barrier logged 'waiting cannot schedule it' every two minutes, forever."""
    from batcher._internal.errors import ResourceError

    run, calls, cancelled = _stalled_barrier(monkeypatch, _UNSATISFIABLE, autoscale_wait_s=0.0)

    with pytest.raises(ResourceError, match="no node advertises"):
        run()
    assert len(calls) == 1  # failed on the first stall report, with no autoscaler to wait on
    assert len(cancelled) == 4  # and took its never-placeable tasks with it


def test_an_unsatisfiable_stall_waits_out_the_autoscale_budget(monkeypatch):
    """An autoscaler may yet add a node of the missing shape, so the budget is honored."""
    run, _, cancelled = _stalled_barrier(monkeypatch, _UNSATISFIABLE, autoscale_wait_s=3600.0)

    with pytest.raises(AssertionError, match="never left"):
        run()
    assert not cancelled


def test_a_merely_full_cluster_is_waited_on(monkeypatch):
    """A cluster short of *free* capacity clears when other work finishes; never fail it."""
    run, _, cancelled = _stalled_barrier(
        monkeypatch,
        "the cluster is short of free capacity: 4 outstanding at 1 CPU each",
        autoscale_wait_s=0.0,
    )

    with pytest.raises(AssertionError, match="never left"):
        run()
    assert not cancelled
