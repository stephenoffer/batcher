"""A warm fleet must be reusable without pinning the cluster against everything else.

Batcher keeps the shuffle fleet's actors alive between queries because respawning them
dominates a short query. The fleet is a placement group holding close to every schedulable
core, so the same property that makes the *next* Batcher query cheap makes a stage of plain
Ray tasks -- and any other consumer of the cluster -- wait.

Two mechanisms answer that, and these tests pin the part of each that is a decision rather
than a Ray call:

* `_barrier._relieve_stall` hands the idle fleet back when a stalled stage needs its cores,
  and MUST NOT when the stalled work is running on those very actors.
* `bt.release_cluster()` is the explicit, public form of the same hand-back.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.policies._barrier import _relieve_stall

pytestmark = pytest.mark.unit


def test_a_pinned_barrier_never_releases_the_actors_it_is_waiting_on(monkeypatch):
    """The guard that separates a slow query from a failed one.

    A barrier over the fleet's own actors is stalled *on those actors*. Releasing the fleet
    there does not unblock it, it kills the workers mid-stage — so the pinned case must
    refuse before it reaches the yield at all, whatever the yield would have said.
    """
    called: list[float] = []

    def _yield(needed):  # pragma: no cover - must never run in the pinned case
        called.append(needed)
        return True

    monkeypatch.setattr("batcher.dist.fleet.yield_session_fleet", _yield)
    assert _relieve_stall(4.0, pinned=True) is False
    assert called == [], "a pinned barrier must not even ask to yield the fleet"
    # Positive control: the same call unpinned does ask, and reports what the yield said.
    assert _relieve_stall(4.0, pinned=False) is True
    assert called == [4.0]


def test_a_stateless_barrier_reports_whether_the_fleet_was_actually_freed(monkeypatch):
    """`False` must mean "nothing changed", so the caller falls back to reporting the stall."""
    monkeypatch.setattr("batcher.dist.fleet.yield_session_fleet", lambda needed: False)
    assert _relieve_stall(1.0, pinned=False) is False


def test_relief_never_raises_into_the_stage_it_is_trying_to_help(monkeypatch):
    """A stalled stage is already in trouble; the remedy failing must not replace its error."""

    def _boom(needed):
        raise RuntimeError("ray is unreachable")

    monkeypatch.setattr("batcher.dist.fleet.yield_session_fleet", _boom)
    assert _relieve_stall(1.0, pinned=False) is False


def test_release_cluster_is_public_and_safe_with_nothing_warm():
    """The explicit hand-back: importable from the top level and a no-op when idle."""
    import batcher as bt

    assert "release_cluster" in bt.__all__
    bt.release_cluster()  # no fleet, no pools — must be a silent no-op
    bt.release_cluster()  # and idempotent


def test_release_cluster_releases_both_the_fleet_and_the_inference_pools(monkeypatch):
    """Both halves, and a failure in one must not skip the other.

    They are separate warm resources with separate owners, so a single `try` around the pair
    would let a fleet that cannot be reached strand every loaded model behind it.
    """
    import batcher as bt

    freed: list[str] = []

    def _bad_fleet():
        freed.append("fleet-attempted")
        raise RuntimeError("unreachable")

    monkeypatch.setattr("batcher.dist.fleet.release_session_fleet", _bad_fleet)
    monkeypatch.setattr(
        "batcher.dist.executors.map.release_inference_pools", lambda: freed.append("pools")
    )
    bt.release_cluster()
    assert freed == ["fleet-attempted", "pools"]


def test_a_map_task_that_lost_a_shuffle_peer_is_retried_not_re_raised():
    """The map barrier must treat the transport's own "retry me" as retryable.

    `_faults` documents the assumption this violates: "a map task that fails reports worker
    loss as a *Ray* error". That held until a map task could read a **Flight intermediate** --
    a stage scanning what a previous stage published fetches from a peer inside the task, so a
    lost peer arrives as a `RetryableShuffleError` wrapped in a `RayTaskError`, which the
    barrier re-raised. Seen as a windowed rank over a hot key dying with `transport error`
    while the same query on a uniform key passed.
    """
    from batcher._internal.errors import FatalShuffleError, RetryableShuffleError
    from batcher.dist.executors.ray_runtime.policies._faults import is_recoverable_task_failure

    assert is_recoverable_task_failure(RetryableShuffleError("transport error"))
    # And the discrimination that makes it safe: a deterministic bug must still surface at
    # once rather than burn the recovery budget and report a resource error for a code error.
    assert not is_recoverable_task_failure(FatalShuffleError("schema mismatch"))
    assert not is_recoverable_task_failure(ZeroDivisionError("in a udf"))


def test_the_barrier_consults_that_classifier_on_its_task_error_branch():
    """A guard on the wiring, not the classifier: the fix is one `or` in one branch."""
    import inspect

    from batcher.dist.executors.ray_runtime.policies import _barrier

    src = inspect.getsource(_barrier.gather_map_results)
    assert "is_recoverable_task_failure(exc)" in src, (
        "the RayTaskError branch must consult the shuffle-fault classifier, or a lost peer "
        "during a map task's Flight read fails the whole query"
    )
