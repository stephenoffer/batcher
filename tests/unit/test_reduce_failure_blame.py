"""`blame_host_for_reduce_failure` — a bug in a reducer must not condemn the fleet.

The bucket-reduce barrier gathers through `gather_with_backups`, whose `on_failure` hook
turns a failed slot into a result rather than raising. That is what lets a preempted host
become a recompute instead of a dead query — but it sees *every* exception, and only some
of them mean a host was lost.

The join, sort, and window shuffles blamed the host unconditionally. A deterministic bug in
the reducer therefore failed on worker 0, got worker 0 marked dead, recomputed onto worker 1,
failed identically, and so on until the fleet was gone and the query raised
`ResourceError("shuffle did not recover after 3 attempts (still unreachable: {0, 1, 2, 3})")`
on four healthy workers, with the real traceback three frames down and discarded. Five of the
twenty-two TPC-H queries failed exactly that way.

These pin both directions: a transport-classified loss still yields a host to recompute on
(or recovery stops working), and anything deterministic propagates untouched (or the bug
goes back to wearing a resource error as a disguise).
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import (
    FatalShuffleError,
    PlanError,
    ResourceError,
    RetryableShuffleError,
)
from batcher.dist.executors.ray_runtime import blame_host_for_reduce_failure

pytestmark = pytest.mark.unit


def test_an_unreachable_peer_blames_its_host() -> None:
    assert blame_host_for_reduce_failure(RetryableShuffleError("peer gone"), 3) == 3


def test_a_vanished_spill_file_blames_its_host() -> None:
    """An ephemeral disk reclaimed under a spill file is lost data, so recompute it."""
    assert blame_host_for_reduce_failure(ResourceError("spill file vanished"), 0) == 0


def test_a_dead_actor_blames_its_host() -> None:
    """The loss this path exists to survive. It arrives as a bare Ray error rather than as
    anything the transport classified, so a rule written only around `RetryableShuffleError`
    re-raises it and turns a recoverable preemption into a failed query."""
    ray_exc = pytest.importorskip("ray.exceptions")
    assert blame_host_for_reduce_failure(ray_exc.RayActorError("worker preempted"), 2) == 2


def test_a_cancelled_task_is_not_a_death() -> None:
    """A speculation loser we cancelled ourselves must not condemn the host that ran it."""
    ray_exc = pytest.importorskip("ray.exceptions")
    with pytest.raises(ray_exc.TaskCancelledError):
        blame_host_for_reduce_failure(ray_exc.TaskCancelledError(), 1)


def test_an_unknown_host_stays_unknown() -> None:
    """`None` means the barrier could not attribute the loss; that is not a worker id."""
    assert blame_host_for_reduce_failure(RetryableShuffleError("peer gone"), None) is None


def test_a_type_error_in_the_reducer_propagates() -> None:
    """The regression: the spilling join's `TypeError` reached the driver as four dead
    workers, because it was charged to a host instead of being raised."""
    boom = TypeError("Schema must be an instance of pyarrow.Schema")
    with pytest.raises(TypeError, match=r"pyarrow\.Schema"):
        blame_host_for_reduce_failure(boom, 2)


def test_a_plan_error_propagates() -> None:
    with pytest.raises(PlanError):
        blame_host_for_reduce_failure(PlanError("no such column"), 1)


def test_a_fatal_shuffle_error_propagates() -> None:
    """The transport said retrying cannot help; blaming a host would override it."""
    with pytest.raises(FatalShuffleError):
        blame_host_for_reduce_failure(FatalShuffleError("schema mismatch"), 1)
