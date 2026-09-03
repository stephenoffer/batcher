"""A mapper reports the bytes it published across the whole map phase, not its last call.

`published_bucket_bytes` is what `flight_aggregate` sums to decide where each reducer should
run: a bucket concentrated on one node wants its reducer there, so the fetch is local. The
figure has to describe the worker, and a worker maps more than one partition whenever the
shuffle asks for more of them than there are workers (`map_partitions`), which is the
ordinary case rather than the exception.

It used to be overwritten by every `map_publish`, so the placement was decided on one
partition's bytes. Accumulating also makes it safe under `FLEET_CONCURRENCY`, where two map
calls run in different threads of the same actor and a read-modify-write of a shared dict is
not atomic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ray", reason="the fleet actor lives behind the optional ray extra")

from batcher.dist.flight_worker import _FlightWorker

pytestmark = pytest.mark.unit

_CLASS = _FlightWorker.__ray_metadata__.modified_class


class _Worker:
    """The accumulator half of `_FlightWorker`, without spawning an actor.

    Only the three attributes `_record_bucket_bytes` touches are set up, so this exercises
    the real methods rather than a re-implementation of them.
    """

    _record_bucket_bytes = _CLASS._record_bucket_bytes
    published_bucket_bytes = _CLASS.published_bucket_bytes

    def __init__(self) -> None:
        import threading

        self._bucket_bytes: dict[int, int] = {}
        self._bucket_bytes_plan: int | None = None
        self._bucket_lock = threading.Lock()


@pytest.fixture
def plan_id(monkeypatch):
    """Pin the ambient shuffle plan id the accumulator keys on."""
    import batcher.dist.flight_worker as fw

    def _set(value: int) -> None:
        monkeypatch.setattr(fw, "current_plan_id", lambda: value)

    _set(1)
    return _set


def test_several_map_calls_of_one_plan_are_summed(plan_id) -> None:
    """The whole point: one worker, several partitions, one honest total per bucket."""
    w = _Worker()
    w._record_bucket_bytes({0: 100, 1: 200})
    w._record_bucket_bytes({0: 30, 1: 5, 2: 7})
    assert w.published_bucket_bytes() == {0: 130, 1: 205, 2: 7}


def test_a_new_plan_starts_the_totals_over(plan_id) -> None:
    """A fleet actor outlives the query that spawned it, so the totals must not carry.

    Without the reset the next query's placement would be decided partly on the previous
    query's bytes, which is a wrong answer that looks like a plausible one.
    """
    w = _Worker()
    w._record_bucket_bytes({0: 100})
    plan_id(2)
    w._record_bucket_bytes({0: 7})
    assert w.published_bucket_bytes() == {0: 7}


def test_the_accessor_hands_back_a_copy(plan_id) -> None:
    """The caller must not be able to mutate the worker's running totals."""
    w = _Worker()
    w._record_bucket_bytes({0: 5})
    got = w.published_bucket_bytes()
    got[0] = 999
    assert w.published_bucket_bytes() == {0: 5}


def test_concurrent_map_calls_lose_no_bytes(plan_id) -> None:
    """Under `FLEET_CONCURRENCY` two map calls land in different threads of one actor.

    `d[k] = d.get(k, 0) + v` is three bytecodes, so without the lock a thread can read a
    total its sibling is about to replace. Run enough increments that a lost update is
    near-certain if the lock is not doing its job, and check the exact sum.
    """
    import threading

    w = _Worker()
    per_thread, threads = 400, 8

    def hammer() -> None:
        for _ in range(per_thread):
            w._record_bucket_bytes({0: 1, 1: 2})

    workers = [threading.Thread(target=hammer) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()

    assert w.published_bucket_bytes() == {0: per_thread * threads, 1: 2 * per_thread * threads}
