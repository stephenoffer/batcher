"""The shuffle REDUCE barrier's recovery contract (`run_bucket_reduce`).

`map_barrier` keeps the map stage alive across a preemption and is pinned by
`test_map_recovery.py`. This is the other half — the reduce stage — and it had no test at
all, which is how the module implementing it went missing from the tree entirely while every
`collect(spill=True)` and `collect(distributed=True)` in the suite failed at *import*. A
contract this load-bearing needs a test that does not require a real cluster to run.

The sort, join, and window shuffles all reduce through this one loop, so what is pinned here
is what all three inherit:

* a clean run reduces each bucket exactly once and recomputes nothing;
* a reducer that reports unreachable mappers gets those mappers republished on a survivor,
  and only the affected buckets re-run;
* a bucket that already reported `ok` is never re-run (recovery costs one recompute, not a
  re-reduce of the whole stage);
* a source and the *host* holding it are different numbers after a relocation, and it is the
  host that dies;
* recovery is bounded — a shuffle that never heals raises rather than looping.

Ray is faked (refs are thunks, `ray.get` calls them), so every branch runs deterministically
without worker crashes — the same approach `test_map_recovery.py` takes.
"""

from __future__ import annotations

import collections
import sys
import types

import pytest

from batcher._internal.errors import ResourceError

pytestmark = pytest.mark.unit


def _install_fake_ray(monkeypatch):
    """A `ray` whose object refs are thunks: `ray.get(ref)` calls it, `wait` returns all ready.

    `gather_with_backups` (which the reduce loop collects through) calls
    `ray.wait(pending, num_returns=..., timeout=...)` and expects `(done, still_pending)`.
    Returning everything as done immediately means no straggler backup is ever launched, which
    is the default speculation policy anyway — so these tests exercise the recovery branches
    without the timing-dependent one.
    """
    exc = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    class RayActorError(RayError):
        pass

    exc.RayError = RayError
    exc.RayTaskError = RayTaskError
    exc.RayActorError = RayActorError

    ray_mod = types.ModuleType("ray")
    ray_mod.exceptions = exc
    ray_mod.get = lambda ref: ref()
    ray_mod.wait = lambda pending, num_returns=1, timeout=None: (list(pending), [])
    ray_mod.cancel = lambda ref, **kw: None
    ray_mod.kill = lambda actor: None

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return exc


class _Actor:
    """Stand-in worker actor. Only `draining_workers` would touch it, and that is inert off
    the spot profile, so it needs no behavior — its identity is the whole contract."""

    def __init__(self, wid: int) -> None:
        self.wid = wid


def _run(monkeypatch, *, n_buckets, workers, reduce_fn, republish_fn, dead=None):
    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    return run_bucket_reduce(
        kind="sort",
        n_buckets=n_buckets,
        workers=workers,
        actors=[_Actor(i) for i in range(workers)],
        remote_reduce=reduce_fn,
        republish=republish_fn,
        dead=dead,
    )


def test_a_clean_run_reduces_every_bucket_exactly_once(monkeypatch):
    _install_fake_ray(monkeypatch)
    launches: collections.Counter = collections.Counter()
    republished: list[tuple[int, int]] = []

    def reduce_fn(host, bucket):
        launches[bucket] += 1
        return lambda b=bucket: ("ok", [f"rows-{b}"])

    out = _run(
        monkeypatch,
        n_buckets=4,
        workers=4,
        reduce_fn=reduce_fn,
        republish_fn=lambda t, s: republished.append((t, s)),
    )

    assert out == {b: [f"rows-{b}"] for b in range(4)}
    assert all(launches[b] == 1 for b in range(4)), launches
    assert republished == []  # nothing failed, so nothing was recomputed


def test_an_empty_bucket_is_reported_rather_than_dropped(monkeypatch):
    """`("ok", None)` means "this bucket is legitimately empty", not "this bucket failed".

    The sort caller keys its output by bucket to concatenate ranges in key order, so an empty
    range must still appear — a missing key would silently shift the ranges.
    """
    _install_fake_ray(monkeypatch)
    out = _run(
        monkeypatch,
        n_buckets=3,
        workers=3,
        reduce_fn=lambda h, b: lambda b=b: ("ok", None) if b == 1 else ("ok", [b]),
        republish_fn=lambda t, s: None,
    )
    assert set(out) == {0, 1, 2}
    assert out[1] is None


def test_an_unreachable_mapper_is_republished_and_only_that_bucket_retries(monkeypatch):
    """The core recovery: a reducer names the mappers it could not reach; those are recomputed.

    Bucket 1 cannot reach source 2 on its first attempt. Source 2 must be republished onto a
    live worker and bucket 1 retried — while buckets 0 and 2, which already succeeded, are
    left alone.
    """
    _install_fake_ray(monkeypatch)
    launches: collections.Counter = collections.Counter()
    republished: list[tuple[int, int]] = []

    def reduce_fn(host, bucket):
        launches[bucket] += 1
        if bucket == 1 and launches[1] == 1:
            return lambda: ("retry", {2})  # mapper 2 unreachable
        return lambda b=bucket: ("ok", [f"rows-{b}"])

    out = _run(
        monkeypatch,
        n_buckets=3,
        workers=3,
        reduce_fn=reduce_fn,
        republish_fn=lambda t, s: republished.append((t, s)),
    )

    assert out == {b: [f"rows-{b}"] for b in range(3)}
    assert [s for _, s in republished] == [2], republished
    assert launches[1] == 2  # retried once
    assert launches[0] == 1 and launches[2] == 1  # completed buckets never re-reduced


def test_a_republish_target_is_never_a_worker_known_to_be_dead(monkeypatch):
    """Recompute must land on a survivor — `dead` seeds what the map barrier already lost."""
    _install_fake_ray(monkeypatch)
    targets: list[int] = []
    launches: collections.Counter = collections.Counter()

    def reduce_fn(host, bucket):
        launches[bucket] += 1
        assert host != 0, "a reducer was hosted on a worker seeded as dead"
        if launches[bucket] == 1:
            return lambda: ("retry", {1})
        return lambda b=bucket: ("ok", [b])

    _run(
        monkeypatch,
        n_buckets=3,
        workers=3,
        reduce_fn=reduce_fn,
        republish_fn=lambda t, s: targets.append(t),
        dead={0},
    )
    assert targets and all(t != 0 for t in targets), targets


def test_recovery_is_bounded_and_raises_rather_than_looping(monkeypatch):
    """A shuffle that never heals fails loudly, and does not retry forever."""
    _install_fake_ray(monkeypatch)
    rounds: collections.Counter = collections.Counter()

    def reduce_fn(host, bucket):
        rounds[bucket] += 1
        return lambda: ("retry", {1})  # never recovers

    with pytest.raises(ResourceError):
        _run(
            monkeypatch,
            n_buckets=2,
            workers=3,
            reduce_fn=reduce_fn,
            republish_fn=lambda t, s: None,
        )
    # Bounded by the recovery policy, not unbounded — a handful of rounds, not hundreds.
    assert 0 < rounds[0] <= 8, rounds


def test_a_dead_reducer_host_is_recovered(monkeypatch):
    """A reducer whose *host* dies is a lost host, not a lost bucket — recompute and retry."""
    exc = _install_fake_ray(monkeypatch)
    launches: collections.Counter = collections.Counter()

    def reduce_fn(host, bucket):
        launches[bucket] += 1
        if bucket == 1 and launches[1] == 1:

            def _die():
                raise exc.RayActorError("worker preempted")

            return _die
        return lambda b=bucket: ("ok", [f"rows-{b}"])

    out = _run(
        monkeypatch,
        n_buckets=3,
        workers=3,
        reduce_fn=reduce_fn,
        republish_fn=lambda t, s: None,
    )
    assert out == {b: [f"rows-{b}"] for b in range(3)}
    assert launches[1] == 2


def test_the_host_that_died_is_recomputed_not_the_source_id(monkeypatch):
    """After a relocation, source id and host id are different numbers — and the host dies.

    This is the invariant `SourcePlacement` exists to hold, and the one a reduce loop that
    reuses a single number for both silently breaks (see its docstring). Source 1 is relocated
    onto worker 2 by the first recovery round; worker 2 then dies. What worker 2 was holding is
    source 1, so source 1 is what must be republished — republishing "source 2" regenerates
    data nobody lost and leaves the query missing source 1's rows.
    """
    exc = _install_fake_ray(monkeypatch)
    # Where each source's output currently lives; the republish closure moves it.
    placement = {s: s for s in range(3)}
    republished: list[int] = []
    state = {"killed_relocation_host": False}

    def republish(target, src):
        republished.append(src)
        placement[src] = target

    def reduce_fn(host, bucket):
        def _run_bucket():
            # Round 1: bucket 0's reducer cannot reach source 1 -> source 1 is relocated.
            if placement[1] == 1:
                return ("retry", {1})
            # Round 2: the worker source 1 was relocated ONTO dies, once.
            if not state["killed_relocation_host"] and host == placement[1]:
                state["killed_relocation_host"] = True
                raise exc.RayActorError("the relocation target was preempted")
            return ("ok", [f"rows-{bucket}"])

        return _run_bucket

    _run(
        monkeypatch,
        n_buckets=3,
        workers=3,
        reduce_fn=reduce_fn,
        republish_fn=republish,
    )

    # Source 1 moved, then its new host died, so source 1 is republished a second time.
    # A loop that conflates the source with its host republishes the host's *id* instead.
    assert republished.count(1) >= 2, (
        f"expected source 1 to be recomputed after its host died, got {republished}. "
        "A host id was almost certainly used as a source id."
    )
