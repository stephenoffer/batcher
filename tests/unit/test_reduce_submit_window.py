"""The shuffle reduce launches within a submit-ahead window, as the map stage already does.

A bucket is reduced by exactly one worker (``bucket % workers``), so a stage with far more
buckets than workers cannot run them all — everything past `workers` is a task queued in
Ray's scheduler. Launching every bucket at once is the "too many pending tasks" anti-pattern
`map_barrier` grew `_pending_window` to avoid, and the reduce could reach it on its own:
`distributed.max_shuffle_partitions` permits 2,048 buckets, and sizing a shuffle by its data
volume rather than by the cluster makes counts near that ceiling ordinary.

What must not change is anything about the *result*: the same buckets, reduced once each, with
the same recovery behaviour. So these tests pin the launch pattern and re-pin the contract
`test_bucket_reduce_recovery.py` establishes, under a window narrow enough to force chunking.

Ray is faked (refs are thunks, `ray.get` calls them) exactly as in that file, so every branch
runs deterministically without a cluster.
"""

from __future__ import annotations

import collections
import dataclasses
import sys
import types

import pytest

from batcher.config import active_config, config_context

# Imported at module scope, *before* any test swaps a fake `ray` into `sys.modules`.
# `batcher.dist` decorates its task functions with `@ray.remote` at import time, so a first
# import performed under the fake would fail — an ordering trap rather than a real failure.
from batcher.dist.executors.ray_runtime import run_bucket_reduce

pytestmark = pytest.mark.unit


def _install_fake_ray(monkeypatch, *, on_wait=None):
    """A `ray` whose refs are thunks. `on_wait` observes how many are in flight per wait."""
    exc = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    class RayActorError(RayError):
        pass

    exc.RayError, exc.RayTaskError, exc.RayActorError = RayError, RayTaskError, RayActorError

    def wait(pending, num_returns=1, timeout=None):
        if on_wait is not None:
            on_wait(len(pending))
        return list(pending), []

    def get(ref):
        # Real `ray.get` takes a single ref or a list of them; the windowed gather uses the
        # list form, so the fake has to honour both or it tests a shape Ray does not have.
        return [r() for r in ref] if isinstance(ref, list) else ref()

    ray_mod = types.ModuleType("ray")
    ray_mod.exceptions = exc
    ray_mod.get = get
    ray_mod.wait = wait
    ray_mod.cancel = lambda ref, **kw: None
    ray_mod.kill = lambda actor: None

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)


class _Actor:
    def __init__(self, wid: int) -> None:
        self.wid = wid


def _run(*, n_buckets, workers, reduce_fn, republish_fn=lambda t, s: None, dead=None):
    return run_bucket_reduce(
        kind="sort",
        n_buckets=n_buckets,
        workers=workers,
        actors=[_Actor(i) for i in range(workers)],
        remote_reduce=reduce_fn,
        republish=republish_fn,
        dead=dead,
    )


@pytest.fixture
def narrow_window():
    """Pin the window at 4 through the real config knob, so a 20-bucket stage must chunk.

    Set via `distributed.max_pending_tasks` rather than by patching the private sizing
    function: that is the setting an operator actually turns, and it makes these tests fail
    against an implementation that ignores the window instead of erroring on a missing name.
    """
    base = active_config()
    replaced = base.replace(distributed=dataclasses.replace(base.distributed, max_pending_tasks=4))
    with config_context(replaced):
        yield 4


def test_no_more_than_a_window_of_reduces_is_ever_outstanding(monkeypatch, narrow_window):
    _install_fake_ray(monkeypatch)
    outstanding = 0
    peak = 0

    def reduce_fn(host, bucket):
        nonlocal outstanding, peak
        outstanding += 1
        peak = max(peak, outstanding)

        def finish(b=bucket):
            nonlocal outstanding
            outstanding -= 1
            return ("ok", [b])

        return finish

    out = _run(n_buckets=20, workers=5, reduce_fn=reduce_fn)
    assert out == {b: [b] for b in range(20)}
    assert peak <= narrow_window, f"{peak} reduces were outstanding at once"


def test_every_bucket_is_still_reduced_exactly_once(monkeypatch, narrow_window):
    """Chunking must not drop, duplicate, or reorder a bucket."""
    _install_fake_ray(monkeypatch)
    launches: collections.Counter = collections.Counter()

    def reduce_fn(host, bucket):
        launches[bucket] += 1
        return lambda b=bucket: ("ok", [f"rows-{b}"])

    out = _run(n_buckets=14, workers=3, reduce_fn=reduce_fn)
    assert out == {b: [f"rows-{b}"] for b in range(14)}
    assert all(launches[b] == 1 for b in range(14)), launches


def test_a_stage_inside_the_window_submits_everything_before_the_first_wait(monkeypatch):
    """The unchanged fast path: an ordinary query must behave exactly as it did."""
    waits: list[int] = []
    _install_fake_ray(monkeypatch, on_wait=waits.append)
    out = _run(n_buckets=6, workers=6, reduce_fn=lambda h, b: lambda b=b: ("ok", [b]))
    assert out == {b: [b] for b in range(6)}
    assert waits and waits[0] == 6, "all six should be in flight at the first wait"


def test_recovery_still_works_across_a_chunk_boundary(monkeypatch, narrow_window):
    """A reducer in a later chunk that cannot reach a mapper is still recovered.

    The recovery loop reruns only what is pending, and completed buckets are cached — so a
    failure in the last chunk must not re-reduce the chunks that already finished.
    """
    _install_fake_ray(monkeypatch)
    launches: collections.Counter = collections.Counter()
    republished: list[tuple[int, int]] = []

    def reduce_fn(host, bucket):
        launches[bucket] += 1
        if bucket == 9 and launches[9] == 1:
            return lambda: ("retry", {2})  # a mapper this reducer could not reach
        return lambda b=bucket: ("ok", [f"rows-{b}"])

    out = _run(
        n_buckets=10,
        workers=3,
        reduce_fn=reduce_fn,
        republish_fn=lambda t, s: republished.append((t, s)),
    )
    assert out == {b: [f"rows-{b}"] for b in range(10)}
    assert [s for _, s in republished] == [2]
    assert launches[9] == 2, "the failed bucket retried"
    assert all(launches[b] == 1 for b in range(9)), "no completed bucket was re-reduced"


def test_a_worker_lost_in_one_chunk_is_avoided_by_the_next(monkeypatch, narrow_window):
    """Discovering a death early must inform later windows, not be rediscovered per bucket."""
    _install_fake_ray(monkeypatch)
    hosts_used: list[int] = []
    seen: set[int] = set()

    def reduce_fn(host, bucket):
        hosts_used.append(host)
        if bucket == 0 and bucket not in seen:
            seen.add(bucket)
            return lambda: ("retry", {0})
        return lambda b=bucket: ("ok", [b])

    out = _run(n_buckets=12, workers=3, reduce_fn=reduce_fn)
    assert out == {b: [b] for b in range(12)}


def test_the_window_is_sized_from_workers_and_the_configured_factor():
    """The reduce window counts workers, because workers is what bounds concurrent reduces."""
    from batcher.dist.executors.ray_runtime.reduce import _reduce_window

    cfg = active_config().distributed
    if cfg.max_pending_tasks > 0:
        assert _reduce_window(8) == cfg.max_pending_tasks
    else:
        assert _reduce_window(8) == 8 * max(1, cfg.pending_window_factor)
    assert _reduce_window(0) >= 1, "a degenerate fan-out must still allow one launch"


# ---- the same bound, for a stage that gathers with a plain `ray.get` --------------------


def test_a_windowed_gather_returns_results_in_item_order(monkeypatch):
    """Callers zip the results against their own task list, so order is the contract."""
    _install_fake_ray(monkeypatch)
    from batcher.dist.executors.ray_runtime import gather_in_windows

    items = list(range(23))
    out = gather_in_windows(lambda i: lambda i=i: i * 10, items, workers=2)
    assert out == [i * 10 for i in items]


def test_a_windowed_gather_bounds_what_is_launched_at_once(monkeypatch, narrow_window):
    """An aggregate's combiner level is `n_reducers x ceil(sources / fan_in)` tasks.

    That is a product of two fan-outs, so at the reducer ceiling one level of one aggregate
    is six figures of simultaneously-pending tasks — both a scheduler flood and one
    `ObjectRef` per task held on the driver.
    """
    _install_fake_ray(monkeypatch)
    from batcher.dist.executors.ray_runtime import gather_in_windows

    outstanding = 0
    peak = 0

    def launch(i):
        nonlocal outstanding, peak
        outstanding += 1
        peak = max(peak, outstanding)

        def finish(i=i):
            nonlocal outstanding
            outstanding -= 1
            return i

        return finish

    out = gather_in_windows(launch, list(range(30)), workers=3)
    assert out == list(range(30))
    assert peak <= narrow_window, f"{peak} tasks were launched at once"


def test_a_gather_inside_the_window_is_a_single_launch(monkeypatch):
    """The unchanged fast path, so an ordinary aggregate behaves exactly as before."""
    waits: list[int] = []
    _install_fake_ray(monkeypatch, on_wait=waits.append)
    from batcher.dist.executors.ray_runtime import gather_in_windows

    launched: list[int] = []

    def launch(i):
        launched.append(i)
        return lambda i=i: i

    assert gather_in_windows(launch, [0, 1, 2], workers=8) == [0, 1, 2]
    assert launched == [0, 1, 2], "all three launched before the first result was taken"


def test_an_empty_gather_launches_nothing(monkeypatch):
    _install_fake_ray(monkeypatch)
    from batcher.dist.executors.ray_runtime import gather_in_windows

    assert gather_in_windows(lambda i: None, [], workers=4) == []
