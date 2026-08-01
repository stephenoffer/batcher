"""WS4: the bounded pending-task submission window in `gather_map_results`.

A high-fan-out map stage must not seed Ray with every partition task at once (the
"too many pending tasks" anti-pattern). The gather bounds outstanding tasks to a
window, refilling a slot per completion, while keeping results partition-ordered and
the preemption-resubmit path intact. These exercise the window against a fake Ray that
tracks peak concurrency, so the bound is verified deterministically.
"""

from __future__ import annotations

import collections
import sys
import types

import pytest

from batcher.carbonite.resilience import RecoveryPolicy


def _raise(exc: BaseException):
    raise exc


def _install_tracking_ray(monkeypatch) -> tuple[dict, type, type]:
    """A fake `ray` whose refs are thunks and that tracks live (submitted-not-gathered)
    task count so a test can assert the peak stays within the window. `ray.get` decrements
    the counter; the test's `submit` increments it."""
    state = {"live": 0, "peak": 0}
    exc = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    exc.RayError = RayError
    exc.RayTaskError = RayTaskError

    ray_mod = types.ModuleType("ray")
    ray_mod.exceptions = exc
    ray_mod.wait = lambda refs, num_returns=1, timeout=None: ([refs[0]], refs[1:])

    def _get(ref):
        state["live"] -= 1
        return ref()

    ray_mod.get = _get
    ray_mod.cluster_resources = lambda: {"CPU": 0.0}
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return state, RayError, RayTaskError


def _counting_submit(state: dict, fail: dict | None = None):
    """A `submit(idx)` that returns a thunk producing `[idx]`, tracking peak live count.
    `fail` maps idx -> remaining number of preemption failures to inject."""
    fail = fail or {}
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        if fail.get(idx, 0) > 0:
            fail[idx] -= 1
            return lambda: _raise(_RAY_ERR[0]("preempted"))
        return lambda i=idx: [i]

    return submit, calls


_RAY_ERR: list = []


def test_small_n_submits_everything_up_front(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    state, RayError, _ = _install_tracking_ray(monkeypatch)
    _RAY_ERR[:] = [RayError]
    submit, _ = _counting_submit(state)
    # window >= n: every task is in flight at once — the unchanged submit-all fast path.
    out = gather_map_results(submit, 5, RecoveryPolicy(max_attempts=3), max_pending=100)
    assert out == [[0], [1], [2], [3], [4]]
    assert state["peak"] == 5


def test_window_caps_inflight(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    state, RayError, _ = _install_tracking_ray(monkeypatch)
    _RAY_ERR[:] = [RayError]
    submit, calls = _counting_submit(state)
    out = gather_map_results(submit, 20, RecoveryPolicy(max_attempts=3), max_pending=3)
    assert out == [[i] for i in range(20)]  # complete + partition-ordered
    assert state["peak"] <= 3  # never more than the window outstanding
    assert sum(calls.values()) == 20  # each partition submitted exactly once


def test_window_survives_and_bounds_a_preemption(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    state, RayError, _ = _install_tracking_ray(monkeypatch)
    _RAY_ERR[:] = [RayError]
    submit, calls = _counting_submit(state, fail={2: 1})  # partition 2 preempts once
    out = gather_map_results(submit, 6, RecoveryPolicy(max_attempts=3), max_pending=2)
    assert out == [[i] for i in range(6)]
    assert state["peak"] <= 2  # the resubmit still respects the window
    assert calls[2] == 2  # failed once, resubmitted once


def test_window_derives_from_cluster_cores(monkeypatch):
    from batcher.dist.executors.ray_runtime import policies

    # 8 cores x default factor 4 = 32; n above that -> the window engages at 32.
    fake_ray = types.ModuleType("ray")
    fake_ray.cluster_resources = lambda: {"CPU": 8.0}
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    assert policies._pending_window() == 32


def test_window_falls_back_when_topology_unreadable(monkeypatch):
    from batcher.dist.executors.ray_runtime import policies

    fake_ray = types.ModuleType("ray")

    def _boom():
        raise RuntimeError("ray down")

    fake_ray.cluster_resources = _boom
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    assert policies._pending_window() == policies._DEFAULT_PENDING_WINDOW


def test_explicit_max_pending_tasks_config(monkeypatch):
    import dataclasses

    from batcher.config import Config, config_context
    from batcher.dist.executors.ray_runtime import policies

    cfg = Config()
    cfg = cfg.replace(distributed=dataclasses.replace(cfg.distributed, max_pending_tasks=7))
    with config_context(cfg):
        assert policies._pending_window() == 7


def test_empty_partition_set(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    _, RayError, _ = _install_tracking_ray(monkeypatch)
    _RAY_ERR[:] = [RayError]
    assert gather_map_results(lambda i: None, 0, RecoveryPolicy(max_attempts=1)) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
