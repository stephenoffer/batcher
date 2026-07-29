"""Straggler-speculation decision logic (pure, no Ray)."""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import SpeculationPolicy, stragglers_to_backup

pytestmark = pytest.mark.unit


def test_disabled_when_max_backups_zero():
    pol = SpeculationPolicy(max_backups=0)
    # Even a wild straggler is not backed up when speculation is off.
    assert stragglers_to_backup(4, {0: 1.0, 1: 1.0, 2: 1.0}, {3: 100.0}, pol) == []


def test_no_backup_before_min_finished_fraction():
    pol = SpeculationPolicy(max_backups=2, min_finished_frac=0.75)
    # Only 1 of 4 finished (<75%): too early to judge a straggler.
    assert stragglers_to_backup(4, {0: 1.0}, {1: 50.0, 2: 50.0, 3: 50.0}, pol) == []


def test_backs_up_slowest_running_beyond_threshold():
    pol = SpeculationPolicy(max_backups=1, min_finished_frac=0.5, straggler_factor=2.0)
    # 2 of 4 finished (median 1.0); threshold = 2.0. Tasks 2 (3x) and 3 (5x) qualify;
    # max_backups=1 → only the slowest (task 3).
    out = stragglers_to_backup(4, {0: 1.0, 1: 1.0}, {2: 3.0, 3: 5.0}, pol)
    assert out == [3]


def test_backs_up_multiple_up_to_cap():
    pol = SpeculationPolicy(max_backups=2, min_finished_frac=0.5, straggler_factor=2.0)
    out = stragglers_to_backup(4, {0: 1.0, 1: 1.0}, {2: 3.0, 3: 5.0}, pol)
    assert out == [3, 2]  # slowest first, both over threshold, within cap


def test_no_backup_when_running_within_threshold():
    pol = SpeculationPolicy(max_backups=2, min_finished_frac=0.5, straggler_factor=3.0)
    # median 1.0, threshold 3.0; the one running task at 2.5x is not a straggler.
    assert stragglers_to_backup(4, {0: 1.0, 1: 1.0, 2: 1.0}, {3: 2.5}, pol) == []


# --- The stalled-barrier diagnostic -------------------------------------------------
#
# `gather_with_backups` waits forever by design, and forever is reachable: a placement
# group that could not be reserved falls back to default scheduling, whose tasks carry the
# same per-task CPU demand and so may never schedule. A silent hang is the worst shape that
# can take, so the barrier says something. These pin *when* it speaks, without Ray.


class _FakeRay:
    """Enough `ray` for `gather_with_backups`, and the test's clock.

    `ready_after` maps a ref to the `wait` call number on which it becomes ready, so a ref
    that is never listed models a task the scheduler will never run.

    The clock advances **inside `wait`**, by `step` seconds per call, and `now()` merely
    reads it. That matters: `gather_with_backups` reads the clock once per wake, but so does
    anything else running in the same process — the logging handler publishes to the event
    bus, which timestamps with `time.monotonic()`. A clock that yielded a scripted sequence
    per *call* therefore drifted or ran out depending on whether logging happened to be
    configured by an earlier test. Tying time to the barrier's own wakes makes the timeline
    the test asserts on independent of who else reads the clock.
    """

    def __init__(self, ready_after: dict[int, int], step: float = 130.0) -> None:
        self.ready_after = ready_after
        self.step = step
        self.calls = 0
        self.clock = 0.0

    def now(self) -> float:
        return self.clock

    def wait(self, pending, num_returns=1, timeout=None):
        self.calls += 1
        self.clock += self.step
        done = [r for r in pending if self.ready_after.get(r, 10**9) <= self.calls]
        return done, [r for r in pending if r not in done]

    def get(self, ref):
        return f"result-{ref}"

    def cancel(self, ref, force=False):
        pass

    def cluster_resources(self):
        return {"CPU": 96.0}

    def available_resources(self):
        return {"CPU": 0.0}


def _run_barrier(monkeypatch, fake_ray, refs):
    import sys
    import time

    from batcher.carbonite.resilience import gather_with_backups

    # `gather_with_backups` imports `ray` and `time` inside the call, so both resolve
    # through `sys.modules` at run time — patch there rather than on the module object.
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(time, "monotonic", fake_ray.now)
    return gather_with_backups(refs, lambda i: refs[i])


def _stall_warnings(caplog):
    return [r for r in caplog.records if "distributed barrier has waited" in r.message]


def test_a_barrier_that_finishes_inside_the_window_says_nothing(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="batcher.carbonite")
    # Both tasks land on the first wake, five seconds in — well inside the 120s window.
    out = _run_barrier(monkeypatch, _FakeRay({0: 1, 1: 1}, step=5.0), [0, 1])
    assert out == ["result-0", "result-1"]
    assert not _stall_warnings(caplog)


def test_one_completion_silences_the_stall_warning(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="batcher.carbonite")
    # Ref 0 lands on the first wake; ref 1 runs another three windows. That is a straggler,
    # which speculation owns — not a starved barrier — so only the window that elapsed
    # before ref 0 landed is reported, however long the second task then takes.
    out = _run_barrier(monkeypatch, _FakeRay({0: 1, 1: 4}), [0, 1])
    assert out == ["result-0", "result-1"]
    assert len(_stall_warnings(caplog)) == 1


def test_a_barrier_with_nothing_finished_reports_each_window(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="batcher.carbonite")
    # Nothing is ready until the third wake. The barrier is checked at each wake, before
    # that wake's completions are drained, so it sees an empty result set at 130s, 260s and
    # 390s — one report per elapsed window, not one per poll (there is no fourth).
    out = _run_barrier(monkeypatch, _FakeRay({0: 3, 1: 3}), [0, 1])
    assert out == ["result-0", "result-1"]
    warnings = _stall_warnings(caplog)
    assert len(warnings) == 3
    assert "0/2 tasks finished" in warnings[0].getMessage()
    # It names the cluster's own view, which is what distinguishes "slow" from "starved".
    assert "cluster CPU 96/96 in use" in warnings[0].getMessage()
