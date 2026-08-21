"""The cheap way to find memory, and the guard that stops it becoming expensive.

Spilling writes operator state to disk and reads it back. There is a strictly cheaper source
of bytes in the same process, and nothing reached for it: the data plane's allocator retains
freed pages by design — that retention is what keeps a parallel scan from serializing on TLB
shootdowns — and the live pressure reading is the resident set, which counts every one of
them. So a query could be sent out of core on the strength of memory nobody was using.

`release_retained_memory` has been able to hand that arena back for a while, and its own
docstring said it "is the thing to try first when an envelope is about to force a spill".
Nothing called it. These tests cover the call and, at least as importantly, the three places
it must *not* fire: inside its cooldown, on a plan whose estimate already exceeds the budget,
and after it has proved the arena is live.
"""

from __future__ import annotations

import contextlib

import pytest

from batcher.carbonite.memory import reclaim

pytestmark = pytest.mark.unit

_MIB = 1 << 20


@pytest.fixture(autouse=True)
def _fresh():
    reclaim.reset_reclaim_state()
    yield
    reclaim.reset_reclaim_state()


@pytest.fixture
def releases(monkeypatch):
    """Stand in for the engine's allocator, returning a scripted number of bytes each call."""
    calls: list[bool] = []
    scripted = {"bytes": 0}

    def release(force: bool = False) -> int:
        calls.append(force)
        return scripted["bytes"]

    monkeypatch.setattr(
        "batcher._internal.hardware.engine.allocator.release_retained_memory", release
    )
    return calls, scripted


def test_a_paying_release_is_reported_and_counted(releases):
    calls, scripted = releases
    scripted["bytes"] = 64 * _MIB
    assert reclaim.reclaim_before_spill() == 64 * _MIB
    assert calls == [True], (
        "an unforced collect walks only the calling thread's heap, and the engine allocates "
        "its operator state on rayon workers — measured 0 MiB unforced against 408 MiB forced"
    )
    stats = reclaim.reclaim_stats()
    assert stats["attempts"] == 1
    assert stats["released_bytes"] == 64 * _MIB
    assert stats["cooldown_s"] == reclaim.RECLAIM_COOLDOWN_S


def test_a_second_attempt_inside_the_cooldown_does_not_touch_the_allocator(releases):
    calls, scripted = releases
    scripted["bytes"] = 64 * _MIB
    reclaim.reclaim_before_spill()
    assert reclaim.reclaim_before_spill() == 0
    assert len(calls) == 1, "the trim costs an unmap; a decision path must not repeat it"


def test_an_empty_release_backs_off_and_a_paying_one_resets(releases, monkeypatch):
    # Returning pages re-imposes the unmapping cost retention exists to avoid, so a release
    # that frees nothing is pure loss. A process whose arena is genuinely live must stop
    # paying for the trim.
    _calls, scripted = releases
    clock = {"now": 1000.0}
    monkeypatch.setattr(reclaim.time, "monotonic", lambda: clock["now"])

    scripted["bytes"] = 0
    for expected in (2, 4, 8):
        reclaim.reclaim_before_spill()
        assert reclaim.reclaim_stats()["cooldown_s"] == reclaim.RECLAIM_COOLDOWN_S * expected
        clock["now"] += reclaim.RECLAIM_COOLDOWN_S * expected

    scripted["bytes"] = 64 * _MIB
    reclaim.reclaim_before_spill()
    assert reclaim.reclaim_stats()["cooldown_s"] == reclaim.RECLAIM_COOLDOWN_S


def test_the_backoff_is_capped(releases, monkeypatch):
    _calls, scripted = releases
    clock = {"now": 0.0}
    monkeypatch.setattr(reclaim.time, "monotonic", lambda: clock["now"])
    scripted["bytes"] = 0
    for _ in range(20):
        reclaim.reclaim_before_spill()
        clock["now"] += reclaim.RECLAIM_COOLDOWN_MAX_S
    assert reclaim.reclaim_stats()["cooldown_s"] == reclaim.RECLAIM_COOLDOWN_MAX_S


def test_a_release_too_small_to_change_a_decision_counts_as_a_miss(releases, monkeypatch):
    # Freeing less than a morsel's worth cannot move the reading, so counting it as a success
    # would keep re-trying a trim that never helps.
    _calls, scripted = releases
    monkeypatch.setattr(reclaim.time, "monotonic", lambda: 0.0)
    scripted["bytes"] = reclaim.RECLAIM_WORTHWHILE_BYTES - 1
    reclaim.reclaim_before_spill()
    assert reclaim.reclaim_stats()["cooldown_s"] == reclaim.RECLAIM_COOLDOWN_S * 2


def test_a_failing_allocator_leaves_the_caller_free_to_spill(monkeypatch):
    def boom(force: bool = False) -> int:
        raise RuntimeError("no engine")

    monkeypatch.setattr("batcher._internal.hardware.engine.allocator.release_retained_memory", boom)
    assert reclaim.reclaim_before_spill() == 0


# --- Where it is called from --------------------------------------------------------------------


def test_the_executor_hands_the_arena_back_when_it_commits_to_disk(releases):
    # The trim belongs at the moment the query is *committed* to the out-of-core path, not at
    # the gate that decides it: `run.py` routes to disk on any of three independent signals --
    # admission's counter-offer, the spill estimate, and the resident input size -- and only the
    # middle one passes through a live pressure reading. A trim hung off that reading covered
    # one route of three, and missed the estimate, which is the ordinary way a large query
    # spills.
    calls, scripted = releases
    scripted["bytes"] = 408 * _MIB
    from batcher.carbonite.manager import ResourceManager

    rm = ResourceManager.__new__(ResourceManager)
    assert rm.going_out_of_core() == 408 * _MIB
    assert calls == [True], "forced: the engine allocates its state on rayon workers"


def test_the_spill_gate_itself_never_trims(releases):
    # `should_spill` is asked on every query, including the ones that comfortably fit. Paying a
    # forced walk of every heap there would be a pure regression on the common path, so the
    # decision stays free of side effects and the executor pays for the trim once it spills.
    from batcher.carbonite.policies.spill_advice import SpillAdvisor

    calls, _ = releases
    for peak, budget in ((10 * _MIB, _MIB), (_MIB, 10 * _MIB)):
        advisor = SpillAdvisor.__new__(SpillAdvisor)
        advisor.peak_bytes = lambda plan, v=peak: v  # type: ignore[method-assign]
        advisor.hard_budget = lambda v=budget: v  # type: ignore[method-assign]
        advisor._pressure = type("P", (), {"classify": staticmethod(lambda: _NORMAL())})()
        advisor._oom_history_reason = lambda estimated: None  # type: ignore[method-assign]
        advisor.spill_reason(object())
    assert calls == []


def _NORMAL():
    from batcher.carbonite.memory.pressure import PressureLevel

    return PressureLevel.NORMAL


def test_the_out_of_core_stage_trims_before_it_writes(monkeypatch):
    # The ordering is the whole point: the pages have to go back *before* the spill starts
    # writing, or the process holds the old arena and the new buckets at once -- which is the
    # peak that gets a swapless node OOM-killed.
    from batcher.api.orchestration import stages

    order: list[str] = []

    class _RM:
        def recommend_spill_partitions(self, opt):
            return 4

        def partitions_for_bounds(self, opt, bounds):
            return 4

        def going_out_of_core(self):
            order.append("trim")
            return 0

        def spill_reason(self, opt):
            return "estimated peak exceeds the budget"

    monkeypatch.setattr(stages, "partitions_from_physical", lambda opt: 4, raising=False)
    import batcher.dist.spill as spill_mod

    def _collect(logical_opt, sources, partitions):
        order.append("spill")
        return None

    monkeypatch.setattr(spill_mod, "spill_collect", _collect)
    import batcher.api.tuning as tuning

    monkeypatch.setattr(tuning, "spill_compression_scope", lambda rm, opt: contextlib.nullcontext())
    ctx = type("C", (), {"profile": None})()
    verdict = type("V", (), {"suggested_bounds": None})()
    stages.spill_to_disk(object(), object(), ctx, _RM(), object(), verdict)
    assert order == ["trim", "spill"], "the arena goes back before the first bucket is written"


# --- What the diagnostic carries ----------------------------------------------------------------


def test_the_manager_reports_the_trim_and_what_overshooting_means_here():
    """Three memory facts that had no reader, made visible to the one reader there is.

    A rising attempt count with no released bytes says the allocator's arena is genuinely
    live, which is what separates "the engine is holding memory it does not need" from "the
    box is full" -- opposite problems with opposite fixes. And whether the node has swap says
    what overshooting the budget *means*: with it the query slows, without it the kernel kills
    the largest process, which is this one.
    """
    from batcher.carbonite.manager import ResourceManager

    stats = ResourceManager().stats()
    assert set(stats["reclaim"]) == {"attempts", "released_bytes", "cooldown_s"}
    assert isinstance(stats["swap"], bool)
