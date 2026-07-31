"""The fan-out narrowing chain is recorded, so "why did it use 6 of 40 nodes" is answerable.

The worker count is the end of a chain of narrowings across five modules — Carbonite's
data-driven want, the cluster-fill rewrite, the even-share raise, the schedulable clamp, and
the drain and node-class exclusions inside it. Each step is documented and none of it was
observable, so the most common distributed complaint could only be answered by reading the
modules and guessing which branch ran.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.trace import FanoutTrace

pytestmark = pytest.mark.unit


def test_no_steps_reports_the_requested_count():
    trace = FanoutTrace(8)
    assert trace.final == 8
    assert trace.summary() == "fan-out 8 -> 8"


def test_the_final_count_is_the_last_step():
    trace = FanoutTrace(8)
    trace.step("cluster_fill", 40, "one per node slice")
    trace.step("clamp", 6, "memory grant")
    assert trace.final == 6


def test_the_summary_names_every_step_in_order():
    trace = FanoutTrace(8)
    trace.step("cluster_fill", 40, "one per node slice")
    trace.step("clamp", 6, "memory grant")
    assert trace.summary() == "fan-out 8 -> 6: cluster_fill 40, clamp 6"


def test_a_step_that_changes_nothing_is_still_recorded():
    """'The clamp ran and agreed' and 'the clamp never ran' are different answers, and a
    reader must be able to tell them apart."""
    trace = FanoutTrace(8)
    trace.step("clamp", 8, "capacity was sufficient")
    detail = trace.to_decision().detail
    assert [s["step"] for s in detail["steps"]] == ["clamp"]
    assert detail["final"] == 8


def test_the_decision_carries_the_structured_chain():
    trace = FanoutTrace(8)
    trace.step("cluster_fill", 40, "one per node slice")
    trace.step("clamp", 6, "memory grant fits 6")
    decision = trace.to_decision()
    assert (decision.subsystem, decision.category) == ("core", "scheduling")
    assert decision.detail["requested"] == 8
    assert decision.detail["final"] == 6
    assert decision.detail["steps"][1] == {
        "step": "clamp",
        "workers": 6,
        "why": "memory grant fits 6",
    }


def test_reporting_logs_the_chain(caplog):
    """The chain has to reach an operator. It is a log record rather than a bus event
    because the bus attributes by a `query_id` minted in `api` that `dist` cannot see, and
    `observe.store` silently drops any event whose id matches no live query."""
    import logging

    trace = FanoutTrace(8)
    trace.step("clamp", 6, "memory grant")
    with caplog.at_level(logging.INFO, logger="batcher.dist"):
        trace.report()
    assert any("fan-out decided" in r.message for r in caplog.records)


def test_a_broken_logger_never_disturbs_the_schedule(monkeypatch):
    """This describes a schedule that has already been decided; it must not be able to
    break the query it is describing."""
    from batcher.dist.executors.ray_runtime import trace as trace_mod

    def _boom(*a, **k):
        raise RuntimeError("logging is down")

    monkeypatch.setattr(trace_mod, "log_kv", _boom)
    trace = FanoutTrace(8)
    trace.step("clamp", 6, "memory grant")
    trace.report()  # must not raise
