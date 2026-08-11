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
    """The chain has to reach an operator, and the log record is the copy that always does.

    The bus copy needs a subscriber and a live query; this one needs neither, so it is what a
    reader has when they turn up after the fact with only a log file."""
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


# --- reaching the event bus ---------------------------------------------------------------


def test_the_chain_reaches_the_bus_attributed_to_the_query_in_flight():
    """`dist` cannot see the query id, so the ambient scope has to carry it.

    Without one, the `Decision` published here reached the bus with an empty id and
    `observe.store` dropped it — silently, by design, since an event naming no live query
    must not resurrect an aged-out record as a ghost. The result was that the single most
    useful distributed diagnostic never appeared beside Kyber's and Carbonite's.
    """
    from batcher._internal import events

    seen: list = []
    stop = events.subscribe(seen.append)
    try:
        with events.query_scope("q-under-test"):
            trace = FanoutTrace(8)
            trace.step("clamp", 6, "memory grant")
            trace.report()
    finally:
        stop()
    decisions = [e for e in seen if e.kind == events.DECISION]
    assert decisions, "the fan-out chain must reach the bus"
    assert decisions[0].query_id == "q-under-test"
    assert decisions[0].fields["category"] == "scheduling"
    assert "fan-out 8 -> 6" in decisions[0].fields["summary"]


def test_outside_a_query_scope_the_event_is_engine_level():
    """No conductor, no id — and that must be an empty id rather than an error.

    `dist` is reachable without `api` (a worker-side call, a direct test), so the scope has
    to be optional in both directions.
    """
    from batcher._internal import events

    seen: list = []
    stop = events.subscribe(seen.append)
    try:
        FanoutTrace(4).report()
    finally:
        stop()
    decisions = [e for e in seen if e.kind == events.DECISION]
    assert decisions and decisions[0].query_id == ""


def test_a_nested_scope_attributes_to_the_inner_query():
    """An adaptive stage re-run or a `ds.dq` probe inside a query is its own query."""
    from batcher._internal import events

    seen: list = []
    stop = events.subscribe(seen.append)
    try:
        with events.query_scope("outer"), events.query_scope("inner"):
            FanoutTrace(2).report()
        assert events.current_query_id() == ""
    finally:
        stop()
    assert [e.query_id for e in seen if e.kind == events.DECISION] == ["inner"]


def test_an_empty_scope_id_leaves_the_surrounding_one_alone():
    """A caller that has not minted an id must not blank out the one already in force."""
    from batcher._internal import events

    with events.query_scope("outer"), events.query_scope(""):
        assert events.current_query_id() == "outer"
