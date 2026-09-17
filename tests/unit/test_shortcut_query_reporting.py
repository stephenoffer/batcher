"""A query answered without executing still ran, and every sink has to be told.

Seven routes out of a terminal op return a finished answer before reaching the executor: a
prepared-plan replay, a bounded peek off a streaming source, a `limit(n)` that stopped
reading early, a keyless aggregate answered from source statistics, a provably-empty result,
the device tier, and the opt-in small-query fast path. `count()` adds three more of its own.

Every one of them is a correct answer to a real query, and every one was invisible to *all*
of the observability surfaces at once -- the console printed no line, the dashboard showed no
row, the OTel exporter emitted no span, and `batcher_queries_total` did not count it. So a
job built out of `agg()`, `count()` and `head()` over statistics-bearing sources reported
that it had run no queries at all. That is the shape of wrongness that reads as healthy: no
error, no gap in a chart, just a number that is quietly too small.

These assert on the *event bus* rather than on any one sink, because the bus is what all of
them consume: a shape reported here is reported everywhere, and one missing here is missing
everywhere.
"""

from __future__ import annotations

from collections import Counter

import pytest

import batcher as bt
from batcher._internal import events
from batcher.api.terminal.event_log import SHORTCUT_REASONS

pytestmark = pytest.mark.unit


@pytest.fixture
def bus():
    """Collect every event published inside the test, then detach."""
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        yield seen
    finally:
        unsubscribe()


@pytest.fixture
def frame():
    return bt.from_pydict({"k": [1, 2, 3, 4] * 50, "v": [1.0, 2.0, 3.0, 4.0] * 50})


#: Every terminal op that executes. `head`/`limit` without a terminal are excluded on
#: purpose -- they are lazy, and reporting a query for a plan nobody ran would be the
#: opposite defect.
EXECUTING = {
    "collect": lambda ds: ds.collect(),
    "filter_collect": lambda ds: ds.filter(bt.col("v") > 1).collect(),
    "grouped_agg": lambda ds: ds.group_by("k").agg(t=bt.col("v").sum()).collect(),
    "global_agg": lambda ds: ds.agg(t=bt.col("v").sum()).collect(),
    "scalar_reduction": lambda ds: ds.sum("v"),
    "count": lambda ds: ds.count(),
    "limit_collect": lambda ds: ds.limit(3).collect(),
    "sort_collect": lambda ds: ds.sort("v").collect(),
    "distinct_collect": lambda ds: ds.distinct().collect(),
    "to_pydict": lambda ds: ds.to_pydict(),
    "iter_batches": lambda ds: list(ds.iter_batches()),
}


@pytest.mark.parametrize("op", sorted(EXECUTING))
def test_every_executing_terminal_op_opens_and_closes_a_query(op, frame, bus):
    """One `query_start` and one `query_end`, whichever route produced the answer."""
    EXECUTING[op](frame)
    kinds = Counter(e.kind for e in bus)
    assert kinds[events.QUERY_START] == 1, f"{op} published {kinds[events.QUERY_START]} starts"
    assert kinds[events.QUERY_END] == 1, f"{op} published {kinds[events.QUERY_END]} ends"


def test_a_lazy_operation_reports_nothing():
    """The control. `head` builds a plan and runs nothing, so a query event would be a lie."""
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        bt.from_pydict({"x": [1, 2, 3]}).limit(2)
    finally:
        unsubscribe()
    assert seen == []


@pytest.mark.parametrize(
    ("op", "route"),
    [
        (lambda ds: ds.agg(t=bt.col("v").sum()).collect(), "metadata_aggregate"),
        (lambda ds: ds.count(), "metadata_count"),
        (lambda ds: ds.limit(3).collect(), "streaming_limit"),
    ],
)
def test_a_shortcut_says_which_route_answered_it(op, route, frame, bus):
    """The route is the point: it is why the query was fast, and it was unknowable."""
    op(frame)
    ends = [e for e in bus if e.kind == events.QUERY_END]
    assert len(ends) == 1
    assert ends[0].fields.get("shortcut") == route
    assert ends[0].fields.get("note") == SHORTCUT_REASONS[route]


def test_a_shortcut_reports_ok_and_a_row_count(frame, bus):
    """A sink must be able to treat a shortcut answer like any other completed query."""
    frame.agg(t=bt.col("v").sum()).collect()
    end = next(e for e in bus if e.kind == events.QUERY_END)
    assert end.fields["ok"] is True
    assert end.fields["rows"] == 1
    assert end.fields["total_ms"] >= 0.0


def test_a_shortcut_query_is_counted_in_the_metrics_export(frame):
    """The number this was really about: the counter a capacity decision is made from."""
    from batcher.observe import metrics_snapshot, reset_metrics, start_metrics, stop_metrics

    start_metrics()
    try:
        reset_metrics()
        frame.agg(t=bt.col("v").sum()).collect()
        frame.count()
        assert metrics_snapshot()["queries"]["total"] == 2
    finally:
        reset_metrics()
        stop_metrics()


def test_a_shortcut_query_carries_a_pipeline_signature(frame, bus):
    """Without it the dashboard files every run as its own singleton pipeline.

    Which is worst exactly here: a `count()` in a loop is the shape these routes serve, and
    grouping repeated runs is the whole point of the pipeline view.
    """
    frame.count()
    start = next(e for e in bus if e.kind == events.QUERY_START)
    assert start.fields.get("signature"), "no signature on a shortcut-answered query"
    assert start.fields.get("label") == "scan"


def test_every_route_has_a_reason_a_reader_can_act_on():
    """A route with no phrase would render its bare identifier on the console."""
    for route, reason in SHORTCUT_REASONS.items():
        assert reason and reason != route
        assert reason == reason.lower() or reason[0].isupper()
        assert "\n" not in reason


#: The terminals that sense configuration before they run. Named here rather than derived
#: from the module, because a test that reads the decorator off the function it is checking
#: cannot fail -- it would agree with whatever the file happens to say.
AUTO_CONFIGURED = ("_collect", "_explain", "_stats", "_write")


@pytest.mark.parametrize("name", AUTO_CONFIGURED)
def test_the_sensing_terminals_keep_their_decorator(name):
    """A decorator separated from its function by an insertion is silent and expensive.

    Adding a helper immediately above `_collect` put `@with_auto_config` on the *helper*:
    `collect()` silently stopped sensing configuration and the helper paid for a resolve on
    every call. It compiled, `import batcher` worked, and the observability and autoconfig
    suites passed -- it was found by disassembling the wrong function while chasing a
    performance regression, which is not a method anyone should have to rely on.

    Asserted through the bytecode because that is what the decorator actually changes; a
    source-level check would pass on the broken arrangement, since the `@` line is still
    there and still adjacent to *a* function.
    """
    import dis
    import io

    from batcher.api.terminal import core

    listing = io.StringIO()
    dis.dis(getattr(core, name), file=listing)
    assert "resolve_auto_config" in listing.getvalue(), f"{name} lost @with_auto_config"


def test_the_decorator_guard_can_fail():
    """The positive control: an undecorated function really does read as undecorated."""
    import dis
    import io

    from batcher.api.terminal import core

    listing = io.StringIO()
    dis.dis(core._shortcut, file=listing)
    assert "resolve_auto_config" not in listing.getvalue()


def test_nothing_is_published_when_nobody_is_listening(frame):
    """Reporting must stay free on the hot path, which is what `listening()` guards.

    Asserted by calling `report_shortcut` with no subscriber and checking it mints no id --
    the counter is the only observable side effect it would have.
    """
    from batcher.api.terminal import event_log

    before = event_log._counter
    plan = bt.from_pydict({"x": [1]})._plan
    event_log.report_shortcut(plan, route="metadata_count", rows=1, total_ms=0.1)
    assert event_log._counter is before
    assert not events.listening()
