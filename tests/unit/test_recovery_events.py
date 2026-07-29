"""Fault tolerance must be observable, or it cannot be distinguished from slowness.

The distributed path recovers from worker loss transparently: it recomputes a lost
mapper's output from lineage, fences the stale copy with an epoch bump, speculates on
stragglers, and migrates work off a spot node before it is reclaimed. All of that used to
happen in complete silence. `ShuffleRecovery.recomputes` counted its rounds into an
attribute nothing read, and the proactive preemption migration sat inside a bare
`contextlib.suppress`, so it was silent whether it worked or failed.

The consequence was not a bug, it was worse: a query that survived losing two workers and
a query that was simply four times too slow produced the same output and the same logs.

These tests pin that the events fire, that they carry enough to attribute the action to a
shuffle, and — the part most likely to rot — that a broken sink cannot turn observability
into an outage.
"""

from __future__ import annotations

import pytest

from batcher._internal import events
from batcher._internal.errors import ResourceError
from batcher.carbonite.resilience import RecoveryPolicy, ShuffleRecovery

pytestmark = pytest.mark.unit


@pytest.fixture
def bus() -> list[events.Event]:
    """Collect every event published during the test, then detach."""
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        yield seen
    finally:
        unsubscribe()


def _recovery(seen: list[events.Event], event: str) -> list[events.Event]:
    return [e for e in seen if e.kind == events.RECOVERY and e.fields.get("event") == event]


class TestShuffleRecoveryEvents:
    def test_a_clean_run_publishes_nothing(self, bus: list[events.Event]) -> None:
        # The cost has to land on the failure path only. A per-round event on the happy
        # path would put a publish in every distributed query.
        result = ShuffleRecovery(label="aggregate").run(lambda: ("done", None), lambda _f: None)
        assert result == "done"
        assert [e for e in bus if e.kind == events.RECOVERY] == []

    def test_one_recovered_round_publishes_one_recompute(self, bus: list[events.Event]) -> None:
        attempts = iter([("partial", {3}), ("done", None)])
        recomputed: list[set[int]] = []
        result = ShuffleRecovery(label="join").run(
            lambda: next(attempts), lambda failed: recomputed.append(failed)
        )
        assert result == "done"
        assert recomputed == [{3}]

        published = _recovery(bus, "recompute")
        assert len(published) == 1
        assert published[0].fields["shuffle"] == "join"
        assert published[0].fields["round"] == 0
        # The label must reach the event, or on a plan with three shuffles a recompute is
        # observable but unattributable.
        assert published[0].name == "join"

    def test_exhausting_the_budget_publishes_give_up(self, bus: list[events.Event]) -> None:
        policy = RecoveryPolicy(max_attempts=2)
        with pytest.raises(ResourceError):
            ShuffleRecovery(policy, label="sort").run(lambda: ("x", {1}), lambda _f: None)

        assert len(_recovery(bus, "give_up")) == 1
        # `run` deliberately does NOT recompute on its final round, so a two-attempt budget
        # yields exactly one recompute. Pinning this keeps that optimization honest.
        assert len(_recovery(bus, "recompute")) == 1

    def test_the_recompute_count_still_agrees_with_the_events(
        self, bus: list[events.Event]
    ) -> None:
        attempts = iter([("a", {1}), ("b", {1}), ("c", None)])
        recovery = ShuffleRecovery(RecoveryPolicy(max_attempts=5), label="window")
        recovery.run(lambda: next(attempts), lambda _f: None)
        assert recovery.recomputes == len(_recovery(bus, "recompute")) == 2


class TestObservabilityCannotBreakRecovery:
    def test_a_sink_that_raises_does_not_fail_the_query(self, bus: list[events.Event]) -> None:
        """A broken dashboard must never turn a recovered query into a failed one.

        `publish` promises this generally; it is worth pinning *here* because this is the
        one publisher on a path that is already handling a failure. A sink raising during
        recovery would convert a survivable worker loss into a query error, which is
        exactly backwards.
        """

        def explode(_event: events.Event) -> None:
            raise RuntimeError("the dashboard is on fire")

        unsubscribe = events.subscribe(explode)
        try:
            attempts = iter([("partial", {1}), ("done", None)])
            result = ShuffleRecovery(label="aggregate").run(lambda: next(attempts), lambda _f: None)
        finally:
            unsubscribe()
        assert result == "done"


class TestObserveFoldsRecovery:
    def test_metrics_counts_by_event_kind(self) -> None:
        from batcher.observe.metrics import metrics_snapshot, reset_metrics, start_metrics

        start_metrics()
        reset_metrics()
        events.publish(events.RECOVERY, event="worker_lost", worker=2)
        events.publish(events.RECOVERY, event="recompute", shuffle="join", round=0)
        events.publish(events.RECOVERY, event="recompute", shuffle="join", round=1)
        snap = metrics_snapshot()
        assert snap["recovery"]["worker_lost"] == 1
        assert snap["recovery"]["recompute"] == 2

    def test_prometheus_exports_a_labelled_series(self) -> None:
        from batcher.observe.metrics import prometheus_text, reset_metrics, start_metrics

        start_metrics()
        reset_metrics()
        events.publish(events.RECOVERY, event="backup_won", slot=1)
        assert 'batcher_recovery_total{event="backup_won"} 1' in prometheus_text()

    def test_the_live_job_view_reports_worker_loss(self) -> None:
        from batcher.observe.inference import InferenceProgress

        progress = InferenceProgress()
        for kind in ("worker_lost", "worker_lost", "recompute"):
            progress.handle(
                events.Event(
                    kind=events.RECOVERY,
                    ts=1.0,
                    wall=1.0,
                    query_id="q1",
                    name="join",
                    fields={"event": kind},
                )
            )
        snapshot = progress.snapshot("q1")
        assert snapshot["recovery"]["workers_lost"] == 2
        assert snapshot["recovery"]["events"] == {"worker_lost": 2, "recompute": 1}
        # And it must reach the human-readable line, since that is where someone asking
        # "why is this slow" actually looks.
        assert "recovering (2 lost)" in progress.render("q1")


def test_every_published_discriminator_is_declared() -> None:
    """The `event` values the engine publishes must all be in `RECOVERY_EVENTS`.

    The vocabulary crosses to the web UI verbatim, so an undeclared value renders as an
    unknown series with no documentation. This is a cheap guard against the constant list
    drifting behind the publishers.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "python" / "batcher"
    published: set[str] = set()
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "events.RECOVERY" not in text:
            continue
        for block in re.findall(r"events\.publish\((.*?)\n\s*\)", text, re.DOTALL):
            if "events.RECOVERY" not in block:
                continue
            match = re.search(r'event="([a-z_]+)"', block)
            if match:
                published.add(match.group(1))
    assert published, "no RECOVERY publishers found — did the call shape change?"
    undeclared = published - set(events.RECOVERY_EVENTS)
    assert not undeclared, f"undeclared RECOVERY event values: {sorted(undeclared)}"
