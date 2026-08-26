"""The bus must survive a broken sink without turning the hot path into a log firehose.

A sink that raises is skipped for that emit and kept, which is right for a transient
failure and wrong for a permanent one. Progress is published per morsel, so a reporter
whose stream has closed raised on every event: one DEBUG record per batch, forever,
attributed to a subsystem with nothing to do with the failure, and a `publish` that stayed
expensive because the sink that can never succeed was still in the tuple.
"""

from __future__ import annotations

import logging

import pytest

from batcher._internal import events

pytestmark = pytest.mark.unit


class _Broken:
    """A sink that always raises, and counts how many times it was asked to."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, event: events.Event) -> None:
        self.calls += 1
        raise ValueError("stream is closed")


class _Flaky:
    """A sink that raises on the first call of every pair — never twice in a row."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, event: events.Event) -> None:
        self.calls += 1
        if self.calls % 2 == 1:
            raise ValueError("transient")


def _emit(n: int) -> None:
    for i in range(n):
        events.publish(events.PROGRESS, query_id="q", rows=i)


def test_a_permanently_broken_sink_is_detached_rather_than_retried_forever():
    broken = _Broken()
    detach = events.subscribe(broken)
    try:
        _emit(50)
    finally:
        detach()
    assert broken.calls == events.MAX_SINK_FAILURES, (
        "the sink must stop being called once it has proven it cannot succeed"
    )


def test_a_flaky_sink_is_kept_because_its_failures_are_not_consecutive():
    """Three strikes keeps a sink that blips and removes one that is gone."""
    flaky = _Flaky()
    detach = events.subscribe(flaky)
    try:
        _emit(20)
    finally:
        detach()
    assert flaky.calls == 20


def test_a_healthy_sink_beside_a_broken_one_keeps_receiving_everything():
    seen: list[events.Event] = []
    broken = _Broken()
    detach_broken = events.subscribe(broken)
    detach_ok = events.subscribe(seen.append)
    try:
        _emit(10)
    finally:
        detach_broken()
        detach_ok()
    assert len(seen) == 10


def test_the_detach_is_reported_once_at_warning_and_names_the_failure(caplog):
    """Losing a whole observability surface mid-run is worth knowing about — once."""
    broken = _Broken()
    detach = events.subscribe(broken)
    with caplog.at_level(logging.WARNING, logger="batcher.observe"):
        try:
            _emit(30)
        finally:
            detach()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "detached" in warnings[0].getMessage()
    fields = getattr(warnings[0], "batcher_fields", {})
    assert fields["error"] == "ValueError"
    assert fields["failures"] == events.MAX_SINK_FAILURES


def test_unsubscribing_clears_the_failure_count_so_a_resubscribe_starts_clean():
    broken = _Broken()
    detach = events.subscribe(broken)
    _emit(1)
    detach()
    detach2 = events.subscribe(broken)
    try:
        _emit(1)
    finally:
        detach2()
    # Two emits, two calls: the first failure did not carry over into the new subscription.
    assert broken.calls == 2


def test_publishing_builds_no_event_when_nothing_is_attached(monkeypatch):
    """The guard on the hot path is a tuple truthiness check and nothing else.

    Asserted by making `Event` explode: if `publish` reached the constructor with no sink
    attached, the per-morsel cost this module promises to be zero would not be.
    """

    def _explode(*_a, **_k):
        raise AssertionError("publish built an Event with no sink attached")

    monkeypatch.setattr(events, "_subscribers", ())
    monkeypatch.setattr(events, "Event", _explode)
    _emit(1000)
