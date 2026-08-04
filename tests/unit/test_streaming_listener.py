"""`StreamingQueryListener`: registration, dispatch, and the promise that it cannot
break a query.

The registry and the fire path are pure control plane, so these drive them directly
rather than through a running query — which is where the *contract* lives anyway: a
listener that raises must be logged and skipped, both spellings of each callback must be
dispatched, and a listener registered twice must not receive everything twice.
"""

from __future__ import annotations

import pytest

from batcher.plan.streaming import (
    QueryProgressEvent,
    QueryStartedEvent,
    QueryTerminatedEvent,
    StreamingQueryListener,
    StreamingQueryProgress,
    add_streaming_listener,
    notify_query_progress,
    notify_query_started,
    notify_query_terminated,
    remove_streaming_listener,
    streaming_listeners,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-wide; leave it exactly as it was found."""
    before = streaming_listeners()
    yield
    for listener in streaming_listeners():
        remove_streaming_listener(listener)
    for listener in before:
        add_streaming_listener(listener)


class _Recorder(StreamingQueryListener):
    def __init__(self) -> None:
        self.seen: list[tuple[str, object]] = []

    def on_query_started(self, event):
        self.seen.append(("started", event))

    def on_query_progress(self, event):
        self.seen.append(("progress", event))

    def on_query_terminated(self, event):
        self.seen.append(("terminated", event))


def _progress(batch_id: int = 0) -> StreamingQueryProgress:
    return StreamingQueryProgress(batch_id, 10, 10, 1.0, 0.0)


def test_a_registered_listener_receives_all_three_events():
    listener = _Recorder()
    add_streaming_listener(listener)

    notify_query_started("q", 1.0)
    notify_query_progress("q", _progress())
    notify_query_terminated("q", None)

    kinds = [kind for kind, _ in listener.seen]
    assert kinds == ["started", "progress", "terminated"]
    assert isinstance(listener.seen[0][1], QueryStartedEvent)
    assert isinstance(listener.seen[1][1], QueryProgressEvent)
    assert isinstance(listener.seen[2][1], QueryTerminatedEvent)


def test_registering_the_same_listener_twice_delivers_once():
    listener = _Recorder()
    add_streaming_listener(listener)
    add_streaming_listener(listener)

    notify_query_started("q", 1.0)
    assert len(listener.seen) == 1
    assert streaming_listeners().count(listener) == 1


def test_a_removed_listener_stops_receiving():
    listener = _Recorder()
    add_streaming_listener(listener)
    assert remove_streaming_listener(listener) is True

    notify_query_started("q", 1.0)
    assert listener.seen == []


def test_removing_an_unregistered_listener_says_so_rather_than_raising():
    assert remove_streaming_listener(_Recorder()) is False


def test_the_spark_spelling_is_dispatched_too():
    """A listener ported from PySpark overrides `onQueryProgress`, not the snake_case
    name. Dispatching only one spelling would make it silently never fire."""

    class _Spark(StreamingQueryListener):
        def __init__(self) -> None:
            self.count = 0

        def onQueryProgress(self, event):
            self.count += 1

    listener = _Spark()
    add_streaming_listener(listener)
    notify_query_progress("q", _progress())
    assert listener.count == 1


def test_a_listener_that_raises_does_not_break_the_query_or_the_others():
    """Monitoring that can fail the thing it monitors is worse than no monitoring."""

    class _Broken(StreamingQueryListener):
        def on_query_progress(self, event):
            raise RuntimeError("metrics endpoint down")

    healthy = _Recorder()
    add_streaming_listener(_Broken())
    add_streaming_listener(healthy)

    notify_query_progress("q", _progress())  # must not raise
    assert [kind for kind, _ in healthy.seen] == ["progress"]


def test_a_termination_carries_the_failure_as_a_serializable_string():
    listener = _Recorder()
    add_streaming_listener(listener)
    notify_query_terminated("q", ValueError("broker gone"))

    event = listener.seen[0][1]
    assert event.exception == "ValueError: broker gone"
    assert event.name == "q" and event.id == "q"


def test_a_clean_stop_carries_no_exception():
    listener = _Recorder()
    add_streaming_listener(listener)
    notify_query_terminated("q", None)
    assert listener.seen[0][1].exception is None


def test_firing_with_no_listeners_is_a_no_op():
    notify_query_started("q", 1.0)
    notify_query_progress("q", _progress())
    notify_query_terminated("q", None)
