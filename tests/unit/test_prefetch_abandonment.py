"""`prefetch` must wind down when its consumer walks away, and never truncate silently.

Abandonment is the normal case: a `limit`, a `stop()`, an error upstream, a plain `break` —
and a streaming inference chain wraps the read *and every stage* in a prefetch, so one
abandoned pipeline is several at once.
"""

from __future__ import annotations

import threading
import time

import pytest

from batcher._internal.prefetch import prefetch


def _settle(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_producer_stops_when_the_consumer_stops():
    """The producer parks on a full queue, so with no stop signal it sat there for the life
    of the process — one thread and `depth` buffered items per abandoned stream."""
    produced = []
    finished = threading.Event()

    def endless():
        try:
            i = 0
            while True:
                produced.append(i)
                yield i
                i += 1
        finally:
            finished.set()

    stream = prefetch(endless(), depth=2)
    assert next(stream) == 0
    stream.close()  # what a `break` out of a `for` does

    assert _settle(finished.is_set), "the source generator was never closed"
    settled = len(produced)
    time.sleep(0.3)
    assert len(produced) == settled, "the producer kept running after the consumer left"


def test_the_source_is_closed_so_its_finally_releases_what_it_holds():
    """A broker's `iter_batches` closes its consumer in a `finally`; on the abandonment
    path nothing else runs it."""
    released = []

    def holding():
        try:
            yield 1
            yield 2
            yield 3
        finally:
            released.append("closed")

    stream = prefetch(holding(), depth=1)
    assert next(stream) == 1
    stream.close()
    assert _settle(lambda: released == ["closed"])


def test_a_fully_drained_stream_still_closes_its_source():
    released = []

    def finite():
        try:
            yield from (1, 2, 3)
        finally:
            released.append("closed")

    assert list(prefetch(iter(finite()), depth=2)) == [1, 2, 3]
    assert _settle(lambda: released == ["closed"])


def test_an_exception_reaches_the_consumer_in_order():
    def boom():
        yield 1
        yield 2
        raise RuntimeError("upstream failed")

    stream = prefetch(boom(), depth=4)
    assert [next(stream), next(stream)] == [1, 2]
    with pytest.raises(RuntimeError, match="upstream failed"):
        next(stream)


def test_a_base_exception_is_not_swallowed_into_a_clean_end_of_stream():
    """Catching only `Exception` let a `KeyboardInterrupt` fall through to the end-of-stream
    marker, so the consumer saw a clean finish and the stream was silently truncated."""

    def interrupted():
        yield 1
        raise KeyboardInterrupt

    stream = prefetch(interrupted(), depth=4)
    assert next(stream) == 1
    with pytest.raises(KeyboardInterrupt):
        next(stream)


def test_zero_depth_bypasses_the_thread_entirely():
    def gen():
        yield from (1, 2, 3)

    assert list(prefetch(gen(), depth=0)) == [1, 2, 3]
    assert threading.active_count() >= 1  # nothing was spawned to be counted


def test_an_empty_source_ends_cleanly():
    assert list(prefetch(iter(()), depth=2)) == []


def test_nested_prefetches_all_wind_down_together():
    """`stream_linear_chain` wraps the read *and every stage*, so an abandoned inference
    pipeline is several prefetches deep. Closing the outermost must reach every source."""
    closed: list[str] = []

    def source():
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            closed.append("source")

    def stage(upstream):
        try:
            for item in upstream:
                yield item * 2
        finally:
            closed.append("stage")

    # Built as one expression, which is the shape `stream_linear_chain` produces: each
    # stage's frame is the only reference to the prefetch below it, so closing the outermost
    # drops that frame and the cascade continues. Holding an intermediate in a local — as a
    # test naturally would — pins it and the inner producer never learns it is unwanted.
    chain = prefetch(stage(prefetch(source(), depth=2)), depth=2)
    assert next(chain) == 0
    chain.close()

    assert _settle(lambda: set(closed) == {"stage", "source"}), closed


def test_an_abandoned_stream_leaves_no_producer_thread_behind():
    before = threading.active_count()
    for _ in range(20):
        stream = prefetch(iter(range(1_000_000)), depth=2)
        next(stream)
        stream.close()
    assert _settle(lambda: threading.active_count() <= before + 1), (
        f"threads grew from {before} to {threading.active_count()}"
    )
