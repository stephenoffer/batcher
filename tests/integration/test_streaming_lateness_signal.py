"""A streaming query must be able to say whether it is keeping up with its trigger.

Throughput says how fast a micro-batch ran. It cannot say whether that was fast *enough*,
because "enough" is the trigger interval — and the progress record did not carry it, so a
query falling hours behind its source looked healthy right up until someone checked the
source lag by hand.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.core.streaming_query import StreamingQueryEngine
from batcher.plan.streaming import StreamingQueryProgress, Trigger


def test_a_batch_inside_its_cadence_is_not_behind():
    p = StreamingQueryProgress(0, 10, 10, duration_ms=25.0, timestamp=0.0)
    assert p.behind_by_ms == 0.0
    assert p.is_behind is False
    assert "behind" not in str(p)


def test_a_batch_that_overran_says_so_in_its_summary():
    """A throughput figure alone reads as healthy; the summary must not hide the overrun."""
    p = StreamingQueryProgress(3, 10, 10, duration_ms=250.0, timestamp=0.0, behind_by_ms=150.0)
    assert p.is_behind is True
    assert "150ms behind" in str(p)


class _Source:
    bounded = True

    def __init__(self, batches):
        self._batches = batches

    def schema(self):
        return self._batches[0].schema

    def row_count(self):
        return sum(b.num_rows for b in self._batches)

    def read(self, projection=None):
        return list(self._batches)

    def iter_batches(self, projection=None):
        yield from self._batches


class _Slow:
    """A processor that takes measurably longer than the trigger interval."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def process(self, batch):
        import time

        time.sleep(self._seconds)
        return [batch]


def _run(trigger, processor, sink=None):
    class _Sink:
        def open(self):
            pass

        def write_batch(self, batch_id, table):
            return None

        def close(self):
            pass

    engine = StreamingQueryEngine(
        name="lateness",
        source=_Source([pa.record_batch({"a": [1, 2]})]),
        sink=sink or _Sink(),
        processor=processor,
        trigger=trigger,
        output_mode="append",
    )
    engine.start()
    assert engine.await_termination(30) is True
    return engine.recent_progress()


@pytest.mark.integration
def test_the_engine_measures_the_overrun_against_the_trigger_interval():
    progress = _run(Trigger.processing_time("0.01 seconds"), _Slow(0.15))
    assert progress and progress[0].is_behind
    # Roughly the excess over the 10ms cadence — asserted as a floor, since a loaded box
    # only ever makes it larger.
    assert progress[0].behind_by_ms >= 100.0


@pytest.mark.integration
def test_a_query_that_keeps_up_reports_zero():
    class _Fast:
        def process(self, batch):
            return [batch]

    progress = _run(Trigger.processing_time("30 seconds"), _Fast())
    assert progress and all(p.behind_by_ms == 0.0 for p in progress)


@pytest.mark.integration
def test_a_trigger_with_no_cadence_is_never_behind():
    """`once` / `available_now` / `continuous` have no interval to be late for."""
    progress = _run(Trigger.available_now(), _Slow(0.05))
    assert progress and all(p.is_behind is False for p in progress)


def test_the_field_defaults_so_an_old_construction_still_works():
    """The record is public and frozen; adding a field must not break a positional build."""
    p = StreamingQueryProgress(1, 2, 3, 4.0, 5.0)
    assert (p.batch_id, p.num_input_rows, p.behind_by_ms) == (1, 2, 0.0)


def test_the_signal_is_reachable_the_way_a_user_gets_it():
    """Users never construct a progress record — they read one off `recent_progress()`."""

    class _Fast:
        def process(self, batch):
            return [batch]

    progress = _run(Trigger.processing_time("30 seconds"), _Fast())
    assert hasattr(progress[0], "behind_by_ms")
    assert hasattr(progress[0], "is_behind")
